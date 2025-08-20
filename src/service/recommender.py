# src/service/recommender.py (UPDATED with New User Logic)
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional, Tuple, List, Dict
import pandas as pd
from pathlib import Path
import numpy as np

from src.utils.paths import get_processed_path
from src.models.fusion import l2norm

try:
    import faiss
    _FAISS_OK = True
except ImportError:
    _FAISS_OK = False

# --- Dataclasses & configuration ---
@dataclass
class FusionWeights:
    text: float = 1.0
    image: float = 1.0
    meta: float = 0.0

@dataclass
class RecommendConfig:
    dataset: str
    user_id: str
    k: int = 10
    fusion: str = "concat"
    weights: FusionWeights = field(default_factory=FusionWeights)
    use_faiss: bool = False
    faiss_name: Optional[str] = None
    exclude_seen: bool = True

__all__ = ["FusionWeights", "RecommendConfig", "recommend_for_user"]

# --- Helper Functions (loading, scoring, etc.) ---
def _load_vectors(fp: Path, id_col: str) -> Tuple[pd.DataFrame, np.ndarray, List[str]]:
    df = pd.read_parquet(fp)
    mat = np.vstack(df["vector"].to_numpy()).astype(np.float32)
    ids = df[id_col].astype(str).tolist()
    return df, mat, ids

def _load_user_vectors(proc: Path) -> Tuple[pd.DataFrame, np.ndarray, List[str]]:
    fp = proc / "user_text_emb.parquet"
    df = pd.read_parquet(fp)
    U = np.vstack(df["vector"].to_numpy()).astype(np.float32)
    users = df["user_id"].astype(str).tolist()
    return df, U, users

def _read_items_table(proc: Path) -> pd.DataFrame:
    fp = proc / "items_with_meta.parquet"
    return pd.read_parquet(fp)

def _seen_items(proc: Path, user_id: str) -> set[str]:
    fp = proc / "reviews.parquet"
    if not fp.exists(): return set()
    df = pd.read_parquet(fp, columns=["user_id", "item_id"])
    return set(df.loc[df["user_id"].astype(str) == str(user_id), "item_id"].astype(str).tolist())

def _faiss_topk(index_fp: Path, ids_fp: Path, queries: np.ndarray, k: int) -> Tuple[np.ndarray, List[str]]:
    if not _FAISS_OK: raise RuntimeError("FAISS not available.")
    index = faiss.read_index(str(index_fp))
    D, I = index.search(queries.astype(np.float32), k)
    item_ids = np.load(ids_fp, allow_pickle=True).tolist()
    id_list = [str(x) for x in item_ids]
    flat_ids: List[str] = [id_list[i] for i in I[0]]
    return D[0], flat_ids

# --- NEW FUNCTION FOR NEW USERS ---
def _get_popular_items(proc: Path, k: int) -> Dict:
    """Gets the top K most popular items based on review count."""
    print("INFO: Generating popularity-based recommendations for a new user.")
    
    # 1. Count reviews for each item
    reviews_df = pd.read_parquet(proc / "reviews.parquet", columns=["item_id"])
    popularity = reviews_df.groupby("item_id").size().sort_values(ascending=False)
    
    # 2. Get top K item IDs
    top_k_ids = popularity.head(k).index.tolist()
    
    # 3. Get metadata for these items
    items_df = _read_items_table(proc)
    top_k_items = items_df[items_df['item_id'].isin(top_k_ids)].to_dict(orient="records")
    
    # 4. Add a dummy score and format for the UI
    for item in top_k_items:
        item['score'] = 1.0  # Assign a default score
    
    return {"recommendations": top_k_items}


def recommend_for_user(cfg: RecommendConfig) -> Dict:
    proc = get_processed_path(cfg.dataset)
    _, U, users = _load_user_vectors(proc)

    # --- UPDATED LOGIC: Check if user is new ---
    if cfg.user_id not in users:
        # If the user is new, call the popularity function and return its result
        popular_recs = _get_popular_items(proc, cfg.k)
        # We also add some metadata to the response for clarity
        popular_recs.update({
            "dataset": cfg.dataset, "user_id": cfg.user_id, "fusion": "popularity",
            "weights": {}, "k": cfg.k, "exclude_seen": False, "use_faiss": False, "faiss_name": None
        })
        return popular_recs

    # --- Existing User Logic (unchanged from here) ---
    print(f"INFO: Generating personalized recommendations for existing user: {cfg.user_id}")
    u_idx = users.index(cfg.user_id)
    u = U[u_idx : u_idx + 1]
    items_df = _read_items_table(proc)
    
    # (The rest of the personalized recommendation logic...)
    items_order = _load_vectors(proc / "item_text_emb.parquet", "item_id")[2]
    id2pos = {iid: i for i, iid in enumerate(items_order)}
    
    mods = {}
    fpt = proc / "item_text_emb.parquet"; fpi = proc / "item_image_emb.parquet"; fpm = proc / "item_meta_emb.parquet"
    if fpt.exists(): mods["text"] = l2norm(_load_vectors(fpt, "item_id")[1])
    if fpi.exists(): mods["image"] = l2norm(_load_vectors(fpi, "item_id")[1])
    if fpm.exists(): mods["meta"] = l2norm(_load_vectors(fpm, "item_id")[1])

    mats, wts = [], []
    if "text" in mods: mats.append(mods["text"]); wts.append(cfg.weights.text)
    if "image" in mods: mats.append(mods["image"]); wts.append(cfg.weights.image)
    if "meta" in mods: mats.append(mods["meta"]); wts.append(cfg.weights.meta)

    Vf = l2norm(np.hstack([w * m for w, m in zip(wts, mats)])) if cfg.fusion == "concat" else None

    seen = _seen_items(proc, cfg.user_id)
    u_img_meta_parts = []
    for mod_name in ["image", "meta"]:
        if mod_name in mods:
            take = [id2pos[s] for s in seen if s in id2pos]
            u_mod_vec = l2norm(np.mean(mods[mod_name][take, :], axis=0, keepdims=True)) if take else np.zeros((1, mods[mod_name].shape[1]))
            u_img_meta_parts.append(u_mod_vec)
    
    u_parts = [u] + u_img_meta_parts
    uf = l2norm(np.hstack([w * m for w, m in zip(wts, u_parts)]))

    scores, top_ids = _faiss_topk(
        proc / "index" / f"items_{cfg.faiss_name}.faiss",
        proc / "index" / f"items_{cfg.faiss_name}.npy",
        uf, cfg.k + 200
    )

    filtered = [(iid, float(scores[i])) for i, iid in enumerate(top_ids) if iid not in seen][:cfg.k]
    top_ids, top_scores = zip(*filtered) if filtered else ([], [])
    
    meta_map = items_df.set_index("item_id").to_dict(orient="index")
    recs = [{**{"item_id": iid, "score": sc}, **meta_map.get(iid, {})} for iid, sc in zip(top_ids, top_scores)]

    return {
        "dataset": cfg.dataset, "user_id": cfg.user_id, "fusion": cfg.fusion,
        "weights": {"text": cfg.weights.text, "image": cfg.weights.image, "meta": cfg.weights.meta},
        "k": cfg.k, "exclude_seen": cfg.exclude_seen, "use_faiss": cfg.use_faiss,
        "faiss_name": cfg.faiss_name, "recommendations": recs,
    }
