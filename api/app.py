
from typing import Optional, Literal, Dict, Any
from pydantic import BaseModel, Field
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import os
import sys

# Ensure we can import our src package
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from src.service.recommender import recommend_for_user, RecommendConfig, FusionWeights

class RecRequest(BaseModel):
    dataset: str = Field(default="beauty", description="Dataset key (e.g., 'beauty').")
    user_id: str = Field(description="User ID as seen in processed reviews.")
    k: int = Field(default=10, ge=1, le=1000)
    fusion: Literal["concat", "weighted"] = "concat"
    w_text: float = 1.0
    w_image: float = 1.0
    w_meta: float = 0.0
    use_faiss: bool = True
    faiss_name: Optional[str] = Field(default=None, description="Name of the FAISS index.")
    exclude_seen: bool = True

class RecItem(BaseModel):
    item_id: str
    score: float
    brand: Optional[str] = None
    price: Optional[float] = None
    categories: Optional[list[str]] = None
    image_url: Optional[str] = None

class RecResponse(BaseModel):
    recommendations: list[RecItem]

# Use FastAPI with its default, reliable docs
app = FastAPI(
    title="MMR Recommender API",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_methods=["*"], allow_headers=["*"],
)

@app.post("/recommend", response_model=RecResponse)
def recommend(req: RecRequest):
    cfg = RecommendConfig(
        dataset=req.dataset, user_id=req.user_id, k=req.k, fusion=req.fusion,
        weights=FusionWeights(text=req.w_text, image=req.w_image, meta=req.w_meta),
        use_faiss=req.use_faiss, faiss_name=req.faiss_name, exclude_seen=req.exclude_seen
    )
    result = recommend_for_user(cfg)
    return RecResponse(recommendations=result.get("recommendations", []))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api.app:app", host="0.0.0.0", port=8000, reload=True)
