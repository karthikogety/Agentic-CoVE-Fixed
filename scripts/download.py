
from pathlib import Path
import argparse, gzip, shutil, time, os
from src.utils.config import load_config
from src.data.registry import get_dataset_paths

def fetch_one(url: str, dest: Path, timeout=60) -> bool:
    """Download a single file using the robust wget command."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    try:
        print(f"↓ Trying with wget: {url}")
        # Use wget with the --no-check-certificate flag
        if url.endswith(".gz"):
            dest_gz = dest.with_suffix(dest.suffix + ".gz")
            command = f"wget --no-check-certificate -O {dest_gz} --timeout={timeout} --tries=3 '{url}'"
            os.system(command)

            print(f"↪ Decompressing → {dest}")
            with gzip.open(dest_gz, "rb") as fin, open(dest, "wb") as fout:
                shutil.copyfileobj(fin, fout)
            dest_gz.unlink()
        else:
            command = f"wget --no-check-certificate -O {dest} --timeout={timeout} --tries=3 '{url}'"
            os.system(command)

        if dest.exists() and dest.stat().st_size > 0:
            print(f"✅ Saved: {dest}")
            return True
        else:
            # Clean up failed download artifact if it exists
            if dest.exists(): dest.unlink()
            print(f"❌ Download failed or created an empty file for {url}")
            return False
    except Exception as e:
        print(f"An unexpected error occurred with {url}: {e}")
        return False

def fetch_any(urls: list[str], dest: Path) -> None:
    for i, url in enumerate(urls, 1):
        print(f"[{i}/{len(urls)}] Download attempt")
        if fetch_one(url, dest):
            return
        time.sleep(1)
    raise RuntimeError(f"All mirrors failed for {dest.name}")

def main(cfg_path: str):
    cfg = load_config(cfg_path)
    reg_key = cfg.get("registry_key", cfg.get("dataset"))
    paths = get_dataset_paths(reg_key)
    raw_dir = Path(paths["raw"])

    for item in cfg.get("download_urls", []):
        name = item["name"]
        target = raw_dir / item["target"]
        mirrors = item.get("targets") or [item.get("url")]
        print(f"\n=== {name} ===")
        fetch_any(mirrors, target)

    print(f"\n✅ Done. Raw files in: {raw_dir}")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--cfg", required=True)
    args = ap.parse_args()
    main(args.cfg)
