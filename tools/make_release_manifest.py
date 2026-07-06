#!/usr/bin/env python3
"""Generate build/deploy/MODEL_VERSION.txt and release-manifest.json.

Fills the packaging gap noted in v1.2.x: portable.nsi requires
MODEL_VERSION.txt but nothing generated it. Run after build + export,
before makensis.

  python3 tools/make_release_manifest.py \
      --release v1.3.0 --model-name bliss-d12-mem-c20-v2a \
      --steps 37600 --val-bpb 0.817234 \
      --training "curated c20 v2 mixture: multi-turn recall, Notes, Context grounding, reasoning" \
      [--build build] [--deploy build/deploy]
"""
import argparse
import datetime
import hashlib
import json
from pathlib import Path

PAYLOAD_BUILD = ["NC_RUN.EXE", "NC_RUN_SSE2.EXE", "NC_RUN_SSE3.EXE", "XPCHAT.EXE"]
PAYLOAD_DEPLOY = ["MODEL.NCB", "TOKENIZER.NCT"]


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--release", required=True)
    ap.add_argument("--model-name", required=True)
    ap.add_argument("--steps", type=int, required=True)
    ap.add_argument("--val-bpb", type=float, required=True)
    ap.add_argument("--training", required=True)
    ap.add_argument("--build", default="build")
    ap.add_argument("--deploy", default="build/deploy")
    args = ap.parse_args()

    root = Path(__file__).resolve().parent.parent
    build = root / args.build
    deploy = root / args.deploy
    deploy.mkdir(parents=True, exist_ok=True)

    files = {}
    for name in PAYLOAD_BUILD:
        p = build / name
        files[name] = {"bytes": p.stat().st_size, "sha256": sha256(p)}
    for name in PAYLOAD_DEPLOY:
        p = deploy / name
        files[name] = {"bytes": p.stat().st_size, "sha256": sha256(p)}

    today = datetime.date.today().isoformat()
    mv = (
        f"Model: {args.model_name}\n"
        f"Release: {args.release}\n"
        "Architecture: d12, 12 layers, 768 embedding, 6 heads, int8 NCB export\n"
        f"Training: {args.training}\n"
        f"Training steps: {args.steps:,}\n"
        f"Final validation bpb: {args.val_bpb}\n"
        f"Exported: {today}\n"
        "Runtime files:\n"
        f"  MODEL.NCB sha256 {files['MODEL.NCB']['sha256']}\n"
        f"  TOKENIZER.NCT sha256 {files['TOKENIZER.NCT']['sha256']}\n"
    )
    (deploy / "MODEL_VERSION.txt").write_text(mv, encoding="ascii")
    p = deploy / "MODEL_VERSION.txt"
    files["MODEL_VERSION.txt"] = {"bytes": p.stat().st_size, "sha256": sha256(p)}

    manifest = dict(sorted(files.items()))
    manifest["compatibility"] = {
        "backend_selection": "XPCHAT detects SSE3 at runtime; launches NC_RUN_SSE3.EXE on SSE3 CPUs, otherwise NC_RUN_SSE2.EXE for Pentium M/SSE2 systems."
    }
    manifest["files"] = {k: v["sha256"] for k, v in files.items()}
    manifest["final_validation_bpb"] = args.val_bpb
    manifest["model_architecture"] = "d12"
    manifest["model_name"] = args.model_name
    manifest["release"] = args.release
    manifest["training_steps"] = args.steps
    (deploy / "release-manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="ascii")
    print(f"wrote {deploy / 'MODEL_VERSION.txt'}")
    print(f"wrote {deploy / 'release-manifest.json'}")


if __name__ == "__main__":
    main()
