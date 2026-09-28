"""CLI to infer brain small lesions from fastMRI FLAIR h5 files."""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from anatobind.infer.brain import run  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description="Infer brain small lesions from fastMRI FLAIR h5 files")
    parser.add_argument("--h5", required=True, help="Path to fastMRI h5 file")
    parser.add_argument("--out", required=True, help="Output directory path")
    parser.add_argument("--folds", nargs="+", type=int, default=[0, 1, 2, 3, 4], help="Folds to use (default: 0 1 2 3 4)")
    parser.add_argument("--gpu", type=int, default=0, help="GPU index (default: 0)")
    parser.add_argument("--config", default="2d", help="nnU-Net configuration (default: 2d)")
    args = parser.parse_args()

    rows = run(args.h5, args.out, args.folds, args.gpu, config=args.config)
    print(f"Found {len(rows)} lesions. Output: {args.out}")


if __name__ == "__main__":
    main()
