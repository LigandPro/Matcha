from __future__ import annotations

from argparse import ArgumentParser
from pathlib import Path

from matcha.utils.repro_baseline import create_baseline_from_git_ref


def main() -> None:
    parser = ArgumentParser(description="Generate reproducibility baseline JSON from a reference git ref.")
    parser.add_argument("--git-ref", required=True, help="Reference branch, tag, or commit to run.")
    parser.add_argument("--receptor", required=True, help="Path to receptor PDB.")
    parser.add_argument("--ligand", required=True, help="Path to ligand SDF.")
    parser.add_argument("--scorer-path", required=True, help="Path to scorer script or GNINA wrapper.")
    parser.add_argument("--output-json", required=True, help="Where to write the baseline JSON.")
    parser.add_argument("--run-name", default="repro_baseline", help="Run name inside the reference worktree.")
    parser.add_argument("--repo-root", default=".", help="Repository root. Defaults to current directory.")
    parser.add_argument(
        "--cuda-visible-devices",
        default=None,
        help="Optional CUDA_VISIBLE_DEVICES value for the reference run.",
    )
    parser.add_argument(
        "--matcha-arg",
        action="append",
        default=[],
        help="Extra argument passed through to `uv run matcha`. Can be repeated.",
    )
    args = parser.parse_args()

    snapshot = create_baseline_from_git_ref(
        repo_root=Path(args.repo_root),
        git_ref=args.git_ref,
        receptor=Path(args.receptor),
        ligand=Path(args.ligand),
        scorer_path=Path(args.scorer_path),
        output_json=Path(args.output_json),
        run_name=args.run_name,
        cuda_visible_devices=args.cuda_visible_devices,
        extra_args=args.matcha_arg,
    )
    print(f"Wrote baseline JSON to {Path(args.output_json).resolve()}")
    print(f"Collected {len(snapshot['scores'])} scores for run {args.run_name}")


if __name__ == "__main__":
    main()
