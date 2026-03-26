import argparse
import json
from pathlib import Path

import numpy as np


STAGE_FILES = (
    "stage1_astex_conf.npy",
    "stage2_astex_conf.npy",
    "stage3_astex_conf.npy",
)

FINAL_FILES = (
    "astex_conf_final_preds_1stage.npy",
    "astex_conf_final_preds_2stage.npy",
    "astex_conf_final_preds_3stage.npy",
    "astex_conf_final_preds.npy",
)


def compare_numeric(left, right, path):
    left_arr = np.asarray(left)
    right_arr = np.asarray(right)
    if left_arr.shape != right_arr.shape:
        raise AssertionError(f"{path}: shape mismatch {left_arr.shape} != {right_arr.shape}")
    if not np.allclose(left_arr, right_arr, rtol=1e-5, atol=1e-6, equal_nan=True):
        raise AssertionError(f"{path}: numeric values differ")


def compare_stage_metrics(reference_path: Path, candidate_path: Path):
    reference = np.load(reference_path, allow_pickle=True)[0]
    candidate = np.load(candidate_path, allow_pickle=True)[0]

    if set(reference.keys()) != set(candidate.keys()):
        raise AssertionError(f"{candidate_path.name}: metric keys differ")

    for complex_name in sorted(reference.keys()):
        ref_preds = reference[complex_name]
        cand_preds = candidate[complex_name]
        if len(ref_preds) != len(cand_preds):
            raise AssertionError(
                f"{candidate_path.name}:{complex_name}: prediction count differs "
                f"{len(ref_preds)} != {len(cand_preds)}"
            )
        for pred_idx, (ref_pred, cand_pred) in enumerate(zip(ref_preds, cand_preds, strict=True)):
            if set(ref_pred.keys()) != set(cand_pred.keys()):
                raise AssertionError(
                    f"{candidate_path.name}:{complex_name}:{pred_idx}: metric fields differ"
                )
            for key in sorted(ref_pred.keys()):
                ref_value = ref_pred[key]
                cand_value = cand_pred[key]
                if isinstance(ref_value, np.ndarray):
                    compare_numeric(ref_value, cand_value, f"{candidate_path.name}:{complex_name}:{pred_idx}:{key}")
                elif isinstance(ref_value, (float, int, np.floating, np.integer)):
                    compare_numeric(ref_value, cand_value, f"{candidate_path.name}:{complex_name}:{pred_idx}:{key}")
                else:
                    if ref_value != cand_value:
                        raise AssertionError(
                            f"{candidate_path.name}:{complex_name}:{pred_idx}:{key}: values differ"
                        )


def compare_final_predictions(reference_path: Path, candidate_path: Path):
    reference = np.load(reference_path, allow_pickle=True)[0]
    candidate = np.load(candidate_path, allow_pickle=True)[0]

    if set(reference.keys()) != set(candidate.keys()):
        raise AssertionError(f"{candidate_path.name}: prediction keys differ")

    for uid in sorted(reference.keys()):
        ref_samples = reference[uid]["sample_metrics"]
        cand_samples = candidate[uid]["sample_metrics"]
        if len(ref_samples) != len(cand_samples):
            raise AssertionError(
                f"{candidate_path.name}:{uid}: sample count differs {len(ref_samples)} != {len(cand_samples)}"
            )
        for sample_idx, (ref_sample, cand_sample) in enumerate(zip(ref_samples, cand_samples, strict=True)):
            compare_numeric(
                ref_sample["pred_pos"],
                cand_sample["pred_pos"],
                f"{candidate_path.name}:{uid}:{sample_idx}:pred_pos",
            )
            compare_numeric(
                ref_sample["error_estimate_0"],
                cand_sample["error_estimate_0"],
                f"{candidate_path.name}:{uid}:{sample_idx}:error_estimate_0",
            )


def compare_config(reference_path: Path, candidate_path: Path):
    reference = json.loads(reference_path.read_text(encoding="utf-8"))
    candidate = json.loads(candidate_path.read_text(encoding="utf-8"))
    if reference != candidate:
        raise AssertionError("config.json differs")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--reference-run", required=True)
    parser.add_argument("--candidate-run", required=True)
    parser.add_argument("--results-root", required=True)
    args = parser.parse_args()

    results_root = Path(args.results_root)
    reference_dir = results_root / args.reference_run
    candidate_dir = results_root / args.candidate_run

    if not reference_dir.exists():
        raise FileNotFoundError(f"Reference run does not exist: {reference_dir}")
    if not candidate_dir.exists():
        raise FileNotFoundError(f"Candidate run does not exist: {candidate_dir}")

    compare_config(reference_dir / "config.json", candidate_dir / "config.json")

    for file_name in STAGE_FILES:
        compare_stage_metrics(reference_dir / file_name, candidate_dir / file_name)

    for file_name in FINAL_FILES:
        compare_final_predictions(reference_dir / file_name, candidate_dir / file_name)

    print("parity_ok")


if __name__ == "__main__":
    main()
