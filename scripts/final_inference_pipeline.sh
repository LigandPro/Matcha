#!/bin/bash

# Usage: ./final_inference_pipeline.sh -n <exp_name> -c <exp_config> -p <paths_config> [-d <device_id>] [-s <n_samples>] [-g <gnina_script>] [--compute_final_metrics]
#   -n: Experiment name (required)
#   -c: Experiment config file (required)
#   -p: Paths config file (required)
#   -d: GPU device ID (optional, default: 0)
#   -s: Number of samples (optional, default: 40)
#   -g: Path to GNINA script (required for GNINA affinity steps; no default)
#   --compute_final_metrics: If set, run step 5 (compute metrics from best SDF predictions). Default: false
#
# Example:
#   ./final_inference_pipeline.sh -n my_experiment -c configs/default.yaml -p configs/paths/paths.yaml -d 0 -s 20 -g /path/to/gnina.sh --compute_final_metrics

# Default values
device_id="0"
n_samples="40"
gnina_script=""
compute_final_metrics="false"

# Preprocess args to support --compute_final_metrics (getopts only handles single-letter options)
args=()
for arg in "$@"; do
  if [ "$arg" = "--compute_final_metrics" ]; then
    compute_final_metrics="true"
  else
    args+=("$arg")
  fi
done
set -- "${args[@]}"

# Parse arguments
while getopts "n:c:p:d:s:g:h" opt; do
  case $opt in
    n)
      exp_name="$OPTARG"
      ;;
    c)
      exp_config="$OPTARG"
      ;;
    p)
      paths_config="$OPTARG"
      ;;
    d)
      device_id="$OPTARG"
      ;;
    s)
      n_samples="$OPTARG"
      ;;
    g)
      gnina_script="$OPTARG"
      ;;
    h)
      echo "Usage: $0 -n <exp_name> -c <exp_config> -p <paths_config> [-d <device_id>] [-s <n_samples>] [-g <gnina_script>] [--compute_final_metrics]"
      echo ""
      echo "Options:"
      echo "  -n  Experiment name (required)"
      echo "  -c  Experiment config file (required)"
      echo "  -p  Paths config file (required)"
      echo "  -d  GPU device ID (optional, default: 0)"
      echo "  -s  Number of samples (optional, default: 40)"
      echo "  -g  Path to GNINA script (required for GNINA steps)"
      echo "  --compute_final_metrics  Run step 5: compute metrics from best SDF predictions (default: false)"
      echo "  -h  Show this help message"
      echo ""
      echo "Example:"
      echo "  $0 -n my_experiment -c configs/default.yaml -p configs/paths/paths.yaml -d 0 -s 20 -g /path/to/gnina.sh"
      echo "  $0 -n my_experiment -c configs/default.yaml -p configs/paths/paths.yaml -g /path/to/gnina.sh"
      echo "  $0 -n my_experiment -c configs/default.yaml -p configs/paths/paths.yaml -g /path/to/gnina.sh --compute_final_metrics"
      exit 0
      ;;
    \?)
      echo "Invalid option: -$OPTARG" >&2
      echo "Use -h for help"
      exit 1
      ;;
    :)
      echo "Option -$OPTARG requires an argument." >&2
      exit 1
      ;;
  esac
done

# Check required arguments
if [ -z "$exp_name" ] || [ -z "$exp_config" ] || [ -z "$paths_config" ]; then
    echo "Error: Missing required arguments"
    echo "Usage: $0 -n <exp_name> -c <exp_config> -p <paths_config> [-d <device_id>] [-s <n_samples>] [-g <gnina_script>] [--compute_final_metrics]"
    echo "Use -h for help"
    exit 1
fi

echo "=========================================="
echo "Final Inference Pipeline"
echo "=========================================="
echo "Experiment name: $exp_name"
echo "Experiment config: $exp_config"
echo "Paths config: $paths_config"
echo "Device ID: $device_id"
echo "Number of samples: $n_samples"
if [ -n "$gnina_script" ]; then
    echo "GNINA script: $gnina_script"
fi
echo "Compute final metrics: $compute_final_metrics"
echo "=========================================="
echo ""

# Step 1: Run full inference
echo "Step 1: Running full inference..."
CUDA_VISIBLE_DEVICES=$device_id python scripts/full_inference.py -c $exp_config -p $paths_config -n $exp_name --merge-stages

# Step 2: Compute GNINA affinity with minimization
echo "Step 2: Computing GNINA affinity with minimization..."
# Extract dataset names from config and trim _conf suffix
datasets=$(python -c "
from omegaconf import OmegaConf
conf = OmegaConf.load('$exp_config')
paths_conf = OmegaConf.load('$paths_config')
conf = OmegaConf.merge(conf, paths_conf)
dataset_names = [name.replace('_conf', '') for name in conf.test_dataset_types]
print(' '.join(dataset_names))
")
# Build gnina command
gnina_cmd="bash scripts/gnina/compute_affinity_batch.sh $exp_name --config $exp_config --paths-config $paths_config"
if [ -n "$gnina_script" ]; then
    gnina_cmd="$gnina_cmd --gnina-script $gnina_script"
fi
gnina_cmd="$gnina_cmd --minimize --device $device_id $datasets"
eval $gnina_cmd

# Step 3: Compute fast filters from SDF
echo "Step 3: Computing fast filters from SDF..."
CUDA_VISIBLE_DEVICES=$device_id python scripts/fast_filters_from_sdf.py -c $exp_config -p $paths_config -n $exp_name --n_samples $n_samples

# Step 4: Select top GNINA poses
echo "Step 4: Selecting top GNINA poses..."
python scripts/gnina/select_top_gnina_poses.py -p $paths_config -n $exp_name --n-samples $n_samples

# Step 5: Compute metrics from best SDF predictions (only if --compute_final_metrics)
if [ "$compute_final_metrics" = "true" ]; then
    echo "Step 5: Computing metrics from best SDF predictions..."
    python scripts/compute_metrics_from_sdf.py -p $paths_config -n $exp_name --prediction-type best_minimized_predictions_${n_samples}_filtered
else
    echo "Step 5: Skipped (use --compute_final_metrics to run)"
fi

echo ""
echo "=========================================="
echo "Pipeline completed successfully!"
echo "Results saved to inference_results_folder/$exp_name"
echo "=========================================="
