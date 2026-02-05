#!/bin/bash

# Usage: ./compute_affinity_batch.sh <exp_name> [--config <config_file>] [--paths-config <paths_config_file>] [--receptors-base <path>] [--gnina-script <path>] [--minimize] [--device <device_id>] [dataset1] [dataset2] [dataset3] ...
#   exp_name: Experiment name (required)
#   --config <config_file>: Path to experiment config file (optional)
#   --paths-config <paths_config_file>: Path to paths config file (optional). Set preprocessed_receptors_base there, or use --receptors-base.
#   --receptors-base <path>: Root path to preprocessed protein structures (optional, overrides config)
#   --gnina-script <path>: Path to gnina script (required when running)
#   --minimize: Optional flag to minimize ligands during scoring
#   --device <device_id>: GPU device ID for gnina (default: 0)
#   dataset1, dataset2, ...: List of dataset names to process (optional, defaults to: astex pdbbind posebusters)
#
# Example: ./compute_affinity_batch.sh PIPELINE_stage2_synthAll_2995k
# Example: ./compute_affinity_batch.sh PIPELINE_stage2_synthAll_2995k astex posebusters
# Example: ./compute_affinity_batch.sh PIPELINE_stage2_synthAll_2995k --config configs/default.yaml --paths-config configs/paths/paths.yaml --minimize
# Example: ./compute_affinity_batch.sh PIPELINE_stage2_synthAll_2995k --gnina-script /path/to/gnina.sh --device 1

# Get the directory where this script is located
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
COMPUTE_AFFINITY_SCRIPT="${SCRIPT_DIR}/compute_affinity.sh"

# Default datasets
DEFAULT_DATASETS=("astex" "pdbbind" "posebusters")

# Check if exp_name is provided
if [ $# -lt 1 ]; then
    echo "Error: Missing required argument: exp_name"
    echo "Usage: $0 <exp_name> [--minimize] [--device <device_id>] [dataset1] [dataset2] [dataset3] ..."
    echo "Example: $0 PIPELINE_stage2_synthAll_2995k"
    echo "Example: $0 PIPELINE_stage2_synthAll_2995k astex posebusters"
    echo "Example: $0 PIPELINE_stage2_synthAll_2995k --minimize"
    echo "Example: $0 PIPELINE_stage2_synthAll_2995k --device 1"
    exit 1
fi

# Get exp_name (first argument)
exp_name="$1"
shift  # Remove exp_name from arguments

# Initialize defaults
minimize_flag=""
device_id="0"
config_file=""
paths_config_file=""
receptors_base=""
gnina_script=""

# Parse flags (can appear in any order)
while [[ $# -gt 0 ]]; do
    case $1 in
        --config)
            if [ -z "$2" ]; then
                echo "Error: --config requires a file path"
                exit 1
            fi
            config_file="$2"
            shift 2
            ;;
        --paths-config)
            if [ -z "$2" ]; then
                echo "Error: --paths-config requires a file path"
                exit 1
            fi
            paths_config_file="$2"
            shift 2
            ;;
        --receptors-base)
            if [ -z "$2" ]; then
                echo "Error: --receptors-base requires a path"
                exit 1
            fi
            receptors_base="$2"
            shift 2
            ;;
        --gnina-script)
            if [ -z "$2" ]; then
                echo "Error: --gnina-script requires a file path"
                exit 1
            fi
            gnina_script="$2"
            shift 2
            ;;
        --minimize)
            minimize_flag="minimize"
            shift
            ;;
        --device)
            if [ -z "$2" ]; then
                echo "Error: --device requires a device ID"
                exit 1
            fi
            device_id="$2"
            shift 2
            ;;
        --*)
            echo "Error: Unknown option: $1"
            echo "Usage: $0 <exp_name> [--config <config_file>] [--paths-config <paths_config_file>] [--gnina-script <path>] [--minimize] [--device <device_id>] [dataset1] [dataset2] [dataset3] ..."
            exit 1
            ;;
        *)
            # Not a flag, break out of loop to collect datasets
            break
            ;;
    esac
done

# Get list of datasets (use defaults if none provided)
if [ $# -eq 0 ]; then
    datasets=("${DEFAULT_DATASETS[@]}")
else
    datasets=("$@")
fi
total_datasets=${#datasets[@]}

echo "=========================================="
echo "Batch Affinity Computation"
echo "=========================================="
echo "Experiment: $exp_name"
if [ -n "$minimize_flag" ]; then
    echo "Mode: Minimize ligands"
fi
echo "GPU Device ID: $device_id"
echo "Datasets to process: ${total_datasets}"
echo "Datasets: ${datasets[*]}"
echo "=========================================="
echo ""

# Process each dataset
dataset_num=0
for dataset_name in "${datasets[@]}"; do
    ((dataset_num++))
    echo ""
    echo "=========================================="
    echo "[$dataset_num/$total_datasets] Processing dataset: $dataset_name"
    echo "=========================================="
    echo ""
    
    # Run the compute_affinity script for this dataset
    # Build command with flags
    cmd_args=("$exp_name" "$dataset_name")
    if [ -n "$config_file" ]; then
        cmd_args+=("--config" "$config_file")
    fi
    if [ -n "$paths_config_file" ]; then
        cmd_args+=("--paths-config" "$paths_config_file")
    fi
    if [ -n "$receptors_base" ]; then
        cmd_args+=("--receptors-base" "$receptors_base")
    fi
    if [ -n "$gnina_script" ]; then
        cmd_args+=("--gnina-script" "$gnina_script")
    fi
    if [ -n "$minimize_flag" ]; then
        cmd_args+=("--minimize")
    fi
    if [ -n "$device_id" ]; then
        cmd_args+=("--device" "$device_id")
    fi
    
    if bash "$COMPUTE_AFFINITY_SCRIPT" "${cmd_args[@]}"; then
        echo ""
        echo "✓ Successfully completed dataset: $dataset_name"
    else
        echo ""
        echo "✗ Error processing dataset: $dataset_name"
        echo "Continuing with next dataset..."
    fi
    
    echo ""
done

echo "=========================================="
echo "Batch processing complete!"
echo "Processed $dataset_num/$total_datasets datasets"
echo "=========================================="
