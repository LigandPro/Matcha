#!/bin/bash

# Usage: ./compute_affinity.sh [exp_name] [dataset_name] [--config <config_file>] [--paths-config <paths_config_file>] [--receptors-base <path>] [--gnina-script <path>] [--minimize] [--device <device_id>]
#   exp_name: Experiment name (default: PIPELINE_stage2_synthAll_2995k)
#   dataset_name: Dataset name (default: astex)
#   --config <config_file>: Path to experiment config file (optional)
#   --paths-config <paths_config_file>: Path to paths config file (optional). Must set preprocessed_receptors_base there, or use --receptors-base.
#   --receptors-base <path>: Root path to preprocessed protein structures (overrides config). Required if paths-config is not provided.
#   --gnina-script <path>: Path to gnina script (optional)
#   --minimize: Optional flag to minimize ligands and save to minimized_sdf_predictions
#   --device <device_id>: GPU device ID for gnina (default: 0)
#
# Example: ./compute_affinity.sh PIPELINE_stage2_synthAll_2995k astex --config configs/default.yaml --paths-config configs/paths/paths.yaml
# Example: ./compute_affinity.sh PIPELINE_stage2_synthAll_2995k astex --receptors-base /path/to/preprocessed_proteins
# Example: ./compute_affinity.sh PIPELINE_stage2_synthAll_2995k astex --gnina-script /path/to/gnina.sh --minimize --device 1

# Parse command-line arguments
# Get positional arguments (exp_name and dataset_name)
exp_name="${1:-PIPELINE_stage2_synthAll_2995k}"
dataset_name="${2:-astex}"

# Remove positional arguments, handling cases where they might not be provided
if [ $# -ge 2 ]; then
    shift 2
elif [ $# -ge 1 ]; then
    shift 1
fi

# Initialize defaults
minimize=""
device_id="0"
config_file=""
paths_config_file=""
receptors_base=""
gnina_script=""

# Parse flags
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
            minimize="minimize"
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
        *)
            echo "Error: Unknown option: $1"
            echo "Usage: $0 [exp_name] [dataset_name] [--config <config_file>] [--paths-config <paths_config_file>] [--receptors-base <path>] [--gnina-script <path>] [--minimize] [--device <device_id>]"
            exit 1
            ;;
    esac
done

# Directories configuration
# Parse inference_results_folder and preprocessed_receptors_base from config if provided
if [ -n "$config_file" ] && [ -n "$paths_config_file" ]; then
    INFERENCE_RESULTS_BASE=$(python -c "
from omegaconf import OmegaConf
conf = OmegaConf.load('$config_file')
paths_conf = OmegaConf.load('$paths_config_file')
conf = OmegaConf.merge(conf, paths_conf)
print(conf.inference_results_folder)
")
    RESULTS_FOLDER="${INFERENCE_RESULTS_BASE}/${exp_name}"
    if [ -z "$receptors_base" ] && python -c "
from omegaconf import OmegaConf
conf = OmegaConf.load('$config_file')
paths_conf = OmegaConf.load('$paths_config_file')
conf = OmegaConf.merge(conf, paths_conf)
exit(0 if getattr(conf, 'preprocessed_receptors_base', None) not in (None, '', '<path_to_preprocessed_proteins>') else 1)
" 2>/dev/null; then
        receptors_base=$(python -c "
from omegaconf import OmegaConf
conf = OmegaConf.load('$config_file')
paths_conf = OmegaConf.load('$paths_config_file')
conf = OmegaConf.merge(conf, paths_conf)
print(conf.preprocessed_receptors_base)
")
    fi
fi

if [ -z "$receptors_base" ]; then
    echo "Error: Preprocessed receptors path is required. Set preprocessed_receptors_base in paths config or pass --receptors-base <path>."
    exit 1
fi

RECEPTORS_BASE="$receptors_base"

if [ -z "$RESULTS_FOLDER" ]; then
    echo "Error: inference_results_folder is required. Provide --config and --paths-config."
    exit 1
fi

if [ -z "$gnina_script" ]; then
    echo "Error: GNINA script is required. Pass --gnina-script <path>."
    exit 1
fi

LIGANDS_DIR="${RESULTS_FOLDER}/${dataset_name}_conf/sdf_predictions"

# Set output file and minimized directory based on minimize flag
if [ "$minimize" = "minimize" ]; then
    OUTPUT_FILE="${RESULTS_FOLDER}/${dataset_name}_affinity_results_minimized.csv"
    MINIMIZED_LIGANDS_DIR="${RESULTS_FOLDER}/${dataset_name}_conf/minimized_sdf_predictions"
    mkdir -p "$MINIMIZED_LIGANDS_DIR"
else
    OUTPUT_FILE="${RESULTS_FOLDER}/${dataset_name}_affinity_results.csv"
    OUTPUT_LIGANDS_DIR="${RESULTS_FOLDER}/${dataset_name}_conf/base_sdf_predictions"
    mkdir -p "$OUTPUT_LIGANDS_DIR"
fi


# Function to count molecules in an SDF file
count_molecules_in_sdf() {
    local sdf_file="$1"
    # Count the number of "$$$$" separators (standard SDF delimiter)
    # In standard SDF format, each molecule block ends with "$$$$"
    local count=$(grep -c '^\$\$\$\$$' "$sdf_file" 2>/dev/null || echo "0")
    
    # If no separators found, assume single molecule
    # Otherwise, the count equals the number of molecules
    if [ "$count" -eq 0 ]; then
        echo "1"
    else
        echo "$count"
    fi
}

# Function to extract molecule name from SDF (first line of each molecule block)
get_molecule_name_from_sdf() {
    local sdf_file="$1"
    local mol_index="$2"
    
    # For first molecule, get first line
    if [ "$mol_index" -eq 0 ]; then
        local name=$(head -n 1 "$sdf_file" 2>/dev/null | tr -d '\r\n' | sed 's/^[[:space:]]*//;s/[[:space:]]*$//')
        # If name is empty or just whitespace, use default
        if [ -z "$name" ] || [ "$name" = "" ]; then
            echo "mol_0"
        else
            echo "$name"
        fi
    else
        # For subsequent molecules, find the (mol_index-1)th $$$$ separator and get the line after it
        local line_num=$(awk -v target=$((mol_index - 1)) 'BEGIN{count=0} /^\$\$\$\$$/{if(count==target){print NR+1; exit} count++}' "$sdf_file" 2>/dev/null)
        if [ -n "$line_num" ] && [ "$line_num" -gt 0 ]; then
            local name=$(sed -n "${line_num}p" "$sdf_file" 2>/dev/null | tr -d '\r\n' | sed 's/^[[:space:]]*//;s/[[:space:]]*$//')
            # If name is empty or just whitespace, use default
            if [ -z "$name" ] || [ "$name" = "" ]; then
                echo "mol_${mol_index}"
            else
                echo "$name"
            fi
        else
            echo "mol_${mol_index}"
        fi
    fi
}

# CSV header - CNNaffinity added
echo "uid,ligand_number,affinity,cnnscore,cnnaffinity,ligand_filename,mol_index,mol_name" > $OUTPUT_FILE

# Count total number of SDF files for progress tracking
total_files=0
for ligand in "$LIGANDS_DIR"/*.sdf; do
    if [ -f "$ligand" ]; then
        ((total_files++))
    fi
done

echo "Found $total_files SDF files to process"
echo ""

# Counter for processed files
processed_files=0

# Set real_dataset_name based on dataset_name
if [ "$dataset_name" = "dockgen_full" ]; then
    real_dataset_name="dockgen"
else
    real_dataset_name="$dataset_name"
fi

# Iterate through all SDF files directly in LIGANDS_DIR
for ligand in "$LIGANDS_DIR"/*.sdf; do
    if [ -f "$ligand" ]; then
        ligand_filename=$(basename "$ligand")
        ligand_basename="${ligand_filename%.sdf}"  # Remove .sdf extension
        
        # Use full filename as UID (e.g., "1G9V_RQ3" from "1G9V_RQ3.sdf")
        uid="$ligand_basename"
        ligand_identifier="$ligand_basename"
        
        # Construct receptor path
        receptor_path="$RECEPTORS_BASE/${real_dataset_name}_${uid}/${real_dataset_name}_${uid}_protein.pdb"
        
        if [[ -f "$receptor_path" ]]; then
            ((processed_files++))
            progress_percent=$((processed_files * 100 / total_files))
            printf "\r[%3d%%] Processing [%d/%d]: %s" "$progress_percent" "$processed_files" "$total_files" "$ligand_filename"
            echo ""  # New line for detailed output
            
            echo "  UID: $uid"
            
            # Count molecules in the SDF file
            num_molecules=$(count_molecules_in_sdf "$ligand")
            
            if [ "$num_molecules" -gt 1 ]; then
                echo "  Processing multi-SDF file: $ligand_filename ($num_molecules molecules)"
            else
                echo "  Processing single molecule: $ligand_identifier"
            fi

            # Run GNINA scoring
            if [ "$minimize" = "minimize" ]; then
                # Create output path for minimized ligand
                minimized_output="${MINIMIZED_LIGANDS_DIR}/${ligand_filename}"
                result=$("$gnina_script" --receptor "$receptor_path" --ligand "$ligand" --device "$device_id" --cnn_scoring none --minimize -o "$minimized_output" 2>/dev/null)
            else
                scored_output="${OUTPUT_LIGANDS_DIR}/${ligand_filename}"
                result=$("$gnina_script" --receptor "$receptor_path" --ligand "$ligand" --device "$device_id" --cnn_scoring none --score_only -o "$scored_output" 2>/dev/null)
            fi

        else
            ((processed_files++))
            progress_percent=$((processed_files * 100 / total_files))
            printf "\r[%3d%%] Skipping [%d/%d]: %s (receptor not found)\n" "$progress_percent" "$processed_files" "$total_files" "$ligand_filename"
        fi
    fi
done

echo ""
echo "Done! Processed $processed_files/$total_files files. Results saved to $OUTPUT_FILE"
