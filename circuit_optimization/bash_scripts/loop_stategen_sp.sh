#!/usr/bin/env bash
###############################################################################
# Nested‑loop job launcher
#   • Stored in   : project_root/bash_scripts/loop_stategen_sp.sh
#   • Executed as : bash bash_scripts/loop_stategen_sp.sh
###############################################################################

# ───────────────────────────── Parameter space ────────────────────────────── #
num_channels_list=(3) # the number of waveguides to scan
num_layers_list=(3 5 7 9) # the layer depth
par_list=(0.0 0.02 0.04 0.06 0.08 0.10) # the parameter to scan (nonlinearity U)
circuit_length=5.0 # unit is 100 um, circuit length fixed
startdev=0 # start cuda device
command="python -m experiments.gen_state.main --config config_sp.yaml" # the python command for running

set -euo pipefail

# ──────────────────────────── Directory handling ──────────────────────────── #
# Absolute path to *this* script (inside experiment/)
SCRIPT_DIR="$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
# Project root is the parent directory of experiment/
ROOT_DIR="$( realpath "${SCRIPT_DIR}/.." )"

# Ensure we run from the root so that `python -m …` resolves correctly
cd "${ROOT_DIR}"

# Create a place for logs next to the bash script
LOG_DIR="${SCRIPT_DIR}/logs"
mkdir -p "${LOG_DIR}"

timestamp=$(date "+%Y%m%d_%H%M%S")

# ───────────────────────────── Launching logic ────────────────────────────── #
for num_channels in "${num_channels_list[@]}"; do
  for num_layers in "${num_layers_list[@]}"; do
    echo "========= num_channels = ${num_channels} num_layers = ${num_layers} (working dir: ${PWD}) ========="

    for idx in "${!par_list[@]}"; do
      par="${par_list[$idx]}"
      dev="cuda:$((startdev + idx))"

      # create write dir
      writedir="bash_scripts/data_loop/${timestamp}/${num_channels}_${num_layers}_${par}"

      echo "  ↪︎ Launching num_channels=${num_channels}  num_layers=${num_layers} par=${par}  device=${dev}"
          $command \
          --num_channels   "${num_channels}" \
          --num_layers   "${num_layers}" \
          --layer_depth "$(echo "$circuit_length / $num_layers" | bc -l)" \
          --par   "${par}" \
          --device "${dev}" \
          --writedir "${writedir}" \
          >  "${LOG_DIR}/${num_channels}_${num_layers}_${par}_${dev//:/}.log" 2>&1 &
    done

    # Wait until all par jobs finished for this for num_layers iteration to complete
    wait
    echo "----- finished batch for num = ${num_channels} -----"
  done
done

echo "🏁  All experiments completed."
