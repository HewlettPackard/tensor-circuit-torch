#!/usr/bin/env bash
###############################################################################
# Nested‑loop job launcher
#   • Stored in   : project_root/bash_scripts/loop_stategen_cat.sh
#   • Executed as : bash bash_scripts/loop_state
###############################################################################

# ───────────────────────────── Parameter space ────────────────────────────── #
# hg_list=(10 50 700) # ueV um^2: the interaction constant                    
# hg_list=(0.0 10.0 20.0 30.0 40.0 50.0 60.0 70.0 80.0 90.0 100.0 700.0) # ueV um^2: the interaction constant   
# hg_list=(200. 300. 400. 500. 600.)  
hg_list=(3.3 16.6 233.3)  
gamma_list=(0.00 0.02 0.04 0.06 0.08 0.10) #  ps^-1 : the decay rate per picosecond
phi_list=(0.0 0.2 0.4 0.6 0.8 1.0) # (units of pi) : the relative pulse phase shift

startdev=0 # cuda start
#command="python -m experiments.g2_experiment.g2_simulator" # the python command
command="python -m experiments.g2_experiment.g2_simulator_scan_vg" # the python command
pi=3.141592

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
for hg in "${hg_list[@]}"; do
  for gamma in "${gamma_list[@]}"; do
    echo "========= hg = ${hg} gamma = ${gamma} (working dir: ${PWD}) ========="

    for idx in "${!phi_list[@]}"; do
      phi="${phi_list[$idx]}"
      phi_pi=$(echo "$phi * $pi" | bc -l)

      dev="cuda:$((startdev + idx))"

      # create write dir
      writedir="bash_scripts/data_g2/${timestamp}/${hg}_${gamma}_${phi}"

      echo "  ↪︎ Launching hg=${hg}  gamma=${gamma} phi=${phi} on device=${dev}"
          $command \
          --hg   "${hg}" \
          --gamma   "${gamma}" \
          --phi "${phi_pi}" \
          --device "${dev}" \
          --writedir "${writedir}" \
          >  "${LOG_DIR}/${hg}_${gamma}_${phi}_${dev//:/}.log" 2>&1 &
    done

    # Wait until all par2 jobs for this par1 complete
    wait
    echo "----- finished batch for gamma = ${gamma} -----"
  done
done

echo "🏁  All experiments completed."