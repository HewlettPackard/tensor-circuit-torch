#!/usr/bin/env bash
###############################################################################
# Ising solver batch launcher (CPU)
#   • Stored in   : project_root/experiments/ising_solver/ising_batch.sh
#   • Executed as : bash experiments/ising_solver/ising_batch.sh
###############################################################################

# ───────────────────────────── Parameter space ────────────────────────────── #
instance_list=(0 1 2 3 4 5 6 7 8 9)

command="python -m experiments.ising_solver.run_ising_solver_unitary" # the python command

set -euo pipefail

# ──────────────────────────── Directory handling ──────────────────────────── #
SCRIPT_DIR="$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
ROOT_DIR="$( realpath "${SCRIPT_DIR}/.." )"

cd "${ROOT_DIR}"

LOG_DIR="${SCRIPT_DIR}/logs"
mkdir -p "${LOG_DIR}"

timestamp=$(date "+%Y%m%d_%H%M%S")

# avoid BLAS/OMP thread oversubscription across parallel processes
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1

# ───────────────────────────── Launching logic ────────────────────────────── #
echo "========= launching ${#instance_list[@]} instances (working dir: ${PWD}) ========="

for instance in "${instance_list[@]}"; do
  echo "  ↪︎ Launching instance=${instance}"
  $command \
      --instance "${instance}" \
      --timestamp "${timestamp}" \
      > "${LOG_DIR}/instance_${instance}.log" 2>&1 &
done

# Wait until all jobs complete
wait

echo "🏁  All experiments completed."