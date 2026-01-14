#!/bin/bash
# Launch distributed examples
# Usage: ./launch.sh [example] [nproc]
#   ./launch.sh all 4      - Run all examples
#   ./launch.sh matvec 4   - Distributed matvec
#   ./launch.sh solve 4    - Distributed CG solve
#   ./launch.sh eigsh 4    - Distributed LOBPCG

set -e
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EXAMPLE=${1:-all}
NPROC=${2:-4}

run() {
    echo -e "\n========== $1 ==========\n"
    torchrun --standalone --nproc_per_node="$NPROC" "$SCRIPT_DIR/$2"
}

case $EXAMPLE in
    matvec) run "Distributed Matvec" "distributed_matvec.py" ;;
    solve)  run "Distributed Solve" "distributed_solve.py" ;;
    eigsh)  run "Distributed Eigsh" "distributed_eigsh.py" ;;
    all)
        run "Distributed Matvec" "distributed_matvec.py"
        run "Distributed Solve" "distributed_solve.py"
        run "Distributed Eigsh" "distributed_eigsh.py"
        ;;
    *) echo "Usage: $0 {matvec|solve|eigsh|all} [nproc]"; exit 1 ;;
esac

echo -e "\nAll examples completed!"
