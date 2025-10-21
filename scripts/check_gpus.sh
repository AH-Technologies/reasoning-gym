#!/bin/bash
#
# check_gpus.sh - Check GPU availability on SLURM cluster
#
# Usage:
#   ./check_gpus.sh              # Show all GPU nodes
#   ./check_gpus.sh h100         # Show only H100 nodes
#   ./check_gpus.sh --free       # Show only nodes with free GPUs
#   ./check_gpus.sh -w           # Watch mode (updates every 5s)
#

# Colors for better readability
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color
BOLD='\033[1m'

# Parse arguments
FILTER_TYPE=""
ONLY_FREE=false
WATCH_MODE=false

while [[ $# -gt 0 ]]; do
    case $1 in
        -w|--watch)
            WATCH_MODE=true
            shift
            ;;
        --free)
            ONLY_FREE=true
            shift
            ;;
        -h|--help)
            echo "Usage: $0 [OPTIONS] [GPU_TYPE]"
            echo ""
            echo "Options:"
            echo "  -w, --watch     Watch mode (updates every 5 seconds)"
            echo "  --free          Show only nodes with free GPUs"
            echo "  -h, --help      Show this help message"
            echo ""
            echo "GPU_TYPE: Filter by GPU type (h100, h200, a100, v100, p100)"
            echo ""
            echo "Examples:"
            echo "  $0              # Show all GPU nodes"
            echo "  $0 h100         # Show only H100 nodes"
            echo "  $0 --free       # Show only nodes with free GPUs"
            echo "  $0 -w h100      # Watch H100 nodes"
            exit 0
            ;;
        *)
            FILTER_TYPE="$1"
            shift
            ;;
    esac
done

show_gpu_status() {
    clear 2>/dev/null || true

    echo -e "${BOLD}${CYAN}=== GPU NODE STATUS ===${NC}"
    echo -e "${BOLD}Partition: GPUQ | Account: share-ie-idi | User: $USER${NC}"
    echo ""
    printf "${BOLD}%-15s | %-8s | %5s | %4s | %4s | %9s | %8s | %-12s | %s${NC}\n" \
           "Node" "GPU" "Total" "Used" "Free" "CPUs Free" "Mem Free" "State" "Users"
    echo "----------------|----------|-------|------|------|-----------|----------|--------------|----------------"

    # Get list of GPU nodes
    nodes=$(sinfo -p GPUQ -N -h -o "%N" | sort -u)

    total_gpus=0
    total_used=0
    total_free=0
    node_count=0

    while IFS= read -r node; do
        # Get node details
        node_info=$(scontrol show node "$node" 2>/dev/null)

        if [[ -z "$node_info" ]]; then
            continue
        fi

        # Extract GPU information
        if [[ $node_info =~ Gres=gpu:([^:]+):([0-9]+) ]]; then
            gpu_type="${BASH_REMATCH[1]}"
            gpu_total="${BASH_REMATCH[2]}"
        else
            continue
        fi

        # Filter by GPU type if specified
        if [[ -n "$FILTER_TYPE" ]] && [[ ! "$gpu_type" =~ $FILTER_TYPE ]]; then
            continue
        fi

        # Extract allocated GPUs
        gpu_used=0
        if [[ $node_info =~ AllocTRES=.*gres/gpu[^=]*=([0-9]+) ]]; then
            gpu_used="${BASH_REMATCH[1]}"
        fi

        # Calculate free GPUs
        gpu_free=$((gpu_total - gpu_used))

        # Filter only free if specified
        if [[ "$ONLY_FREE" == true ]] && [[ $gpu_free -eq 0 ]]; then
            continue
        fi

        # Extract CPU information
        if [[ $node_info =~ CPUAlloc=([0-9]+) ]]; then
            cpu_alloc="${BASH_REMATCH[1]}"
        else
            cpu_alloc=0
        fi

        if [[ $node_info =~ CPUTot=([0-9]+) ]]; then
            cpu_tot="${BASH_REMATCH[1]}"
        else
            cpu_tot=0
        fi

        cpu_free=$((cpu_tot - cpu_alloc))

        # Extract memory information
        if [[ $node_info =~ RealMemory=([0-9]+) ]]; then
            mem_tot="${BASH_REMATCH[1]}"
        else
            mem_tot=0
        fi

        if [[ $node_info =~ AllocMem=([0-9]+) ]]; then
            mem_alloc="${BASH_REMATCH[1]}"
        else
            mem_alloc=0
        fi

        mem_free=$(( (mem_tot - mem_alloc) / 1024 ))  # Convert to GB

        # Extract state
        if [[ $node_info =~ State=([A-Z+]+) ]]; then
            state="${BASH_REMATCH[1]}"
        else
            state="UNKNOWN"
        fi

        # Get users running on this node
        users=$(squeue -h -w "$node" -o "%u" | sort -u | tr '\n' ',' | sed 's/,$//' | sed 's/,/, /g')
        if [[ -z "$users" ]]; then
            users="-"
        fi

        # Color coding based on availability
        if [[ $gpu_free -eq 0 ]]; then
            color=$RED
        elif [[ $gpu_free -eq $gpu_total ]]; then
            color=$GREEN
        else
            color=$YELLOW
        fi

        # Print node information
        printf "${color}%-15s | %-8s | %5d | %4d | %4d | %9d | %6dG | %-12s | %s${NC}\n" \
               "$node" "$gpu_type" "$gpu_total" "$gpu_used" "$gpu_free" "$cpu_free" "$mem_free" "$state" "$users"

        # Update totals
        total_gpus=$((total_gpus + gpu_total))
        total_used=$((total_used + gpu_used))
        total_free=$((total_free + gpu_free))
        node_count=$((node_count + 1))

    done <<< "$nodes"

    echo "--------------------------------------------------------------------------------------------"
    printf "${BOLD}%-15s | %-8s | %5d | %4d | %4d | %9s | %8s | %-12s | %s${NC}\n" \
           "TOTAL" "($node_count nodes)" "$total_gpus" "$total_used" "$total_free" "" "" "" ""

    echo ""
    echo -e "${CYAN}Legend:${NC}"
    echo -e "  ${GREEN}Green${NC}  = All GPUs free"
    echo -e "  ${YELLOW}Yellow${NC} = Partially allocated"
    echo -e "  ${RED}Red${NC}    = Fully allocated"

    if [[ "$WATCH_MODE" == true ]]; then
        echo ""
        echo -e "${BLUE}Press Ctrl+C to exit watch mode${NC}"
    fi
}

# Main execution
if [[ "$WATCH_MODE" == true ]]; then
    while true; do
        show_gpu_status
        sleep 5
    done
else
    show_gpu_status
fi
