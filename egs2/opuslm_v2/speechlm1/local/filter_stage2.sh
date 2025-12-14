#!/usr/bin/env bash
# Set bash to 'debug' mode, it will exit on :
# -e 'error', -u 'undefined variable', -o ... 'error in pipeline', -x 'print commands',
set -e
set -u
set -o pipefail

log() {
    local fname=${BASH_SOURCE[1]##*/}
    echo -e "$(date '+%Y-%m-%dT%H:%M:%S') (${fname}:${BASH_LINENO[0]}:${FUNCNAME[1]}) $*"
}
SECONDS=0

# Distributed processing
rank=0
world_size=1

# vLLM service configuration
vllm_launch_script="/work/nvme/bbjs/jtian1/tools/vllm/launch_vllm_qwen3_32b.sh"
vllm_port=8001
vllm_host="localhost"
vllm_timeout=600  # Max seconds to wait for vLLM to start

# Input/output directories
text_dir=""
audio_dir=""
output_dir=""

log "$0 $*"
. utils/parse_options.sh

if [ -z "${text_dir}" ] || [ -z "${output_dir}" ]; then
    log "Usage: $0 --text_dir <dir> --output_dir <dir> [options]"
    log "  --text_dir             # Input text directory"
    log "  --audio_dir            # Input audio directory (for CLAP scoring)"
    log "  --output_dir           # Output directory for filtered results"
    log "  --rank                 # Current process rank (default: 0)"
    log "  --world_size           # Total number of processes (default: 1)"
    log "  --vllm_port            # vLLM service port (default: 8001)"
    log "  --vllm_host            # vLLM service host (default: localhost)"
    log "  --vllm_timeout         # Max seconds to wait for vLLM (default: 600)"
    exit 1
fi

mkdir -p "${output_dir}"

# Store vLLM process ID for cleanup
VLLM_PID=""

cleanup() {
    log "Cleaning up..."
    if [ -n "${VLLM_PID}" ] && kill -0 "${VLLM_PID}" 2>/dev/null; then
        log "Stopping vLLM service (PID: ${VLLM_PID})..."
        kill "${VLLM_PID}" 2>/dev/null || true
        wait "${VLLM_PID}" 2>/dev/null || true
        log "vLLM service stopped."
    fi
}

# Set up trap to cleanup on exit
trap cleanup EXIT INT TERM

check_vllm_health() {
    # Check if vLLM service is ready by querying the health endpoint
    local response
    response=$(curl -s -o /dev/null -w "%{http_code}" \
        "http://${vllm_host}:${vllm_port}/health" 2>/dev/null) || return 1
    [ "${response}" = "200" ]
}

start_vllm_service() {
    log "Starting vLLM service on port ${vllm_port}..."

    # Check if port is already in use
    if check_vllm_health; then
        log "vLLM service is already running on port ${vllm_port}"
        return 0
    fi

    # Launch vLLM service in background
    bash "${vllm_launch_script}" "${vllm_port}" \
        > "${output_dir}/vllm_server_rank${rank}.log" 2>&1 &
    VLLM_PID=$!
    log "vLLM service started with PID: ${VLLM_PID}"

    # Wait for vLLM to be ready
    log "Waiting for vLLM service to be ready (timeout: ${vllm_timeout}s)..."
    local elapsed=0
    local check_interval=10

    while [ ${elapsed} -lt ${vllm_timeout} ]; do
        if ! kill -0 "${VLLM_PID}" 2>/dev/null; then
            log "Error: vLLM process died unexpectedly. Check logs:"
            log "  ${output_dir}/vllm_server_rank${rank}.log"
            tail -50 "${output_dir}/vllm_server_rank${rank}.log" || true
            return 1
        fi

        if check_vllm_health; then
            log "vLLM service is ready after ${elapsed}s"
            return 0
        fi

        sleep ${check_interval}
        elapsed=$((elapsed + check_interval))
        log "  Still waiting... (${elapsed}s / ${vllm_timeout}s)"
    done

    log "Error: vLLM service failed to start within ${vllm_timeout}s"
    log "Check logs: ${output_dir}/vllm_server_rank${rank}.log"
    tail -50 "${output_dir}/vllm_server_rank${rank}.log" || true
    return 1
}

# =============================================================================
# Main execution
# =============================================================================

log "Stage 2: Summarization and CLAP scoring"
log "  Rank: ${rank} / World size: ${world_size}"
log "  Text directory: ${text_dir}"
log "  Audio directory: ${audio_dir:-'(not specified)'}"
log "  Output directory: ${output_dir}"
log "  vLLM host:port: ${vllm_host}:${vllm_port}"

# Step 1: Start vLLM service
start_vllm_service || exit 1

# Step 2: Summarization using vLLM
log "Step 2: Summarization using vLLM..."
python3 local/filter_stage2_client.py \
    --text_dir "${text_dir}" \
    --output_dir "${output_dir}" \
    --rank ${rank} \
    --world_size ${world_size} \
    --vllm_host "${vllm_host}" \
    --vllm_port ${vllm_port}

# Step 3: CLAP scoring (TODO)
log "Step 3: CLAP scoring..."
# TODO: Implement CLAP scoring logic

log "Successfully finished. [elapsed=${SECONDS}s]"
