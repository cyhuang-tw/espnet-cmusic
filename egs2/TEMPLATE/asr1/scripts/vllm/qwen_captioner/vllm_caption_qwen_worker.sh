#!/usr/bin/env bash
set -euo pipefail
SECONDS=0
log() {
    local fname=${BASH_SOURCE[1]##*/}
    echo -e "$(date '+%Y-%m-%dT%H:%M:%S') (${fname}:${BASH_LINENO[0]}:${FUNCNAME[1]}) $*"
}

# Function to wait for vLLM service to be ready
wait_for_vllm_service() {
    local port=$1
    local service_pid=$2

    log "Waiting for vLLM service to be ready on port ${port}..."
    local max_attempts=180  # 180 * 10 seconds = 30 minutes max wait
    local attempt=0

    while [ ${attempt} -lt ${max_attempts} ]; do
        # Check if service process is still running
        if ! kill -0 ${service_pid} 2>/dev/null; then
            log "ERROR: Service process died unexpectedly"
            exit 1
        fi

        # Try to connect to the health endpoint
        if curl -s -o /dev/null -w "%{http_code}" \
            http://localhost:${port}/health 2>/dev/null | grep -q "200"; then
            log "Service is ready!"

            # Double-check with models endpoint
            if curl -s http://localhost:${port}/v1/models 2>/dev/null | \
                grep -q "Qwen3-Omni-30B-A3B-Captioner"; then
                log "Model loaded successfully"
                return 0
            fi
        fi

        attempt=$((attempt + 1))
        if [ ${attempt} -eq ${max_attempts} ]; then
            log "ERROR: Service failed to start after 30 minutes"
            kill -9 ${service_pid} 2>/dev/null || true
            exit 1
        fi

        log "Attempt ${attempt}/${max_attempts}: Service not ready yet, waiting 10s..."
        sleep 10
    done
}

nj=
rank=
parquet_path=
output_dir=
base_port=

. utils/parse_options.sh

port=$((base_port + rank))

# (1) launch service
nvidia-smi
log "Launch service at port ${port}"
./scripts/vllm/qwen_captioner/launch_service.sh ${port} &
SERVICE_PID=$!
log "Service launched with PID ${SERVICE_PID}"

# (2) Wait for service to be ready
wait_for_vllm_service ${port} ${SERVICE_PID}
nvidia-smi

# (3) launch client
python3 scripts/vllm/qwen_captioner/launch_client.py \
    --parquet ${parquet_path} \
    --output ${output_dir} \
    --n-jobs ${nj} \
    --rank ${rank} \
    --base-url http://localhost:${port}/v1 \
    --workers 1024 \
    --save-freq 10 \
    --timeout 1800

# (4) Kill service
log "Finish all queries. Terminate the service"
kill $SERVICE_PID
