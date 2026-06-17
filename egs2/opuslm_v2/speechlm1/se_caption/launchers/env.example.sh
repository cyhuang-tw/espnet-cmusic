# Copy to `env.sh` and edit for your machine. All se_caption launchers source it.
#   cp env.example.sh env.sh && $EDITOR env.sh
#
# env.sh is gitignored so your local paths never get committed.

# ---- required paths ---------------------------------------------------------
export ESPNET_ROOT=/path/to/espnet                 # this espnet checkout (repo root)
export PYTHON_ENV=/path/to/conda/env               # env with torch + deepspeed + espnet deps
# Working / output root: holds data/<name>/dataset.json, stats/, exp/, eval/.
export SE_ROOT="$ESPNET_ROOT/egs2/opuslm_v2/speechlm1/se_caption"

# Pretrain checkpoint to fine-tune from (the SpeechLM/OpusLM-v2 base, "step_260000"):
#   - full-FT  -> the *universal* checkpoint DIR (has latest_universal + step_*_universal/)
#   - LoRA/peft-> the single weights file step_*_universal/mp_rank_00_model_states.pt
export PRETRAIN_CKPT_DIR=/path/to/ckpt                                   # for full-FT
export PRETRAIN_CKPT_FILE="$PRETRAIN_CKPT_DIR/step_260000_universal/mp_rank_00_model_states.pt"  # for LoRA

# HF cache (models pulled by tag: Qwen/Qwen3-8B-Base, Qwen/Qwen3-Omni-..., Xcodec).
export HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}"

# ---- SLURM (only used by the .sbatch wrappers) ------------------------------
# You can also pass these on the CLI: sbatch --account=A --partition=P launcher.sbatch
export SLURM_ACCOUNT=your-account
export SLURM_PARTITION=your-gpu-partition

# ---- common runtime exports -------------------------------------------------
export PATH="$PYTHON_ENV/bin:$PATH"
export PYTHONPATH="$ESPNET_ROOT"
export CUDA_HOME="${CUDA_HOME:-$PYTHON_ENV}"
export PYTHONUNBUFFERED=1
# GH200/H100 = 9.0; A100 = 8.0; set to your GPU's compute capability.
export TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-9.0}"

# ---- captioning (only for the caption-conditioned pipeline) -----------------
export BAGPIPER_MODEL_DIR=/path/to/vllm_captioner_model   # vLLM export of the SpeechLM base
export VLLM_PY=/path/to/vllm-speechlm-env/bin/python
# export VLLM_REPO=/path/to/vllm-fork                     # if the server must run from a checkout
# export CAPTION_PORT=9011

# ---- URGENT-2026 scoring (only for eval) ------------------------------------
export URGENT_DATA=/path/to/urgent2026_challenge_track1   # contains evaluation_metrics/, data/validation/
export URGENT_PY=/path/to/urgent-metrics-env/bin/python   # env with the URGENT metric deps
export SCOREQ_PY="$URGENT_PY"                              # separate env if torchaudio clashes
export DNSMOS_MODELS=/path/to/dnsmos_onnx_models          # sig_bak_ovr.onnx + model_v8.onnx
# export ESPEAK_DIR=/path/to/espeak-ng-install            # phoneme-similarity metric
# export ORTPATCH=/path/to/ortpatch                        # onnxruntime thread shim (only if SCOREQ fails)
