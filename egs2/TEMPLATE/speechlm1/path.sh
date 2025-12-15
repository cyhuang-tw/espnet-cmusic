MAIN_ROOT=$PWD/../../..

export PATH=$PWD/utils/:$PATH
export LC_ALL=C
export OMP_NUM_THREADS=1

# NOTE(kan-bayashi): Use UTF-8 in Python to avoid UnicodeDecodeError when LC_ALL=C
export PYTHONIOENCODING=UTF-8
export PYTHONPATH=${MAIN_ROOT}:PYTHONPATH

# You need to change or unset NCCL_SOCKET_IFNAME according to your network environment
# https://docs.nvidia.com/deeplearning/sdk/nccl-developer-guide/docs/env.html#nccl-socket-ifname
export NCCL_SOCKET_IFNAME="^lo,docker,virbr,vmnet,vboxnet"

# NOTE(kamo): Source at the last to overwrite the setting
. local/path.sh

# NOTE(Jinchuan): avoid pytorch memory segmentation
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

ESPNET_DATASET_REGISTRY=
# NOTE(Jinchuan): selectively enable this for wavlab internal usage.
if [[ "$(hostname)" == dt* ]] || [[ "$(hostname)" == gh* ]] || [[ "$(hostname)" == gpu* ]] ; then # For Delta/DeltaAI
    ESPNET_DATASET_REGISTRY+=":/work/nvme/bbjs/shared/data_registry/train_shared.yaml:/work/nvme/bbjs/shared/opuslm_v2_data/data_jsons/opuslm_v2.yaml"
    ESPNET_DATASET_REGISTRY+=":/work/nvme/bbjs/shared/data_registry/valid_shared.yaml"
fi

export ESPNET_DATASET_REGISTRY
