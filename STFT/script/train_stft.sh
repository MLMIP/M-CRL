#!/bin/bash

python -m torch.distributed.launch \
    --nproc_per_node=1 \
    ./tools//train_net.py \
    --master_port="$RANDOM" \
    --config-file ./STFT/configs/STFT/kumc_R_50_STFT.yaml \
    OUTPUT_DIR ''