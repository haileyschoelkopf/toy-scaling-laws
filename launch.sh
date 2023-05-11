#!/bin/bash
conda activate toyscaling

# Loop over the hyperparameter combinations
for k in {1..10}; do
      # Override the configuration values using OmegaConf
      overrides="{train: {k: $k}, checkpointing: {dir: run-$k}, wandb: {group: k_is_$k}}"

      # Run your training script with the current hyperparameters
      python toy_scaling/main.py --config configs/main.yaml train.k=$k wandb.group="k_is_$k" checkpointing.dir="run-$k"
done

