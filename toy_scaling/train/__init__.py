import argparse 
from omegaconf import OmegaConf
from tqdm import tqdm
from dataclasses import dataclass

import torch

from transformer_lens import HookedTransformer

from toy_scaling.data import get_dataset
from toy_scaling.utils import set_seeds
from toy_scaling.train import TrainingConfig


@dataclass
class TrainingConfig:
    batch_size: int = 128, 
    lr_scheduler: callable = lambda t: 1.0
    num_steps: int 
    optimizer: str = "AdamW"

def train_loop(
        model: HookedTransformer, 
        optim: torch.optim.Optimizer, 
        config: TrainingConfig, 
        train_dataloader: torch.utils.data.DataLoader, 
        test_dataloader: torch.utils.data.DataLoader,
):

    cfg = cfg.train

    for iteration in tqdm.tqdm(cfg.train_iters):

        # run fwd on a data batch here (should we do full-batch training?)

        # loss.backward()

        optim.step()
        optim.zero_grad()

        if iteration % cfg.eval_every == 0:
            # run test loop
            for test_iter in range(cfg.eval_iters):
                # get test loss on a batch
                pass
            
            # log avg test loss to wandb 
            # TODO: should we run 1 epoch on test data each eval loop?

        if iteration % cfg.save_every == 0:
            # save a checkpoint (and metadata + config?)
            pass
