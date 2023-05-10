import argparse 
from omegaconf import OmegaConf
from tqdm import tqdm
from dataclasses import dataclass
import os
import wandb

import torch

from transformer_lens import HookedTransformer

from toy_scaling.data import get_dataset
from toy_scaling.train.utils import set_seeds
from toy_scaling.scheduler import TransformerScalingScheduler, TrainingConfig


def train_loop(
        model: HookedTransformer, 
        optim: torch.optim.Optimizer, 
        config: TrainingConfig, 
        train_dataloader: torch.utils.data.DataLoader, 
        test_dataloader: torch.utils.data.DataLoader,
):
    train_dataloader = iter(train_dataloader)
    test_dataloader = iter(test_dataloader)

    cfg = config.train

    for iteration in tqdm(range(1, cfg.train_iters + 1)):

        # run fwd on a data batch here (should we do full-batch training?)
        if config.data.task == "wikitext":
            batch = next(train_dataloader)["inputs"].to("cuda")

            logits = model(batch)
            loss = model.loss_fn(logits, batch)
        elif config.data.task == "sparse_parity":
            batch = next(train_dataloader)

            logits = model(batch["inputs"].to("cuda"))
            # print(logits[:,-1,:].shape)
            loss_fn = torch.nn.CrossEntropyLoss()
            # print(batch["labels"].squeeze().shape)
            loss = loss_fn(logits[:,-1,:], batch["labels"].squeeze().to("cuda"))
            print(loss)

        loss.backward()

        optim.step()
        optim.zero_grad()

        wandb.log({'train/loss': loss, 'train/ppl': torch.exp(loss)}, step=iteration)
        wandb.log(
            {'train-tokens/loss': loss, 'train-tokens/ppl': torch.exp(loss)}, 
            step=iteration * config.data.task_params.n_ctx * cfg.batch_size
        )
        del batch

        if iteration % cfg.eval_every == 0:
            # run test loop
            sum_loss = 0
            with torch.no_grad():
                model.eval()
                for test_iter in range(cfg.eval_iters):
                    # get test loss on a batch
                    test_batch = next(test_dataloader)["inputs"].to("cuda")
                    logits = model(test_batch)
                    test_loss = model.loss_fn(logits, test_batch)
                    # print("test loss:", test_loss)
                    # log the loss and ppl?

                    sum_loss += test_loss.item()
                
                wandb.log({
                    'test/avg_loss': sum_loss / cfg.eval_iters, 
                    'test/avg_ppl': torch.exp((torch.Tensor([sum_loss]) / cfg.eval_iters))
                }, step=iteration)
                # also create plots with tokens processed on y axis
                wandb.log({
                    'test-tokens/avg_loss': sum_loss / cfg.eval_iters, 
                    'test-tokens/avg_ppl': torch.exp((torch.Tensor([sum_loss]) / cfg.eval_iters))
                }, step=iteration * config.data.task_params.n_ctx * cfg.batch_size)

        if iteration % cfg.save_every == 0:
            # save a checkpoint (and metadata + config?)
            
            save_dir = "./" + config.checkpointing.dir + "/" + config.wandb.name + "/" + f"step{iteration}"
            os.makedirs(save_dir, exist_ok=True)

            torch.save({
                "model": model.state_dict(),
                "config": config,
            },
            save_dir + "/model.pt"
            )
            # TODO: copy config file to directory?