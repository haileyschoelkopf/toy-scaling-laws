import argparse 
from omegaconf import OmegaConf
from tqdm import tqdm

import torch

from toy_scaling.models import get_model
from toy_scaling.data import get_dataset


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config", 
        required=True, 
        type=str, 
        help="path to a config YAML file."
    )

    args, _ = parser.parse_known_args()
    return args


def build_config(config):
    cfg = OmegaConf.load(config) # load from yaml in args 
    cli = OmegaConf.from_cli() # load overrides from command line

    merge = OmegaConf.merge(cfg, cli)
    # TODO: make sure this merged config is saved alongside each saved checkpoint!
    return merge


def train_loop(model, optim, config, train_dataloader, test_dataloader):

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

        


def main():
    args = parse_args()
    
    config = build_config(args.config)
    device = config.train.device # "cuda:0" if torch.cuda.is_available() else "cpu"

    train_dataloader, test_dataloader = get_dataset(
        config,    
    )

    model = get_model().to(device)

    optim = torch.optim.AdamW(
        model.parameters(),
        lr=config.optimizer.lr,
        weight_decay=config.optimizer.wd,
        betas=config.optimizer.betas,
    )

    # TODO: add wandb logging

    train_loop(
        model,
        optim,
        config,
        train_dataloader,
        test_dataloader,
    )


if __name__ == '__main__':
    main()