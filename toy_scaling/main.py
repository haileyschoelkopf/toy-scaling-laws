import argparse 
from omegaconf import OmegaConf
from tqdm import tqdm
import wandb

import torch

from transformer_lens import HookedTransformer

# from toy_scaling.models import get_transformer_config
from toy_scaling.train.utils import set_seeds 
from toy_scaling.train import train_loop
from toy_scaling.data import get_dataset
from toy_scaling.scheduler import TransformerScalingScheduler

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

def main():
    args = parse_args()
    
    config = build_config(args.config)
    device = config.train.device
    np_rng = set_seeds(config.train.seed)

    train_dataloader, test_dataloader = get_dataset(
        config,    
        np_rng=np_rng, 
    )

    scale_scheduler = TransformerScalingScheduler()
    model_config, train_config = scale_scheduler.get_hparams(
            k=config.train.k, 
            tokens=10**6, 
            n_ctx=32,
            batch_size=config.train.batch_size,
            d_vocab=2, 
            tokenizer_name=None,
            seed=37,
    )
    print(model_config, train_config)
    # model_config = get_transformer_config(config.k)
    model = HookedTransformer(cfg=model_config)
    print(model)

    optim = torch.optim.AdamW(
        model.parameters(),
        lr=config.optimizer.lr,
        weight_decay=config.optimizer.wd,
        betas=config.optimizer.betas,
    )

    # TODO: add wandb logging
    run = wandb.init(
        project=config.wandb.project,
        entity=config.wandb.entity,
        name=config.wandb.name,
        group=config.wandb.group,
        config=config,
        )
    train_loop(
        model,
        optim,
        config,
        train_dataloader,
        test_dataloader,
    )


if __name__ == '__main__':
    main()
