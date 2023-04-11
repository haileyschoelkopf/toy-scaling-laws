import argparse 
from omegaconf import OmegaConf
from tqdm import tqdm

from transformer_lens import HookedTransformer

from toy_scaling.models import get_transformer_config
from toy_scaling.models import 

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
    
    model_config = get_transformer_config(config.k)
    model = 

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
