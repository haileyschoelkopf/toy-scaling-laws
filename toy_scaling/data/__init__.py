import torch

import math

from .sparse_parity import SparseParityDataset
from .wikitext import WikitextDataset

DATASET_REGISTRY = {
    "sparse_parity": SparseParityDataset,
    "wikitext": WikitextDataset,
}


def get_dataset(
    config,
    np_rng=None,
):

    cfg = config.train
    task_cfg = config.data


    # calculate total samples to generate: 
    use_train_frac = False
    if (cfg.train_frac < 1.0 and cfg.train_frac > 0.0):
        # we are dividing data based on train % and valid %
         # n_samples consumed by train loop * (1 / train_frac) --> after train-test split, get slightly more than the desired
        n_samples = math.ceil(cfg.train_iters * cfg.batch_size * (1.0 / cfg.train_frac))
        use_train_frac = True
    else:
        n_samples = cfg.train_iters * cfg.batch_size
    # train samples. 1.005 factor to avoid running out of genned samples / avoiding rounding error.
    n_samples = math.ceil(n_samples * 1.005)


    # offload to task-specific dataset to construct samples
    dataset = DATASET_REGISTRY[task_cfg.task](
        config,
        n_samples,
        np_rng=np_rng,
        split="train"
    )

    if use_train_frac: 
        train_data, test_data, np_rng = train_test_split(dataset, cfg.train_frac, np_rng)
    else:
        train_data = dataset
        test_data = DATASET_REGISTRY[task_cfg.task](
            config,
            math.ceil((cfg.eval_iters * cfg.batch_size * (cfg.train_iters // cfg.eval_every)) * 1.005),
            np_rng=np_rng,
            split="validation"
        )

    train_dataloader = torch.utils.data.DataLoader(train_data, cfg.batch_size, shuffle=True)
    test_dataloader = torch.utils.data.DataLoader(test_data, cfg.batch_size, shuffle=True)

    return train_dataloader, test_dataloader


def train_test_split(dataset, train_pct, np_rng):

    # create a random shuffle order
    shuffle_indices = np_rng.permutation(len(dataset))
    cutoff = math.ceil(train_pct * len(dataset)) 

    # get the sample indices for each of train and test splits
    train_indices = shuffle_indices[:cutoff]
    test_indices = shuffle_indices[cutoff:]

    # subset the dataset's data by these index lists
    train_data = torch.utils.data.Subset(dataset, train_indices)
    test_data = torch.utils.data.Subset(dataset, test_indices)

    print(len(train_data), len(test_data))
    return train_data, test_data, np_rng
