import torch
import numpy as np

from tqdm import tqdm

# TODO: turn this into a subclass of a general ToyDataset class?
class SparseParityDataset(torch.utils.data.Dataset):

    """
    The sparse parity task.
    for more information on this task, 
    please see https://arxiv.org/abs/2207.08799 or 
    https://arxiv.org/abs/2303.13506 .

    In particular, we implement the formulation from Michaud et al.
    To use the version more closely following (n,k) sparse parity in Barak et al. 
    please use `n_tasks=1`, `n=n`, `k=k`.
    """

    def __init__(
        self,
        config, 
        n_samples,
        np_rng,
    ):

        self.config = config
        self.n_samples = n_samples
        self.np_rng = np_rng

        self.construct_samples()

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]

    def construct_samples(self):

        # validate the task-specific hyperparameters
        hparams = self.config.data.task_params

        n_tasks = hparams.get("n_tasks", 1)
        n = hparams.get("n", 8)
        k = hparams.get("k", 2)
        task_freqs = hparams.get("task_freqs", n_tasks * [1.0])
        task_freqs = [prob / sum(task_freqs) for prob in task_freqs] # normalize sum of subtask probs to 1

        assert n_tasks * k <= n, \
        "n_tasks * k must not exceed n for sparse parity task!"
        assert len(task_freqs) == n_tasks, "if passing individual subtask frequencies, must have same length as num. subtasks"

        # determine k relevant bits for each subtask (non-overlapping size k subsets). 
        self.active_indices = np.split(self.np_rng.choice(n, n_tasks * k, replace=False), n_tasks)
        # self.task_bits[i] gives np.array of the k (non-control bit) indices to consider for i-th subtask!


        self.samples = []
        # gen samples. 
        # TODO: add a name to progress bar
        for i in tqdm(range(self.n_samples)):
            
            # select a subtask, with probability task_freqs[i] for each task
            active_task = self.np_rng.choice(n_tasks, p=task_freqs)
            
            ctrl_bits = np.zeros(n_tasks)
            ctrl_bits[active_task] = 1

            # draw a string of n 0's and 1's, with uniform freq.
            task_bits = self.np_rng.choice(2, n)

            # sum the active task bits. 
            bitsum = task_bits[self.active_indices[active_task]].sum()

            label = bitsum % 2

            # don't feed in the "indicator" bit if we have one task only
            bitstring = np.concatenate([ctrl_bits, task_bits]) if n_tasks > 1 else task_bits

            sample = {
                "inputs": torch.Tensor(bitstring),
                "labels": torch.Tensor([label]),
                "ctrl_bits": torch.Tensor(ctrl_bits),
                "task_bits": torch.Tensor(task_bits),
                "active_task": torch.Tensor([active_task]),
                "active_indices": torch.Tensor(self.active_indices[active_task]) 
            }
            self.samples.append(sample)
