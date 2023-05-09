import torch
import numpy as np

import datasets
import transformers

from tqdm import tqdm
import re

def wikitext_detokenizer(string):
    """
    Taken from https://github.com/EleutherAI/lm-evaluation-harness/blob/master/lm_eval/tasks/wikitext.py . 
    The wikitext dataset currently available has weird spacing issues, fixed here. 
    """
    # contractions
    string = string.replace("s '", "s'")
    string = re.sub(r"/' [0-9]/", r"/'[0-9]/", string)
    # number separators
    string = string.replace(" @-@ ", "-")
    string = string.replace(" @,@ ", ",")
    string = string.replace(" @.@ ", ".")
    # punctuation
    string = string.replace(" : ", ": ")
    string = string.replace(" ; ", "; ")
    string = string.replace(" . ", ". ")
    string = string.replace(" ! ", "! ")
    string = string.replace(" ? ", "? ")
    string = string.replace(" , ", ", ")
    # double brackets
    string = re.sub(r"\(\s*([^\)]*?)\s*\)", r"(\1)", string)
    string = re.sub(r"\[\s*([^\]]*?)\s*\]", r"[\1]", string)
    string = re.sub(r"{\s*([^}]*?)\s*}", r"{\1}", string)
    string = re.sub(r"\"\s*([^\"]*?)\s*\"", r'"\1"', string)
    string = re.sub(r"'\s*([^']*?)\s*'", r"'\1'", string)
    # miscellaneous
    string = string.replace("= = = =", "====")
    string = string.replace("= = =", "===")
    string = string.replace("= =", "==")
    string = string.replace(" " + chr(176) + " ", chr(176))
    string = string.replace(" \n", "\n")
    string = string.replace("\n ", "\n")
    string = string.replace(" N ", " 1 ")
    string = string.replace(" 's", "'s")

    return string


class WikitextDataset(torch.utils.data.Dataset):

    """
    Wikitext-103 dataset.

    Original source:
    
    Pointer Sentinel Mixture Models
    https://arxiv.org/pdf/1609.07843.pdf
    https://www.salesforce.com/products/einstein/ai-research/the-wikitext-dependency-language-modeling-dataset/
    """

    def __init__(
        self,
        config, 
        n_samples,
        np_rng,
        split="validation"
    ):

        self.config = config
        self.n_samples = n_samples
        self.np_rng = np_rng
        self.split = split

        self.construct_samples()

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]

    def construct_samples(self):

        # validate the task-specific hyperparameters
        hparams = self.config.data.task_params

        n_ctx = hparams.get("n_ctx", 1) + 1 # add 1 because we need to shift labels right
        split = self.split

        # load + shuffle HF dataset
        hf_dataset = datasets.load_dataset("EleutherAI/wikitext_document_level", name="wikitext-2-raw-v1")[split]
        tokenizer = transformers.AutoTokenizer.from_pretrained("gpt2")
        # map tokenization over dataset (truncate = False, pad = False)
        def tokenize(x):
            x["detok"] = wikitext_detokenizer(x["page"])
            x["tokenized"] = tokenizer(x["detok"], padding=False, truncation=False, return_tensors="pt")["input_ids"][0]
            return x
        hf_dataset = hf_dataset.map(tokenize)
        # pack seqs into ctxlen chunks 
        # (keep cursor = curr. idx we draw from)
        # (if cursor > number of rows in dataset, reshuffle dataset and take it mod number of rows)
        cursor = 0
        n_rows = len(hf_dataset)
        # also save in this class how many epochs we go over

        self.samples = []
        # gen samples. 
        # TODO: add a name to progress bar
        buffer = torch.empty((0,))
        for i in tqdm(range(self.n_samples)):
            
            while buffer.shape[0] < n_ctx:
                buffer = torch.cat([
                    buffer, 
                    torch.Tensor([tokenizer.eos_token_id]), 
                    torch.Tensor(hf_dataset[cursor]["tokenized"])
                ], axis=0)
                
                cursor += 1
                if cursor >= n_rows:
                    # we're hitting a new epoch. reshuffle dataset
                    hf_dataset = hf_dataset.shuffle(seed=cursor)
                    cursor = cursor % n_rows
                    print("new epoch...")

            tokens = buffer[:n_ctx]
            buffer = buffer[n_ctx:]
            # print(tokens.shape, buffer.shape)
            sample = {
                # right-shift outputs
                "inputs": tokens[:-1].long(),
                "labels": tokens[1:].long(),
            }
            self.samples.append(sample)
