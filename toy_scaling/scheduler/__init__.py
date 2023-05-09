from abc import ABC, abstractmethod
from dataclasses import dataclass

import math

from transformer_lens.HookedTransformerConfig import HookedTransformerConfig


@dataclass
class TrainingConfig:
    batch_size: int = 128, 
    lr_scheduler: callable = lambda t: 1.0
    num_steps: int = 1000
    optimizer: str = "AdamW"


class ScalingScheduler(ABC): 
    """
    To achieve predictable scaling, one must increment all the parameters that 
    determine a model's parameter count and its training hyperparameters carefully. 

    The goal of the `ScalingScheduler` abstraction is to express how each of these 
    parameters should be increased in code, rather than in a manner exogeneous 
    to the code (e.g through a config file with hardcoded numbers). 
    """
    @abstractmethod
    def get_hparams(self, model_k: int, tokens: int, **kwargs): 
        """
        The `get_hparams` method returns a tuple: its first element is a HookedMLPConfig 
        or a HookedTransformerConfig, while the second element is a TrainingConfig.
        """
        pass

class TransformerScalingScheduler(ScalingScheduler): 
    def __init__(self): 
        pass 

    def _get_transformer_hparams(self, k: int):
        assert k >= 1

        arch_params = dict()

        arch_params["n_ctx"] = 512

        arch_params["d_model"] = 64 + 32 * (k // 2)
        arch_params["n_layers"] = 1 + 2 * (k - 1)

        arch_params["n_heads"] = arch_params["d_model"] // 8
        arch_params["d_head"] = 8 * (1 + k // 8)

        arch_params["d_mlp"] = 4 * arch_params["d_model"]

        assert arch_params["d_model"] % arch_params["n_heads"] == 0

        return arch_params

    def _get_train_hparams(self, tokens: int, n_ctx: int, batch_size: int): 
        """
        [1] shows that while large batch sizes may be a problem, we should be able to 
        get away with small ones. I also want to see if we can get away with *not* 
        using a learning rate scheduler. 

        1. McCandlish, S., Kaplan, J., Amodei, D., & Team, O.D. (2018). 
        An Empirical Model of Large-Batch Training. ArXiv, abs/1812.06162.
        """
        return {
            "batch_size": 128, 
            "lr_scheduler": lambda t: 1.0,
            "num_steps": tokens//(n_ctx * batch_size),
            "optimizer": "AdamW",
        }

    def get_hparams(
            self,
            k: int, 
            tokens: int, 
            n_ctx: int,
            batch_size: int,
            d_vocab: int, 
            tokenizer_name: str, 
            seed: int,
    ): 
        arch_params = self._get_transformer_hparams(k)

        model_config = HookedTransformerConfig(
            **arch_params,
            d_vocab=d_vocab,
            act_fn="solu_ln",
            eps=1e-5,
            use_attn_result=False,
            use_split_qkv_input=False,
            use_attn_scale=True,
            tokenizer_name=tokenizer_name,
            normalization_type="LN",
            seed=seed,
            initializer_range=0.8 / math.sqrt(arch_params["d_model"]),
            init_weights=True,
            positional_embedding_type="standard",
            parallel_attn_mlp=False,
        )

        train_config = TrainingConfig(**self._get_train_hparams(tokens, n_ctx, batch_size))

        return model_config, train_config

def main():
    for k in range(1, 30):
        scale_scheduler = TransformerScalingScheduler()
        model_config, train_config = scale_scheduler.get_hparams(
                k=k, 
                tokens=10**6, 
                d_vocab=6, 
                tokenizer_name="gpt2",
                seed=37,
        )

        print(model_config)
        print(train_config)


if __name__ == "__main__":
    main()
