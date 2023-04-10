import math
from transformer_lens.HookedTransformerConfig import HookedTransformerConfig

def get_transformer_config(k: int, d_vocab: int, tokenizer_name: str, seed: int):
    """
    Given a parameter count, how should we define each transformer architecture
    hyperparameter in order to achieve predictable scaling? This function defines 
    a sequence of transformers hyperparameter configurations that should scale 
    predictably. 

    Right now, we use an incrementing scheme similar to that use by Chinchilla. 
    However, we will experiment with this to achieve the cleanest scaling curve
    possible. 

    Right now, we don't quite embedding and layernorm parameters. This might change. 
    """
    assert k >= 1

    n_ctx = 32

    d_model = 64 + 32*(k//3)
    n_layers = 1 + 2*(k-1)

    n_heads = d_model//16
    d_head = 8 * (1+k//8)

    d_mlp = 4 * d_model

    assert d_model % n_heads == 0 

    param_count = n_layers * (2*d_model*d_mlp + \
            3 * n_heads * (d_model//n_heads) * d_head
            ) + d_model * d_vocab 

    return HookedTransformerConfig(
            d_model=d_model, 
            d_head=d_head, 
            n_layers=n_layers, 
            n_ctx=n_ctx, 
            n_heads=n_heads, 
            d_mlp=d_mlp, 
            d_vocab=d_vocab, 
            act_fn='solu_ln', 
            eps=1e-5, 
            use_attn_result=False, 
            use_split_qkv_input=False, 
            use_attn_scale=True, 
            tokenizer_name=tokenizer_name, 
            normalization_type='LN', 
            seed=seed, 
            initializer_range=0.8/math.sqrt(d_model),
            init_weights=True, 
            positional_embedding_type='standard', 
            parallel_attn_mlp=False, 
            )


def main(): 
    for k in range(1, 10):
        cfg = get_transformer_config(k, 2, "gpt2", 42)
        print(cfg)

if __name__=="__main__": 
    main()
