import jax
import jax.numpy as jnp
from jransformers.nano_gpt import model, config


def test_gpt_decode_shapes():
    key = jax.random.PRNGKey(0)
    gpt_conf = config.GPTConfig(block_size=8, n_layers=1, vocab_size=10, n_head=2, n_embed=8, dropout=0.0)
    gpt = model.GPT(key, gpt_conf)
    tokens = jnp.array([1, 2, 3])
    key, subkey = jax.random.split(key)
    out = gpt.decode(subkey, tokens, max_new_tokens=2)
    assert out.shape[0] == len(tokens) + 2
