import jax
import jax.numpy as jnp
from jransformers import attention

def test_scaled_dot_product():
    key = jax.random.PRNGKey(0)
    q = jax.random.normal(key, (2, 4))
    key, subkey = jax.random.split(key)
    k = jax.random.normal(subkey, (2, 4))
    key, subkey = jax.random.split(key)
    v = jax.random.normal(subkey, (2, 4))
    values, attn = attention.scaled_dot_product(q[None, :], k[None, :], v[None, :])
    assert values.shape == (1, 2, 4)
    assert attn.shape == (1, 2, 2)
    sums = attn.sum(axis=-1)
    assert jnp.allclose(sums, jnp.ones_like(sums))

def test_expand_mask():
    mask = jnp.ones((2, 3))
    out = attention.expand_mask(mask)
    assert out.ndim == 4
    assert out.shape[-2:] == (2, 3)

def test_multi_head_attention_output_shape():
    key = jax.random.PRNGKey(0)
    mha = attention.MultiHeadAttention(key, n_embed=8, n_heads=2)
    x = jax.random.normal(key, (3, 8))
    values, attn = mha(x)
    assert values.shape == (3, 8)
    assert attn.shape == (2, 3, 3)
