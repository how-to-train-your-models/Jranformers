import equinox as eqx
import jax

from equinox import nn
from dataclasses import dataclass
from jax import numpy as jnp
from jaxtyping import Integer, Float, Array, PRNGKeyArray
from typing import List

from .. import attention
from .config import GPTConfig


class SwiGLU(eqx.Module):
    """
    https://azizbelaweid.substack.com/p/what-is-swiglu-how-to-implement-it
    """

    W: Float[Array, "in_features out_features"]
    V: Float[Array, "in_features out_features"]
    b: Float[Array, "out_features"]
    c: Float[Array, "out_features"]

    def __init__(self, key: PRNGKeyArray, in_features: int, out_features: int):
        k1, k2, k3, k4 = jax.random.split(key, 4)
        self.W = jax.random.normal(k1, (in_features, out_features))
        self.V = jax.random.normal(k2, (in_features, out_features))
        self.b = jax.random.normal(k3, (out_features,))
        self.c = jax.random.normal(k4, (out_features,))

    def __call__(self, x):
        return jax.nn.swish(jnp.dot(x, self.W) + self.b) * (jnp.dot(x, self.V) + self.c)


class MLP(eqx.Module):
    c_fc: nn.Linear
    swiglu: SwiGLU
    c_proj: nn.Linear
    dropout: nn.Dropout

    def __init__(self, key: PRNGKeyArray, model_config: GPTConfig):
        key_fc, key_swiglu, key_proj, key_dropout = jax.random.split(key, 4)

        self.c_fc = nn.Linear(
            key=key_fc,
            in_features=model_config.n_embed,
            out_features=4 * model_config.n_embed,
            use_bias=model_config.bias,
        )

        self.swiglu = SwiGLU(
            key=key_swiglu,
            in_features=4 * model_config.n_embed,
            out_features=4 * model_config.n_embed,
        )

        self.c_proj = nn.Linear(
            key=key_proj,
            in_features=4 * model_config.n_embed,
            out_features=model_config.n_embed,
            use_bias=model_config.bias,
        )

        self.dropout = nn.Dropout(model_config.dropout)

    def __call__(self, x):
        x = self.c_fc(x)
        x = self.swiglu(x)
        x = self.c_proj(x)
        x = self.dropout(x)
        return x


class CasualSelfAttention(eqx.Module):
    mha: attention.MultiHeadAttention

    def __init__(self, key: PRNGKeyArray, model_config: GPTConfig):
        self.mha = attention.MultiHeadAttention(
            key=key, n_embed=model_config.n_embed, n_heads=model_config.n_head
        )

    def __call__(
        self, x: Float[Array, "n_tokens n_embed"]
    ) -> Float[Array, "n_tokens n_embed"]:
        n_tokens = x.shape[0]
        mask = jnp.tril(jnp.ones((n_tokens, n_tokens)))
        self.mha(x, mask=mask)


class Block(eqx.Module):
    ln_1: nn.LayerNorm
    attn: CasualSelfAttention
    ln_2: nn.LayerNorm
    mlp: MLP

    def __init__(self, key: PRNGKeyArray, model_config: GPTConfig):
        key_attn, key_mlp = jax.random.split(key, 2)

        self.ln_1 = nn.LayerNorm(model_config.n_embed, use_bias=model_config.bias)
        self.attn = CasualSelfAttention(key=key_attn, model_config=model_config)
        self.ln_2 = nn.LayerNorm(model_config.n_embed, use_bias=model_config.bias)
        self.mlp = MLP(key=key_mlp, model_config=model_config)

    def __call__(
        self, x: Float[Array, "n_tokens n_embed"]
    ) -> Float[Array, "n_tokens n_embed"]:
        x = jax.vmap(self.ln_1)(x)
        x = x + jax.vmap(self.attn)(x)
        x = jax.vmap(self.ln_2)(x)
        x = jax.vmap(self.mlp)(x)
        return x


class Transformer(eqx.Module):
    wte: nn.Embedding
    wpe: nn.Embedding
    drop: nn.Dropout
    h: List[Block]
    ln_f: nn.LayerNorm

    def __init__(self, key: PRNGKeyArray, model_config: GPTConfig):
        te_key, pe_key, h_key = jax.random.split(key, 3)

        # token embeddings
        self.wte = nn.Embedding(
            key=te_key,
            num_embeddings=model_config.vocab_size,
            embedding_size=model_config.n_embed,
        )
        # positional embeddings
        self.wpe = nn.Embedding(
            key=pe_key,
            num_embeddings=model_config.block_size,
            embedding_size=model_config.n_embed,
        )
        self.drop = nn.Dropout(model_config.dropout)
        block_keys = jax.random.split(h_key, model_config.n_layers)
        self.h = [
            Block(key=block_keys[i], model_config=model_config)
            for i in range(model_config.n_layers)
        ]
        self.ln_f = nn.LayerNorm(model_config.n_embed, use_bias=model_config.bias)

    def __call__(
        self,
        key: PRNGKeyArray,
        tokens: Integer[Array, "n_tokens"],
        inference: bool = False,
    ) -> Float[Array, "n_tokens n_embed"]:
        pos = jnp.arange(0, len(tokens), dtype=jnp.int64)
        t_embed = jax.vmap(self.wte)(tokens)  # token embeddings
        p_embed = jax.vmap(self.wpe)(pos)  # positional embedidngs
        x = self.drop(
            t_embed + p_embed, inference=inference, key=key
        )  # TODO: confirm why is key optional in params
        for block in self.h:
            x = block(x)
        x = jax.vmap(self.ln_f)(x)
        return x


class GPT(eqx.Module):
    transformer: Transformer
    lm_head: nn.Linear

    def __init__(self, key: PRNGKeyArray, model_config: GPTConfig):
        key_transformer, key_lm_head = jax.random.split(key, 2)

        self.transformer = Transformer(key=key_transformer, model_config=model_config)
        self.lm_head = nn.Linear(
            key=key_lm_head,
            in_features=model_config.n_embed,
            out_features=model_config.vocab_size,
            use_bias=True,
        )

    def __call__(
        self,
        key: PRNGKeyArray,
        tokens: Integer[Array, "n_tokens"],
        inference: bool = False,
    ) -> Float[Array, "n_tokens vocab_size"]:
        x = self.transformer(key, tokens, inference=inference)
        if not inference:
            logits = jax.vmap(self.lm_head)(x)  # (n_tokens, vocab_size)
        else:
            last_token_embedding = x[[-1], :]
            # during inference we only care about the last token
            # vmap is not needed here, because it's only single token
            logits = self.lm_head(last_token_embedding)
        return logits
