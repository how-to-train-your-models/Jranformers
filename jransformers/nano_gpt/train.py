import jax
import jax.numpy as jnp
import einops
import equinox as eqx
import jax
import jax.numpy as jnp
import optax
import os
import pickle

from jaxtyping import Float, Array, PRNGKeyArray
from simple_parsing import ArgumentParser
from typing import Tuple
from . import model, data, config

seed = 42

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1" # second GPU

def get_optimizers(
    model: model.GPT, weight_decay: float, learning_rate: float, betas: Tuple
):

    # Weight decay optimizer
    decay_optimizer = optax.chain(
        optax.add_decayed_weights(weight_decay),  # Apply weight decay
        optax.adamw(learning_rate=learning_rate, b1=betas[0], b2=betas[1]),
    )
    # No weight decay optimizer
    nodecay_optimizer = optax.adamw(
        learning_rate=learning_rate, b1=betas[0], b2=betas[1]
    )

    # Divide parameters into groups based on their dimensionality
    def is_decay_weight(leaf):
        # Any parameters that is 2D will be weight decayed, otherwise no.
        # all weight tensors in matmuls + embeddings decay.
        # all biases and layernorms don't.
        return eqx.is_array(leaf) and leaf.ndim >= 2

    def get_param_label(leaf):
        if is_decay_weight(leaf):
            return "decay"
        return "nodecay"

    param_labels = jax.tree_util.tree_map(get_param_label, model)
    # eqx.tree_pprint(param_labels)
    # Combine the two optimizers into one
    optimizer = optax.multi_transform(
        transforms={
            "decay": decay_optimizer,
            "nodecay": nodecay_optimizer,
        },
        param_labels=lambda _: param_labels,
    )
    return optimizer


def get_loss(logits: Float[Array, "batch seq_len"], y: Float[Array, "batch seq_len"]):
    logits = einops.rearrange(logits, "batch seq_len logits -> (batch seq_len) logits")
    y = einops.rearrange(y, "batch seq_len -> (batch seq_len)")
    return optax.softmax_cross_entropy_with_integer_labels(logits, y)
    # return jnp.mean(jax.nn.log_softmax(logits) * y)


@eqx.filter_value_and_grad
def compute_grads(
    model: model.GPT,
    key: PRNGKeyArray,
    x: Float[Array, "batch seq_len"],
    y: Float[Array, "batch seq_len"],
):
    batch_size = x.shape[0]
    sub_keys = jax.random.split(key, batch_size)    
    logits = jax.vmap(model, in_axes=(0, 0))(sub_keys, x)  # (batch_size,)
    loss = get_loss(logits, y)
    return jnp.mean(loss)


@eqx.filter_jit
def step(
    key: PRNGKeyArray,
    model: model.GPT,
    optimizer: optax.GradientTransformation,
    state: optax.OptState,
    batch_data: Tuple[Float[Array, "batch seq_len"], Float[Array, "batch seq_len"]],
):
    x, y = batch_data
    loss, grads = compute_grads(model, key, x, y)
    model_params = eqx.filter(model, eqx.is_inexact_array)
    updates, new_state = optimizer.update(grads, state, model_params)
    model = eqx.apply_updates(model, updates)
    return model, new_state, loss


def eval(
    key: PRNGKeyArray,
    model: model.GPT,
    val_data: Tuple[Float[Array, "batch seq_len"], Float[Array, "batch seq_len"]],
):
    x, y = val_data
    logits = jax.vmap(model, in_axes=(None, 0))(key, x)  # (batch_size,)
    return jnp.mean(get_loss(logits, y))


def train(train_config: config.TrainConfig, model_config: config.GPTConfig):
    key = jax.random.PRNGKey(seed)
    train_key, eval_key, data_key, model_key = jax.random.split(key, 4)

    vocab_info = data.get_vocabulary_info()
    model_config.vocab_size = vocab_info["vocab_size"]  # TODO provide a way to override this
    
    gpt = model.GPT(model_key, model_config)
    model_params = eqx.filter(gpt, eqx.is_inexact_array)
    # Initialize the optimizer
    optimizer = get_optimizers(
        model_params,  # model_params has a __call__ method, which causes an error so we wrap it in a lambda
        train_config.weight_decay,
        train_config.learning_rate,
        (train_config.beta1, train_config.beta2),
    )

    # Initialize the optimizer state
    state = optimizer.init(model_params)

    # Get the infinite dataloader
    train_dataloader = data.get_infinite_dataloader(
        data_key, "train", train_config.batch_size, model_config.block_size
    )
    val_dataloader = data.get_infinite_dataloader(
        data_key, "validation", train_config.batch_size, model_config.block_size
    )

    for i in range(train_config.num_steps):
        train_batch = next(train_dataloader)
        gpt, state, loss = step(train_key, gpt, optimizer, state, train_batch)

        if i % train_config.log_interval == 0:
            print(f"Step {i}, Loss: {loss}")

        if i % train_config.eval_interval == 0:
            val_batch = next(val_dataloader)
            eval_loss = eval(eval_key, gpt, val_batch)
            print(f"Eval Loss: {eval_loss}")
            if train_config.always_save_checkpoint:
                if not os.path.exists(train_config.out_dir):
                    os.makedirs(train_config.out_dir, exist_ok=True)
                ckpt_path = os.path.join(train_config.out_dir, f"ckpt_step_{i}.eqx")
                eqx.tree_serialise_leaves(ckpt_path, gpt)
                print(f"Saved checkpoint to {ckpt_path}")
                
                # Save meta.pkl alongside the checkpoint
                meta_path = os.path.join(train_config.out_dir, "meta.pkl")
                with open(meta_path, 'wb') as f:
                    pickle.dump(vocab_info, f)
                print(f"Saved vocabulary metadata to {meta_path}")


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_arguments(config.TrainConfig, dest="train_config")
    parser.add_arguments(config.GPTConfig, dest="model_config")
    args = parser.parse_args()
    train(args.train_config, args.model_config)
