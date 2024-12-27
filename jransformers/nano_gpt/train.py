import jax
import jax.numpy as jnp
import equinox as eqx
import jax
import jax.numpy as jnp
import optax

from jaxtyping import Float, Array, PRNGKeyArray
from simple_parsing import ArgumentParser
from typing import Tuple
from . import model, data, config

seed = 42


def get_optimizers(
    model: model.GPT, weight_decay: float, learning_rate: float, betas: Tuple
):
    # Collect all parameters from the model
    param_dict = eqx.filter(model, eqx.is_array)

    # Divide parameters into groups based on their dimensionality    
    decay_params = {k: v for k, v in param_dict.items() if v.ndim >= 2}
    nodecay_params = {k: v for k, v in param_dict.items() if v.ndim < 2}

    # Weight decay optimizer
    decay_optimizer = optax.chain(
        optax.add_decayed_weights(weight_decay),  # Apply weight decay
        optax.adamw(learning_rate=learning_rate, b1=betas[0], b2=betas[1]),
    )

    # No weight decay optimizer
    nodecay_optimizer = optax.adamw(
        learning_rate=learning_rate, b1=betas[0], b2=betas[1]
    )

    # Combine the two optimizers into one
    optimizer = optax.multi_transform(
        transforms={
            "decay": decay_optimizer,
            "nodecay": nodecay_optimizer,
        },
        param_labels={
            **{k: "decay" for k in decay_params},
            **{k: "nodecay" for k in nodecay_params},
        },
    )

    # Count parameters for logging
    num_decay_params = sum(v.size for v in decay_params.values())
    num_nodecay_params = sum(v.size for v in nodecay_params.values())
    print(
        f"num decayed parameter tensors: {len(decay_params)}, with {num_decay_params:,} parameters"
    )
    print(
        f"num non-decayed parameter tensors: {len(nodecay_params)}, with {num_nodecay_params:,} parameters"
    )
    return optimizer


def get_loss(logits: Float[Array, "batch seq_len"], y: Float[Array, "batch seq_len"]):
    return jnp.mean(jax.nn.log_softmax(logits) * y)


@eqx.filter_value_and_grad
def compute_grads(
    model: model.GPT,
    key: PRNGKeyArray,
    x: Float[Array, "batch seq_len"],
    y: Float[Array, "batch seq_len"],
):
    sub_keys = jax.random.split(key, x.shape[0])
    logits = jax.vmap(model, in_axes=(0, 0, 0))(sub_keys, x, y)  # (batch_size,)
    loss = get_loss(logits, y)
    return jnp.mean(loss)


@eqx.filter_jit
def step(
    model: model.GPT,
    optimizer: optax.GradientTransformation,
    state: optax.OptState,
    batch_data: Float[Array, "batch seq_len"],
):
    x, y = batch_data
    loss, grads = compute_grads(model, x, y)
    updates, new_state = optimizer.update(grads, state)
    model = eqx.apply_updates(model, updates)
    return model, new_state, loss


def eval(
    model: model.GPT,
    val_data: Float[Array, "batch seq_len"],
):
    x, y = val_data
    logits = jax.vmap(model, in_axes=(0, 0, 0))(x, y)  # (batch_size,)
    return get_loss(logits, y)


def train(train_config: config.TrainConfig, model_config: config.GPTConfig):
    key = jax.random.PRNGKey(seed)
    key, data_key, model_key = jax.random.split(key, 3)

    # Initialize the model
    gpt = model.GPT(model_key, model_config)

    # Initialize the optimizer
    optimizer = get_optimizers(
        gpt,
        train_config.weight_decay,
        train_config.learning_rate,
        (train_config.beta1, train_config.beta2),
    )

    # Initialize the optimizer state
    state = optimizer.init(gpt)

    # Get the infinite dataloader
    train_dataloader = data.get_infinite_dataloader(
        data_key, "train", train_config.batch_size, train_config.block_size
    )
    val_dataloader = data.get_infinite_dataloader(
        data_key, "validation", train_config.batch_size, train_config.block_size
    )

    for i in range(train_config.num_steps):
        train_batch = next(train_dataloader)
        gpt, state, loss = step(gpt, optimizer, state, train_batch)

        if i % train_config.log_interval == 0:
            print(f"Step {i}, Loss: {loss}")

        if i % train_config.eval_interval == 0:
            val_batch = next(val_dataloader)
            eval_loss = eval(gpt, val_batch)
            print(f"Eval Loss: {eval_loss}")


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_arguments(config.TrainConfig, dest="train_config")
    parser.add_arguments(config.GPTConfig, dest="model_config")
    args = parser.parse_args()
    train(args.train_config, args.model_config)
