import jax 
from datasets import load_dataset
from jax import numpy as jnp
from jaxtyping import Float, Array, PRNGKeyArray

dataset_name = 'karpathy/tiny_shakespeare'
_cached_vocab_info = None

def get_dataset():
    return load_dataset(dataset_name, trust_remote_code=True)

def get_vocabulary_info():
    global _cached_vocab_info
    if _cached_vocab_info is not None:
        return _cached_vocab_info

    dataset = get_dataset()
    # Use 'train' split to define the vocabulary
    train_text_list = dataset['train']['text']
    full_train_text = "".join(train_text_list)
    
    chars = sorted(list(set(full_train_text)))
    stoi = {ch: i for i, ch in enumerate(chars)}
    itos = {i: ch for ch, i in stoi.items()} 
    vocab_size = len(chars)
    
    _cached_vocab_info = {"stoi": stoi, "itos": itos, "vocab_size": vocab_size}
    return _cached_vocab_info

def get_encoder():
    return get_vocabulary_info()["stoi"]

def get_decoder():
    return get_vocabulary_info()["itos"]

def get_vocab_size():
    return get_vocabulary_info()["vocab_size"]

def get_infinite_dataloader(key: PRNGKeyArray, split_type: str, batch_size: int, seq_len: int):
    dataset_obj = get_dataset()
    assert split_type in dataset_obj.keys(), f"Split {split_type} not found in dataset"
    
    text_list_for_split = dataset_obj[split_type]['text']
    full_text_for_split = "".join(text_list_for_split)

    vocab_info = get_vocabulary_info()
    stoi = vocab_info["stoi"]
    
    def encode_fn(s):
        return [stoi[c] for c in s if c in stoi] 
    
    # Convert encoded data to a JAX array once so we can use vectorized slicing
    data_encoded = jnp.array(encode_fn(full_text_for_split), dtype=jnp.int32)
    n = data_encoded.shape[0]

    if n < seq_len + 1:
        # Attempt to provide a more robust check for empty or too short data_encoded
        min_data_len = seq_len + 1
        if n < min_data_len:
            error_msg = (
                f"Dataset split '{split_type}' has insufficient data ({n} tokens after encoding) "
                f"for sequence length {seq_len}. Minimum required is {min_data_len}. "
                "This might be due to the split being too small, or `encode_fn` filtering "
                "out too many characters (e.g., if the split contains characters not in the 'train' vocab)."
            )
            raise ValueError(error_msg)
    
    def get_block(start_idx):
        """Vectorized slice helper using ``dynamic_slice`` for JIT efficiency."""
        return jax.lax.dynamic_slice(data_encoded, (start_idx,), (seq_len + 1,))

    while True:
        # create a new key every iteration to avoid returning the same batch
        key, subkey = jax.random.split(key)
        ix = jax.random.randint(subkey, (batch_size,), 0, n - seq_len - 1)

        seq = jax.vmap(get_block)(ix)
        x, y = seq[:, :seq_len], seq[:, 1:]
        yield x, y
