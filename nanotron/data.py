import jax
from datasets import load_dataset
from jax import numpy as jnp
from jaxtyping import Float, Array, PRNGKeyArray
from typing import Any, Dict, Generator, Optional, Tuple, Callable

dataset_name = 'karpathy/tiny_shakespeare'
_cached_vocab_info: Optional[Dict[str, Any]]  = None

def get_dataset() -> Any:
    return load_dataset(dataset_name, trust_remote_code=True)

def get_vocabulary_info() -> Dict[str, Any]:
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

def get_encoder() -> Dict[str, int]:
    return get_vocabulary_info()["stoi"]

def get_decoder() -> Dict[int, str]:
    return get_vocabulary_info()["itos"]

def get_vocab_size() -> int:
    return get_vocabulary_info()["vocab_size"]

def get_infinite_dataloader(
    key: PRNGKeyArray, split_type: str, batch_size: int, seq_len: int
) -> Generator[Tuple[jnp.ndarray, jnp.ndarray], None, None]:
    dataset_obj = get_dataset()
    assert split_type in dataset_obj.keys(), f"Split {split_type} not found in dataset"
    
    text_list_for_split = dataset_obj[split_type]['text']
    full_text_for_split = "".join(text_list_for_split)

    vocab_info = get_vocabulary_info()
    stoi = vocab_info["stoi"]
    
    def encode_fn(s: str) -> list[int]:
        return [stoi[c] for c in s if c in stoi]
    
    data_encoded = encode_fn(full_text_for_split)
    n = len(data_encoded)

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
    
    while True:
        # create a new key every iteration to avoid returning the same batch
        key, subkey = jax.random.split(key)
        # sample random indices (batch_size, )
        ix = jax.random.randint(subkey, (batch_size,), 0, n - seq_len)
        # for every index, extract a sequence of length seq_len
        x = jnp.stack([jnp.array(data_encoded[i:i+seq_len]) for i in ix])
        # for every index, extract the sequence that follows the previous one
        y = jnp.stack([jnp.array(data_encoded[i+1:i+seq_len+1]) for i in ix])
        yield x, y
