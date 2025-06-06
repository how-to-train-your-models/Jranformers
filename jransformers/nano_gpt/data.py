import jax 

from datasets import load_dataset
from jax import numpy as jnp
from jaxtyping import Float, Array, PRNGKeyArray

dataset_name = 'karpathy/tiny_shakespeare'

def get_dataset():
    return load_dataset(dataset_name, trust_remote_code=True)

def get_infinite_dataloader(key: PRNGKeyArray, split_type: str, batch_size: int, seq_len: int):
    """Creates an infinite dataloader that yields batches of data.    
    
    Yields:
        batch: Array of shape (batch_size, seq_len) containing token sequences
    """    
    dataset = get_dataset()
    assert split_type in dataset.keys(), f"Split {split_type} not found in dataset"
    data = dataset[split_type]['text']
    chars = sorted(list(set(''.join(data))))
    stoi = {ch: i for i, ch in enumerate(chars)}
    
    def encode(s):
        return [stoi[c] for c in s]
    
    data = encode(''.join(data))
    n = len(data)
    
    while True:
        # create a new key every iteration to avoid returning the same batch
        key, subkey = jax.random.split(key)
        # sample random indices (batch_size, )
        ix = jax.random.randint(subkey, (batch_size,), 0, n - seq_len)
        # for every index, extract a sequence of length seq_len
        x = jnp.stack([jnp.array(data[i:i+seq_len]) for i in ix])
        # for every index, extract the sequence that follows the previous one
        y = jnp.stack([jnp.array(data[i+1:i+seq_len+1]) for i in ix])
        yield x, y
