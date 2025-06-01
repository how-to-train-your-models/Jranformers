"""
Sample from a trained JAX-based GPT model.
"""
import os
import jax
import jax.numpy as jnp
import equinox as eqx
from simple_parsing import ArgumentParser
from dataclasses import dataclass, fields

# Project imports
from jransformers.nano_gpt import model as nano_gpt_model
from jransformers.nano_gpt import config as nano_gpt_config
from jransformers.nano_gpt import data as nano_gpt_data


def get_latest_checkpoint(out_dir: str) -> str:
    """Loads the latest .eqx checkpoint from the given directory."""
    if not os.path.isdir(out_dir):
        raise FileNotFoundError(f"Checkpoint directory {out_dir} not found.")
    ckpts = [f for f in os.listdir(out_dir) if f.endswith('.eqx')]
    if not ckpts:
        raise FileNotFoundError(f"No checkpoint (.eqx file) found in {out_dir}")
    # Sort by modification time to get the latest
    ckpts.sort(key=lambda x: os.path.getmtime(os.path.join(out_dir, x)), reverse=True)
    return os.path.join(out_dir, ckpts[0])


def get_char_tokenizer():
    """Returns encoding and decoding functions based on the dataset's vocabulary."""
    vocab_info = nano_gpt_data.get_vocabulary_info()
    stoi = vocab_info["stoi"]
    itos = vocab_info["itos"]
    
    def encode_fn(s: str) -> jnp.ndarray:
        return jnp.array([stoi[c] for c in s if c in stoi], dtype=jnp.int32)

    def decode_fn(arr: jnp.ndarray) -> str:
        return "".join([itos[int(t)] for t in arr if int(t) in itos])
    
    return encode_fn, decode_fn

@dataclass
class SampleScriptConfig:
    """Configuration for the sampling script."""
    out_dir: str = "out"  # Directory to load checkpoint from. Assumed to be set by user if not 'out'.
    start: str = "\n"  # Prompt string, or "FILE:prompt.txt"
    num_samples: int = 3 # Number of samples to generate
    max_new_tokens: int = 100 # Number of tokens generated in each sample
    temperature: float = 0.8 # Sampling temperature (1.0 = no change)
    top_k: int = 200 # Retain only the top_k most likely tokens, 0 for no top-k filtering
    seed: int = 1337 # Random seed

def main():
    parser = ArgumentParser()
    parser.add_arguments(nano_gpt_config.GPTConfig, dest="gpt_config")
    # TrainConfig is not strictly needed if out_dir is managed by SampleScriptConfig
    # parser.add_arguments(nano_gpt_config.TrainConfig, dest="train_config") 
    parser.add_arguments(SampleScriptConfig, dest="sample_config")
    
    args = parser.parse_args()

    gpt_config: nano_gpt_config.GPTConfig = args.gpt_config
    sample_config: SampleScriptConfig = args.sample_config
    
    # Setup JAX key
    key = jax.random.PRNGKey(sample_config.seed)
    model_init_key, generation_key = jax.random.split(key)
    
    # Load tokenizer and actual vocabulary size
    encode_fn, decode_fn = get_char_tokenizer()
    actual_vocab_size = nano_gpt_data.get_vocab_size()

    # Ensure GPTConfig uses the actual vocab_size from the data
    if gpt_config.vocab_size != actual_vocab_size:
        print(f"Warning: GPTConfig.vocab_size ({gpt_config.vocab_size}) differs from actual data vocab_size ({actual_vocab_size}).")
        print(f"Overriding GPTConfig.vocab_size to {actual_vocab_size}.")
        
        current_gpt_config_fields = {f.name: getattr(gpt_config, f.name) for f in fields(nano_gpt_config.GPTConfig)}
        current_gpt_config_fields['vocab_size'] = actual_vocab_size
        gpt_config = nano_gpt_config.GPTConfig(**current_gpt_config_fields)

    # Initialize model structure
    # This model instance is a template; weights will be loaded from checkpoint.
    # Ensure this gpt_config is the one with corrected vocab_size.
    empty_model = nano_gpt_model.GPT(model_init_key, gpt_config)
    
    # Load checkpoint
    try:
        ckpt_path = get_latest_checkpoint(sample_config.out_dir)
        print(f"Loading checkpoint from {ckpt_path}...")
        # tree_deserialise_leaves loads the checkpoint into the structure of empty_model
        loaded_model = eqx.tree_deserialise_leaves(ckpt_path, empty_model)
    except FileNotFoundError as e:
        print(f"Error: {e}")
        print(f"Please ensure --out_dir ('{sample_config.out_dir}') contains a valid .eqx checkpoint or specify the correct directory.")
        return
        
    print(f"Model loaded. Generating {sample_config.num_samples} samples...")

    # Prepare prompt
    prompt_text = sample_config.start
    if prompt_text.startswith('FILE:'):
        file_path = prompt_text[5:]
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                prompt_text = f.read()
            print(f"Loaded prompt from {file_path}")
        except FileNotFoundError:
            print(f"Error: Prompt file {file_path} not found. Using the literal string '{sample_config.start}' as prompt.")
            # Fallback to using the string itself if file not found
            prompt_text = sample_config.start 
    
    start_ids = encode_fn(prompt_text)
    if start_ids.shape[0] == 0 and len(prompt_text) > 0:
        print(f"Warning: Prompt '{prompt_text}' encoded to an empty sequence. Check if characters are in vocabulary.")
    
    # Run generation loop
    for i in range(sample_config.num_samples):
        generation_key, sample_key = jax.random.split(generation_key) # New key for each sample
        
        print(f"--- Sample {i+1}/{sample_config.num_samples} ---")
        
        # Ensure top_k is None if 0, as per model.decode's expectation for no top-k
        current_top_k = sample_config.top_k if sample_config.top_k > 0 else None

        generated_tokens = loaded_model.decode(
            key=sample_key, 
            initial_tokens=start_ids, 
            max_new_tokens=sample_config.max_new_tokens,
            temperature=sample_config.temperature,
            top_k=current_top_k
        )
        
        generated_text = decode_fn(generated_tokens)
        print(generated_text)
    print('---------------')

if __name__ == "__main__":
    main()
