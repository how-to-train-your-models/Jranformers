from dataclasses import dataclass

@dataclass
class GPTConfig:
    """Model configuration"""
    block_size: int = 256
    n_layers: int = 6
    vocab_size: int = -1 # to be set later, e.g. from meta.pkl
    n_head: int = 6
    n_embed: int = 384
    dropout: float = 0.2
    bias: bool = True  # True: bias in Linears and LayerNorms, like GPT-2. False: a bit better and faster

@dataclass
class TrainConfig:
    """Training configuration"""
    # Settings adapted from nanoGPT's tiny Shakespeare configuration
    num_steps: int = 5000
    batch_size: int = 64
    # Use the original output directory path
    out_dir = '/mnt/nvme9n1/huggingface/hub/out/'
    eval_interval = 250
    log_interval = 10
    eval_iters = 200
    eval_only = False  # if True, script exits right after the first eval
    always_save_checkpoint = False
    init_from = 'scratch'  # 'scratch' or 'resume' or 'gpt2*'

    gradient_accumulation_steps = 1  # nanoGPT uses no accumulation here

    learning_rate = 1e-3  # max learning rate

    weight_decay = 1e-1
    beta1 = 0.9
    beta2 = 0.99
    grad_clip = 1.0  # clip gradients at this value, or disable if == 0.0
    # learning rate decay settings
    decay_lr = True  # whether to decay the learning rate
    warmup_iters = 100
    lr_decay_iters = 5000
    min_lr = 1e-4
    

