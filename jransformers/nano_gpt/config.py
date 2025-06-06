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
    num_steps: int = 500
    batch_size: int = 1024
    out_dir = '/mnt/nvme9n1/huggingface/hub/out/'
    eval_interval = 100
    log_interval = 1
    eval_iters = 200
    eval_only = False # if True, script exits right after the first eval
    always_save_checkpoint = True # if True, always save a checkpoint after each eval
    init_from = 'scratch' # 'scratch' or 'resume' or 'gpt2*'    

    gradient_accumulation_steps = 5 * 8 # used to simulate larger batch sizes

    learning_rate = 1e-4 # max learning rate

    weight_decay = 1e-1
    beta1 = 0.9
    beta2 = 0.95
    grad_clip = 1.0 # clip gradients at this value, or disable if == 0.0
    # learning rate decay settings
    decay_lr = True # whether to decay the learning rate
    warmup_iters = 2000 # how many steps to warm up for
    lr_decay_iters = 600000 # should be ~= max_iters per Chinchilla
    min_lr = 6e-5 # minimum learning rate, should be ~= learning_rate/10 per Chinchilla    
    

