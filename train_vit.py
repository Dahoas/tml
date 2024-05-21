"""
Implements SimMIM (https://arxiv.org/pdf/2111.09886) training. 
"""
import os
import time
import math
import pickle
from contextlib import nullcontext
from pathlib import Path
import json
import datetime

import numpy as np
import torch
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group

from transformers import ViTImageProcessor, ViTConfig, ViTForMaskedImageModeling
from PIL import Image
from datasets import load_dataset

def configurator():
    import sys
    from ast import literal_eval
    for arg in sys.argv[1:]:
        if '=' not in arg:
            # assume it's the name of a config file
            assert not arg.startswith('--')
            config_file = arg
            print(f"Overriding config with {config_file}:")
            with open(config_file) as f:
                print(f.read())
            exec(open(config_file).read())
        else:
            # assume it's a --key=value argument
            assert arg.startswith('--')
            key, val = arg.split('=')
            key = key[2:]
            if key in globals():
                try:
                    # attempt to eval it it (e.g. if bool, number, or etc)
                    attempt = literal_eval(val)
                except (SyntaxError, ValueError):
                    # if that goes wrong, just use the string
                    attempt = val
                # ensure the types match ok
                assert type(attempt) == type(globals()[key])
                # cross fingers
                print(f"Overriding: {key} = {attempt}")
                globals()[key] = attempt
            else:
                raise ValueError(f"Unknown config key: {key}!!!")

# -----------------------------------------------------------------------------
# default config values designed to train a gpt2 (124M) on OpenWebText
# I/O
eval_interval = 100
log_interval = 1
eval_only = False # if True, script exits right after the first eval
always_save_checkpoint = True # if True, always save a checkpoint after each eval
init_from = 'hybrid' # 'scratch' # 'scratch' or 'resume' or 'gpt2*'
# wandb logging
wandb_log = False # disabled by default
wandb_project = 'owt'
wandb_run_name = 'gpt2' # 'run' + str(time.time())
# data
dataset = 'imagenet-1k'
dataset_frac = 1.0
dataset_max_len = 1_281_167
dataset_len = int(dataset_frac * dataset_max_len)
image_size = 224
num_channels = 3
patch_size = 32
encoder_stride = patch_size
patches_per_row = 7
assert patches_per_row * patch_size == image_size

mask_rate = 0.6 # probability of randomly masking a patch

gradient_accumulation_steps = 32 # used to simulate larger batch sizes
batch_size = 16 # if gradient_accumulation_steps > 1, this is the micro-batch size
eval_iters = 2048 // batch_size
# Target global batch size is 2048
# (ViT) model
num_hidden_layers = 12
num_attention_heads = 12
hidden_size = 768
dropout = 0.0 # for pretraining 0 is good, for finetuning try 0.1+
bias = False # do we use bias inside LayerNorm and Linear layers?
# adamw optimizer
learning_rate = 8e-4 # max learning rate
max_iters = 600000 # 600000 # total number of training iterations
weight_decay = 5e-2
beta1 = 0.9
beta2 = 0.999
grad_clip = 1.0 # clip gradients at this value, or disable if == 0.0
# learning rate decay settings
decay_lr = True # whether to decay the learning rate
warmup_iters = 2000 # how many steps to warm up for
lr_decay_iters = 600000 # should be ~= max_iters per Chinchilla
min_lr = 6e-5 # minimum learning rate, should be ~= learning_rate/10 per Chinchilla
# DDP settings
backend = 'nccl' # 'nccl', 'gloo', etc.
# system
device = 'cuda' if torch.cuda.is_available() else 'cpu'
#device = 'cuda' # examples: 'cpu', 'cuda', 'cuda:0', 'cuda:1' etc., or try 'mps' on macbooks
dtype = 'bfloat16' if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else 'float16' # 'float32', 'bfloat16', or 'float16', the latter will auto implement a GradScaler
compile = False # use PyTorch 2.0 to compile the model to be faster
log_path = "logs/tml_train/vit_test"
out_dir = str(log_path)
# -----------------------------------------------------------------------------
config_keys = [k for k,v in globals().items() if not k.startswith('_') and isinstance(v, (int, float, bool, str))]
configurator() # overrides from command line or config file
if init_from == 'scratch':
    log_path = os.path.join(log_path, datetime.datetime.now().strftime("%Y-%m-%d-%H:%M:%S"))
elif init_from == 'hybrid':
    files = Path(log_path).glob("*.pt")
    if len(list(files)) > 0:
        init_from = 'resume'
    else:
        init_from = 'scratch'
out_dir = str(log_path)
config = {k: globals()[k] for k in config_keys} # will be useful for logging
print("Config: ", config)
log_path = Path(log_path)
# -----------------------------------------------------------------------------

# various inits, derived attributes, I/O setup
ddp = int(os.environ.get('RANK', -1)) != -1 # is this a ddp run?
if ddp:
    init_process_group(backend=backend)
    ddp_rank = int(os.environ['RANK'])
    ddp_local_rank = int(os.environ['LOCAL_RANK'])
    ddp_world_size = int(os.environ['WORLD_SIZE'])
    device = f'cuda:{ddp_local_rank}'
    torch.cuda.set_device(device)
    master_process = ddp_rank == 0 # this process will do logging, checkpointing etc.
    seed_offset = ddp_rank # each process gets a different seed
    # world_size number of processes will be training simultaneously, so we can scale
    # down the desired gradient accumulation iterations per process proportionally
    assert gradient_accumulation_steps % ddp_world_size == 0
    gradient_accumulation_steps //= ddp_world_size
else:
    # if not ddp, we are running on a single gpu, and one process
    ddp_rank = 0
    ddp_world_size = 1
    master_process = True
    seed_offset = 0
    ddp_world_size = 1

if master_process:
    os.makedirs(out_dir, exist_ok=True)
# logging
if wandb_log and master_process:
    import wandb
    wandb.init(project=wandb_project, name=wandb_run_name, config=config)
elif master_process and not eval_only:
    log_path.mkdir(exist_ok=True, parents=True,)
    with open(os.path.join(log_path, "config.json"), "w") as f:
        json.dump(config, f, indent=2)

torch.manual_seed(1337 + seed_offset)
torch.backends.cuda.matmul.allow_tf32 = True # allow tf32 on matmul
torch.backends.cudnn.allow_tf32 = True # allow tf32 on cudnn
device_type = 'cuda' if 'cuda' in device else 'cpu' # for later use in torch.autocast
# note: float16 data type will automatically use a GradScaler
ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[dtype]
ctx = nullcontext() if device_type == 'cpu' else torch.amp.autocast(device_type=device_type, dtype=ptdtype)

# poor man's data loader
train_data = load_dataset(dataset, split="train", streaming=True)
train_iter = iter(train_data)
val_data = load_dataset(dataset, split="validation", streaming=True)
val_iter = iter(val_data)

processor = ViTImageProcessor.from_pretrained("google/vit-base-patch16-224")

forward = -1

def is_rank_data(forward, ddp_rank):
    return (forward + (ddp_world_size - ddp_rank)) % ddp_world_size == 0

def get_batch(split):
    global train_iter, val_iter, forward
    data = train_iter if split == 'train' else val_iter
    xs = []
    while len(xs) < batch_size:
        try:
            x = next(data)["image"]
            forward += 1
            if forward >= dataset_len and split == "train":
                train_iter = iter(train_data)
                forward = -1
        except StopIteration:
            if split == "train":
                train_iter = iter(train_data)
                data = train_iter
            else:
                val_iter = iter(val_data)
                data = val_iter
            x = next(data)["image"]
            forward = 0
        # Ensure image is RGB
        if x.mode == "RGB" and is_rank_data(forward, ddp_rank):
            xs.append(x)
    xs = processor(images=xs, return_tensors="pt")["pixel_values"]
    # Randomly select patches to mask
    bool_masked_pos = np.random.choice([0, 1], 
                                       p=[1-mask_rate, mask_rate], 
                                       size=(batch_size, patches_per_row * patches_per_row),)
    bool_masked_pos = torch.tensor(bool_masked_pos, dtype=torch.bool)
    if device_type == 'cuda':
        # pin arrays x,y, which allows us to move them to GPU asynchronously (non_blocking=True)
        xs = xs.pin_memory().to(device, non_blocking=True)
        bool_masked_pos = bool_masked_pos.pin_memory().to(device, non_blocking=True)
    else:
        xs = xs.to(device)
        bool_masked_pos = bool_masked_pos.to(device)
    return xs, bool_masked_pos

# init these up here, can override if init_from='resume' (i.e. from a checkpoint)
iter_num = 0
best_val_loss = 1e9

# model init
model_args = dict(hidden_size=hidden_size,
                num_hidden_layers=num_hidden_layers,
                num_attention_heads=num_attention_heads,
                image_size=image_size,
                patch_size=patch_size,
                num_channels=num_channels,
                encoder_stride=encoder_stride,)
if init_from == 'scratch':
    config = ViTConfig(**model_args)
    model = ViTForMaskedImageModeling(config)
elif init_from == 'resume':
    print(f"Resuming training from {out_dir}")
    with open(os.path.join(out_dir, "config.json")) as f:
        pre_config = json.load(f)
    # Remove keys we allow to change after resuming
    config.pop("init_from"), pre_config.pop("init_from")
    config.pop("max_iters"), pre_config.pop("max_iters")
    pre_key_set = set(pre_config.keys())
    key_set = set(config.keys())
    if pre_key_set != key_set:
        raise ValueError(f"Key diff: \n\
                           New - Pre: {key_set - pre_key_set}\n\
                           Pre - New: {pre_key_set - key_set}")
    pre_values_set = set(pre_config.values())
    values_set = set(config.values())
    if pre_values_set != values_set:
        raise ValueError(f"Key diff: \n\
                           New - Pre: {values_set - pre_values_set}\n\
                           Pre - New: {pre_values_set - values_set}")
    # resume training from a checkpoint.
    ckpt_path = os.path.join(out_dir, 'ckpt.pt')
    checkpoint = torch.load(ckpt_path, map_location=device)
    checkpoint_model_args = checkpoint['model_args']
    # force these config attributes to be equal otherwise we can't even resume training
    # the rest of the attributes (e.g. dropout) can stay as desired from command line
    for k in ["hidden_size", "num_hidden_layers", "num_attention_heads", "image_size", "patch_size", "num_channels", "encoder_stride"]:
        model_args[k] = checkpoint_model_args[k]
    config = ViTConfig(**model_args)
    # create the model
    model = ViTForMaskedImageModeling(config)
    state_dict = checkpoint['model']
    # fix the keys of the state dictionary :(
    # honestly no idea how checkpoints sometimes get this prefix, have to debug more
    unwanted_prefix = '_orig_mod.'
    for k,v in list(state_dict.items()):
        if k.startswith(unwanted_prefix):
            state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)
    model.load_state_dict(state_dict)
    iter_num = checkpoint['iter_num']
    best_val_loss = checkpoint['best_val_loss']
    
    # also need to iterate through the train dataset when resuming training
    dataset_offset = iter_num % dataset_len
    for _ in range(dataset_offset):
        get_batch(split="train")
model.to(device)

# initialize a GradScaler. If enabled=False scaler is a no-op
scaler = torch.cuda.amp.GradScaler(enabled=(dtype == 'float16'))

# optimizer
def get_optimizer(model, weight_decay, learning_rate, betas, device_type):
    param_dict = {pn: p for pn, p in model.named_parameters()}
    # filter out those that do not require grad
    param_dict = {pn: p for pn, p in param_dict.items() if p.requires_grad}
    # create optim groups. Any parameters that is 2D will be weight decayed, otherwise no.
    # i.e. all weight tensors in matmuls + embeddings decay, all biases and layernorms don't.
    decay_params = [p for n, p in param_dict.items() if p.dim() >= 2]
    nodecay_params = [p for n, p in param_dict.items() if p.dim() < 2]
    optim_groups = [
        {'params': decay_params, 'weight_decay': weight_decay},
        {'params': nodecay_params, 'weight_decay': 0.0}
    ]
    num_decay_params = sum(p.numel() for p in decay_params)
    num_nodecay_params = sum(p.numel() for p in nodecay_params)
    print(f"num decayed parameter tensors: {len(decay_params)}, with {num_decay_params:,} parameters")
    print(f"num non-decayed parameter tensors: {len(nodecay_params)}, with {num_nodecay_params:,} parameters")
    # Create AdamW optimizer and use the fused version if it is available
    optimizer = torch.optim.AdamW(optim_groups, lr=learning_rate, betas=betas)
    return optimizer

optimizer = get_optimizer(model, weight_decay, learning_rate, (beta1, beta2), device_type)
if init_from == 'resume':
    optimizer.load_state_dict(checkpoint['optimizer'])
checkpoint = None # free up memory

# wrap model into DDP container
if ddp:
    model = DDP(model, device_ids=[ddp_local_rank])

# helps estimate an arbitrarily accurate loss over either split using many batches
@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, bool_masked_pos = get_batch(split)
            with ctx:
                outputs = model(X, bool_masked_pos=bool_masked_pos)
                loss, pred_pixels = outputs.loss, outputs.reconstruction
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out

# learning rate decay scheduler (cosine with warmup)
def get_lr(it):
    # 1) linear warmup for warmup_iters steps
    if it < warmup_iters:
        return learning_rate * it / warmup_iters
    # 2) if it > lr_decay_iters, return min learning rate
    if it > lr_decay_iters:
        return min_lr
    # 3) in between, use cosine decay down to min learning rate
    decay_ratio = (it - warmup_iters) / (lr_decay_iters - warmup_iters)
    assert 0 <= decay_ratio <= 1
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio)) # coeff ranges 0..1
    return min_lr + coeff * (learning_rate - min_lr)

# training loop
X, bool_masked_pos = get_batch('train') # fetch the very first batch
t0 = time.time()
local_iter_num = 0 # number of iterations in the lifetime of this process
raw_model = model.module if ddp else model # unwrap DDP container if needed
running_mfu = -1.0
dt = 0
while True:

    # determine and set the learning rate for this iteration
    lr = get_lr(iter_num) if decay_lr else learning_rate
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

    # evaluate the loss on train/val sets and write checkpoints
    if iter_num % eval_interval == 0 and master_process:
        losses = estimate_loss()
        print(f"step {iter_num}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")
        stats = {
                  "iter": iter_num,
                  "train/loss": losses['train'].item(),
                  "val/loss": losses['val'].item(),
                  "lr": lr,
                  "mfu": running_mfu*100, # convert to percentage
                  "time": dt,
                }
        if wandb_log:
            wandb.log(stats)
        elif master_process:
            with open(os.path.join(log_path, "stats.jsonl"), "a+") as f:
                json.dump(stats, f)
                f.write("\n")
        if losses['val'] < best_val_loss or always_save_checkpoint:
            best_val_loss = losses['val']
            if iter_num > 0:
                checkpoint = {
                    'model': raw_model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'model_args': model_args,
                    'iter_num': iter_num,
                    'best_val_loss': best_val_loss,
                    'config': config,
                }
                print(f"saving checkpoint to {out_dir}")
                torch.save(checkpoint, os.path.join(out_dir, 'ckpt.pt'))
    if iter_num == 0 and eval_only:
        break

    # forward backward update, with optional gradient accumulation to simulate larger batch size
    # and using the GradScaler if data type is float16
    for micro_step in range(gradient_accumulation_steps):
        if ddp:
            # in DDP training we only need to sync gradients at the last micro step.
            # the official way to do this is with model.no_sync() context manager, but
            # I really dislike that this bloats the code and forces us to repeat code
            # looking at the source of that context manager, it just toggles this variable
            model.require_backward_grad_sync = (micro_step == gradient_accumulation_steps - 1)
        with ctx:
            outputs = model(X, bool_masked_pos=bool_masked_pos)
            loss, pred_pixels = outputs.loss, outputs.reconstruction
            loss = loss / gradient_accumulation_steps # scale the loss to account for gradient accumulation
        # immediately async prefetch next batch while model is doing the forward pass on the GPU
        X, bool_masked_pos = get_batch('train')
        # backward pass, with gradient scaling if training in fp16
        scaler.scale(loss).backward()
    # clip the gradient
    if grad_clip != 0.0:
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
    # step the optimizer and scaler if training in fp16
    scaler.step(optimizer)
    scaler.update()
    # flush the gradients as soon as we can, no need for this memory anymore
    optimizer.zero_grad(set_to_none=True)

    # timing and logging
    t1 = time.time()
    dt = t1 - t0
    t0 = t1
    if iter_num % log_interval == 0 and master_process:
        # get loss as float. note: this is a CPU-GPU sync point
        # scale up to undo the division above, approximating the true total loss (exact would have been a sum)
        lossf = loss.item() * gradient_accumulation_steps
        print(f"iter {iter_num}: loss {lossf:.4f}, time {dt*1000:.2f}ms")
    iter_num += 1
    local_iter_num += 1

    # termination conditions
    if iter_num > max_iters:
        break

if ddp:
    destroy_process_group()

