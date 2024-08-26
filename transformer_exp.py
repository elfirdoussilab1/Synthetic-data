from model import *
from dataset import *
import torch.optim as optim
import math
import wandb
from utils import fix_seed
from tqdm.auto import tqdm

wandb.login(key='7c2c719a4d241a91163207b8ae5eb635bc0302a4')

wandb.init(
        # set the wandb project where this run will be logged
        project=f"Transformer-Grammar",

        # track hyperparameters and run metadata
        config={
        "architecture": "Transformer",
        "dataset": "Grammar"
        }
    )

# Set random seeds for reproducibility
fix_seed(1337)
 
# Hyperparameters
vocab_size = 1000
seq_length = 15
d_embd = 64
num_heads = 4
n_layer = 2
batch_size = 32

device ="cuda" if torch.cuda.is_available() else "cpu"
print("Using device : ", device)

# Datasets
grammar = SimpleGrammar(vocab_size)
train_data = generate_grammar_data(1000000, grammar, seq_length + 1) # samples of length seq_length + 1
val_data = generate_grammar_data(10000, grammar, seq_length + 1)

def get_batch(split, batch_size):
    data = train_data if split == 'train' else val_data
    idx = random.randint(0, len(train_data) - 1)
    x = data[idx : idx + batch_size, :-1]
    y = data[idx : idx + batch_size , 1:]
    return x, y

def evaluate_loss(split, model, batch):
    model.eval()
    data = train_data if split == 'train' else val_data
    n = len(data) // batch
    loss_accum = 0
    for idx in range(0, len(data), batch):
        x = data[idx : idx + batch_size, :-1].to(device)
        y = data[idx : idx + batch_size , 1:].to(device)
        logits, loss = model(x, y)
        loss_accum += loss.item()
    return loss_accum / n


# Model
model = GPT(vocab_size, d_embd, num_heads, seq_length, n_layer, device).to(device)

# Learning-rate scheduler
max_lr = 6e-4
min_lr = max_lr * 0.1
warmup_steps = 1000
max_steps = 10000
def get_lr(step):
    # 1) Liner warmup for warmup_iters steps
    if step < warmup_steps:
        return max_lr * (step + 1)/ warmup_steps
    # 2) if step > lr_decay_iters, return min learning rate
    if step > max_steps:
        return min_lr
    # 3) in between, use cosine decay down to min learning rate
    decay_ratio = (step - warmup_steps) / (max_steps - warmup_steps)
    assert 0 <= decay_ratio <= 1
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio)) # coeff starts at 1 and gets to 0
    return min_lr + coeff * (max_lr - min_lr)

# Optimizer
optimizer = optim.AdamW(model.parameters(), lr = max_lr, fused= True)

# # First evaluation
# model.eval()
# with torch.no_grad():
#     train_loss = evaluate_loss('train', model, 100)
#     val_loss = evaluate_loss('val', model, 100)
#     print("Train loss", train_loss)
#     print("Train loss", val_loss)
#     wandb.log({"Train Loss" : train_loss, "Test Loss": val_loss})


for step in tqdm(range(max_steps)):
    if step % 50 == 0:
        model.eval()
        with torch.no_grad():
            train_loss = evaluate_loss('train', model, 100)
            val_loss = evaluate_loss('val', model, 100)
            print("Train loss", train_loss)
            wandb.log({"Train Loss" : train_loss, "Test Loss": val_loss})
    
    model.train()
    optimizer.zero_grad()
    # Update lr

    lr = get_lr(step)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

    x, y = get_batch('train', batch_size)
    x = x.to(device)
    y = y.to(device)
    
    logits, loss = model(x, y)
    loss.backward()
    optimizer.step()

# Saving the final model
torch.save(model.state_dict(), f'transformer-grammar_p_{d_embd}.pth')










