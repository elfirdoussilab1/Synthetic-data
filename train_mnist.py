# In this file, we will make the experiments using the MNIST dataset
import numpy as np
import wandb
from dataset import *
from model import *
from  utils import fix_seed
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
import pandas as pd
import math

wandb.login(key='7c2c719a4d241a91163207b8ae5eb635bc0302a4')

# Set random seeds for reproducibility
fix_seed(1337)

device ="cuda" if torch.cuda.is_available() else "cpu"
print("Using device : ", device)

# Hyperparameters
batch_size = 64
weight_decay = 1e-5
num_steps = 500
eval_delta = 10
grad_clip = 1

# Learning rate
lr = 1e-3

# Datasets & DataLoaders
m = 0
m_estim = None
train_data = MNIST_generator(m, device, train = True)
test_data = MNIST_generator(m = 0, device = device, train = False)

# Dataloaders
train_loader = DataLoader(train_data, batch_size= batch_size, shuffle= True)
test_loader = DataLoader(test_data, batch_size= batch_size, shuffle= False)

wandb.init(
        # set the wandb project where this run will be logged
        project=f"Simple-NN",

        # track hyperparameters and run metadata
        config={
        "architecture": "NN",
        "dataset": "MNIST"
        },
        name = f"m = {m}, m_estim = {m_estim}"
    )
results = pd.DataFrame()
# Model
model = Mnist_Model().to(device)

# Learning rate schedular
# max_lr = 4e-3
# min_lr = 1e-4
# warmup_steps = 100
# def get_lr(step):
#     # 1) Liner warmup for warmup_iters steps
#     if step < warmup_steps:
#         return max_lr * (step + 1)/ warmup_steps
#     # 2) if step > lr_decay_iters, return min learning rate
#     if step > num_steps:
#         return min_lr
#     # 3) in between, use cosine decay down to min learning rate
#     decay_ratio = (step - warmup_steps) / (num_steps - warmup_steps)
#     assert 0 <= decay_ratio <= 1
#     coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio)) # coeff starts at 1 and gets to 0
#     return min_lr + coeff * (max_lr - min_lr)

# Loss and optimizer
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr = lr, weight_decay=weight_decay)

# Evaluation
def evaluate_accuracy(model, split):
    model.eval()
    acc = 0
    loader = train_loader if split in "train" else test_loader
    n = len(train_data) if split in "train" else len(test_data)

    for X, y in loader:
        logits = model(X)
        _, predicted = torch.max(logits, dim = 1)
        acc += (predicted == y).sum().item()

    return (acc / n) * 100

def evaluate_loss(model, split):
    model.eval()
    loss_accum = 0
    loader = train_loader if split in "train" else test_loader
    n = len(train_data) if split in "train" else len(test_data)
    for X, y in loader:
        logits = model(X)
        loss = loss_fn(logits, y)
        loss_accum += loss.item()
    
    return loss_accum / n

# Training Loop
train_iter = iter(train_loader)

for step in tqdm(range(num_steps)):
    if step % eval_delta == 0:
        with torch.no_grad():
            train_loss = evaluate_loss(model, "train")
            test_loss = evaluate_loss(model, "test")
            train_acc = evaluate_accuracy(model, "train")
            test_acc = evaluate_accuracy(model, "test")
            wandb.log({"Train Loss": train_loss, "Test Loss" : test_loss, "Train Accuracy": train_acc, "Test Accuracy": test_acc})
    
    model.train()
    optimizer.zero_grad()

    # Update learning rate
    #lr = get_lr(step)
    #for param_group in optimizer.param_groups:
    #    param_group['lr'] = lr

    batch = next(train_iter)
    if batch is None:
        train_iter = iter(train_loader)
        batch = next(train_iter)
    x, y = batch
    logits = model(x)
    loss = loss_fn(logits, y)
    loss.backward()
    optimizer.step()

# Saving the final model
torch.save(model.state_dict(), f'mnist-m-{m}-m_estim-{m_estim}.pth')
