# In this file, we will train a Discriminator to differentiate from MNIST data and GAN generated ones
import torch
import wandb
from dataset import *
from model import *
from  utils import fix_seed, get_lr
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
import pandas as pd

wandb.login(key='7c2c719a4d241a91163207b8ae5eb635bc0302a4')

# Set random seeds for reproducibility
fix_seed(1337)

device ="cuda" if torch.cuda.is_available() else "cpu"
print("Using device : ", device)

# Hyperparameters
batch_size = 128
weight_decay = 1e-4
num_steps = 4000
eval_delta = 50
#Learning rate schedular
max_lr = 5e-3
min_lr = max_lr * 0.1
warmup_steps = 800
threshold = 0.

# Datasets
m = 12000
train_data = MNIST_verifier_data(m = m, train = True, device= device)
test_data = MNIST_verifier_data(m = m // 6, train = False, device= device)

# DataLoaders
train_loader = DataLoader(train_data, batch_size= batch_size, shuffle = True)
test_loader = DataLoader(test_data, batch_size= batch_size, shuffle = False)

def evaluate_accuracy(model, split):
    model.eval()
    acc = 0
    loader = train_loader if split in "train" else test_loader
    n = len(train_data) if split in "train" else len(test_data)

    for X, y in loader:
        logits = model(X)
        predictions = (logits > 0.).long()
        acc += (predictions == y).sum().item()

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

wandb.init(
            # set the wandb project where this run will be logged
            project=f"Training-Verifier",

            # track hyperparameters and run metadata
            config={
            "architecture": "Discriminator",
            "dataset": "MNIST"
            },
            name = f"Verifier, m = {m}"
        )

# Model
model = Discriminator(28*28, 1)

# Loss and optimizer
loss_fn = nn.BCELoss()
optimizer = torch.optim.AdamW(model.parameters(), lr = max_lr, weight_decay=weight_decay)

# Training Loop
train_iter = iter(train_loader)

for step in tqdm(range(num_steps)):
    # Update learning rate
    lr = get_lr(step, max_lr, min_lr, warmup_steps, num_steps)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

    if step % eval_delta == 0:
        with torch.no_grad():
            train_loss = evaluate_loss(model, "train")
            test_loss = evaluate_loss(model, "test")
            train_acc = evaluate_accuracy(model, "train")
            test_acc = evaluate_accuracy(model, "test")
            wandb.log({"Train Loss": train_loss, "Test Loss" : test_loss, "Train Accuracy": train_acc, "Test Accuracy": test_acc, 'lr': lr})

    model.train()
    optimizer.zero_grad()

    try :
        batch = next(train_iter)
    except:
        train_iter = iter(train_loader)
        batch = next(train_iter)
    x, y = batch
    logits = model(x)
    loss = loss_fn(logits, y)
    loss.backward()

    # Gradient clipping
    #norm = torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip) # clipping threshold is grad_clip
    optimizer.step()

    if step == num_steps -1: # Last step
        with torch.no_grad():
            train_loss = evaluate_loss(model, "train")
            test_loss = evaluate_loss(model, "test")
            train_acc = evaluate_accuracy(model, "train")
            test_acc = evaluate_accuracy(model, "test")
            wandb.log({"Train Loss": train_loss, "Test Loss" : test_loss, "Train Accuracy": train_acc, "Test Accuracy": test_acc})

# Saving the final model
torch.save(model.state_dict(), f'verifier-m-{m}.pth')
wandb.finish()
