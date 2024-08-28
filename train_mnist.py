# In this file, we will make the experiments using the MNIST dataset
import numpy as np
import wandb
from dataset import *
from model import *
from  utils import fix_seed, get_lr
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
num_steps = 2000
eval_delta = 10
#Learning rate schedular
max_lr = 5e-3
min_lr = max_lr * 0.1
warmup_steps = 200

# Datasets & DataLoaders
n = 1000
ms = [0, 500, 1000, 4000, 10000]
for m in ms:
    m_estim = None
    train_data = MNIST_generator(n, m, device, train = True)
    test_data = MNIST_generator(n, m = 0, device = device, train = False)

    # Dataloaders
    train_loader = DataLoader(train_data, batch_size= batch_size, shuffle= True)
    test_loader = DataLoader(test_data, batch_size= batch_size, shuffle= False)

    wandb.init(
            # set the wandb project where this run will be logged
            project=f"Simple-NN-ReLU",

            # track hyperparameters and run metadata
            config={
            "architecture": "Linear",
            "dataset": "MNIST"
            },
            name = f"n = {n}, m = {m}, m_estim = {m_estim}"
        )

    # Model
    model = Mnist_Model().to(device)
    #model = log_reg().to(device)

    # Loss and optimizer
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr = max_lr, weight_decay=weight_decay)

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
        lr = get_lr(step, max_lr, min_lr, warmup_steps, num_steps)
        for param_group in optimizer.param_groups:
           param_group['lr'] = lr

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

                # Add to results csv
                row = {'n': n, 'm': m, 'm_estim': m_estim,
                       'Train Loss': train_loss, 'Test Loss': test_loss,
                       'Train Accuracy': train_acc, 'Test Accuracy': test_acc}
                result = pd.read_csv('results.csv')
                df = pd.concat([result, pd.DataFrame([row])], ignore_index= True)
                df.to_csv('results.csv', index = False)

    # Saving the final model
    torch.save(model.state_dict(), f'mnist-n-{n}-m-{m}-m_estim-{m_estim}.pth')
    wandb.finish()
