# In this file, we will make the experiments using the MNIST dataset
import numpy as np
import wandb
from dataset import *
from model import *
from  utils import fix_seed
from torchvision import datasets
from torchvision.transforms import ToTensor
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

wandb.login(key='7c2c719a4d241a91163207b8ae5eb635bc0302a4')

# Set random seeds for reproducibility
fix_seed(1337)

device ="cuda" if torch.cuda.is_available() else "cpu"
print("Using device : ", device)

# Hyperparameters
batch_size = 64
weight_decay = 1e-5
num_steps = 1000
eval_delta = 10

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

# Model
model = Mnist_Model().to(device)

# Loss and optimizer
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr = lr, weight_decay=weight_decay)

# Evaluation
def evaluate_accuracy(model, split):
    model.eval()
    acc = 0
    loader = train_data if split == "train" else test_loader
    n = len(train_data) if split == "train" else len(test_data)

    for X, y in loader:
        logits = model(X)
        _, predicted = torch.max(logits, dim = 1)
        acc += (predicted == y).sum().item()

    return (acc / n) * 100

def evaluate_loss(model, split):
    model.eval()
    loss_accum = 0
    loader = train_data if split == "train" else test_loader
    n = len(train_data) if split == "train" else len(test_data)
    for X, y in loader:
        logits = model(X)
        loss = loss_fn(logits, y)
        loss_accum += loss.item()
    
    return loss_accum / n

# Training Loop
train_iter = iter(train_loader)

for step in tqdm(range(num_steps)):
    if step % eval_delta:
        with torch.no_grad():
            train_loss = evaluate_loss(model, "train")
            test_loss = evaluate_loss(model, "test")
            train_acc = evaluate_accuracy(model, "train")
            test_acc = evaluate_accuracy(model, "test")
            wandb.log({"Train Loss": train_loss, "Test Loss" : test_loss, "Train Accuracy": train_acc, "Test Accuracy": test_acc})
    
    model.train()
    optimizer.zero_grad()

    batch = next(iter)
    if batch is None:
        train_iter = iter(train_loader)
        batch = next(iter)
    x, y = batch
    logits = model(x)
    loss = loss_fn(logits, y)
    loss.backward()
    optimizer.step()

# Saving the final model
torch.save(model.state_dict(), f'mnist-m-{m}-m_estim-{m_estim}.pth')
