import torch
from torch.utils.tensorboard import SummaryWriter
writer = SummaryWriter()
"""
This script contains the train loop for training the ENet
"""

def train_one_epoch(epoch, dataloader: DataLoader, device, model, criterion, optimizer):
    """
    Trains the Model for one epoch
    """
    model.train()
    train_loss = 0
    num_train_batches = 0
    for image, label in dataloader: 
        num_train_batches += 1
        image, label = image.to(device, dtype=torch.float32), target.to(device, dtype=torch.long)

        optimizer.zero_grad()
        pred = model(image)
        loss = criterion(pred, label)
        train_loss += loss.item()
        loss.backward()
        optimizer.step

    avg_loss = train_loss / num_train_batches

    return avg_loss


def eval_one_epoch(epoch, dataloader: DataLoader, device, model, criterion):
    """
    Evaluates the models performance on the validation datset.
    Called after training for one epoch
    """
    val_loss = 0
    num_val_batches = 0
    model.eval()
    with torch.no_grad():
        for image, label in dataloader:
            image, label = image.to(device, dtype=torch.float32), label.to(device, dtype=torch.long)

            pred = model(image)
            loss = criterion(pred, label)
            val_loss += loss.item()

        avg_loss = val_loss / num_val_batches
    
    return avg_loss


def run(num_epochs, dataloader, device, model, criterion, optimizer):
    """
    Training script
    """

    return None

def main():
    return None

if __name__ == '__main__':
    main()