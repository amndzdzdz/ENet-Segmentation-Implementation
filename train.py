import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
from torch.utils.tensorboard import SummaryWriter
writer = SummaryWriter()
from dataset import GTAData
from model import ENet
from tqdm import tqdm
from utils import create_dataloaders, probability_to_class, visualize_sample
import argparse

"""
This script contains the train loop for training the ENet
"""

def train_one_epoch(epoch, dataloader, device, model, criterion, optimizer):
    """
    Trains the Model for one epoch
    """
    model.train()
    train_loss = 0
    num_train_batches = 0
    for image, target in tqdm(dataloader, "iterating over train batches..."): 
        num_train_batches += 1
        image, target = image.to(device, dtype=torch.float32), target.to(device, dtype=torch.long)

        optimizer.zero_grad()
        pred = model(image)
        loss = criterion(pred, target)
        train_loss += loss.item()
        loss.backward()
        #visualize_sample(image, target, pred)

        optimizer.step()

    avg_loss = train_loss / num_train_batches

    return avg_loss


def eval_one_epoch(epoch, dataloader, device, model, criterion):
    """
    Evaluates the models performance on the validation datset.
    Called after training for one epoch
    """
    val_loss = 0
    num_val_batches = 0
    model.eval()
    with torch.no_grad():
        for image, target in tqdm(dataloader, "iterating over val batches"):
            num_val_batches += 1
            image, target = image.to(device, dtype=torch.float32), target.to(device, dtype=torch.long)

            pred = model(image)
            loss = criterion(pred, target)
            val_loss += loss.item()

        avg_loss = val_loss / num_val_batches
    
    return avg_loss


def train(num_epochs, train_loader, val_loader, device, model, criterion, optimizer):
    """
    Training script
    """
    for epoch in range(num_epochs):
        train_loss = train_one_epoch(epoch, train_loader, device, model, criterion, optimizer)
        val_loss = eval_one_epoch(epoch, val_loader, device, model, criterion)
        print(f"Epoch {epoch}: train_loss={train_loss}, val_loss={val_loss}")
        writer.add_scalar("Loss/train", train_loss, epoch)
        writer.add_scalar("Loss/val", val_loss, epoch)


def main(args):
    num_epochs, lr, batch_size, weight_decay = args.num_epochs, args.lr, args.batch_size, args.weight_decay

    train_loader, val_loader = create_dataloaders(batch_size=batch_size, test_size=0.2)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = ENet(in_channels=3, out_channels=20)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    train(num_epochs, train_loader, val_loader, device, model, criterion, optimizer)

    torch.save({
            'epoch': num_epochs,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            }, "model_checkpoints/checkpoint.tar")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_epochs", default=20, help='specify how long the model will be trained')
    parser.add_argument("--lr", default=0.0005, help='specify the learning rate')
    parser.add_argument("--batch_size", default=10, help='specify the batch size')
    parser.add_argument("--weight_decay", default=0.000, help='specify the weight decay')
    args = parser.parse_args()

    main(args)