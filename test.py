import torch
from model import ENet
from utils import visualize_sample, create_dataloaders
"""
This script is for inference testing etc.
"""

def run_test(dataloader):
    checkpoint = torch.load("model_checkpoints\checkpoint.tar", weights_only=True)
    
    model = ENet(in_channels=3, out_channels=20)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    device = 'cpu'
    
    for image, label in dataloader:
        image, label = image.to(device, dtype=torch.float32), label.to(device, dtype=torch.long)

        pred = model(image)
        visualize_sample(image, label, pred)
        break

def run_inference():
    return None
    

if __name__ == '__main__':
    _, val_loader = create_dataloaders(batch_size=1, test_size=0.2)
    run_inference(val_loader)