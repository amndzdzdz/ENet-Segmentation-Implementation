import os
import torch
import torch.nn.functional as F
import numpy as np
from torch.utils.data import random_split, DataLoader

def get_image_names(path):
    image_names = []
    image_dir = os.path.join(path, "images")
    i = 0
    for imagename in os.listdir(image_dir):
        i += 1 
        image_names.append(imagename)
        if i > 499:
            break
    return image_names

def create_dataloaders(dataset, batch_size=16, test_size=0.2):
    train_size = int((1-test_size) * dataset.__len__())
    test_size = int(test_size * dataset.__len__())

    train_data, val_data = random_split(dataset, [train_size, test_size])

    print("length Traindata:", train_data.__len__())
    print("length Valdata:", val_data.__len__())

    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=False, drop_last=True)

    return train_loader, val_loader

def probability_to_class(tensor, categories):
    softmaxed_tensor = F.softmax(tensor, dim=1)
    argmaxed_tensor = softmaxed_tensor.argmax(dim=1)
    mask = np.array(categories)[argmaxed_tensor]

    return mask


if __name__ == "__main__":
    categories = [0, 7, 8, 11, 12, 13, 17, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 31, 32, 33]
    tensor = torch.rand((1, 19, 512, 512))
    probability_to_class(tensor, categories)

