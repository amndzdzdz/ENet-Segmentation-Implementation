import os
from torch.utils.data import random_split, DataLoader

def get_image_names(path):
    image_names = []
    image_dir = os.path.join(path, "images")
    for imagename in os.listdir(image_dir):
        image_names.append(imagename)

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


