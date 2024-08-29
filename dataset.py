import torch.nn as nn
import torch
import torchvision.transforms as transforms
from utils import get_image_names
from PIL import Image
import numpy as np
import os

class GTAData(nn.Module):
    """ 
    Custom Dataset for the GTA5 Data
    """
    def __init__(self, root_dir, transforms, categories):
        self.image_dir = os.path.join(root_dir, "images")
        self.label_dir = os.path.join(root_dir, "labels")
        self.transforms = transforms
        self.categories = categories
        self.image_names = get_image_names(root_dir)

    def __getitem__(self, idx):
        image_name = self.image_names[idx]
        
        image_path = os.path.join(self.image_dir, image_name)
        label_path = os.path.join(self.label_dir, image_name)
        
        image = Image.open(image_path).convert("RGB")
        label = Image.open(label_path)
        label = label.resize((512, 512))
        label = np.asarray(label)
        normalized_label = np.zeros(label.shape)

        if self.transforms:
            image = self.transforms(image)

        #avoid indexation error
        for i in range(len(self.categories)):
            cat = self.categories[i]
            normalized_label[label == cat] = i 

        normalized_label = torch.from_numpy(normalized_label).type(torch.IntTensor)

        return image, normalized_label.squeeze()

    def __len__(self):
        return len(self.image_names)