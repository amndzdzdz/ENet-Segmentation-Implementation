import torch.nn as nn
import torch
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
import os

class GTAData(nn.Module):
    """ 
    Custom Dataset for the GTA5 Data
    """
    NUM_CLASSES = 20

    def __init__(self, root_dir, transforms, categories):
        self.image_dir = os.path.join(root_dir, "images")
        self.label_dir = os.path.join(root_dir, "labels")
        self.transforms = transforms
        self.categories = categories
        self.image_names = self._get_image_names(root_dir)
        self.class_map = dict(zip(self.categories, range(self.NUM_CLASSES)))

    def __getitem__(self, idx):
        image_name = self.image_names[idx]
        
        image_path = os.path.join(self.image_dir, image_name)
        label_path = os.path.join(self.label_dir, image_name)
        
        image = Image.open(image_path).convert("RGB")
        label = Image.open(label_path)

        label = label.resize((256, 256))
        label = np.asarray(label)
        normalized_label = np.zeros(label.shape)

        if self.transforms:
            image = self.transforms(image)

        #avoid indexation error
        for category in self.categories:
            normalized_label[label == category] = self.class_map[category] 

        normalized_label = torch.from_numpy(normalized_label).type(torch.IntTensor)

        return image, normalized_label

    def __len__(self):
        return len(self.image_names)

    def _get_image_names(self, path):
        image_names = []
        image_dir = os.path.join(path, "images")
        i = 0
        for imagename in os.listdir(image_dir):
            i += 1 
            image_names.append(imagename)
            if i > 69:
                break
        return image_names