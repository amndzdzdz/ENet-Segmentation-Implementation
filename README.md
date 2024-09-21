# ENet Segmentation

This repository contains an implementation of the ENet model for image segmentation tasks. ENet is a deep learning architecture designed for efficient semantic segmentation, particularly known for its speed and lightweight nature. It is widely used for real-time segmentation tasks such as urban street scene segmentation. You can read more about ENet in the original paper [here](https://arxiv.org/abs/1606.02147).

## Project Overview

The goal of this project is to provide a flexible and easy-to-use implementation of the ENet architecture for segmentation tasks. This implementation is well-suited for scenarios where real-time performance and efficiency are important, such as autonomous driving applications.

Please note that this model has only been trained on a small subset of the **GTA5 Synthetic dataset** to verify that the implementation works correctly. No hyperparameter tuning has been performed, as the primary objective was to ensure the model was properly implemented.

## Features

- **ENet Architecture**: A fully functional ENet model ready for training and evaluation.
- **Training Loop**: A basic training loop has been implemented.

## Installation

To install the necessary dependencies, you can use the following command:

```bash
pip install -r requirements.txt
