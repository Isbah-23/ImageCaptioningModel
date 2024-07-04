import torch
import torchvision.transforms as transforms
from data_loader import get_loader
from model import CNNtoRNN

def train():
    transform = transforms.Compose([transforms.Resize(356,356),
                                    transforms.RandomCrop(299,299),
                                    transforms.ToTensor(),
                                    transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))])