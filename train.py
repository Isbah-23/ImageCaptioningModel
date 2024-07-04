import torch
import torch.utils
import torch.utils.tensorboard
import torchvision.transforms as transforms
from data_loader import data_loader
from model import CNNtoRNN

def train():
    transform = transforms.Compose([transforms.Resize(356,356),
                                    transforms.RandomCrop(299,299),
                                    transforms.ToTensor(),
                                    transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))])
    dataloader, dataset = data_loader('random_dataset','image_captions.csv',transform)

    # torch.backends.cudnn.benchmark()
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device("cpu") # i dont have a gpu available atm

    # Settiing the hyperparameters
    embed_size = 256
    hidden_size = 256
    vocab_size = len(dataset.vocabulary)
    num_lstm_layers = 1
    learning_rate = 5e-4 # 0.0005
    num_epochs = 100

    writer = torch.utils.tensorboard.SummaryWriter()
    step = 0

    