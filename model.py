# holds the model
# seq2seq kind of model with CNN encoder and RNN decoder - like an autoencoder

import torch
import torchvision.models as models

class CNN(torch.nn.Module):
    def __init__(self, embedding_size, train=False):
        super(CNN,self).__init__()
        self.train = train # do we want to retrain the entire model?
        self.inception = models.inception_v3(pretrained=True, aux_logits=False) # using a pretrained inception model
        self.inception.fc = torch.nn.Linear(self.inception.fc.in_features, embedding_size) # change the last layer to a custom layer whose output dims match our embedding size
        self.relu = torch.nn.ReLU()
        self.dropout = torch.nn.Dropout(0.5) # set dropout rate to 50% to avoid overfitting

    def forward(self, input_images):
        features = self.inception(input_images) # run the forward pass to get features from pictures
        # time for setting some constraints for backward pass - we dont neccessarily want to train any layers cept the last custom layer we added
        for name, param in self.inception.named_parameters():
            if "fc.weight" in name or "fc.bias" in name:
                param.requires_grad = True # always train the last layer
            else:
                param.requires_grad = self.train # train only if the train is set to true by user
        return self.dropout(self.relu(features)) # run activation then dropout on the output from forward pass and return
    
class RNN(torch.nn.Module):
    def __init__(self, embedding_size, hidden_layer_size, vocabulary_size, num_layers):
        super(RNN, self).__init__()
        self.embed = torch.nn.Embedding(vocabulary_size, embedding_size) # convert input encodings to embeddings
        self.lstm = torch.nn.LSTM(embedding_size,hidden_layer_size, num_layers) # pass through n = num_layers of lstm layers with each having num_neurons = hidden_layer_size
        self.fc = torch.nn.Linear(hidden_layer_size, vocabulary_size) # output size = vocabulary_size
        self.dropout = torch.nn.Dropout(0.5)
    
    def forward(self, features, captions): # features from CNN, captions from data
        embeddings = self.dropout(self.embed(features)) # forward pass to embedding layer then apply dropout
        embeddings = torch.cat((features.unsqueeze(0),embeddings),dim=0)
        # add a dim of size 1 at 0th indx of the shape vector [dim1, dim2, dim3, ..., dimn] -> [1, dim1, dim2, ..., dimn] - dim at indx 0 will be used to keep idea of timestep
        # concat this new vector with embeddings (with similar dims) along dim = 0 (timestep dim)
        layer_output, _ = self.lstm(embeddings) # output, cell = output of lstm layer at processed embeddings
        layer_output = self.fc(layer_output)
        return layer_output
    
# join the components into one model
class CNNtoRNN(torch.nn.Module):
    def __init__(self):
        pass