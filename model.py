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
        self.embed = torch.nn.Embedding(vocabulary_size, embedding_size) # convert input encodings (codes from words) to embeddings
        self.lstm = torch.nn.LSTM(embedding_size,hidden_layer_size, num_layers) # pass through n = num_layers of lstm layers with each having num_neurons = shidden_layer_size
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
    def __init__(self, embedding_size, hidden_layer_size, vocabulary_size, num_layers):
        super(CNNtoRNN,self).__init__()
        self.encoder_CNN = CNN(embedding_size)
        self.decoder_RNN = RNN(embedding_size, hidden_layer_size, vocabulary_size, num_layers)
    
    def forward(self, input_images, captions):
        features = self.encoder_CNN(input_images)
        output = self.decoder_RNN(features, captions)
        return output
    
    # when evaluating the function the next input would be prev output instead of ground truth
    def caption_image(self, input_images, vocabulary, max_length_caption=50):
        generated_caption = [] # stores codes for generated words uptil a certain step
        with torch.no_grad():
            features = self.encoder_CNN(input_images).unsqueeze(0) # adding a dim at indx 0 for batch num
            cell_states = None # to store prev cell states of the LSTM - initialized to all 0
            for _ in range(max_length_caption): # get 50 outputs (generated words) from model
                hidden, cell_states = self.decoder_RNN.lstm(features,cell_states)
                possible_options = self.decoder_RNN.fc(hidden.unsqueeze(0)) # add 1 at 0th indx to fix vector dims since we're passing just 1 img instead of batch of images
                best_option = possible_options.argmax(1) # get the most probable next word
                generated_caption.append(best_option.item())
                features = self.decoder_RNN.embed(best_option).unsqueeze(0) # next input is current output
                if vocabulary.itos[best_option.item()] == "<EOS>": # end of sentence token predicted, caption ended
                    break
        return [vocabulary.itos[indx] for indx in generated_caption] # convert codes back to words and return