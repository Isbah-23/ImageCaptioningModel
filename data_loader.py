import os
import pandas as pd
import spacy # for tokenization
import torch
from PIL import Image
import torchvision.transforms as transforms

spacy_eng = spacy.load('en_core_web_sm') # english language model for tokenization, named entity recognition, POS tagging, parsing, word vector conversions

class Vocabulary:
    def __init__(self, frequency_threshold):
        # special tokens in vocab:
        # PAD - Padding element - usually 0s
        # SOS - start of sentence elem, before the first word in text
        # EOS - end of sentence elem, after last word in text
        # UNK - unknown elem, not present in vocabulary words - (words that model knows)
        self.stoi = {'<PAD>':0, '<SOS>':1, '<EOS>':2, '<UNK>':3}
        self.itos = {0:'<PAD>', 1:'<SOS>', 2:'<EOS>', 3:'<UNK>'}
        self.freq_threshold = frequency_threshold
    
    def __len__(self):
        return len(self.stoi) # return the total num of elems in vocabulary
    
    @staticmethod
    def tokenizer_eng(text):
        return [tok.text.lower() for tok in spacy_eng.tokenizer(text)] # returns list of lowercase tokens
    
    def build_vocabulary(self, sentence_list):
        frequencies = {}
        idx = 4 # start after index 3 because special tokens are till 3
        for sentence in sentence_list:
            for word in self.tokenizer_eng(sentence):
                if word not in frequencies:
                    frequencies[word] = 1
                else:
                    frequencies[word] += 1
                if frequencies[word] == self.freq_threshold: # if word occurs a certain number of times add it to the language's vocab
                    self.stoi[word] = idx
                    self.itos[idx] = word
                    idx += 1

    def encode(self, text): # encode the text to code using the vocabulary
        tokenized_text = self.tokenizer_eng(text)
        return [self.stoi[token] if token in self.stoi else self.stoi['UNK'] for token in tokenized_text]
    
class RandomDataset(torch.utils.data.Dataset):
    def __init__(self, root_directory, captions_file_name, transform=None, freq_threshold=5):
        self.root_dir = root_directory
        self.df = pd.read_csv(os.path.join(root_directory,'images',captions_file_name))
        self.image_files = self.df["img_name"]
        self.captions = self.df["img_caption"]
        self.transform = transform
        self.vocabulary = Vocabulary(freq_threshold)
        self.vocabulary.build_vocabulary(self.captions.to_list())
    
    def __len__(self):
        return len(self.df) # returns the input sample size
    
    def __getitem__(self, idx):
        image_file, caption = self.image_files[idx], self.captions[idx]
        image = Image.open(os.path.join(self.root_dir,'images',image_file)).convert("RGB")
        if self.transform is not None:
            image = self.transform(image)
        encoded_caption = [self.vocabulary.stoi['<SOS>']]
        encoded_caption.extend(self.vocabulary.encode(caption))
        encoded_caption.append(self.vocabulary.stoi['<EOS>'])
        return image, torch.tensor(encoded_caption)
    
class Collate: # custom dataloader collate (converts data into batches for processing)
    def __init__(self, padding_index):
        self.pad_indx = padding_index
    def __call__(self, batch):
        imgs = [input_sample[0].unsqueeze(0) for input_sample in batch] # get the first item(img) from the (img, caption) tuple of each input_sample in batch, add dim = 1 at indx 0
        imgs = torch.cat(imgs, dim=0) # concat along added dim
        captions = [input_sample[1] for input_sample in batch] # caption from each (img, caption) in batch
        captions = torch.nn.utils.rnn.pad_sequence(captions, batch_first=False, padding_value=self.pad_indx)
        return imgs, captions


if __name__ == "__main__":
    rd = RandomDataset('random_dataset','image_captions.csv')
    print(rd.df.head)
    print(rd.__getitem__(3))