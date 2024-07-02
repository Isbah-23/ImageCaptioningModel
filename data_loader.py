import os
import pandas as pd
import spacy # for tokenization
import torch
from PIL import Image
import torchvision.transforms as transforms

spacy_eng = spacy.load('en_core_web_sm') # english language model for tokenization, named entitiy recognising, POS tagging, parsing, word vector conversions

class Vocabulary:
    def __init__(self, frequency_threshold):
        # special tokens in vocab:
        # PAD - Padding element - usually 0s
        # SOS - start of sentence elem, before the first word in text
        # EOS - end of sentence elem, after last word in text
        # UNK - unkknown elem, not present in vocabulary words - (words that model knows)
        self.stoi = {'<PAD>':0, '<SOS>':1, '<EOS>':2, '<UNK>':3}
        self.itos = {0:'<PAD>', 1:'<SOS>', 2:'<EOS>', 3:'<UNK>'}
        self.freq_threshold = frequency_threshold
    
    def __len__(self):
        return len(self.stoi) # return the total num of elems in vocabulary
    
    @staticmethod
    def tokenizer_eng(text):
        return [tok.text.lower() for tok in spacy_eng.tokenizer(text)]