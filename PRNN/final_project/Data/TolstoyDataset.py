import torch
import numpy as np
from torch.utils.data import Dataset
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
import os
import warnings
warnings.filterwarnings("ignore")

class TolstoyDataset(Dataset):
    """Tolstoy dataset."""

    def __init__(self, len_seq):
        """
        Args:
            txt_file (string): Path to the txt file for the entire book
            len_seq (int): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.len_seq = len_seq
        dir_path = os.path.dirname(os.path.realpath(__file__))
        self.txt_file_1 = os.path.join(dir_path,'anna.txt')
        self.txt_file_2 = os.path.join(dir_path,'war_and_peace.txt')
        # load the whole book
        file_1 = open(self.txt_file_1)
        alltxt = file_1.read()
        file_2 = open(self.txt_file_2)
        alltxt += file_2.read()
        # remove newline formmating
        alltxt = alltxt.replace("\n\n", "&").replace("\n", " ").replace("&", "\n")
        # define categories
        unique_chars = list(sorted(set(alltxt)))
        self.categories = unique_chars
        # integer encode
        label_encoder = LabelEncoder()
        label_encoder.fit(self.categories)
        self.integer_encoded = torch.LongTensor(label_encoder.transform(list(alltxt)))
        #self.invert = lambda X: label_encoder.inverse_transform(X)
        # onehot encode
        onehot_encoder = OneHotEncoder(sparse=False)
        onehot_encoder.fit(np.arange(len(label_encoder.classes_)).reshape(-1, 1))
        integer_encoded = self.integer_encoded.reshape(len(self.integer_encoded), 1)
        self.onehot_encoded = torch.FloatTensor(onehot_encoder.transform(integer_encoded))
        
    def __len__(self):
        return int(np.floor(len(self.integer_encoded)/self.len_seq))

    def __getitem__(self, idx):
        x = self.onehot_encoded[idx*self.len_seq:(idx+1)*self.len_seq,:]
        y = self.integer_encoded[idx*self.len_seq+1:(idx+1)*self.len_seq+1]
        return  x,  y
    
def InvertTolstoy(X):
    dir_path = os.path.dirname(os.path.realpath(__file__))
    txt_file_1 = os.path.join(dir_path,'anna.txt')
    txt_file_2 = os.path.join(dir_path,'war_and_peace.txt')
    # load the whole book
    file_1 = open(txt_file_1)
    alltxt = file_1.read()
    file_2 = open(txt_file_2)
    alltxt += file_2.read()
    # remove newline formmating
    alltxt = alltxt.replace("\n\n", "&").replace("\n", " ").replace("&", "\n")
    # define categories
    unique_chars = list(sorted(set(alltxt)))
    categories = unique_chars
    # integer encode
    label_encoder = LabelEncoder()
    label_encoder.fit(categories)
    letters = label_encoder.inverse_transform(X)
    
    return letters