import torch
from torch import nn
import os
import pandas as pd
import pickle
from collections import defaultdict
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from torch.nn.utils.rnn import pad_sequence
import ast

class Summary_dataset(Dataset):
    def __init__(self):
        super().__init__()
        self.vocab = {"<PAD>": 0, "<UNK>":1, "<START>": 2, "<STOP>":3}
        self.inverse_vocab = None
        self.encoder_inputs = []
        self.decoder_inputs = []
        self.targets = []
    def build_vocab(self, filename): # this assumes the df has 2 columns, src and target
        print("starting build vocab")
        df = pd.read_csv(filename)
        word_counts = defaultdict(int) # for setting unk tokens
        for _, row in tqdm(df.iterrows(), total=len(df), desc="Getting word counts"):
            src_tokens = ast.literal_eval(row['src'])
            tgt_tokens = ast.literal_eval(row['tgt'])
            
            for token in src_tokens:
                word_counts[token] += 1
            for token in tgt_tokens:
                word_counts[token] += 1

        for token, count in tqdm(word_counts.items(), total=len(word_counts), desc="Building vocab"):
            if count >= 3 and token not in self.vocab: # we are only including items that occured
                self.vocab[token] = len(self.vocab)

        os.makedirs('dataset', exist_ok=True)
        with open('dataset/vocab.pkl', 'wb') as f:
            pickle.dump(self.vocab, f)

        self.inverse_vocab = {v: k for k, v in self.vocab.items()}

    def load_pickled_vocab(self):
        print("loading pickled data")
        vocab_path = 'dataset/vocab.pkl'
        if os.path.exists(vocab_path):
            with open(vocab_path, 'rb') as f:
                self.vocab = pickle.load(f)
            print("Vocabulary loaded successfully.")
            self.inverse_vocab = {v: k for k, v in self.vocab.items()}
        else:
            print(f"Error: {vocab_path} not found.")
            
        

    def encode_data(self, filename):
        df = pd.read_csv(filename)
        counter = 0
        for _, row in tqdm(df.iterrows(), total=len(df), desc="Encoding Data"):
            src_tokens = ast.literal_eval(row['src'])
            tgt_tokens = ast.literal_eval(row['tgt'])
            if filename == "processed_data/final_df_train.csv":

                # 🥺 the gpu's are full. I'm cpu'ing this monster. Im using a subset, please dont be mad 🥺
                if counter >= 45000: # training on 45,000 split
                    break
                else:
                    counter += 1
            

            encoder_input = [self.vocab.get(token, self.vocab["<UNK>"]) for token in src_tokens]
            self.encoder_inputs.append(torch.tensor(encoder_input))
            decoder_input = [self.vocab.get(token, self.vocab["<UNK>"]) for token in tgt_tokens]
            decoder_input = [self.vocab["<START>"]] + decoder_input
            decoder_target = decoder_input[1:] + [self.vocab["<STOP>"]]
            self.decoder_inputs.append(torch.tensor(decoder_input))
            self.targets.append(torch.tensor(decoder_target))

    def __len__(self):
        return len(self.encoder_inputs)

    def __getitem__(self, idx):
        return {
            'encoder_input': self.encoder_inputs[idx],
            'decoder_input': self.decoder_inputs[idx],
            'target': self.targets[idx]
        }


def collate_fn(batch):
    encoder_inputs = [item['encoder_input'] for item in batch]
    decoder_inputs = [item['decoder_input'] for item in batch]
    targets = [item['target'] for item in batch]

    encoder_inputs_padded = pad_sequence(encoder_inputs, batch_first=True, padding_value=0)

    decoder_inputs_padded = pad_sequence(decoder_inputs, batch_first=True, padding_value=0)

    targets_padded = pad_sequence(targets, batch_first=True, padding_value=0)


    return {
        'encoder_input': encoder_inputs_padded,
        'decoder_input': decoder_inputs_padded,
        'target': targets_padded
        }


train_file = "processed_data/final_df_train.csv"
val_file = "processed_data/final_df_val.csv"
test_file = "processed_data/final_df_test.csv"

train_dataset = Summary_dataset()
val_dataset = Summary_dataset()
test_dataset = Summary_dataset()

exists_vocab = input("Does a vocab already exist? y/n: ")
if exists_vocab == "y":
    train_dataset.load_pickled_vocab()
elif exists_vocab == "n":
    train_dataset.build_vocab(filename=train_file)
val_dataset.load_pickled_vocab()
test_dataset.load_pickled_vocab()

train_dataset.encode_data(filename=train_file)
val_dataset.encode_data(filename=val_file)
test_dataset.encode_data(filename=test_file)

train_loader = DataLoader(train_dataset, batch_size=32, collate_fn=collate_fn)
val_loader = DataLoader(val_dataset, batch_size=32, collate_fn=collate_fn)
test_loader = DataLoader(test_dataset, batch_size=32, collate_fn=collate_fn)


