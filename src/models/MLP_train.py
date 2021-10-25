#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import random


import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split, RandomSampler, Subset
import glob
from tqdm import tqdm

class commits_dataset(Dataset):
    def __init__(self, csv_path, embeddings_path):
        self.df = pd.read_csv(csv_path,lineterminator='\n')
        self.hashes = list(self.df['COMMIT_HASH'])
        self.embeddings_path = embeddings_path
        self.embeddings = [np.load(os.path.join(self.embeddings_path,h + '.npy')) for h in self.hashes]

    def __len__(self):
        return len(self.hashes)

    def __getitem__(self, index):
        h = self.hashes[index]
        #target = self.df[self.df['COMMIT_HASH'] == h][['inc_complexity','inc_violations','inc_development_cost']].values.tolist()[0]
        target = self.df[self.df['COMMIT_HASH'] == h]['inc_complexity'].values.tolist()
        #embedding = np.load(os.path.join(self.embeddings_path,h + '.npy'))
        embedding = self.embeddings[index]
        return torch.Tensor(embedding),torch.Tensor(target)

# Multilayer perceptron
class MultilayerPerceptron(nn.Module):
    def __init__(self):
        super().__init__()
        self.lin1 = nn.Linear(384, 1024, bias=True)
        self.lin2 = nn.Linear(1024, 120, bias=True)
        self.lin3 = nn.Linear(120, 1, bias=True)

    def forward(self, xb):
        x = xb.float()
        #x = xb.view(250, -1)
        x = F.relu(self.lin1(x))
        x = F.relu(self.lin2(x))
        return self.lin3(x)

if __name__ == '__main__':
    predictionDB_path = "../../data/processed/predictionDB.csv"
    embeddings_path = '../../data/processed/embeddings'
    n = len(os.listdir(embeddings_path))
    print(n)

    df = pd.read_csv(predictionDB_path,lineterminator='\n')
    df

    dataset = commits_dataset(predictionDB_path,embeddings_path)
    print(dataset[0])

    bs = 32
    num_workers = 2

    n=len(dataset)
    n_train = int(n*0.7)
    n_valid = n-n_train

    '''
    Use first 70% of data as training and last 30% as validation
    '''
    train_sample, valid_sample = Subset(dataset, range(n_train)), Subset(dataset, range(n_train,n))
    train_sampler = RandomSampler(train_sample)
    valid_sampler = RandomSampler(valid_sample)
    train_loader = DataLoader(train_sample, sampler = train_sampler, batch_size=bs, num_workers = num_workers)
    valid_loader = DataLoader(valid_sample, sampler = valid_sampler, batch_size=bs, num_workers = num_workers)

    model = MultilayerPerceptron()
    print(model)


    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
    loss_fn = nn.SmoothL1Loss()


    mean_train_losses = []
    mean_valid_losses = []
    valid_acc_list = []
    epochs = 50
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model.to(device)
    for epoch in range(epochs):
        model.train()

        train_losses = []
        valid_losses = []
        for i, (embeddings, labels) in tqdm(enumerate(train_loader)):

            embeddings = embeddings.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()

            outputs = model(embeddings)
            loss = loss_fn(outputs.squeeze(0),labels)
            loss.backward()
            optimizer.step()

            train_losses.append(loss.item())


        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for i, (embeddings, labels) in tqdm(enumerate(valid_loader)):
                embeddings = embeddings.to(device)
                labels = labels.to(device)
                outputs = model(embeddings)
                loss = loss_fn(outputs.squeeze(0), labels)

                valid_losses.append(loss.item())

                _, predicted = torch.max(outputs.data, 1)

                total += labels.size(0)

        mean_train_losses.append(np.mean(train_losses))
        mean_valid_losses.append(np.mean(valid_losses))
        print('epoch : {}, train loss : {:.4f}, valid loss : {:.4f}'         .format(epoch+1, mean_train_losses[-1], mean_valid_losses[-1]))
    torch.save(model,'weights.pt')
