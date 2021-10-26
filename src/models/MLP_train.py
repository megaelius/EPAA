#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
import torch
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
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
        self.df['inc_complexity_binary'] =  self.df['inc_complexity'].transform(lambda x:x > 0)
        self.df['inc_violations_binary'] =  self.df['inc_violations'].transform(lambda x:x > 0)
        self.df['inc_development_cost_binary'] =  self.df['inc_development_cost'].transform(lambda x:x > 0)
        print(sum(self.df['inc_complexity_binary'] == 1))
        print(sum(self.df['inc_complexity_binary'] == 0))
    def __len__(self):
        return len(self.hashes)

    def __getitem__(self, index):
        h = self.hashes[index]
        #
        #target = self.df[self.df['COMMIT_HASH'] == h][['inc_complexity_binary','inc_violations_binary','inc_development_cost_binary']].values.tolist()[0]
        target = self.df[self.df['COMMIT_HASH'] == h]['inc_complexity_binary'].values.tolist()
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
        self.drop = nn.Dropout(0.5)

    def forward(self, xb):
        x = xb.float()
        #x = xb.view(250, -1)
        x = F.relu(self.lin1(x))
        #x = self.drop(x)
        x = F.relu(self.lin2(x))
        #x = self.drop(x)
        #x = F.relu(self.lin3(x))
        return self.lin3(x)

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', default="../../data/processed/predictionDB.csv")
    parser.add_argument('--embed_path', default='../../data/processed/embeddings')
    parser.add_argument('--output_path', default='../../models/Model_pruebas')

    args = parser.parse_args()
    predictionDB_path = args.data_path
    embeddings_path = args.embed_path

    if not Path(args.output_path).exists():
        Path(args.output_path).mkdir()

    n = len(os.listdir(embeddings_path))
    print(n)

    df = pd.read_csv(predictionDB_path,lineterminator='\n')
    df

    torch.multiprocessing.set_sharing_strategy('file_system')

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


    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001, weight_decay = 0.0005)
    #loss_fn = nn.SmoothL1Loss()
    loss_fn = nn.BCEWithLogitsLoss()


    mean_train_losses = []
    mean_valid_losses = []
    train_acc = []
    valid_acc = []

    epochs = 20
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model.to(device)
    for epoch in range(epochs):
        model.train()

        train_losses = []
        valid_losses = []
        train_preds = []
        valid_preds = []
        train_targets = []
        valid_targets = []
        for i, (embeddings, labels) in tqdm(enumerate(train_loader)):

            embeddings = embeddings.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()

            outputs = model(embeddings)
            loss = loss_fn(outputs.squeeze(0),labels)

            train_preds += list(outputs.squeeze(0).sigmoid() > 0.5)
            train_targets += list(labels)

            loss.backward()
            optimizer.step()

            train_losses.append(loss.item())

        acc = accuracy_score(train_preds,train_targets)
        train_acc.append(acc)
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for i, (embeddings, labels) in tqdm(enumerate(valid_loader)):
                embeddings = embeddings.to(device)
                labels = labels.to(device)
                outputs = model(embeddings)
                loss = loss_fn(outputs.squeeze(0), labels)

                valid_preds += list(outputs.squeeze(0).sigmoid() > 0.5)
                valid_targets += list(labels)

                valid_losses.append(loss.item())

                _, predicted = torch.max(outputs.data, 1)

                total += labels.size(0)

        acc = accuracy_score(valid_preds,valid_targets)
        valid_acc.append(acc)
        mean_train_losses.append(np.mean(train_losses))
        mean_valid_losses.append(np.mean(valid_losses))
        x = list(range(len(mean_train_losses)))
        plt.plot(x,mean_train_losses)
        plt.plot(x,mean_valid_losses)
        plt.legend(['Train','Valid'])
        plt.savefig(os.path.join(args.output_path,'Loss.png'))
        plt.close()
        plt.plot(x,train_acc)
        plt.plot(x,valid_acc)
        plt.axhline(y=sum(dataset.df['inc_complexity_binary'] == 0)/len(dataset), color='r', linestyle='-')
        plt.legend(['Train','Valid','Frequency of 0s'])
        plt.savefig(os.path.join(args.output_path,'Accuracy.png'))
        plt.close()
        print('epoch : {}, train loss : {:.4f}, valid loss : {:.4f}'.format(epoch+1, mean_train_losses[-1], mean_valid_losses[-1]))
    torch.save(model,os.path.join(args.output_path,'weights.pt'))
