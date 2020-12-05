import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
import pandas as pd
import os
import os.path
import shutil
import time

df = pd.read_csv('train_labels.csv', header=0)

df['label'] = df['label'].astype(int)
df['label'] = df['label'] + 100

# df['label'] = df['label'].astype(str)

fileDir = os.path.dirname(os.path.realpath('__file__'))

def move_files(mylist, listtype, label):

    for x in mylist:
        string = f'{fileDir}/{listtype}/class{label}'

        if not os.path.exists(f'{fileDir}/{listtype}/class{label}'):
            os.makedirs(f'{fileDir}/{listtype}/class{label}')
            time.sleep(1)
        shutil.move(f'{fileDir}/train_set/train_set/{x}', f'{fileDir}/{listtype}/class{label}/{x}')
        # shutil.move(f'/home/hulu/Desktop/AML_FR/Test/{x}', f'/home/hulu/Desktop/AML_FR/{listtype}/class{label}/{x}')

# for i in range(100):
for i in range(200):
    file_list = []
    print(i)
    file_list = df.loc[df['label'] == i, 'img_name'].to_list()
    file_list.sort()
    lengte = len(file_list)
    train_len = int(lengte * 0.8)

    train_list = file_list[:train_len]
    val_list = file_list[train_len:]
    move_files(train_list, 'train', i)
    move_files(val_list, 'validate', i)

