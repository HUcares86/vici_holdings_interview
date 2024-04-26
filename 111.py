import torch
# import torchvision
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
from sklearn import datasets



eval_data = np.load('/Users/huzuwang/我的雲端硬碟/code/python/vici_holdings_test/Test/eval_data.npy')
eval_labels = np.load('/Users/huzuwang/我的雲端硬碟/code/python/vici_holdings_test/Test/eval_labels.npy')
train_data = np.load('/Users/huzuwang/我的雲端硬碟/code/python/vici_holdings_test/Test/train_data.npy')
train_labels = np.load('/Users/huzuwang/我的雲端硬碟/code/python/vici_holdings_test/Test/train_labels.npy')
df_eval_data = pd.DataFrame(eval_data)
df_eval_labels = pd.DataFrame(eval_labels)
df_train_data = pd.DataFrame(train_data)
df_train_labels = pd.DataFrame(train_labels)

# ----------------------------
params = {'axes.titlesize':'5',
          'xtick.labelsize':'5',
          'ytick.labelsize':'5'}
matplotlib.rcParams.update(params)
df_train_data.hist(bins=50000, figsize=(30,30))
# plt.tight_layout()
plt.show()