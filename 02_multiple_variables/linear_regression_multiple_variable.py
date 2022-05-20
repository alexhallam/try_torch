import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from pprint import pprint
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures
from sklearn.preprocessing import StandardScaler
from torch.nn.modules.loss import MSELoss
torch.manual_seed(42)
df = pd.read_csv('../simple_data.csv')
# with categoricals
X = df.loc[:,['x', 'x2', 'x3']].to_numpy()
y = df.loc[:,['y']].to_numpy()
# train_test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state=42)
X_train_pre = X_train
# sc.fit_transform (train) sc.transform (test)
sc = StandardScaler()  
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)
# torch.from_numpy(a.astype(np.float32))
X_train = torch.from_numpy(X_train.astype(np.float32))
X_test = torch.from_numpy(X_test.astype(np.float32))
y_train = torch.from_numpy(y_train.astype(np.float32))
y_test = torch.from_numpy(y_test.astype(np.float32))
# reshape y_train = y_train.view(y_train.shape[0], 1)
y_train = y_train.view(y_train.shape[0], 1)
y_test = y_test.view(y_test.shape[0], 1)
# model, loss, optimizer
samples, n_features = X_train.shape
model = nn.Linear(n_features, 1)
loss = MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr = 0.5)
# train loop: forward, backward, step, zero
num_epoch = 100
for epoch in range(num_epoch):
  predicted = model(X_train)
  resid = loss(y_train, predicted)
  resid.backward()
  optimizer.step()
  optimizer.zero_grad()
  if (epoch + 1) % 10 == 0:
    print(f'epoch: {epoch + 1} params: {model.state_dict()}')

# pull preds: predicted = model(X_test).detach().numpy().flatten()
predicted = model(X_test).detach().numpy().flatten()
dict_params = model.state_dict()
dict_params