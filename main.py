import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from pprint import pprint
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures
from sklearn.preprocessing import StandardScaler
from torch.nn.modules.loss import MSELoss


torch.manual_seed(42)
df = pd.read_csv('simple_data.csv')
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

class LinearRegressionFeedForwad(nn.Module):
  def __init__(self, input_features, hidden_layers1):
    super(LinearRegressionFeedForwad, self).__init__()
    self.l1 = nn.Linear(input_features, hidden_layers1)
    self.relu = nn.ReLU()
    self.l2 = nn.Linear(hidden_layers1, 1)


  def forward(self,x):
    out = self.l1(x)
    out = self.relu(out)
    out = self.l2(out)
    return out


input_size = n_features
hidden_layers1 = 10
model = LinearRegressionFeedForwad(input_size, hidden_layers1)
loss = MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr = 0.2)
# train loop: forward, backward, step, zero
num_epoch = 10
for epoch in range(num_epoch):
  predicted = model(X_train)
  resid = loss(y_train, predicted)
  resid.backward()
  optimizer.step()
  optimizer.zero_grad()
  if (epoch + 1) % 100 == 0:
    print(f'epoch: {epoch + 1}')

# pull preds: predicted = model(X_test).detach().numpy().flatten()
predicted = model(X_test).detach().numpy().flatten()
dict_params = model.state_dict()
#dict_params
print(predicted)