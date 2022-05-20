# single variable linear regression
import torch
import torch.nn as nn
import numpy as np
from sklearn import datasets
import matplotlib.pyplot as plt
from torch.distributions.normal import Normal
from torch.distributions.uniform import Uniform
from pprint import pprint
from sklearn.datasets import fetch_california_housing
housing = fetch_california_housing()

X_numpy, y_numpy = datasets.make_regression(n_samples=100, n_features=1, noise=20, random_state=4) 

X = torch.from_numpy(X_numpy.astype(np.float32))
y = torch.from_numpy(y_numpy.astype(np.float32))

X.shape # torch.Size([1000, 1])
y.shape # torch.Size([1000])
y = y.view(y.shape[0],1)
y.shape # torch.Size([1000, 1])

n_samples, n_features = X.shape

#######################
# 1. model
# 2. loss
# 3. optimizer
######################
# 1. model
model = nn.Linear(in_features=n_features, out_features=1)
# 2. loss
loss = nn.MSELoss()
# 3. optimizer
learning_rate = 0.01
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate) 

#################################
# Training Loop
# for epoch in range(num_epoch)
#################################

num_epochs = 1000

for epoch in range(num_epochs):
  #forward (predict)
  y_predicted = model(X)
  l = loss(y_predicted, y)

  #backward (gradient)
  l.backward()

  #update 
  optimizer.step()

  # zero
  optimizer.zero_grad()


predicted = model(X).detach().numpy().flatten()
print(list(zip(y.numpy().flatten(), predicted)))