import torch
from torch import nn
from torch.utils.data import DataLoader

from src.models import MatrixFactorization
from src.datasets import BaseDataset, NegativeSamplingDataset, train_validation_test_split


def train_loop():
    ...


def test_loop():
    ...


# load data


def load_dataset() -> BaseDataset:
    return NegativeSamplingDataset(1, 1, torch.FloatTensor(torch.empty()), torch.FloatTensor(torch.empty()))


dataset = load_dataset()

# split data

train_dataset, validation_dataset, test_dataset = train_validation_test_split(dataset, 5, 5)

train_loader = DataLoader(train_dataset, batch_size=100)
validation_loader = DataLoader(validation_dataset, batch_size=100)
test_loader = DataLoader(train_dataset, batch_size=100)

# model

model = MatrixFactorization(dataset.no_users, dataset.no_items, embedding_dim=100)
loss = nn.BCEWithLogitsLoss()

# optimizer
epochs = 10
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# training

for epoch in range(epochs):
    train_loop()
    test_loop()
model.eval()

# model saving
torch.save(model.state_dict(), "path")

# evaluation

