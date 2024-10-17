import numpy as np
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
import torch
from torch import nn
import torch.nn.functional as F


class NeuralRatioEstimator(nn.Module):
    def __init__(self, ninput):
        super(NeuralRatioEstimator, self).__init__()
        self.fc1 = nn.Linear(ninput, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, 64)
        self.fc4 = nn.Linear(64, 64)
        self.fcn = nn.Linear(64, 1)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = F.relu(self.fc4(x))
        return torch.sigmoid(self.fcn(x))


data = np.load("sbi_data.npy")

# scramble first half of data (label 0)
rng = np.random.default_rng()
idx = rng.permutation(len(data)//2)
data[:len(idx)] = data[idx]
y = np.zeros(len(data))
y[:len(idx)] = 1

x_train, x_test, y_train, y_test = train_test_split(data, y, test_size=0.2)
x_train = torch.tensor(x_train, dtype=torch.float32)
y_train = torch.tensor(y_train, dtype=torch.float32)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using {device}")

model = NeuralRatioEstimator(ninput=data.shape[1]).to(device)
optimizer = torch.optim.Adam(model.parameters())
loss_fn = nn.BCELoss()

n_epochs = 100  # specify the number of epochs to train for

for epoch in range(n_epochs):
    model.train()  # Set model to training mode

    y_pred = model(x_train.to(device))  # forward pass: compute predicted output
    loss = loss_fn(y_pred.squeeze(), y_train.to(device))  # compute loss

    optimizer.zero_grad()  # clear old gradients
    loss.backward()  # compute gradients
    optimizer.step()  # update parameters

    if epoch % 10 == 0:
        print(f'Epoch {epoch+1}/{n_epochs}, Loss: {loss.item()}')

model.eval()  # Set model to evaluation mode


def logR(x):
    p = model(x.to(device)).squeeze()
    return torch.log(p / (1 - p))


logRs = logR(torch.tensor(x_test, dtype=torch.float32)).detach().cpu().numpy()

fig, ax = plt.subplots()
ax.hist(logRs, bins=30)
ax.set(xlabel=r"$\log R$")
plt.show()
