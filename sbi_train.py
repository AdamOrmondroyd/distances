import sys
import numpy as np
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import scale
import torch
from torch import nn
import torch.nn.functional as F
import torch.nn.init as init

# need to know the number of input dimensions from A and B
ndims_A = int(sys.argv[1])

data = np.load("sbi_data.npy")
# data = data[:, :ndims_A*2]
width = int(data.shape[1] * 1.1 + 20)


class NeuralRatioEstimator(nn.Module):
    def __init__(self, ninput):
        super().__init__()
        self.fc1 = nn.Linear(ninput, width)
        self.lr1 = nn.LeakyReLU(width)
        self.bn1 = nn.BatchNorm1d(width)
        self.fc2 = nn.Linear(width, 64)
        self.lr2 = nn.LeakyReLU(64)
        self.bn2 = nn.BatchNorm1d(64)
        self.fc3 = nn.Linear(64, 64)
        self.lr3 = nn.LeakyReLU(64)
        self.bn3 = nn.BatchNorm1d(64)
        self.fc4 = nn.Linear(64, 64)
        self.lr4 = nn.LeakyReLU(64)
        self.bn4 = nn.BatchNorm1d(64)
        self.fcn = nn.Linear(64, 1)

        # Initialize weights using Xavier initialization
        for m in self.modules():
            if isinstance(m, nn.Linear):
                init.xavier_normal_(m.weight)
                init.constant_(m.bias, 0)

    def forward(self, x):
        x = F.relu(self.bn1(self.lr1(self.fc1(x))))
        x = F.relu(self.bn2(self.lr2(self.fc2(x))))
        x = F.relu(self.bn3(self.lr3(self.fc3(x))))
        x = F.relu(self.bn4(self.lr4(self.fc4(x))))
        return self.fcn(x)


fig, ax = plt.subplots()

data, samples = train_test_split(data, test_size=0.1)

loss_fn = nn.BCEWithLogitsLoss()


# harry says I can double my data by scrambling all of it and using it twice
def scramble(x):
    # mix up half of the samples from data B
    rng = np.random.default_rng()
    x_scrambled = x.copy()
    idx = rng.permutation(len(x_scrambled))
    x_scrambled[:, ndims_A:] = x_scrambled[idx, ndims_A:]
    # unmixed samples are label 1
    x = np.vstack([x_scrambled, x])
    y = np.zeros(len(x))
    y[len(x_scrambled):] = 1
    # then just mix it up for good measure
    idx = rng.permutation(len(x))
    x = x[idx]
    y = y[idx]
    return x, y


for i in range(5):
    # scramble first half of data (label 0)
    x_train, x_test = train_test_split(data, test_size=1/3)

    # scramble second half of train and test data

    x_train, y_train = scramble(x_train)
    x_test, y_test = scramble(x_test)
    x_train = torch.tensor(x_train, dtype=torch.float32)
    y_train = torch.tensor(y_train, dtype=torch.float32)
    x_test = torch.tensor(x_test, dtype=torch.float32)
    y_test = torch.tensor(y_test, dtype=torch.float32)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using {device}")

    model = NeuralRatioEstimator(ninput=data.shape[1]).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    n_epochs = 500  # specify the number of epochs to train for

    patience = 10  # Number of epochs to wait before early stopping
    min_delta = 1e-6  # Minimum change in loss to qualify as an improvement

    counter = 0
    best_loss = np.inf
    best_model_weights = None

    for epoch in range(n_epochs):
        model.train()  # Set model to training mode

        y_pred = model(x_train.to(device))  # forward pass: compute predicted output
        loss = loss_fn(y_pred.squeeze(), y_train.to(device))  # compute loss

        optimizer.zero_grad()  # clear old gradients
        loss.backward()  # compute gradients
        optimizer.step()  # update parameters

        y_pred_test = model(x_test.to(device))
        validation_loss = loss_fn(y_pred_test.squeeze(), y_test.to(device)).item()
        if validation_loss < best_loss - min_delta:
            best_loss = validation_loss
            best_model_weights = model.state_dict()
            counter = 0
        else:
            counter += 1
            if counter > patience:
                print(f"Early stopping at epoch {epoch}")
                model.load_state_dict(best_model_weights)
                break
        if epoch % 10 == 0:
            print(f'Epoch {epoch+1}/{n_epochs}, Loss: {loss.item()}')

    model.eval()  # Set model to evaluation mode

    def logR(x):
        r = model(torch.tensor(x, dtype=torch.float32)).squeeze().detach().cpu().numpy()
        return r

    logRs = logR(samples)
    ax.hist(logRs, bins=30, alpha=0.25)

ax.set(xlabel=r"$\log R$")
plt.show()
