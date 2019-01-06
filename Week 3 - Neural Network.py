import pandas as pd
import numpy as np
import torch
import torch.nn.functional as F
from torch.autograd import Variable
import matplotlib.pyplot as plt

df = pd.read_csv('creditcard.csv')
#print(df)

n = df.shape[1]
m = df.shape[0]

#print(n, m)

alldata = np.array(df)
alldata = (alldata[:, :5], alldata[:, -1])

frauds = np.array(df[df['Class'] == 1])
num_frauds = frauds.shape[0]
print(num_frauds)
legit = np.array(df[df['Class'] == 0])

traininglegits = np.random.randint(m - num_frauds, size=num_frauds)

trainingdata = np.vstack((frauds, legit[traininglegits]))
trainingdata = (trainingdata[:, :5], trainingdata[:, -1])

lr = 0.01
epochs = 50

class NN(torch.nn.Module):

    def __init__(self):
        super().__init__()
        self.l1 = torch.nn.Linear(5, 6)
        self.l2 = torch.nn.Linear(6, 3)
        self.l3 = torch.nn.Linear(3, 1)
        self.relu = F.relu
        self.sigmoid = F.sigmoid

    def forward(self, x):
        out1 = self.relu(self.l1(x))
        out2 = self.relu(self.l2(out1))
        out3 = self.sigmoid(self.l3(out2))
        return out3

mynet = NN()

criterion = torch.nn.BCELoss()
optimiser = torch.optim.Adam(mynet.parameters(), lr=lr)

def train():
    mynet.train()

    fig = plt.figure()
    ax = fig.add_subplot(111)
    plt.ion()
    plt.show()
    costs = []

    for e in range(epochs):

        features, labels = Variable(torch.Tensor(trainingdata[0])), Variable(torch.Tensor(trainingdata[1]))
        prediction = mynet(features)

        optimiser.zero_grad()
        loss = criterion(prediction, labels)
        loss.backward()
        optimiser.step()

        costs.append(loss.data)
        ax.plot(costs, 'b')
        fig.canvas.draw()

        print('Epoch', e, '\tLoss', loss.data[0])

train()

def test():
    print('\n\n\n')
    test_size = 2048
    test_sample = np.random.randint(m, size=test_size)
    features, labels = Variable(torch.Tensor(alldata[0][test_sample])), Variable(torch.Tensor(alldata[1][test_sample]))

    prediction = np.round(mynet(features).data)

    correct = prediction.eq(labels.data.view_as(prediction)).sum()

    print('Test accuracy', correct/test_size)

test()