import torch
import numpy as np
import matplotlib.pyplot as plt
from torch.autograd import Variable

def makedata(numdatapoints):
    x = np.linspace(-10, 10, numdatapoints)

    coeffs = [2, -30, 0.5, 5]

    y = np.polyval(coeffs, x) + 2 * np.random.rand(numdatapoints)

    return x, y

# center all reatures around their mean and divide by their average
def scale_features(features):

    avg = np.mean(features, axis=1).reshape(-1, 1)
    ranges = np.ptp(features, axis=1).reshape(-1, 1)

    scaled = features - avg
    scaled = np.divide(scaled, ranges)

    return scaled, avg, ranges

numdatapoints = 100

inputs, labels = makedata(numdatapoints)

fig = plt.figure(figsize=(10, 20))

ax1 = fig.add_subplot(121)
ax1.set_xlabel('Input')
ax1.set_ylabel('Output')
ax1.scatter(np.array(inputs), np.array(labels), s=5)
ax1.grid()

ax2 = fig.add_subplot(122)
ax2.set_title('Error vs epoch')
ax2.grid()

line1, = ax1.plot(inputs, inputs)

plt.ion()
plt.show()

def makefeatures(powers):
    features = np.ones((inputs.shape[0], len(powers)))
    for i in range(len(powers)):
        features[:,i] = (inputs**powers[i])
    print(features.shape)

    return features.T

class LinearModel(torch.nn.Module):

    def __init__(self):
        super().__init__()
        self.l = torch.nn.Linear(features.shape[0], 1)

    def forward(self, x):
        out = self.l(x)
        return out

# hyperparaters
epochs = 100
lr = 0.5

powers = [2, 3]

features = makefeatures(powers)
scaled, avg, range = scale_features(features)

datain = Variable(torch.Tensor(scaled.T))

labels = Variable(torch.Tensor(labels.T))

mymodel = LinearModel()

criterion = torch.nn.MSELoss(size_average=True)
optimiser = torch.optim.SGD(mymodel.parameters(), lr=lr)

print(labels.shape)
print(datain.shape)


def train():
    costs=[]
    for e in range(epochs):

        prediction = mymodel(datain)

        cost = criterion(prediction, labels)

        costs.append(cost.data)
        print('Epoch', e, 'Cost', cost.data[0])

        params = [mymodel.state_dict()[i][0] for i in mymodel.state_dict()]

        weights = params[0]
        bias = params[1]
        print('b', bias)
        print('w', weights)

        optimiser.zero_grad()
        cost.backward()
        optimiser.step()

        line1.set_ydata(torch.mm(weights.view(1, -1), datain.data.t()) + bias)
        fig.canvas.draw()
        ax2.plot(costs)
        print(cost)

train()