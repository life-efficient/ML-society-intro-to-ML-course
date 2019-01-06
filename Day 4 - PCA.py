import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn import datasets

data = datasets.load_iris()

print(data)
X = data.data
Y = data.target

print(X.shape)
print(Y.shape)

m = X.shape[0]

def normalise(x):
    x_std = x - np.mean(x, axis=0)
    x_std = np.divide(x_std, np.std(x_std, axis=0))
    return x_std

def decompose(x):
    cov = np.matmul(x.T, x)
    print('\nCovariance matrix')
    print(cov)

    eig_vals, eig_vecs = np.linalg.eig(cov)
    print('\nEigenvectors')
    print(eig_vecs)
    print('\nEigenvalues')
    print(eig_vals)
    return eig_vals, eig_vecs, cov

def whicheigs(eig_vals):    # lets us know which eigenvalues we are going to use
    total = sum(eig_vals)
    var_percent = [(i/total) * 100 for i in sorted(eig_vals, reverse=True)]
    cum_var_percent = np.cumsum(var_percent)

    fig = plt.figure()
    ax =  fig.add_subplot(111)
    plt.title('Variance along different principal components')
    ax.grid()
    plt.xlabel('Principal Component')
    plt.ylabel('Percentage total variance accounted for')

    ax.plot(cum_var_percent, '-ro')
    ax.bar(range(len(eig_vals)), var_percent)
    plt.xticks(np.arange(len(eig_vals)), ('PC{}'.format(i) for i in range(len(eig_vals))))
    plt.show()

def reduce(x, eig_vecs, dims):
    W = eig_vecs[:, :dims]
    print('\nDimension reducing matrix')
    print(W)
    return np.matmul(x, W), W

colour_dict = {0:'r', 1:'g', 2:'b'}
colour_list = [colour_dict[i] for i in list(Y)]

dim = 3 # reduce to 3d

def plotreduced(x):
    dims = x.shape[1]

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(x[:, 0], x[:, 1], x[:, 2], c=colour_list)
    plt.xlabel('PC1 value')
    plt.ylabel('PC2 value')
    ax.set_zlabel('PC3 value')
    plt.grid()
    plt.show()

# centre data around mean and divide by sd
X_std = normalise(X)
eig_vals, eig_vecs, covariance = decompose(X_std)
whicheigs(eig_vals)
X_reduced, transform = reduce(X_std, eig_vecs, dim)
print(X_reduced.shape)
print(transform.shape)
plotreduced(X_reduced)

#plt.show()

print('\n\n\n')

epochs = 30

def k_means(x, y, centroids=3):
    positions = 2* np.random.rand(centroids, dim).reshape(3, 3) - 1
    m = x.shape[0]

    for i in range(epochs):
        assignments = np.zeros(m)
        #assignment_dict = {0:'C1', 1:'C2', 2:'C3'}


        # assign each data point to a centroid
        for datapoint in range(m):
            difference = X_reduced[datapoint] - positions
            norms = np.linalg.norm(difference, 2, axis=1) # retain cols
            assignment = np.argmin(norms)
            print('difference', 'norms, assignment')
            print(difference, norms, assignment)
            assignments[datapoint] = assignment_dict[assignment]

        for c in range(centroids):
            positions[c] = np.mean(x[assignments == c], axis=0)
            #print(x[assignments == c])
            #print(np.mean(x[assignments==c], axis=0))

        print('\nassignments')
        print(assignments)
        print('\nLabels')
        print(Y)

    print(positions)
    return positions

k_means(X_reduced, Y, 3)