import numpy as np      # effective math
import matplotlib.pyplot as plt     # ultimate plotting tool
from mpl_toolkits.mplot3d import Axes3D     # 3D plots
import pandas as pd     # allow us to make dataframes to store our data cleanly

data = pd.read_csv('Iris.csv').set_index('Id')  # read our data into a dataframe and set the index to the ID
print(data)     # last column is a label, all other columns are features

X = np.array(data[data.columns[:-1]])   # set our design matrix to the features (all columns except last)
label_dict = {'Iris-setosa':0, 'Iris-versicolor':1, 'Iris-virginica':2}  # dictionary containing label to number mapping
Y = np.array([label_dict[i] for i in data[data.columns[-1]]])   # put out labels into a np array
print(Y)

print(X.shape)      # 150 rows (datapoints), 4 columns (features)
print(Y.shape)      # 150 single dimension labels

m = X.shape[0]      # 150 rows

def normalise(x):   # centre around mean and divide by range to put all features on similar scale
    x_std = x - np.mean(x, axis=0)      # subtract the mean
    x_std = np.divide(x_std, np.std(x_std, axis=0))     # divide each feature by the range of that feature (-1 < x < 1)
    return x_std    # return our standardised features

def decompose(x):
    cov = np.matmul(x.T, x)     # compute the covariance matrix
    print('\nCovariance matrix')
    print(cov)

    eig_vals, eig_vecs = np.linalg.eig(cov)     # find the eigenvalues and eigenvectors of the covariance matrix
    print('\nEigenvectors')
    print(eig_vecs)
    print('\nEigenvalues')
    print(eig_vals)
    return eig_vals, eig_vecs, cov

def whicheigs(eig_vals):
    """"Plot the variance accounted for by each eigenvector and their cumulative sum"""
    total = sum(eig_vals)   # sum up the eigenvalues so we can compare each one to the total to determine their importance
    var_percent = [(i/total) * 100 for i in eig_vals]   # calculate the percentage variance of the data which this eigenvalue accounts for
    cum_var_percent = np.cumsum(var_percent)    # make a vector of the cumulative sum of the variance percentages

    fig = plt.figure()      # make a figure
    ax =  fig.add_subplot(111)      # add an axis
    plt.title('Variance along different principal components')
    ax.grid()
    plt.xlabel('Principal Component')
    plt.ylabel('Percentage total variance accounted for')

    ax.plot(cum_var_percent, '-ro')     # plot the cumulative sum of the variances accounted for by each eigenvector
    ax.bar(range(len(eig_vals)), var_percent) # position, height # show how much variance individual eig accounts for
    plt.xticks(np.arange(len(eig_vals)), ('PC{}'.format(i) for i in range(len(eig_vals))))  # set the xticks to 'PC1' etc
    plt.show()  # show us the figure

def reduce(x, eig_vecs, dims):      # reduce the number of dimensions of our data by
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

X_std = normalise(X)    # centre data around mean and divide by sd
eig_vals, eig_vecs, covariance = decompose(X_std)      # compute the covariance matrix and find its characteristics
whicheigs(eig_vals)     # visualise the variance of the data for each eigenvector of the covariance matrix
X_reduced, transform = reduce(X_std, eig_vecs, dim)     # transform our data into a lower dimension
print(X_reduced.shape)
print(transform.shape)
plotreduced(X_reduced)      # check out how the data looks in a visualisable dimension

#plt.show()

print('\n\n\n')

epochs = 30

def k_means(x, y, centroids=3):
    """Cluster the data"""
    positions = 2* np.random.rand(centroids, dim).reshape(centroids, 3) - 1 # randomly intialise initial centroid positions
    m = x.shape[0]  # number of training examples

    for i in range(epochs):     # cycle through our data a bunch of times
        assignments = np.zeros(m)   # initialise a list of which datapoints belong to which class
        #assignment_dict = {0:'C1', 1:'C2', 2:'C3'}


        # assign each data point to a centroid
        for datapoint in range(m):  # for each of our datapoints
            difference = X_reduced[datapoint] - positions   # find the distance from each centroid (row) in each dimension (column)
            norms = np.linalg.norm(difference, 2, axis=1) # find the euclidian distance from the datapoint to each centroid
            assignment = np.argmin(norms)   # assign that datapoint to the nearest centroid
            print('difference', 'norms, assignment')
            print(difference, norms, assignment)
            assignments[datapoint] = assignment #assignment_dict[assignment] # update the list of all assignments

        for c in range(centroids):  # update the position of each centroid to the mean of the datapoints currently assigned to it
            positions[c] = np.mean(x[assignments == c], axis=0)     # find which datapoints are assigned to each centroid, index them from the design matrix and find their mean position
            #print(x[assignments == c])
            #print(np.mean(x[assignments==c], axis=0))

        print('\nassignments')
        print(assignments)
        print('\nLabels')
        print(y)

    print('Centroid positions')
    print(positions)
    return positions

k_means(X_reduced, Y, 2)