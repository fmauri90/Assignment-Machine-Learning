import matplotlib.pylab as plt

def plot_mnist_weights(W_fc2):
    assert W_fc2.shape == (784,10)
    # plot weights
    fig = plt.figure()
    for digit in range(W_fc2.shape[1]):
        ax = fig.add_subplot(4,3,digit+1)
        ax.set_aspect('equal')
        plt.imshow(W_fc2.T[digit].reshape(28,28), interpolation='nearest', cmap=plt.cm.ocean)
        plt.plt.colorbar()
    plt.show()
