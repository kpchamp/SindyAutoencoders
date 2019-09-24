import numpy as np
import scipy.io as sio


def get_rd_data(random=True):
    data = sio.loadmat('reaction_diffusion.mat')

    n_samples = data['t'].size
    n = data['x'].size
    N = n*n

    data['uf'] += 1e-6*np.random.randn(data['uf'].shape[0], data['uf'].shape[1], data['uf'].shape[2])
    data['duf'] += 1e-6*np.random.randn(data['duf'].shape[0], data['duf'].shape[1], data['duf'].shape[2])

    if not random:
        # consecutive samples
        training_samples = np.arange(int(.8*n_samples))
        val_samples = np.arange(int(.8*n_samples), int(.9*n_samples))
        test_samples = np.arange(int(.9*n_samples), n_samples)
    else:
        # random samples
        perm = np.random.permutation(int(.9*n_samples))
        training_samples = perm[:int(.8*n_samples)]
        val_samples = perm[int(.8*n_samples):]

        test_samples = np.arange(int(.9*n_samples), n_samples)

    training_data = {'t': data['t'][training_samples],
                     'y1': data['x'].T,
                     'y2': data['y'].T,
                     'x': data['uf'][:,:,training_samples].reshape((N,-1)).T,
                     'dx': data['duf'][:,:,training_samples].reshape((N,-1)).T}
    val_data = {'t': data['t'][val_samples],
                'y1': data['x'].T,
                'y2': data['y'].T,
                'x': data['uf'][:,:,val_samples].reshape((N,-1)).T,
                'dx': data['duf'][:,:,val_samples].reshape((N,-1)).T}
    test_data = {'t': data['t'][test_samples],
                 'y1': data['x'].T,
                 'y2': data['y'].T,
                 'x': data['uf'][:,:,test_samples].reshape((N,-1)).T,
                 'dx': data['duf'][:,:,test_samples].reshape((N,-1)).T}

    return training_data, val_data, test_data
