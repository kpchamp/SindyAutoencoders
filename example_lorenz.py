import numpy as np
from scipy.integrate import odeint
from scipy.special import legendre
from sindy_utils import library_size


def get_lorenz_data(n_training_ics, n_validation_ics, n_test_ics):
    t = np.arange(0, 5, .02)
    n_steps = t.size
    N = 128
    
    ic_means = np.array([0,0,25])
    ic_widths = 2*np.array([36,48,41])

    noise_strength = 1e-6

    # training data
    ics = ic_widths*(np.random.rand(n_training_ics, 3)-.5) + ic_means
    training_data = generate_lorenz_data(ics, t, N, linear=False, normalization=np.array([1/40,1/40,1/40]))
    training_data['u'] = training_data['u'].reshape((-1,N)) + noise_strength*np.random.randn(n_steps*n_training_ics,N)
    training_data['du'] = training_data['du'].reshape((-1,N)) + noise_strength*np.random.randn(n_steps*n_training_ics,N)
    training_data['ddu'] = training_data['ddu'].reshape((-1,N)) + noise_strength*np.random.randn(n_steps*n_training_ics,N)

    # validation data
    ics = ic_widths*(np.random.rand(n_validation_ics, 3)-.5) + ic_means
    val_data = generate_lorenz_data(ics, t, 128, linear=False, normalization=np.array([1/40,1/40,1/40]))
    val_data['u'] = val_data['u'].reshape((-1,N)) + noise_strength*np.random.randn(n_steps*n_validation_ics,N)
    val_data['du'] = val_data['du'].reshape((-1,N)) + noise_strength*np.random.randn(n_steps*n_validation_ics,N)
    val_data['ddu'] = val_data['ddu'].reshape((-1,N)) + noise_strength*np.random.randn(n_steps*n_validation_ics,N)
    
    # test data
    ics = ic_widths*(np.random.rand(n_test_ics, 3)-.5) + ic_means
    test_data = generate_lorenz_data(ics, t, 128, linear=False, normalization=np.array([1/40,1/40,1/40]))
    test_data['u'] = test_data['u'].reshape((-1,N)) + noise_strength*np.random.randn(n_steps*n_test_ics,N)
    test_data['du'] = test_data['du'].reshape((-1,N)) + noise_strength*np.random.randn(n_steps*n_test_ics,N)
    test_data['ddu'] = test_data['ddu'].reshape((-1,N)) + noise_strength*np.random.randn(n_steps*n_test_ics,N)

    return training_data, val_data, test_data


def lorenz_coefficients(normalization, poly_order=3):
    sigma = 10
    beta = 8/3
    rho = 28
    Xi = np.zeros((library_size(3,poly_order),3))
    Xi[1,0] = -sigma
    Xi[2,0] = sigma*normalization[0]/normalization[1]
    Xi[1,1] = rho*normalization[1]/normalization[0]
    Xi[2,1] = -1
    Xi[6,1] = -normalization[1]/(normalization[0]*normalization[2])
    Xi[3,2] = -beta
    Xi[5,2] = normalization[2]/(normalization[0]*normalization[1])
    return Xi


def simulate_lorenz(x0, t):
    sigma = 10.
    beta = 8/3
    rho = 28 
    f = lambda x,t : [sigma*(x[1] - x[0]), x[0]*(rho - x[2]) - x[1], x[0]*x[1] - beta*x[2]]
    df = lambda x,dx,t : [sigma*(dx[1] - dx[0]),
                          dx[0]*(rho - x[2]) + x[0]*(-dx[2]) - dx[1],
                          dx[0]*x[1] + x[0]*dx[1] - beta*dx[2]]

    v = odeint(f, x0, t)

    dt = t[1] - t[0]
    dv = np.zeros(v.shape)
    ddv = np.zeros(v.shape)
    for i in range(t.size):
        dv[i] = f(v[i],dt*i)
        ddv[i] = df(v[i], dv[i], dt*i)
    return v, dv, ddv


def generate_lorenz_data(ics, t, n_points, linear=True, normalization=None):
    sigma = 10.
    beta = 8/3
    rho = 28 
    f = lambda x,t : [sigma*(x[1] - x[0]), x[0]*(rho - x[2]) - x[1], x[0]*x[1] - beta*x[2]]

    n_ics = ics.shape[0]
    n_steps = t.size
    dt = t[1]-t[0]

    d = 3
    v = np.zeros((n_ics,n_steps,d))
    dv = np.zeros(v.shape)
    ddv = np.zeros(v.shape)
    for i in range(n_ics):
        v[i], dv[i], ddv[i] = simulate_lorenz(ics[i], t)


    if normalization is not None:
        v *= normalization
        dv *= normalization
        ddv *= normalization

    n = n_points
    L = 1
    x = np.linspace(-L,L,n)

    modes = np.zeros((2*d, n))
    for i in range(2*d):
        modes[i] = legendre(i)(x)
    u1 = np.zeros((n_ics,n_steps,n))
    u2 = np.zeros((n_ics,n_steps,n))
    u3 = np.zeros((n_ics,n_steps,n))
    u4 = np.zeros((n_ics,n_steps,n))
    u5 = np.zeros((n_ics,n_steps,n))
    u6 = np.zeros((n_ics,n_steps,n))

    u = np.zeros((n_ics,n_steps,n))
    du = np.zeros(u.shape)
    ddu = np.zeros(u.shape)
    for i in range(n_ics):
        for j in range(n_steps):
            u1[i,j] = modes[0]*v[i,j,0]
            u2[i,j] = modes[1]*v[i,j,1]
            u3[i,j] = modes[2]*v[i,j,2]
            u4[i,j] = modes[3]*v[i,j,0]**3
            u5[i,j] = modes[4]*v[i,j,1]**3
            u6[i,j] = modes[5]*v[i,j,2]**3

            u[i,j] = u1[i,j] + u2[i,j] + u3[i,j]
            if not linear:
                u[i,j] += u4[i,j] + u5[i,j] + u6[i,j]

            du[i,j] = modes[0]*dv[i,j,0] + modes[1]*dv[i,j,1] + modes[2]*dv[i,j,2]
            if not linear:
                du[i,j] += modes[3]*3*(v[i,j,0]**2)*dv[i,j,0] + modes[4]*3*(v[i,j,1]**2)*dv[i,j,1] + modes[5]*3*(v[i,j,2]**2)*dv[i,j,2]
            
            ddu[i,j] = modes[0]*ddv[i,j,0] + modes[1]*ddv[i,j,1] + modes[2]*ddv[i,j,2]
            if not linear:
                ddu[i,j] += modes[3]*(6*v[i,j,0]*dv[i,j,0]**2 + 3*(v[i,j,0]**2)*ddv[i,j,0]) \
                          + modes[4]*(6*v[i,j,1]*dv[i,j,1]**2 + 3*(v[i,j,1]**2)*ddv[i,j,1]) \
                          + modes[5]*(6*v[i,j,2]*dv[i,j,2]**2 + 3*(v[i,j,2]**2)*ddv[i,j,2])

    if normalization is None:
        Xi = lorenz_coefficients([1,1,1])
    else:
        Xi = lorenz_coefficients(normalization)

    data = {}
    data['t'] = t
    data['x'] = x
    data['u'] = u
    data['du'] = du
    data['ddu'] = ddu
    data['v'] = v
    data['dv'] = dv
    data['ddv'] = ddv
    data['Xi'] = Xi.astype(np.float32)

    return data
