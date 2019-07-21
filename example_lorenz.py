import numpy as np
from scipy.integrate import odeint
from scipy.special import legendre
from sindy_utils import library_size


def get_lorenz_data(n_ics):
    t = np.arange(0, 5, .02)
    n_steps = t.size
    N = 128
    
    ic_means = np.array([0,0,25])
    ic_widths = 2*np.array([36,48,41])

    noise_strength = 1e-6

    # training data
    ics = ic_widths*(np.random.rand(n_ics, 3)-.5) + ic_means
    data = generate_lorenz_data(ics, t, N, linear=False, normalization=np.array([1/40,1/40,1/40]))
    data['x'] = data['x'].reshape((-1,N)) + noise_strength*np.random.randn(n_steps*n_ics,N)
    data['dx'] = data['dx'].reshape((-1,N)) + noise_strength*np.random.randn(n_steps*n_ics,N)
    data['ddx'] = data['ddx'].reshape((-1,N)) + noise_strength*np.random.randn(n_steps*n_ics,N)

    return data


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


def simulate_lorenz(z0, t):
    sigma = 10.
    beta = 8/3
    rho = 28 
    f = lambda z,t : [sigma*(z[1] - z[0]), z[0]*(rho - z[2]) - z[1], z[0]*z[1] - beta*z[2]]
    df = lambda z,dz,t : [sigma*(dz[1] - dz[0]),
                          dz[0]*(rho - z[2]) + z[0]*(-dz[2]) - dz[1],
                          dz[0]*z[1] + z[0]*dz[1] - beta*dz[2]]

    z = odeint(f, z0, t)

    dt = t[1] - t[0]
    dz = np.zeros(z.shape)
    ddz = np.zeros(z.shape)
    for i in range(t.size):
        dz[i] = f(z[i],dt*i)
        ddz[i] = df(z[i], dz[i], dt*i)
    return z, dz, ddz


def generate_lorenz_data(ics, t, n_points, linear=True, normalization=None):
    sigma = 10.
    beta = 8/3
    rho = 28 
    f = lambda z,t : [sigma*(z[1] - z[0]), z[0]*(rho - z[2]) - z[1], z[0]*z[1] - beta*z[2]]

    n_ics = ics.shape[0]
    n_steps = t.size
    dt = t[1]-t[0]

    d = 3
    z = np.zeros((n_ics,n_steps,d))
    dz = np.zeros(z.shape)
    ddz = np.zeros(z.shape)
    for i in range(n_ics):
        z[i], dz[i], ddz[i] = simulate_lorenz(ics[i], t)


    if normalization is not None:
        z *= normalization
        dz *= normalization
        ddz *= normalization

    n = n_points
    L = 1
    y_spatial = np.linspace(-L,L,n)

    modes = np.zeros((2*d, n))
    for i in range(2*d):
        modes[i] = legendre(i)(y_spatial)
    x1 = np.zeros((n_ics,n_steps,n))
    x2 = np.zeros((n_ics,n_steps,n))
    x3 = np.zeros((n_ics,n_steps,n))
    x4 = np.zeros((n_ics,n_steps,n))
    x5 = np.zeros((n_ics,n_steps,n))
    x6 = np.zeros((n_ics,n_steps,n))

    x = np.zeros((n_ics,n_steps,n))
    dx = np.zeros(x.shape)
    ddx = np.zeros(x.shape)
    for i in range(n_ics):
        for j in range(n_steps):
            x1[i,j] = modes[0]*z[i,j,0]
            x2[i,j] = modes[1]*z[i,j,1]
            x3[i,j] = modes[2]*z[i,j,2]
            x4[i,j] = modes[3]*z[i,j,0]**3
            x5[i,j] = modes[4]*z[i,j,1]**3
            x6[i,j] = modes[5]*z[i,j,2]**3

            x[i,j] = x1[i,j] + x2[i,j] + x3[i,j]
            if not linear:
                x[i,j] += x4[i,j] + x5[i,j] + x6[i,j]

            dx[i,j] = modes[0]*dz[i,j,0] + modes[1]*dz[i,j,1] + modes[2]*dz[i,j,2]
            if not linear:
                dx[i,j] += modes[3]*3*(z[i,j,0]**2)*dz[i,j,0] + modes[4]*3*(z[i,j,1]**2)*dz[i,j,1] + modes[5]*3*(z[i,j,2]**2)*dz[i,j,2]
            
            ddx[i,j] = modes[0]*ddz[i,j,0] + modes[1]*ddz[i,j,1] + modes[2]*ddz[i,j,2]
            if not linear:
                ddx[i,j] += modes[3]*(6*z[i,j,0]*dz[i,j,0]**2 + 3*(z[i,j,0]**2)*ddz[i,j,0]) \
                          + modes[4]*(6*z[i,j,1]*dz[i,j,1]**2 + 3*(z[i,j,1]**2)*ddz[i,j,1]) \
                          + modes[5]*(6*z[i,j,2]*dz[i,j,2]**2 + 3*(z[i,j,2]**2)*ddz[i,j,2])

    if normalization is None:
        Xi = lorenz_coefficients([1,1,1])
    else:
        Xi = lorenz_coefficients(normalization)

    data = {}
    data['t'] = t
    data['y_spatial'] = y_spatial
    data['x'] = x
    data['dx'] = dx
    data['ddx'] = ddx
    data['z'] = z
    data['dz'] = dz
    data['ddz'] = ddz
    data['Xi'] = Xi.astype(np.float32)

    return data
