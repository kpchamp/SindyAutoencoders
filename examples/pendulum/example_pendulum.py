import numpy as np
from scipy.integrate import odeint


def get_pendulum_data(n_ics):
    t,x,dx,ddx,z = generate_pendulum_data(n_ics)
    data = {}
    data['t'] = t
    data['x'] = x.reshape((n_ics*t.size, -1))
    data['dx'] = dx.reshape((n_ics*t.size, -1))
    data['ddx'] = ddx.reshape((n_ics*t.size, -1))
    data['z'] = z.reshape((n_ics*t.size, -1))[:,0:1]
    data['dz'] = z.reshape((n_ics*t.size, -1))[:,1:2]

    return data


def generate_pendulum_data(n_ics):
    f  = lambda z, t : [z[1], -np.sin(z[0])]
    t = np.arange(0, 10, .02)

    z = np.zeros((n_ics,t.size,2))
    dz = np.zeros(z.shape)

    z1range = np.array([-np.pi,np.pi])
    z2range = np.array([-2.1,2.1])
    i = 0
    while (i < n_ics):
        z0 = np.array([(z1range[1]-z1range[0])*np.random.rand()+z1range[0],
            (z2range[1]-z2range[0])*np.random.rand()+z2range[0]])
        if np.abs(z0[1]**2/2. - np.cos(z0[0])) > .99:
            continue
        z[i] = odeint(f, z0, t)
        dz[i] = np.array([f(z[i,j], t[j]) for j in range(len(t))])
        i += 1

    x,dx,ddx = pendulum_to_movie(z, dz)

    # n = 51
    # xx,yy = np.meshgrid(np.linspace(-1.5,1.5,n),np.linspace(1.5,-1.5,n))
    # create_image = lambda theta : np.exp(-((xx-np.cos(theta-np.pi/2))**2 + (yy-np.sin(theta-np.pi/2))**2)/.05)
    # argument_derivative = lambda theta,dtheta : -1/.05*(2*(xx - np.cos(theta-np.pi/2))*np.sin(theta-np.pi/2)*dtheta \
    #                                                   + 2*(yy - np.sin(theta-np.pi/2))*(-np.cos(theta-np.pi/2))*dtheta)
    # argument_derivative2 = lambda theta,dtheta,ddtheta : -2/.05*((np.sin(theta-np.pi/2))*np.sin(theta-np.pi/2)*dtheta**2 \
    #                                                            + (xx - np.cos(theta-np.pi/2))*np.cos(theta-np.pi/2)*dtheta**2 \
    #                                                            + (xx - np.cos(theta-np.pi/2))*np.sin(theta-np.pi/2)*ddtheta \
    #                                                            + (-np.cos(theta-np.pi/2))*(-np.cos(theta-np.pi/2))*dtheta**2 \
    #                                                            + (yy - np.sin(theta-np.pi/2))*(np.sin(theta-np.pi/2))*dtheta**2 \
    #                                                            + (yy - np.sin(theta-np.pi/2))*(-np.cos(theta-np.pi/2))*ddtheta)
        
    # x = np.zeros((n_ics, t.size, n, n))
    # dx = np.zeros((n_ics, t.size, n, n))
    # ddx = np.zeros((n_ics, t.size, n, n))
    # for i in range(n_ics):
    #     for j in range(t.size):
    #         z[i,j,0] = wrap_to_pi(z[i,j,0])
    #         x[i,j] = create_image(z[i,j,0])
    #         dx[i,j] = (create_image(z[i,j,0])*argument_derivative(z[i,j,0], dz[i,j,0]))
    #         ddx[i,j] = create_image(z[i,j,0])*((argument_derivative(z[i,j,0], dz[i,j,0]))**2 \
    #                         + argument_derivative2(z[i,j,0], dz[i,j,0], dz[i,j,1]))

    return t,x,dx,ddx,z


def pendulum_to_movie(z, dz):
    n_ics = z.shape[0]
    n_samples = z.shape[1]
    n = 51
    y1,y2 = np.meshgrid(np.linspace(-1.5,1.5,n),np.linspace(1.5,-1.5,n))
    create_image = lambda theta : np.exp(-((y1-np.cos(theta-np.pi/2))**2 + (y2-np.sin(theta-np.pi/2))**2)/.05)
    argument_derivative = lambda theta,dtheta : -1/.05*(2*(y1 - np.cos(theta-np.pi/2))*np.sin(theta-np.pi/2)*dtheta \
                                                      + 2*(y2 - np.sin(theta-np.pi/2))*(-np.cos(theta-np.pi/2))*dtheta)
    argument_derivative2 = lambda theta,dtheta,ddtheta : -2/.05*((np.sin(theta-np.pi/2))*np.sin(theta-np.pi/2)*dtheta**2 \
                                                               + (y1 - np.cos(theta-np.pi/2))*np.cos(theta-np.pi/2)*dtheta**2 \
                                                               + (y1 - np.cos(theta-np.pi/2))*np.sin(theta-np.pi/2)*ddtheta \
                                                               + (-np.cos(theta-np.pi/2))*(-np.cos(theta-np.pi/2))*dtheta**2 \
                                                               + (y2 - np.sin(theta-np.pi/2))*(np.sin(theta-np.pi/2))*dtheta**2 \
                                                               + (y2 - np.sin(theta-np.pi/2))*(-np.cos(theta-np.pi/2))*ddtheta)
        
    x = np.zeros((n_ics, n_samples, n, n))
    dx = np.zeros((n_ics, n_samples, n, n))
    ddx = np.zeros((n_ics, n_samples, n, n))
    for i in range(n_ics):
        for j in range(n_samples):
            z[i,j,0] = wrap_to_pi(z[i,j,0])
            x[i,j] = create_image(z[i,j,0])
            dx[i,j] = (create_image(z[i,j,0])*argument_derivative(z[i,j,0], dz[i,j,0]))
            ddx[i,j] = create_image(z[i,j,0])*((argument_derivative(z[i,j,0], dz[i,j,0]))**2 \
                            + argument_derivative2(z[i,j,0], dz[i,j,0], dz[i,j,1]))
            
    return x,dx,ddx


def wrap_to_pi(z):
    z_mod = z % (2*np.pi)
    subtract_m = (z_mod > np.pi) * (-2*np.pi)
    return z_mod + subtract_m
