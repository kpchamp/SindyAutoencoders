import numpy as np
from scipy.integrate import odeint


def get_pendulum_data(n_training_ics, n_validation_ics, n_test_ics):
    t,u,du,ddu,v = generate_pendulum_data(n_training_ics)
    training_data = {}
    training_data['t'] = t
    training_data['u'] = u.reshape((n_training_ics*t.size, -1))
    training_data['du'] = du.reshape((n_training_ics*t.size, -1))
    training_data['ddu'] = ddu.reshape((n_training_ics*t.size, -1))
    training_data['v'] = v.reshape((n_training_ics*t.size, -1))[:,0:1]
    training_data['dv'] = v.reshape((n_training_ics*t.size, -1))[:,1:2]

    t,u,du,ddu,v = generate_pendulum_data(n_validation_ics)
    val_data = {}
    val_data['t'] = t
    val_data['u'] = u.reshape((n_validation_ics*t.size, -1))
    val_data['du'] = du.reshape((n_validation_ics*t.size, -1))
    val_data['ddu'] = ddu.reshape((n_validation_ics*t.size, -1))
    val_data['v'] = v.reshape((n_validation_ics*t.size, -1))[:,0:1]
    val_data['dv'] = v.reshape((n_validation_ics*t.size, -1))[:,1:2]

    t,u,du,ddu,v = generate_pendulum_data(n_test_ics)
    test_data = {}
    test_data['t'] = t
    test_data['u'] = u.reshape((n_test_ics*t.size, -1))
    test_data['du'] = du.reshape((n_test_ics*t.size, -1))
    test_data['ddu'] = ddu.reshape((n_test_ics*t.size, -1))
    test_data['v'] = v.reshape((n_test_ics*t.size, -1))[:,0:1]
    test_data['dv'] = v.reshape((n_test_ics*t.size, -1))[:,1:2]

    return training_data, val_data, test_data


def generate_pendulum_data(n_ics):
    f  = lambda x, t : [x[1], -np.sin(x[0])]
    t = np.arange(0, 10, .02)

    x = np.zeros((n_ics,t.size,2))
    dx = np.zeros(x.shape)

    x1range = np.array([-np.pi,np.pi])
    x2range = np.array([-2.1,2.1])
    i = 0
    while (i < n_ics):
        x0 = np.array([(x1range[1]-x1range[0])*np.random.rand()+x1range[0],
            (x2range[1]-x2range[0])*np.random.rand()+x2range[0]])
        if np.abs(x0[1]**2/2. - np.cos(x0[0])) > .99:
            continue
        x[i] = odeint(f, x0, t)
        dx[i] = np.array([f(x[i,j], t[j]) for j in range(len(t))])
        i += 1

    n = 51
    xx,yy = np.meshgrid(np.linspace(-1.5,1.5,n),np.linspace(1.5,-1.5,n))
    create_image = lambda theta : np.exp(-((xx-np.cos(theta-np.pi/2))**2 + (yy-np.sin(theta-np.pi/2))**2)/.05)
    argument_derivative = lambda theta,dtheta : -1/.05*(2*(xx - np.cos(theta-np.pi/2))*np.sin(theta-np.pi/2)*dtheta \
                                                      + 2*(yy - np.sin(theta-np.pi/2))*(-np.cos(theta-np.pi/2))*dtheta)
    argument_derivative2 = lambda theta,dtheta,ddtheta : -2/.05*((np.sin(theta-np.pi/2))*np.sin(theta-np.pi/2)*dtheta**2 \
                                                               + (xx - np.cos(theta-np.pi/2))*np.cos(theta-np.pi/2)*dtheta**2 \
                                                               + (xx - np.cos(theta-np.pi/2))*np.sin(theta-np.pi/2)*ddtheta \
                                                               + (-np.cos(theta-np.pi/2))*(-np.cos(theta-np.pi/2))*dtheta**2 \
                                                               + (yy - np.sin(theta-np.pi/2))*(np.sin(theta-np.pi/2))*dtheta**2 \
                                                               + (yy - np.sin(theta-np.pi/2))*(-np.cos(theta-np.pi/2))*ddtheta)
        
    u = np.zeros((n_ics, t.size, n, n))
    du = np.zeros((n_ics, t.size, n, n))
    ddu = np.zeros((n_ics, t.size, n, n))
    for i in range(n_ics):
        for j in range(t.size):
            x[i,j,0] = wrap_to_pi(x[i,j,0])
            u[i,j] = create_image(x[i,j,0])
            du[i,j] = (create_image(x[i,j,0])*argument_derivative(x[i,j,0], dx[i,j,0]))
            ddu[i,j] = create_image(x[i,j,0])*((argument_derivative(x[i,j,0], dx[i,j,0]))**2 \
                            + argument_derivative2(x[i,j,0], dx[i,j,0], dx[i,j,1]))

    return t,u,du,ddu,x


def wrap_to_pi(x):
    x_mod = x % (2*np.pi)
    subtract_m = (x_mod > np.pi) * (-2*np.pi)
    return x_mod + subtract_m
