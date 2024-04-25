import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np

def bruijn_sequence(resolution):
    # calculate Moser–de Bruijn sequence
    bru_seq = np.array([0,1])
    bru_pow = 4
    while bru_seq.shape[0] <= 4*resolution: 
        bru_new = np.full((bru_seq.shape[0]), bru_pow) + bru_seq
        bru_seq = np.concatenate((bru_seq, bru_new))
        bru_pow *= 4
    
    bru_x = tf.constant(bru_seq)
    bru_y = tf.constant(bru_seq * 2)
    return bru_x, bru_y

def init_population(n, resolution):
    props = np.ones((n, 4), dtype=np.float32)
    # Infection progress 
    props[:,0] = 0.0
    # infectious
    props[:,1] = 0.0
    # hospitalized
    props[:,2] = 0.0
    # recovering
    props[:,3] = 0.0

    props[100:150,0] = 1.0
    #props[100:150,1] = 1.0

    positions = tf.random.uniform(shape=[n, 2], minval=resolution, maxval=2*resolution, dtype=tf.float32, seed=0) # the position of every person as a couple of floats

    return tf.constant(props), positions

#@tf.function
def diffuse_infection(n, n_nearest, positions, props):
    diffusion = tf.constant(0.0, shape=(n + n_nearest,), dtype=tf.float32)

    # pad arrays to account for extreme indexes in the arrays (introducing very far snd so ininfluential people)
    half_nearest = int(n_nearest/2)
    pos_augmented0 = tf.pad(positions, tf.constant([[half_nearest, half_nearest],[0,0]]), "CONSTANT", constant_values=1000000.0)
    pos_augmented1 = tf.pad(positions, tf.constant([[0, n_nearest], [0,0]]), "CONSTANT", constant_values=1000000.0)
    props_augmented1 = tf.pad(props, tf.constant([[0, n_nearest], [0,0]]), "CONSTANT", constant_values=0.0)

    # calculate the sum excluding an element itself
    for i in range(n_nearest+1):
        if i == half_nearest:   continue
        distance = tf.math.reduce_sum(tf.math.square(pos_augmented0 - tf.roll(pos_augmented1, shift=[i, 0], axis=[0,1])), 1)
        distance = tf.math.sqrt(distance)
        hazard = tf.roll(props_augmented1, shift=[i, 0], axis=[0,1])
        hazard = hazard[:, 1:2] # Ip'
        diffusion += 1/(1+distance) * tf.squeeze(hazard)

    # un-pad the arrays
    diffusion = diffusion[half_nearest : -half_nearest] 
    return diffusion


#@tf.function
def simulation_step(n, delta_time, props, positions, bru_x, bru_y):
    n_nearest = 24 # must be multiple of 2
    infect_rate = 10
    progress_threshold = 0.8 # when R goes above this, the infection starts to heal
    hospitalization_threshold = 0.2 # one isolates itself 

    # calculate z curve values
    Zx = tf.gather(bru_x, tf.squeeze(tf.cast(positions[:, 0:1], tf.int32)))
    Zy = tf.gather(bru_y, tf.squeeze(tf.cast(positions[:, 1:], tf.int32)))
    Z_vals = Zx + Zy
    # order people based on position on the Z curve
    sort_indexes = tf.argsort(Z_vals)
    positions = tf.gather(positions, sort_indexes)
    props = tf.gather(props, sort_indexes)

    # repeat ordering but with inverted axis to mitigate z curve problems
    Zx = tf.gather(bru_y, tf.squeeze(tf.cast(positions[:, 0:1], tf.int32)))
    Zy = tf.gather(bru_x, tf.squeeze(tf.cast(positions[:, 1:], tf.int32)))
    Z_vals = Zx + Zy
    sort_indexes1 = tf.argsort(Z_vals)
    positions1 = tf.gather(positions, sort_indexes)
    props1 = tf.gather(props, sort_indexes)

    # Diffuse infection
    # PRp' = (1-Rp) * infect_rate * (sum for nearest people p' (activation(distance pp') * Ip' * (1 - Hp'))) + PRp * (progress_threshold - Rp)
    # the sum term is carried out translating the props vector with respect to itself to get nearest people p'
    diffusion = diffuse_infection(n, n_nearest, positions, props)
    diffusion1 = diffuse_infection(n, n_nearest, positions1, props1)    # repeat with shuffled indexes (to mitigate z curve problems)
    diffusion += tf.gather(diffusion1, tf.gather(sort_indexes1, sort_indexes1)) # unshuffle and add up

    PR, I, H, R = tf.squeeze(props[:, 0:1]),tf.squeeze(props[:, 1:2]), tf.squeeze(props[:, 2:3]), tf.squeeze(props[:, 3:]) 
    diffusion *= infect_rate * (1 - R)  # infect_rate * (1-R)

    # PR * (progress_threshold - R), the infection goes up exponentially until a certain threshold, then it goes down
    PR_der = diffusion + PR * (progress_threshold - R)

    # R' = PR
    R_der = PR

    # I = activation(PR) * H
    I = PR * PR * (1 - H)

    # H goes to 1 if the PR is above a threshold, goes to 0 if below another threshold
    H1 = tf.math.logical_and(tf.less(H, tf.constant([0.5])), tf.greater(PR, tf.constant([hospitalization_threshold])))
    H0 = tf.math.logical_and(tf.greater(H, tf.constant([0.5])), tf.less(PR, tf.constant([hospitalization_threshold])))
    H = tf.where(tf.math.logical_or(H0, H1), x=(1 - H), y= H)

    # progress PR and R of a step
    PR += PR_der * delta_time
    R += R_der * delta_time

    # progress position in a random direction
    speed = 20
    positions += tf.random.uniform(shape=[n, 2], minval=0, maxval=speed, dtype=tf.float32, seed=0) * delta_time

    # re stack properties together
    props = tf.stack([PR, I, H, R], axis=1)

    # plot population
    #p_numpy = positions.numpy()
    #plot_data = {'x': p_numpy[:, 0],
    #            'y': p_numpy[:, 1],
    #            'c': props.numpy()[:,0]}
    #plt.scatter('x', 'y', c='c', s = 4.0, vmin=0.0, vmax=1.0, data=plot_data)    
    ##plt.plot(p_numpy[:,0], p_numpy[:,1], linewidth=.5)
    #plt.show()

    return props, positions


def particles_simulation(n, delta_time):
    # the properties of every person: Pr (infection progress), I (infectious), H (hospitalized), R (recovering)
    resolution = 1000 # possible positions per dimension
    props, positions = init_population(n, resolution)

    bru_x, bru_y = bruijn_sequence(resolution)

    t, S, I, R = [], [], [], []
    for iter in range(50):
        props, positions = simulation_step(n, delta_time, props, positions, bru_x, bru_y)
        # extract info from simulation
        t.append(iter)
        # PR < 0.5 and R < 0.5
        S.append(tf.reduce_sum(tf.cast(tf.math.logical_and(tf.less(props[:, 0:1], tf.constant([0.5])), tf.less(props[:, 3:], tf.constant([0.5]))), tf.float32)).numpy())
        # PR >= 0.5
        I.append(tf.reduce_sum(tf.cast(tf.greater(props[:, 0:1], tf.constant([0.5])), tf.float32)).numpy())
        # PR < 0.5 and R >= 0.5
        R.append(tf.reduce_sum(tf.cast(tf.math.logical_and(tf.less(props[:, 0:1], tf.constant([0.5])), tf.greater(props[:, 3:], tf.constant([0.5]))), tf.float32)).numpy())

    plt.plot(t, S, label = 'S')
    plt.plot(t, I, label = 'I')
    plt.plot(t, R, label = 'R')
    plt.legend()
    plt.show()

particles_simulation(900, 0.1)