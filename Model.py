import numpy as np
import math as m
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf
from gym_super_mario_bros.actions import COMPLEX_MOVEMENT as POSSIBLE_ACTIONS
from Optimizers import Adam

H_size = 150
F_size = 3
NF1_out = 4
NF2_out = 8
NF3_out = 12
NF4_out = 16
NF5_out = 32
NF6_out = 64
z_size = 726
Y_size = len(POSSIBLE_ACTIONS)

def var(x):
    return tf.Variable(x, trainable=False)

class Param:
    def __init__(self, initial_mean, population_size, sigma):
        self.population_size = population_size
        self.mean = var(initial_mean)
        self.population = []

        # during visualization we use the initial_mean as the only parameters in the population
        if population_size == 1:
            self.population = [self.mean]

        # initialize population
        for _ in range(self.population_size//2):
            jitter = tf.random.normal(self.mean.shape, stddev=sigma)
            self.population.append(jitter + self.mean)
            self.population.append(self.mean - jitter)

        self.set_current(0)

    # change reference to current tensor
    def set_current(self, i):
        self.current = self.population[i]

    def get_grad(self, reward):
        jitter = self.population[0] - self.mean
        grad = jitter * reward[0]
        for i in range(1,self.population_size):
            jitter = self.population[i] - self.mean
            grad += jitter * reward[i]
        return grad

    def gen_pop_about_mean(self, sigma):
        for i in range(self.population_size//2):
            jitter = tf.random.normal(self.mean.shape, stddev=sigma)
            self.population[2*i+0] = jitter + self.mean
            self.population[2*i+1] = self.mean - jitter

class Parameters:
    def __init__(self, population_size=1, sigma=0, alpha=0, filename=''):
        self.population_size = population_size
        self.sigma = sigma
        self.alpha = alpha
        self.optimizer = Adam()

        if filename:
            npz = np.load(filename)
            self.F1 = Param(npz['arr_0'], population_size, sigma)
            self.F2 = Param(npz['arr_1'], population_size, sigma)
            self.F3 = Param(npz['arr_2'], population_size, sigma)
            self.F4 = Param(npz['arr_3'], population_size, sigma)
            self.F5 = Param(npz['arr_4'], population_size, sigma)
            self.F6 = Param(npz['arr_5'], population_size, sigma)

            self.g3 = Param(npz['arr_6'], population_size, sigma)
            self.b3 = Param(npz['arr_7'], population_size, sigma)
            self.g4 = Param(npz['arr_8'], population_size, sigma)
            self.b4 = Param(npz['arr_9'], population_size, sigma)
            self.g5 = Param(npz['arr_10'], population_size, sigma)
            self.b5 = Param(npz['arr_11'], population_size, sigma)
            self.g6 = Param(npz['arr_12'], population_size, sigma)
            self.b6 = Param(npz['arr_13'], population_size, sigma)

            self.Wx0 = Param(npz['arr_14'], population_size, sigma)
            self.bx0 = Param(npz['arr_15'], population_size, sigma)
            self.Wx1 = Param(npz['arr_16'], population_size, sigma)
            self.bx1 = Param(npz['arr_17'], population_size, sigma)
            self.Wx2 = Param(npz['arr_18'], population_size, sigma)
            self.bx2 = Param(npz['arr_19'], population_size, sigma)
            self.Wv = Param(npz['arr_20'], population_size, sigma)
            self.bv = Param(npz['arr_21'], population_size, sigma)

            self.lg0 = Param(npz['arr_22'], population_size, sigma)
            self.lb0 = Param(npz['arr_23'], population_size, sigma)
            self.lg1 = Param(npz['arr_24'], population_size, sigma)
            self.lb1 = Param(npz['arr_25'], population_size, sigma)
            self.lg2 = Param(npz['arr_26'], population_size, sigma)
            self.lb2 = Param(npz['arr_27'], population_size, sigma)
        else:
            # filter weight is whdo
            # w = width
            # h = height
            # d = depth (in channels)
            # o = out depth (out channels)?
            self.F1 = Param(tf.random.normal([F_size,F_size,3,NF1_out], stddev=m.sqrt(2/F_size)), population_size, sigma)
            self.F2 = Param(tf.random.normal([F_size,F_size,NF1_out,NF2_out], stddev=m.sqrt(2/F_size)), population_size, sigma)
            self.F3 = Param(tf.random.normal([F_size,F_size,NF2_out,NF3_out], stddev=m.sqrt(2/F_size)), population_size, sigma)
            self.g3 = Param(tf.ones((NF3_out,1)), population_size, sigma)
            self.b3 = Param(tf.zeros((NF3_out,1)), population_size, sigma)
            self.F4 = Param(tf.random.normal([F_size,F_size,NF3_out,NF4_out], stddev=m.sqrt(2/F_size)), population_size, sigma)
            self.g4 = Param(tf.ones((NF4_out,1)), population_size, sigma)
            self.b4 = Param(tf.zeros((NF4_out,1)), population_size, sigma)
            self.F5 = Param(tf.random.normal([F_size,F_size,NF4_out,NF5_out], stddev=m.sqrt(2/F_size)), population_size, sigma)
            self.g5 = Param(tf.ones((NF5_out,1)), population_size, sigma)
            self.b5 = Param(tf.zeros((NF5_out,1)), population_size, sigma)
            self.F6 = Param(tf.random.normal([F_size,F_size,NF5_out,NF6_out], stddev=m.sqrt(2/F_size)), population_size, sigma)
            self.g6 = Param(tf.ones((NF6_out,1)), population_size, sigma)
            self.b6 = Param(tf.zeros((NF6_out,1)), population_size, sigma)

            self.lg0 = Param(tf.ones((H_size,1)), population_size, sigma)
            self.lb0 = Param(tf.zeros((H_size,1)), population_size, sigma)
            self.lg1 = Param(tf.ones((H_size,1)), population_size, sigma)
            self.lb1 = Param(tf.zeros((H_size,1)), population_size, sigma)
            self.lg2 = Param(tf.ones((H_size,1)), population_size, sigma)
            self.lb2 = Param(tf.zeros((H_size,1)), population_size, sigma)

            self.Wx0 = Param(tf.random.normal([H_size * 4, z_size]), population_size, sigma)
            self.bx0 = Param(tf.zeros([H_size * 4, 1]), population_size, sigma)
            self.Wx1 = Param(tf.random.normal([H_size * 4, H_size*2]), population_size, sigma)
            self.bx1 = Param(tf.zeros([H_size * 4, 1]), population_size, sigma)
            self.Wx2 = Param(tf.random.normal([H_size * 4, H_size*2]), population_size, sigma)
            self.bx2 = Param(tf.zeros([H_size * 4, 1]), population_size, sigma)
            self.Wv = Param(tf.random.normal([Y_size, H_size]), population_size, sigma)
            self.bv = Param(tf.zeros([Y_size, 1]), population_size, sigma)

    def all(self):
        return [self.F1, self.F2, self.F3, self.F4, self.F5, self.F6,\
                self.g3, self.b3, self.g4, self.b4, self.g5, self.b5, self.g6, self.b6,\
                self.Wx0, self.bx0, self.Wx1, self.bx1, self.Wx2, self.bx2,\
                self.Wv, self.bv,\
                self.lg0,self.lb0,self.lg1,self.lb1,self.lg2,self.lb2]

    # return reference to current tensors
    def current(self):
        return [param.current for param in self.all()]

    def set_current(self, i):
        for param in self.all():
            param.set_current(i)

    def update(self, reward, reward_mean, reward_std):
        reward = (reward-reward_mean)/(reward_std+.00001)
        grads = []
        means = []
        for param in self.all():
            grads += [param.get_grad(reward) * (self.alpha / (self.population_size*self.sigma))]
            means += [param.mean]
        self.optimizer.update(means, grads)
        for param in self.all():
            param.gen_pop_about_mean(self.sigma)

# For visualization
import viridis
cmap = viridis.Viridis().getmap()
def post_process_activations(a):
    for i in range(len(a)):
        a[i] = tf.image.resize(a[i], (256,256), method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
    for i in range(len(a)):
        mx = tf.reduce_max(a[i])
        mn = tf.reduce_min(a[i])
        a[i]=tf.cast(255.*(a[i]-mn)/(mx-mn+.00001), tf.int32)
    for i in range(len(a)):
        a[i]=255.*tf.gather(params=cmap,indices=a[i])[:,:,0,:]
    return a

def gnorm(x, gamma, beta, G, eps=1e-5):
    N, H, W, C = x.get_shape().as_list()
    G = min(G, C)
    x = tf.reshape(x, [N, H, W, G, C // G])
    mean, var = tf.nn.moments(x, [1, 2, 4], keepdims=True)
    x = (x - mean) / tf.sqrt(var + eps)
    gamma = tf.reshape(gamma,[1,1,1,C])
    beta = tf.reshape(beta,[1,1,1,C])
    x = tf.reshape(x, [N, H, W, C]) * gamma + beta
    return x

def lnorm(x, gamma, beta, eps=1e-5):
    mean, var = tf.nn.moments(x,[0],keepdims=True)
    x=(x-mean)/tf.sqrt(var+eps)
    return x * gamma + beta

def conv(x,F,S):
    x = tf.nn.conv2d(x, F, strides=[1,1,1,1], padding='VALID')
    print('x:',x.shape)
    x = tf.nn.relu(x)
    x = tf.nn.max_pool(x,2,(1,1),padding='VALID')
    S=S//2
    x = tf.image.resize(x, (S,S))
    print('x:',x.shape)
    return x, S

def lstm(x,h,c,W,b,lg,lb):
    hx = tf.concat((x,h), axis=0)
    print('hx:',hx.shape)
    z = tf.matmul(W, hx) + b
    print('z:',z.shape)
    i, f, o, cp = tf.split(z, axis=0, num_or_size_splits=4)
    print('i:{}\nf:{}\no:{}\ncp:{}'.format(i.shape,f.shape,o.shape,cp.shape))
    i = tf.nn.sigmoid(lnorm(i,lg,lb))
    f = tf.nn.sigmoid(lnorm(f,lg,lb))
    o = tf.nn.sigmoid(lnorm(o,lg,lb))
    c = lnorm(f * c + i * tf.nn.tanh(lnorm(cp,lg,lb)),lg,lb)
    h = o * tf.nn.tanh(c)
    return h, c

# see tf.function docs section 'When to retrace?'
@tf.function
def forward(observation,\
        h0,c0,\
        h1,c1,\
        h2,c2,\
        F1,F2,F3,\
        F4,F5,F6,\
        g3,b3,\
        g4,b4,\
        g5,b5,\
        g6,b6,\
        Wx0,bx0,\
        Wx1,bx1,\
        Wx2,bx2,\
        Wv,bv,\
        lg0,lb0,\
        lg1,lb1,\
        lg2,lb2,
        visualize=False):

    print('inputs:',observation)
    x = observation[35:215,38:256-38]/255.
    print('x:',x.shape)
    S,_,_ = x.shape

    x,S = conv([x],F1,S)
    x,S = conv(x,F2,S)
    x,S = conv(x,F3,S)
    if visualize:
        a1 = x[0,:,:,0][:,:,None]
        a2 = x[0,:,:,1][:,:,None]
        a3 = x[0,:,:,2][:,:,None]
        a4 = x[0,:,:,3][:,:,None]
        a5 = x[0,:,:,4][:,:,None]
        a6 = x[0,:,:,5][:,:,None]
        a7 = x[0,:,:,6][:,:,None]
        a8 = x[0,:,:,7][:,:,None]
        a9 = x[0,:,:,8][:,:,None]
        a10 = x[0,:,:,9][:,:,None]
        a11 = x[0,:,:,10][:,:,None]
        a12 = x[0,:,:,11][:,:,None]
    x = gnorm(x,g3,b3,G=4)
    x,S = conv(x,F4,S)
    x = gnorm(x,g4,b4,G=4)
    x,S = conv(x,F5,S)
    x = gnorm(x,g5,b5,G=8)
    x = tf.nn.conv2d(x, F6, strides=[1,1,1,1], padding='VALID')
    x = gnorm(x,g6,b6,G=16)
    print('x:',x.shape)
    x = tf.nn.relu(x)

    x = tf.reshape(x, [-1, 1]) # flatten
    print('x:',x.shape)

    h0,c0 = lstm(x,h0,c0,Wx0,bx0,lg0,lb0)
    h1,c1 = lstm(h0,h1,c1,Wx1,bx1,lg1,lb1)
    h2,c2 = lstm(h1,h2,c2,Wx2,bx2,lg2,lb2)
    a = tf.matmul(Wv, h2) + bv

    if visualize:
        a13 = a[:,:,None]

    a = tf.argmax(a)

    if visualize:
        a14 = tf.reshape(tf.concat((h0,h1,h2),axis=0)[:289], (17,17))[:,:,None]
        a15 = tf.reshape(tf.concat((c0,c1,c2),axis=0)[:289], (17,17))[:,:,None]
        observation = tf.image.resize(observation, (256,256), method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
        a1,a2,a3,a4,a5,a6,a7,a8,a9,a10,a11,a12,a13,a14,a15 = post_process_activations([a1,a2,a3,a4,a5,a6,a7,a8,a9,a10,a11,a12,a13,a14,a15])
        return h0,c0,h1,c1,h2,c2,a,(a1,a2,a3,a4,a5,a6,a7,a8,a9,a10,a11,a12,a13,a14,a15,observation),
    else:
        return h0,c0,h1,c1,h2,c2,a

class Model:
    def __init__(self, population_size, observation, params, visualize=False):
        # TODO what is 6 here? (2 vectors for each lstm layer (have 3 lstm layers))
        self.rnn_states = [[tf.zeros((H_size,1))]*6 for _ in range(population_size)]
        self.params = params

        # forward once just to compile the graph
        forward(observation, *(self.rnn_states[0]), *params.current(), visualize)

        # whether to store activations for visualization
        self.visualize = visualize

    def __call__(self, observation, env_id):
        self.params.set_current(env_id)
        if self.visualize:
            *self.rnn_states[env_id], action, activations = forward(observation, *self.rnn_states[env_id], *self.params.current(), visualize=True)
            return action.numpy()[0], activations
        else:
            *self.rnn_states[env_id], action = forward(observation, *self.rnn_states[env_id], *self.params.current())
            return action.numpy()[0]

    def reset_rnn_states(self):
        for i in range(len(self.rnn_states)):
            for j in range(len(self.rnn_states[i])):
                self.rnn_states[i][j] *= 0






# def gnorm(x, gamma, beta, G, eps=1e-5):
#     # normalize
#     # transpose: [bs, h, w, c] to [bs, c, h, w] folloing the paper
#     x = tf.transpose(x, [0,3,1,2])
#     N,C,H,W=x.shape
#     G=min(G,C)
#     x=tf.reshape(x,[-1,G,C//G,H,W])
#     mean, var = tf.nn.moments(x,[2,3,4],keepdims=True)
#     x=(x-mean)/tf.sqrt(var+eps)
#     # per channel gamma and beta
#     gamma = tf.reshape(gamma,[1,C,1,1])
#     beta = tf.reshape(beta,[1,C,1,1])
#     output = tf.reshape(x,[-1,C,H,W]) * gamma + beta
#     return tf.transpose(output, [0,2,3,1])



