import tensorflow as tf
import numpy as np

class GradientDescent:
    def __init__(self, lr=1, weight_decay=0.0005):
        self.lr = lr
        self.weight_decay = weight_decay

    def update(self, params, grads):
        n = 0
        for i,(p,g) in enumerate(zip(params, grads)):
            if self.weight_decay != 0.:
                p.assign_add(-self.weight_decay*self.lr*p)
            p.assign_add(self.lr*g)
            n += np.prod(p.shape)

# TODO Adam is not initilized correctly after restart
# I have to save Adam.v_t and Adam.m_t

# taken from https://gist.github.com/Harhro94/3b809c5ae778485a9ea9d253c4bfc90a
class Adam:
    """Adam optimizer.
    Default parameters follow those provided in the original paper.
    # Arguments
        lr: float >= 0. Learning rate.
        beta_1: float, 0 < beta < 1. Generally close to 1.
        beta_2: float, 0 < beta < 1. Generally close to 1.
        epsilon: float >= 0. Fuzz factor.
        decay: float >= 0. Learning rate decay over each update.
    # References
        - [Adam - A Method for Stochastic Optimization](http://arxiv.org/abs/1412.6980v8)
    """
    def __init__(self, lr=1, beta_1=0.9, beta_2=0.999,
                epsilon=1e-2, decay=0.0, weight_decay=0.0005):
        self.iterations = 0
        self.lr = lr
        self.beta_1 = beta_1
        self.beta_2 = beta_2
        self.decay = decay
        self.epsilon = epsilon
        self.initial_decay = decay
        self.weight_decay = weight_decay

    def update(self, params, grads):
        """ params and grads are list of numpy arrays
        """

        """ #TODO: implement clipping
        if hasattr(self, 'clipnorm') and self.clipnorm > 0:
            norm = np.sqrt(sum([np.sum(np.square(g)) for g in grads]))
            grads = [clip_norm(g, self.clipnorm, norm) for g in grads]
        if hasattr(self, 'clipvalue') and self.clipvalue > 0:
            grads = [K.clip(g, -self.clipvalue, self.clipvalue) for g in grads]
        """

        lr = self.lr
        if self.initial_decay > 0:
            lr *= (1. / (1. + self.decay * self.iterations))

        t = self.iterations + 1
        # compensate for initial moment estimate bias
        # limit{t->inf} of lr_t = lr
        lr_t = lr*np.sqrt(1. - np.power(self.beta_2, t)) / (1. - np.power(self.beta_1, t))

        if not hasattr(self, 'ms'):
            self.ms = [tf.zeros(p.shape) for p in params]
            self.vs = [tf.zeros(p.shape) for p in params]

        for i, p, g, m, v in zip(range(len(params)), params, grads, self.ms, self.vs):
            m_t = (self.beta_1 * m) + (1. - self.beta_1) * g
            v_t = (self.beta_2 * v) + (1. - self.beta_2) * (g*g)

            # apply weight decay
            if self.weight_decay != 0.:
                p.assign_add(-self.weight_decay*lr*p)
            p.assign_add(lr_t * m_t / (tf.sqrt(v_t) + self.epsilon))
            self.ms[i] = m_t
            self.vs[i] = v_t
        self.iterations += 1

    def get_state(self):
        return [self.iterations, self.ms, self.vs]