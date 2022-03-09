import numpy as np
import math
from tqdm import tqdm


class RLS:
    def __init__(self, num_vars, num_classes, lam=0.98, delta=1):
        '''
        num_vars: numebr of variables, here the number of neurons
        lam: forgetting factor
        delta: control the initial states
        '''
        self.num_vars = num_vars
        self.num_classes = num_classes
        self.A = delta*np.matrix(np.identity(self.num_vars))   # -> n_reservoir x n_reservoir
        self.w = np.matrix(np.zeros((self.num_vars, self.num_classes)))   # -> n_reservoir x num_classes

        # Variables needed for add_obs
        self.lam_inv = lam**(-1)
        self.sqrt_lam_inv = math.sqrt(self.lam_inv)
        self.a_priori_error = 0  # a priori error
        self.num_obs = 0  # count of number of observations added
    
    def _add_obs(self, x, t):
        ''' Add the observation(state) x with label t
        x: observation & state  --> n_reservoir x 1
        t: label vector  --> n_classes x 1
        '''
        z = self.lam_inv * self.A.dot(x)  # -> n_reservoir x 1
        alpha = float((1 + x.T.dot(z))**(-1))  # scalar
        self.a_priori_error = t - self.w.T.dot(x) # n_classes x 1
        pre_cal = alpha* np.dot(x.T, (self.w + z.dot(t.T)))
        self.w = self.w + np.dot((t - pre_cal.T), z.T).T  # -> n_reservoir x num_classes
        self.A -= alpha * z.dot(z.T)  # n_reservoir x n_reservoir
        self.num_obs += 1 
        
    def fit(self, X, Y):
        ''' Fit a model with train_data(states) and train_label
        X: states -> n_samples x n_reservoir
        Y: labels (onehot encoding) -> n_smaples x n_classes
        '''
        for idx, state in tqdm(enumerate(X)):
            state = np.transpose(np.matrix(state))
            label = np.transpose(np.matrix(Y[idx]))
            self._add_obs(state, label)
            
    def predict(self, X):
        '''Predict the value of observation X
        X: observation (e.x. the states of testset) -> n_samples x n_reservoir
        '''
        return float(np.dot(X, self.w))