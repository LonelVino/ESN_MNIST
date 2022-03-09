import numpy as np
from tqdm import tqdm
from sklearn.base import BaseEstimator
from utils import * 
from RLS import RLS

class ESN(BaseEstimator):

    def __init__(self, n_inputs, n_outputs, n_reservoir=200,
                 spectral_radius=0.95, sparsity=0, noise=0.001, ridge_noise=0.001,
                 wash_out = 100, learn_method="ridge", SGD_clf=None,
                 state_activation=np.tanh, leaky_rate=1.0, 
                 is_SLM=False, intensity=0.1,
                 out_activation=identity, inverse_out_activation=identity,
                 W_in_scaling=None, W_feedb_scaling=None,
                 input_shift=None, input_scaling=None, 
                 teacher_forcing=True, teacher_scaling=None, teacher_shift=None,
                 random_state=None, silent=True):
        """
        Args:
        [Initialization]
            n_inputs: nr of input dimensions
            n_outputs: nr of output dimensions
            n_reservoir: nr of reservoir neurons
            spectral_radius: spectral radius of the recurrent weight matrix
            sparsity: proportion of recurrent weights set to zero
            noise: noise added to each neuron when updating (regularization)
            ridge_noise: noise added in the ridge regression (regulatization)
            wash_out: duration/length of the washout (discarding transient from initial conditions)
        [Training]    
            learn_method: the learn method used to compute the output weights matrix 
                        options: ['pinv', 'ridge', 'SGD']
            SGD_clf: sklearn.linear_model.SGDclassifier, if learn_method=='SGD'
            state_activation: activation function used in updating reservoir states  
            leaky_rate: leaky rate of Leaky-Integrator ESN (LIESN), used to improve STM 
            is_SLM: if use SLM as the state activation
            out_activation: output activation function (applied to the readout)
            inverse_out_activation: inverse of the output activation function
        [Scaling & Shifting]
            W_in_scaling: scale of the input weights matrix range 
            W_feedb_scaling: scale of the feedback weights matrix range 
            input_shift: scalar or vector of length n_inputs to add to each
                        input dimension before feeding it to the network.
            input_scaling: scalar or vector of length n_inputs to multiply
                        with each input dimension before feeding it to the network.
            teacher_forcing: if True, feed the target back into output units
            teacher_scaling: factor applied to the target signal
            teacher_shift: additive term applied to the target signal
        [Others]
            random_state: positive integer seed, np.rand.RandomState object,
                          or None to use numpy's builting RandomState.
            silent: suppress messages if silent=True
            """
        # check for proper dimensionality of all arguments and write them down.
        self._estimator_type = 'classifier'
        self.n_inputs = n_inputs
        self.n_reservoir = n_reservoir
        self.n_outputs = n_outputs
        self.spectral_radius = spectral_radius
        self.sparsity = sparsity
        self.noise = noise
        self.ridge_noise = ridge_noise
        self.wash_out = wash_out
        self.learn_method = learn_method
        self.SGD_clf = SGD_clf
        self.state_activation = state_activation
        self.leaky_rate = leaky_rate
        self.is_SLM = is_SLM; self.intensity = intensity
        self.out_activation = out_activation
        self.inverse_out_activation = inverse_out_activation
        self.W_in_scaling = W_in_scaling
        self.W_feedb_scaling = W_feedb_scaling
        self.random_state = random_state
        self.silent = silent

        self.teacher_forcing = teacher_forcing
        self.teacher_scaling = teacher_scaling
        self.teacher_shift = teacher_shift
        self.input_shift = correct_dimensions(input_shift, n_inputs)
        self.input_scaling = correct_dimensions(input_scaling, n_inputs)

        # the given random_state might be either an actual RandomState object,
        # a seed or None (in which case we use numpy's builtin RandomState)
        if isinstance(random_state, np.random.RandomState):
            self.random_state_ = random_state
        elif random_state:
            try:
                self.random_state_ = np.random.RandomState(random_state)
            except TypeError as e:
                raise Exception("Invalid seed: " + str(e))
        else:
            self.random_state_ = np.random.mtrand._rand
        
        self.initweights()

    def initweights(self):
        W = self.random_state_.rand(self.n_reservoir, self.n_reservoir) - 0.5
        # delete the fraction of connections given by (self.sparsity):
        W[self.random_state_.rand(*W.shape) < self.sparsity] = 0
        # rescale them to reach the requested spectral radius:
        radius = np.max(np.abs(np.linalg.eigvals(W))) # compute the spectral radius of these weights
        self.W = W * (self.spectral_radius / radius)

        self.W_in = self.random_state_.rand(self.n_reservoir, self.n_inputs) * 2 - 1 
        if self.W_in_scaling is not None: self.W_in *= self.W_in_scaling 
        self.W_feedb = self.random_state_.rand(self.n_reservoir, self.n_outputs) * 2 - 1
        if self.W_feedb_scaling is not None: self.W_in *= self.W_in_scaling

    def _scale_inputs(self, inputs):
        """For each input dimension j: multiplies by the j'th entry in the
            input_scaling argument, then adds the j'th entry of the input_shift
            argument.
        """
        if self.input_scaling is not None: inputs = np.dot(inputs, np.diag(self.input_scaling))
        if self.input_shift is not None: inputs = inputs + self.input_shift
        return inputs

    def _scale_teacher(self, teacher):
        """Multiplies the teacher/target signal by the teacher_scaling argument,
            then adds the teacher_shift argument to it."""
        if self.teacher_scaling is not None: teacher = teacher * self.teacher_scaling
        if self.teacher_shift is not None: teacher = teacher + self.teacher_shift
        return teacher

    def _unscale_teacher(self, teacher_scaled):
        """Inverse operation of the _scale_teacher method."""
        if self.teacher_shift is not None: teacher_scaled = teacher_scaled - self.teacher_shift
        if self.teacher_scaling is not None: teacher_scaled = teacher_scaled / self.teacher_scaling
        return teacher_scaled


    def _SLM(self, input_pattern, state, intensity=1.0):
        '''An Activation function simulating the photonic reservoir
        Args:
            intensity (float)
        '''
        preactivation = quantize(np.dot(self.W, state) + np.dot(self.W_in, input_pattern), num_bins=2**8)
        return quantize(intensity*(np.sin(preactivation)**2), num_bins=2**10)
    
    def _update(self, state, input_pattern, output_pattern):
        """performs one update step.
            i.e., computes the next network state by applying the recurrent weights
            to the last state & and feeding in the current input and output patterns
        """
        if self.is_SLM:
            return (1-self.leaky_rate)*state \
                    + self._SLM(input_pattern, state, self.intensity) \
                    + self.noise * (self.random_state_.rand(self.n_reservoir) - 0.5)

        if self.teacher_forcing:
            preactivation = (np.dot(self.W, state) + np.dot(self.W_in, input_pattern)
                             + np.dot(self.W_feedb, output_pattern))
        else:
            preactivation = (np.dot(self.W, state) + np.dot(self.W_in, input_pattern))
        
        return (1-self.leaky_rate)*state + self.state_activation(preactivation) \
                + self.noise * (self.random_state_.rand(self.n_reservoir) - 0.5)  # regularization
    

    def fit(self, inputs, outputs):
        """Collect the network's reaction to training data, train readout weights.
        Args:
            inputs: array of dimensions (N_training_samples x n_inputs)
            outputs: array of dimension (N_training_samples x n_outputs)
            inspect: show a visualisation of the collected reservoir states
        Returns:
            the network's output on the training data, using the trained weights
        """
        # transform any vectors of shape (x,) into vectors of shape (x,1):
        if inputs.ndim < 2: inputs = np.reshape(inputs, (len(inputs), -1))
        if outputs.ndim < 2: outputs = np.reshape(outputs, (len(outputs), -1))
        # transform input and teacher signal:
        teachers_scaled = self._scale_teacher(outputs) if self.teacher_forcing else outputs
        inputs_scaled = self._scale_inputs(inputs) if self.input_shift is not None or self.input_scaling is not None else inputs
        
        ### Harvesting States
        if not self.silent: print("[INFO] Harvesting states...")
        states = np.zeros((inputs.shape[0], self.n_reservoir))
        for n in tqdm(range(1, inputs.shape[0]), disable=self.silent):
            states[n, :] = self._update(states[n - 1], inputs_scaled[n, :],teachers_scaled[n - 1, :])
        extended_states = np.hstack((states, inputs_scaled))  # include the raw inputs
        self.states = extended_states
        
        ### Training output matrix (readout layer)
        if not self.silent: print("[INFO] Training Readout Layer...")
        transient = min(int(inputs.shape[1] / 10), self.wash_out)  # Wash-out Length (disregard the first few states)
        train_states = extended_states[transient:, :]
        targets = self.inverse_out_activation(teachers_scaled[transient:, :])
        if self.learn_method == 'ridge':
            self.W_out = ridge(train_states, targets, self.ridge_noise)
        elif self.learn_method == 'pinv':
            self.W_out = pinv(train_states, targets)
        elif self.learn_method == 'SGD':
            self.SGD_clf.fit(train_states, targets)
            self.W_out = self.SGD_clf.coef_
        elif self.learn_method == 'RLS':
            RLS_ = RLS(num_vars=extended_states.shape[1], num_classes=self.n_outputs)
            RLS_.fit(train_states, targets)
            self.W_out = RLS_.w.T
        else:
            raise ValueError("Invalid Learning Method")

        # remember the last state for later:
        self.laststate = states[-1, :]
        self.lastinput = inputs[-1, :]
        self.lastoutput = teachers_scaled[-1, :]

        # Estimate training set by applying learned weights to the extended states
        pred_train = self._unscale_teacher(self.out_activation(
            np.dot(extended_states, self.W_out.T)))
        return pred_train

    def _predict(self, inputs, continuation=False):
        """Apply the learned weights to the network's reactions to new input.
        Args:
            inputs: array of dimensions (N_test_samples x n_inputs)
            continuation: if True, start the network from the last training state
        Returns:
            Array of output activations
        """
        inputs = np.reshape(inputs, (len(inputs), -1)) if inputs.ndim < 2 else inputs
        n_samples = inputs.shape[0]

        if continuation:
            laststate = self.laststate; lastinput = self.lastinput; lastoutput = self.lastoutput
        else:
            laststate = np.zeros(self.n_reservoir); lastinput = np.zeros(self.n_inputs); lastoutput = np.zeros(self.n_outputs)

        inputs = np.vstack([lastinput, self._scale_inputs(inputs)])
        states = np.vstack([laststate, np.zeros((n_samples, self.n_reservoir))])
        outputs = np.vstack([lastoutput, np.zeros((n_samples, self.n_outputs))])

        for n in tqdm(range(n_samples), disable=self.silent):
            states[n + 1, :] = self._update(states[n, :], inputs[n + 1, :], outputs[n, :])
            outputs[n + 1, :] = self.out_activation(np.dot(
                self.W_out, np.concatenate([states[n + 1, :], inputs[n + 1, :]])))

        return self._unscale_teacher(self.out_activation(outputs[1:]))

    def predict(self, inputs, continuation=True):
        pred_prob = self._predict(inputs=inputs, continuation=continuation) 
        return np.argmax(pred_prob, axis=1)
        
        
    def predict_proba(self, inputs, continuation=True):
        return self._predict(inputs=inputs, continuation=continuation)
