import tensorflow as tf
import tensorflow.keras as keras
import tensorflow.keras.backend as backend
from tensorflow.python.platform import tf_logging as logging
from tensorflow.keras.layers import Layer
from tensorflow.keras.layers import RNN
import numpy as np
from tensorflow.math import negative

class LuGreCell(Layer):
    """
    __init__
    any parameter given will not be trained.
    
    dt timestep
    sigma_call call used for prediction prediction
    """
    def __init__(self, dt,
                 sigma_call,
                 sigma_call_state_size,
                 size=1,
                 sigma_1 = None, 
                 sigma_2 = None,
                 alpha = None,
                 v_s = None,
                 a=None,
                 callv=3,
                 sigma_1_regularizer=0,
                 sigma_2_regularizer=0,
                 **kwargs):
        self.callv = callv
        self.sigma_call = sigma_call
        self.dt = dt
        self.units = size
        
        if(callv == 0 or callv == 1):
            self.state_size = [tf.TensorShape([size]), tf.TensorShape([size]), sigma_call_state_size]
        if(callv == 2 or callv == 3 or callv == 4):
            self.state_size = [tf.TensorShape([size])] # for callv3
        
        # self.state_size = [size, size]
        super(LuGreCell, self).__init__(**kwargs)
        initializer = keras.initializers.RandomUniform(minval=0, maxval=.1)
        j = [None]*4
        k = [sigma_1, sigma_2, alpha, v_s]
        names = ["sigma_1", "sigma_2", "alpha", "v_s"]
        
        # initializing sigma_0_star, sigma_1, sigma_2, alpha, v_s
        for i in range(len(j)):
            val = k[i]
            name = names[i]
            if(val == None):
                weight = self.add_weight(
                    shape=(1, size),
                    name=name,
                    trainable=True,
                    initializer=initializer,
                )
            else:
                s = np.full((1, size), val, dtype=np.float32)
                weight = tf.Variable(
                    s,
                    name=name,
                    trainable=False,
                    dtype="float32",
                )
            j[i] = weight
        [self.sigma_1, self.sigma_2, self.alpha, self.v_s]\
            = [j_ for j_ in j]
        
        if(sigma_1_regularizer != 0 and sigma_1 == None):
            self.sigma_1 = self.add_weight(
                shape=(1, size),
                name=name,
                trainable=True,
                regularizer=keras.regularizers.L1(sigma_1_regularizer),
                initializer=initializer,
            )
        if(sigma_2_regularizer != 0 and sigma_2 == None):
            self.sigma_2 = self.add_weight(
                shape=(1, size),
                name=name,
                trainable=True,
                regularizer=keras.regularizers.L1(sigma_2_regularizer),
                initializer=initializer,
            )
        self.built = True
    
    def get_config(self):
        return {"dt": self.dt}
    
    """
    call with lstm state
    """
    def callv1(self, inputs, states):
        sz_0, F_0, lstm_states = states
        v, F_c, F_s = tf.split(inputs, 3, -1)
        
        lstm_in = tf.concat([v, sz_0], -1)
        
        sigma_0, lstm_states = self.sigma_call(lstm_in, lstm_states)
        
        sigma_1 = tf.abs(self.sigma_1); sigma_2 = tf.abs(self.sigma_2);
        v_s = self.v_s; alpha = self.alpha;
        sign = tf.sign(v)
        
        dif = F_s - F_c
        g = F_c + dif*tf.exp(negative(tf.pow(tf.abs(v/v_s), alpha)))
        k = g*sign
        
        sz = k + (sz_0 - k)*tf.exp(tf.scalar_mul(-self.dt, sigma_0*tf.abs(v)/g))
        zdot = v - sz*tf.abs(v)/g
        sz = tf.clip_by_value(sz, tf.negative(F_s), F_s) # protection
        F = sz + sigma_1*zdot + sigma_2*v
        return F, [sz, F, lstm_states]
    
    """
    call with MLP and return sigma_0
    """
    def callv2(self, inputs, states):
        sz_0 = states[0]
        v, F_c, F_s = tf.split(inputs, 3, -1)
        
        model_in = tf.concat([v, sz_0], -1)
        
        sigma_0 = self.sigma_call(model_in)
        
        sigma_1 = tf.abs(self.sigma_1); sigma_2 = tf.abs(self.sigma_2);
        v_s = self.v_s; alpha = self.alpha;
        sign = tf.sign(v)
        
        dif = F_s - F_c
        g = F_c + dif*tf.exp(negative(tf.pow(tf.abs(v/v_s), alpha)))
        k = g*sign
        
        sz = k + (sz_0 - k)*tf.exp(tf.scalar_mul(-self.dt, sigma_0*tf.abs(v)/g))
        zdot = v - sz*tf.abs(v)/g
        sz = tf.clip_by_value(sz, tf.negative(F_s), F_s) # protection
        F = sz + sigma_1*zdot + sigma_2*v
        return sigma_0, [sz]
    
    """
    call with MLP and return F
    """
    def callv3(self, inputs, states):
        sz_0 = states[0]
        v, F_c, F_s = tf.split(inputs, 3, -1)
        
        model_in = tf.concat([v, sz_0], -1)
        
        sigma_0 = self.sigma_call(model_in)
        
        sigma_1 = tf.abs(self.sigma_1); sigma_2 = tf.abs(self.sigma_2);
        v_s = self.v_s; alpha = self.alpha;
        sign = tf.sign(v)
        
        dif = F_s - F_c
        g = F_c + dif*tf.exp(negative(tf.pow(tf.abs(v/v_s), alpha)))
        k = g*sign
        
        sz = k + (sz_0 - k)*tf.exp(tf.scalar_mul(-self.dt, sigma_0*tf.abs(v)/g))
        zdot = v - sz*tf.abs(v)/g
        sz = tf.clip_by_value(sz, tf.negative(F_s), F_s) # protection
        F = sz + sigma_1*zdot + sigma_2*v
        return F, [sz]
    
    """
    call with MLP and return state
    """
    def callv4(self, inputs, states):
        sz_0 = states[0]
        v, F_c, F_s = tf.split(inputs, 3, -1)
        
        model_in = tf.concat([v, sz_0], -1)
        
        sigma_0 = self.sigma_call(model_in)
        
        sigma_1 = tf.abs(self.sigma_1); sigma_2 = tf.abs(self.sigma_2);
        v_s = self.v_s; alpha = self.alpha;
        sign = tf.sign(v)
        
        dif = F_s - F_c
        g = F_c + dif*tf.exp(negative(tf.pow(tf.abs(v/v_s), alpha)))
        k = g*sign
        
        sz = k + (sz_0 - k)*tf.exp(tf.scalar_mul(-self.dt, sigma_0*tf.abs(v)/g))
        zdot = v - sz*tf.abs(v)/g
        sz = tf.clip_by_value(sz, tf.negative(F_s), F_s) # protection
        F = sz + sigma_1*zdot + sigma_2*v
        return sz, [sz]
    
    def call(self, inputs, states):
        if(self.callv == 1):
            return self.callv1(inputs, states)
        if(self.callv == 2):    
            return self.callv2(inputs, states)
        if(self.callv == 3):
            return self.callv3(inputs, states)
        if(self.callv == 4):
            return self.callv4(inputs, states)
    
    def get_sigma_1(self):
        return tf.abs(self.sigma_1).numpy()
    
    def get_sigma_2(self):
        return tf.abs(self.sigma_2).numpy()
    
    def get_alpha(self):
        return self.alpha.numpy()
    
    def get_v_s(self):
        return tf.abs(self.v_s).numpy()
    
    def get_a(self):
        return tf.abs(self.a).numpy()
    
    
""" RNN wrapper for LuGreCell """
class LuGre(RNN):
    
    def __init__(self, dt,
                 sigma_call,
                 sigma_call_state_size,
                 sigma_model_weights,
                 size=1,
                 return_sequences=False,
                 stateful=False, 
                 return_state=False, **kwargs):
        self.cell = LuGreCell(dt,
                              sigma_call,
                              sigma_call_state_size,
                              size=size, **kwargs)
        self.dt = dt
        self.sigma_1 = self.cell.sigma_1
        self.sigma_2 = self.cell.sigma_2
        self.alpha = self.cell.alpha
        self.v_s = self.cell.v_s
        super(LuGre, self).__init__(
            self.cell,
            return_sequences=return_sequences,
            return_state=return_state,
            stateful=stateful,
        )
        for variable in sigma_model_weights:
            self._trainable_weights.append(variable)
    
    def get_config(self):
        return {"dt": self.dt}
    
    # def call(self, inputs, mask=None, training=None, constants=None):
    #     initial_state = [np.zeros((1,1), dtype=np.float32), np.zeros((1, 1), dtype=np.float32)]
    #     super(LuGre, self).call(inputs, mask=mask, training=training, initial_state=initial_state, constants=constants)
    
    def get_F_s(self):
        return self.cell.get_F_s()
    
    def get_F_c(self):
        return self.cell.get_F_c()
    
    def get_sigma_0(self):
        return self.cell.get_sigma_0()
    
    def get_sigma_1(self):
        return self.cell.get_sigma_1()
    
    def get_sigma_2(self):
        return self.cell.get_sigma_2()
    
    def get_alpha(self):
        return self.cell.get_alpha()
    
    def get_v_s(self):
        return self.cell.get_v_s()
    
    def get_a(self):
        return self.cell.get_a()

"""
A generator class for creating training data for the combined model.
"""
class TrainingGenerator(keras.utils.Sequence):
    
    def __init__(self, *args, train_len=400):
        self.args = args
        self.train_len = train_len
        self.length = args[0].shape[1]//train_len
    
    def __len__(self):
        return self.length
    
    def __getitem__(self, index):
        # F_batch = self.F[:,index*self.train_len:(index+1)*self.train_len,:]
        # lstm_batch = self.lstm_input[:,index*self.train_len:(index+1)*self.train_len,:]
        # return lstm_batch, F_batch
        rtrn = [arg[:,index*self.train_len:(index+1)*self.train_len,:] for arg in self.args]
        
        return rtrn[:-1], rtrn[-1] 

class LSTMTrainingGenerator(keras.utils.Sequence):
    
    def __init__(self, lstm_input, F, train_len=400):
        self.train_len = train_len
        self.length = lstm_input.shape[1]//train_len
        self.lstm_input = lstm_input
        self.F = F
    
    def __len__(self):
        return self.length
    
    def __getitem__(self, index):
        F_batch = self.F[:,index*self.train_len:(index+1)*self.train_len,:]
        lstm_batch = self.lstm_input[:,index*self.train_len:(index+1)*self.train_len,:]
        
        return lstm_batch, F_batch

"""
Regularizer which is a ratio of the powers of the LSTM and LuGre outputs.
Add to model with model.add_loss
"""
class PowerRatioRegularizer:
    def __init__(self, coef=0.1):
        coef = 0 if coef is None else coef
        self.coef = backend.cast_to_floatx(coef)
    
    # x is tuple of [lugre_output, lstm_output]
    def __call__(self, x):
        lugre = x[0]
        lstm = x[1]
        return self.coef*tf.reduce_sum(tf.square(lstm))/tf.reduce_sum(tf.square(lugre))
    
    def get_config(self):
        return {'coef': self.coef}

"""
keras Callback for protecting against gradient explosion producing nans.
It doesn't work that well, but I eventually got gradient explosion to stop
happening.
"""
class NanProtection(keras.callbacks.Callback):
    
    def __init__(self, model):
        self.model = model
        super(NanProtection).__init__()
        self.weights = None
        # self.stateful_indices = []
        # self.states = []
        # for i in range(len(self.model.layers)):
        #     if(self.model.layers[i].stateful):
        #         self.stateful_indices += [i]
        #         self.states += [self.model.layers[i].states]
        
    
    # def on_train_begin(self, logs=None):
    #     self.weights = self.model.get_weights()
    
    def on_train_batch_end(self, batch, logs=None):
        if(self.weights is not None and any([np.isnan(w_).any() or np.isinf(w_).any() for w_ in self.weights])):
           print("critical error reached")
        if(any([np.isnan(w_).any() or np.isinf(w_).any() for w_ in self.model.get_weights()])):
            self.model.set_weights(self.weights)
            # for i in range(len(self.stateful_indices)):
            #     self.model.layers[self.stateful_indices[i]].states = self.states[i]
            for layer in self.model.layers:
                if(layer.stateful):
                    layer.reset_states()
            print("Training encountered NaN values.")
            self.model.stop_training = True
        else:
            self.weights = self.model.get_weights()
            # for i in range(len(self.stateful_indices)):
            #     self.model.layers[self.stateful_indices[i]].states = self.states[i]
"""
Reset all stateful layers in a model
"""
class StateResetter(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs={}):
        for layer in self.model.layers:
            if(layer.stateful):
                layer.reset_states()


"""
Basically EarlyStopping's restore_best_weights = True. Restore best
weights at the end of training.
"""
class RestoreBestWeights(keras.callbacks.Callback):
    
    def __init__(self, monitor="val_loss", mode="auto"):
        super().__init__()
        
        self.monitor = monitor
        self.best_weights = None
        
        if mode == "min":
            self.monitor_op = np.less
        elif mode == "max":
            self.monitor_op = np.greater
        else:
            if (
                self.monitor.endswith("acc")
                or self.monitor.endswith("accuracy")
                or self.monitor.endswith("auc")
            ):
                self.monitor_op = np.greater
            else:
                self.monitor_op = np.less
        
    
    def _is_improvement(self, monitor_value, reference_value):
        return self.monitor_op(monitor_value, reference_value)
    
    def get_monitor_value(self, logs):
        logs = logs or {}
        monitor_value = logs.get(self.monitor)
        if monitor_value is None:
            logging.warning(
                "Restore best weights conditioned on metric `%s` "
                "which is not available. Available metrics are: %s",
                self.monitor,
                ",".join(list(logs.keys())),
            )
        return monitor_value
    
    def on_train_begin(self, logs=None):
        self.best = np.Inf if self.monitor_op == np.less else -np.Inf
        self.best_weights = None
        self.best_epoch = 0
    
    def on_epoch_end(self, epoch, logs=None):
        current = self.get_monitor_value(logs)
        
        
        if(self.best_weights is None):
            self.best_weights = self.model.get_weights()
        
        if self._is_improvement(current, self.best):
            self.best = current
            self.best_epoch = epoch
            self.best_weights = self.model.get_weights()
        
    def on_train_end(self, logs=None):
        self.model.set_weights(self.best_weights)