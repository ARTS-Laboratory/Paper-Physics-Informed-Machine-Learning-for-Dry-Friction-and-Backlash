import numpy as np
import tensorflow.keras as keras
import tensorflow as tf
import time

from lugre import LuGre, RestoreBestWeights
"""
Using numerical dataset and lugre_v7, training on only 
harmonic data.
"""
#%% loading and preprocessing data
"""
Data is loading from a 4-dimensional numpy array.
The axes are [Amplitude, frequency, time, feature]. The feature axis is [v, x, F].

The amplitude and frequencies in the experiments were 
"""
def train_numerical_model(epochs=400):
    print("loading and processing data...")
    amplitudes = [.02, .05, .1, .2]
    frequencies = [.1, .2, .5, 1, 2]
    batch_train_time = .4
    dt = 1/200
    batch_len = int(batch_train_time//dt)
    
    # change to version 2
    arr = np.load("./datasets/numerical_generated_data.npy")
    
    arr = arr.reshape((-1, arr.shape[2], 3))
    num_batches = arr.shape[0]
    
    x = arr[:,:,1]
    v = arr[:,:,0]
    F = arr[:,:,2]/1000 # output force
    D = np.full((x.shape[0], x.shape[1], 2), [1.0, 1.05])
    
    # reshape
    x = x[:,:x.shape[1]//batch_len*batch_len]
    v = v[:,:v.shape[1]//batch_len*batch_len]
    F = F[:,:F.shape[1]//batch_len*batch_len]
    D = D[:,:D.shape[1]//batch_len*batch_len]
    
    x = x.reshape(-1, batch_len, 1)
    v = v.reshape(-1, batch_len, 1)
    F = F.reshape(-1, batch_len, 1)
    D = D.reshape(-1, batch_len, 2)
    
    # scale inputs
    v_std = np.std(v)
    v = v/v_std
    y_init = F[:,0]
    
    # prepare validation dataset
    v_val = None
    F_val = None
    D_val = None
    
    filenames = ["DuzceMCE", "ImperialValleyMCE", "KocaeliMCE"]
    for file in filenames:
        data = np.load("./datasets/numerical validation/%s.npy"%file)
        v_test = data[:,0]
        x_test = data[:,1]
        D_test = np.full((v_test.shape[0], 2), [1.0, 1.05])
        F_test = data[:,2]/1000
        
        v_test = v_test[:v_test.size//batch_len*batch_len]
        x_test = x_test[:x_test.size//batch_len*batch_len]
        F_test = F_test[:F_test.size//batch_len*batch_len]
        D_test = D_test[:D_test.shape[0]//batch_len*batch_len]
        
        v_test = v_test.reshape(-1, batch_len, 1)
        x_test = x_test.reshape(-1, batch_len, 1)
        F_test = F_test.reshape(-1, batch_len, 1)
        D_test = D_test.reshape(-1, batch_len, 2)
        
        if(F_val is None):
            v_val = v_test
            F_val = F_test
            D_val = D_test
        else:
            v_val = np.append(v_val, v_test, axis=0)
            F_val = np.append(F_val, F_test, axis=0)
            D_val = np.append(D_val, D_test, axis=0)
    
    # scale validation dataset
    v_val /= v_std
    
    y_val_init = F_val[:,0]
    #%% build model
    print("building LuGre model...")
    alpha = 2
    v_s=0.01/v_std
    sigma_1=100*v_std/1000 # sigma_1 was set in the multiphysics simulation
    sigma_2=0 # zero viscosity assumption
    units=50
    
    
    F_cs_input = keras.Input(shape=[None, 2], name="F_cs_input")
    # X_input = keras.Input(shape=[None, 2], name='X')
    v_input = keras.Input(shape=[None, 1], name='v')
    y_init_input = keras.Input(shape=[1,], name='y_init')
    
    dense1 = keras.layers.Dense(units, use_bias=True, activation='relu')
    dense2 = keras.layers.Dense(units, use_bias=True, activation='relu')
    dense3 = keras.layers.Dense(1, use_bias=True, activation='relu')
    
    dense1.build(input_shape=[1,2])
    dense2.build(input_shape=[1,units])
    dense3.build(input_shape=[1,units])
    
    # states passes through unchanged.
    def sigma_call(inputs):
        y = dense1(inputs)
        y = dense2(y)
        y = dense3(y)
        return y
    
    sigma_call_state_size = 0
    sigma_model_weights = dense1.weights + dense2.weights + dense3.weights
    
    concat = keras.layers.Concatenate()([v_input, F_cs_input])
    
    lugre = LuGre(dt,
                  sigma_call,
                  sigma_call_state_size,
                  sigma_model_weights,
                  alpha=alpha,
                  v_s=v_s,
                  sigma_1=sigma_1,
                  sigma_2=sigma_2,
                  # sigma_1_regularizer=0.02,
                  stateful=False,
                  callv=3,
                  return_sequences=True,
                  name="lugre")(concat, initial_state=y_init_input)
    
    model = keras.Model(
        inputs = [v_input, F_cs_input, y_init_input],
        outputs = [lugre]
    )
    adam = keras.optimizers.Adam(
        learning_rate = 0.0001
    ) #clipnorm=1
    model.compile(
        loss="mse",
        optimizer=adam,
    )
    checkpoint_filepath = "./model_saves/combined_model"
    checkpoint = tf.keras.callbacks.ModelCheckpoint(
        filepath=checkpoint_filepath,
        save_weights_only=True,
        monitor='val_loss',
        mode='auto',
        save_best_only=True,
    )
    early_stopping = tf.keras.callbacks.EarlyStopping(
        monitor="val_loss",
        min_delta=0.000,
        patience=15,
        verbose=0,
        mode="auto",
        baseline=None,
        restore_best_weights=True,
    )
    restore_best_weights = RestoreBestWeights()
    #%% train LuGre model
    print("beginning training...")
    start_train_time = time.perf_counter()
    model.fit(
        [v, D, y_init], F,
        shuffle=False,
        epochs=epochs,
        validation_data = ([v_val, D_val, y_val_init], F_val),
        callbacks=[checkpoint],
    )
    stop_train_time = time.perf_counter()
    elapsed_time = stop_train_time - start_train_time
    print("training required %d minutes, %d seconds"%((int(elapsed_time/60)), int(elapsed_time%60)))
    has_nan = False
    for layer in model.layers:
        for weight in layer.weights:
            if(np.any(np.isnan(weight))):
                has_nan = True
                print("weight has nan")
                print(weight.name)
    print(has_nan)
    #%% save sigma model
    sigma_model = keras.models.Sequential([
        keras.layers.Dense(units, input_shape=[1,2], use_bias=True, activation='relu'),
        keras.layers.Dense(units, use_bias=True, activation='relu'),
        keras.layers.Dense(1, use_bias=True, activation='relu'),
    ])
    
    sigma_model.layers[0].set_weights(dense1.get_weights())
    sigma_model.layers[1].set_weights(dense2.get_weights())
    sigma_model.layers[2].set_weights(dense3.get_weights())
    
    sigma_model.save("./model_saves/numerical model")
#%%
if __name__ == '__main__':
    train_numerical_model()