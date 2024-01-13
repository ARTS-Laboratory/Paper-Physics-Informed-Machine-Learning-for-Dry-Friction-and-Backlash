import numpy as np
import tensorflow.keras as keras
import tensorflow as tf
import json
import time

from lugre import LuGre, RestoreBestWeights

"""
Experimental dataset, looping with different excitations excluded
"""
#%%
def train_experimental_model_loop(epochs=800):
    #%% loading and preprocessing data
    training_files_harmonic = ['/harmonic/test%d.npy'%i for i in range(13)]
    training_earthquakes = ['DuzceDBE',
                            'DuzceMCE',
                            'ImperialValleyDBE',
                            'ImperialValleyMCE',
                            'KocaeliDBE',
                            'KocaeliMCE']
    """ Lugre g function, works on numpy arrays """
    def g(F_c, F_s, v_s, v):
        return F_c + (F_s - F_c)*np.exp(-(v/v_s)**2)
    for earthquake_index in range(6):
        #%%
        """
        Data is loading from a 4-dimensional numpy array.
        The axes are [Amplitude, frequency, time, feature]. The feature axis is [v, x, F].
        
        The amplitude and frequencies in the experiments were
        """
        print("loading and processing data...")
        training_files_earthquake_excluded = training_earthquakes.copy()
        val_earthquake = training_files_earthquake_excluded.pop(earthquake_index)
        training_files_earthquake_excluded = ['/validation/%s.npy'%i for i in \
                                     training_files_earthquake_excluded]
        
        training_files = training_files_harmonic + training_files_earthquake_excluded
        
       
        dt = 1/1024
        batch_train_time = .4
        batch_len = int(batch_train_time//dt)
        
        # these values were found in find_static_parameters.py and would have to be
        # changed if the dataset changes
        F_sneg = 20392.719197631664
        F_spos = 14281.704636699957
        F_cneg = 19763.37223071306
        F_cpos = 14261.88204331606
        
        F_batched = None
        v_batched = None
        D_batched = None
        
        for i, file in enumerate(training_files):
            t_test, F_test, x_test, v_test = np.load('./datasets/dataset-2/' + file).T
            
            F_test = F_test.reshape(-1, 1)
            v_test = v_test.reshape(-1, 1)
            
            F_c_test = np.full(F_test.shape, F_cneg) * (F_test<0) + np.full(F_test.shape, F_cpos) * (F_test>=0)
            F_s_test = np.full(F_test.shape, F_sneg) * (F_test<0) + np.full(F_test.shape, F_spos) * (F_test>=0)
            
            D_test = np.append(F_c_test, F_s_test, axis=-1)
            
            # reshape to be added to batched data
            F_test = F_test[:F_test.size//batch_len*batch_len]
            v_test = v_test[:v_test.size//batch_len*batch_len]
            D_test = D_test[:D_test.shape[0]//batch_len*batch_len]
            
            F_test = F_test.reshape(-1, batch_len, 1)
            v_test = v_test.reshape(-1, batch_len, 1)
            D_test = D_test.reshape(-1, batch_len, 2)
            
            if(F_batched is None):
                F_batched = F_test
                v_batched = v_test
                D_batched = D_test
            else:
                F_batched = np.append(F_batched, F_test, axis=0)
                v_batched = np.append(v_batched, v_test, axis=0)
                D_batched = np.append(D_batched, D_test, axis=0)
        
        F_std = np.std(F_batched)
        v_std = np.std(v_batched)
        sigma_1=24845.007005532887*v_std/F_std # small sigma_1
        sigma_2=0.06661985*v_std/F_std # low viscosity assumption
        v_s=0.01
        
        # normalize data
        F_batched /= F_std
        D_batched /= F_std
        v_batched /= v_std
        
        # fill dictionary with data scalers and save to .json
        scales = dict()
        
        scales['F_std'] = F_std
        scales['v_std'] = v_std
        scales['F_sneg'] = F_sneg
        scales['F_spos'] = F_spos
        scales['F_cneg'] = F_cneg
        scales['F_cpos'] = F_cpos
        
        F = F_batched
        v = v_batched
        D = D_batched
        y_init = (F[:,0] - (sigma_1 + sigma_2)*v[:,0])/(1 - sigma_1*np.abs(v[:,0])/g(D[:,0,0:1], D[:,0,1:2], v_s, v[:,0]))
        
        # here's only one validation test
        val_names = \
            [val_earthquake]
        
        F_val = None
        v_val = None
        D_val = None
        for i, name in enumerate(val_names):
            t_test, F_test, x_test, v_test = np.load("./datasets/dataset-2/validation/%s.npy"%name).T
            
            F_test = F_test.reshape(-1, 1)
            v_test = v_test.reshape(-1, 1)
            
            
            F_c_test = np.full(F_test.shape, F_cneg) * (F_test<0) + np.full(F_test.shape, F_cpos) * (F_test>=0)
            F_s_test = np.full(F_test.shape, F_sneg) * (F_test<0) + np.full(F_test.shape, F_spos) * (F_test>=0)
            
            D_test = np.append(F_c_test, F_s_test, axis=-1)
            
            # reshape to be added to batched data
            F_test = F_test[:F_test.size//batch_len*batch_len]
            v_test = v_test[:v_test.size//batch_len*batch_len]
            D_test = D_test[:D_test.shape[0]//batch_len*batch_len]
            
            F_test = F_test.reshape(-1, batch_len, 1)
            v_test = v_test.reshape(-1, batch_len, 1)
            D_test = D_test.reshape(-1, batch_len, 2)
            
            if(F_val is None):
                F_val = F_test
                v_val = v_test
                D_val = D_test
            else:
                F_val = np.append(F_val, F_test, axis=0)
                v_val = np.append(v_val, v_test, axis=0)
                D_val = np.append(D_val, D_test, axis=0)
        
        # normalize validation data
        F_val /= F_std
        D_val /= F_std
        v_val /= v_std
        # correct initialization values
        y_val_init = (F_val[:,0] - (sigma_1 + sigma_2)*v_val[:,0])/(1 - sigma_1*np.abs(v_val[:,0])/g(D_val[:,0,0:1], D_val[:,0,1:2], v_s, v_val[:,0]))
        #%%
        print("building LuGre model...")
        alpha = 2
        v_s=0.01
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
                      sigma_1=None, # learning sigma_1
                      sigma_2=None, # learning sigma_2
                      # sigma_1_regularizer=0.02,
                      callv=3,
                      stateful=False,
                      return_sequences=True,
                      name="lugre")(concat, initial_state=[y_init_input])
        
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
            min_delta=0,
            patience=15,
            verbose=0,
            baseline=None,
            restore_best_weights=True,
            # start_from_epoch=100,
        )
        # nan_protect = NanProtection(checkpoint_filepath)
        restore_best_weights = RestoreBestWeights()
        #%% load model from save (to continue previous training)
        # s_model = keras.models.load_model("./model_saves/experimental sigma model")
        
        # dense1.set_weights(s_model.layers[0].get_weights())
        # dense2.set_weights(s_model.layers[1].get_weights())
        # dense3.set_weights(s_model.layers[2].get_weights())
        #%% train model
        print("beginning training...")
        start_train_time = time.perf_counter()
        model.fit(
            [v, D, y_init], F,
            shuffle=True,
            epochs=epochs,
            validation_data = ([v_val, D_val, y_val_init], F_val),
            callbacks=[checkpoint, restore_best_weights],
        )
        stop_train_time = time.perf_counter()
        elapsed_time = stop_train_time - start_train_time
        print("training required %d minutes, %d seconds"%((int(elapsed_time/60)), int(elapsed_time%60)))
        lugre_sigmas = np.array([model.layers[-1].get_sigma_1()[0,0], model.layers[-1].get_sigma_2()[0,0]])
        np.save("./model_saves/sigma_saves.npy", lugre_sigmas)
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
        
        sigma_model.save("./model_saves/loop experimental models/%s model"%val_earthquake)
        
        # add sigma_1 and sigma_2 to scales and save to a file
        
        sigma_1 = model.layers[-1].get_sigma_1()[0,0]*F_std/v_std
        sigma_2 = model.layers[-1].get_sigma_2()[0,0]*F_std/v_std
        scales['sigma_1'] = sigma_1
        scales['sigma_2'] = sigma_2
        with open('./model_saves/scaling/exclude %s.json'%val_earthquake, 'w+') as f:
            json.dump(scales, f)
#%%
if __name__ == '__main__':
    train_experimental_model_loop()