import numpy as np
import tensorflow.keras as keras
import json

from lugre import LuGre
"""
for experimental dataset, calculate the results in a loop
for multiphysics dataset: don't train on any earthquakes
"""

""" Lugre g function, works on numpy arrays """
def g(F_c, F_s, v_s, v):
    return F_c + (F_s - F_c)*np.exp(-(v/v_s)**2)
#%% numerical tests
def run_numerical():
    print("loading and processing data...")
    amplitudes = [.02, .05, .1, .2]
    frequencies = [.1, .2, .5, 1, 2]
    batch_train_time = .2
    
    arr = np.load("./datasets/numerical_generated_data.npy")
    
    arr = arr[:,:,:,:] # exclude no data
    
    dt = 1/200
    t = np.array([dt*i for i in range(arr.shape[2])])
    
    arr = arr.reshape((-1, arr.shape[2], 3))
    num_batches = arr.shape[0]
    
    x = arr[:,:,1]
    v = arr[:,:,0]
    F = arr[:,:,2:]/1000 # output force
    D = np.full((x.shape[0], x.shape[1], 2), [1.0, 1.05])
    
    # scale inputs (this number was found in train_model_v26.py)
    v_std = 0.7603725298200057
    
    v = (v)/v_std
    
    x = np.expand_dims(x, -1)
    v = np.expand_dims(v, -1)
    
    # validation tests
    val_v = []
    val_D = []
    val_F = []
    filenames = ["DuzceDBE", "DuzceMCE", "ImperialValleyDBE", "ImperialValleyMCE",
                 "KocaeliDBE", "KocaeliMCE"]
    for file in filenames:
        data = np.load("./datasets/numerical validation/%s.npy"%file)
        v_test = data[:,0]
        v_test = (v_test)/v_std
        D_test = np.full((1, v_test.shape[0], 2), [1.0, 1.05])
        F_test = data[:,2:]/1000
        v_test = np.expand_dims(v_test, (0,-1))
        
        val_v.append(v_test)
        val_D.append(D_test)
        val_F.append(F_test)
    restricted_indices = [(900, 2400), (900, 2400), (600, 1500), (600, 1500), (1200, 2800), (1200, 2800)]
    #%%
    alpha = 2
    v_s=0.01/v_std
    sigma_1=100*v_std/1000 # small sigma_1
    sigma_2=0 # zero viscosity assumption
    units=50
    
    
    F_cs_input = keras.Input(batch_input_shape=[1, None, 2], name="F_cs_input")
    # X_input = keras.Input(shape=[None, 2], name='X')
    v_input = keras.Input(batch_input_shape=[1, None, 1], name='v')
    
    
    dense1 = keras.layers.Dense(units, use_bias=True, activation='relu')
    dense2 = keras.layers.Dense(units, use_bias=True, activation='relu')
    dense3 = keras.layers.Dense(1, use_bias=True, activation='relu')
    
    dense1.build(input_shape=[1,2])
    dense2.build(input_shape=[1,units])
    dense3.build(input_shape=[1,units])
    
    # states passes through unchanged.
    def sigma_call(inputs, states):
        y = dense1(inputs)
        y = dense2(y)
        y = dense3(y)
        return y, states
    
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
                  stateful=True, # not stateful
                  return_sequences=True,
                  callv=1,
                  name="lugre")(concat)
    
    model = keras.Model(
        inputs = [v_input, F_cs_input],
        outputs = [lugre]
    )
    adam = keras.optimizers.Adam(
        learning_rate = 0.0001
    ) #clipnorm=1
    model.compile(
        loss="mse",
        optimizer=adam,
    )
    #%% set weights
    s_model = keras.models.load_model("./model_saves/numerical model")
    
    dense1.set_weights(s_model.layers[0].get_weights())
    dense2.set_weights(s_model.layers[1].get_weights())
    dense3.set_weights(s_model.layers[2].get_weights())
    #%% run tests
    # harmonic
    F_pred = None
    for v_, D_ in zip(v, D):
        v_ = v_.reshape(1, -1, 1)
        D_ = D_.reshape(1, -1, 2)
        F_pred_ = model.predict([v_, D_])
        if(F_pred is None):
            F_pred = F_pred_
        else:
            F_pred = np.append(F_pred, F_pred_, axis=0)
    F_pred = F_pred.squeeze()
    np.save("./model predictions/numerical/predictiondata.npy", F_pred*1000)
    # validation
    for i in range(6):
        for layer in model.layers:
            if(layer.stateful):
                layer.reset_states()
        
        file = filenames[i]
        v_test = val_v[i]
        D_test = val_D[i]
        F_test = val_F[i]
        
        r = restricted_indices[i]
        F_test = F_test[r[0]:r[1]]
        D_test = D_test[:,r[0]:r[1],:]
        v_test = v_test[:,r[0]:r[1],:]
        
        y_init = np.array([F_test[0,0]])
        y_init = y_init.reshape(1,1)
        model.layers[-1].states[0].assign(y_init)
        
        F_pred = model.predict([v_test, D_test]).squeeze()
        np.save("./model predictions/numerical/validation/%s.npy"%file, F_pred*1000)
#%% experimental tests
def run_experimental():
    training_earthquakes = ['DuzceDBE',
                            'DuzceMCE',
                            'ImperialValleyDBE',
                            'ImperialValleyMCE',
                            'KocaeliDBE',
                            'KocaeliMCE']
    for earthquake_index in range(6):
        val_earthquake = training_earthquakes[earthquake_index]
        scales = json.load(open('./model_saves/scaling/exclude %s.json'%val_earthquake))
        
        F_std = scales['F_std']
        v_std = scales['v_std']
        F_sneg = scales['F_sneg']
        F_spos = scales['F_spos']
        F_cneg = scales['F_cneg']
        F_cpos = scales['F_cpos']
        sigma_1 = scales['sigma_1']
        sigma_2 = scales['sigma_2']
        
        # scale the static and kinetic friction values
        # F_sneg /= F_std
        # F_spos /= F_std
        # F_cneg /= F_std
        # F_cpos /= F_std
        
        dt = 1/1024
        
        restricted_indices = [(4608, 12288), (4608, 12288), (3072, 7680), (3072, 7680), (6144, 14336), (6144, 14336)]
        #%% 
        alpha = 2
        v_s=0.01
        units=50
        
        F_cs_input = keras.Input(batch_input_shape=[1, None, 2], name="F_cs_input")
        # X_input = keras.Input(shape=[None, 2], name='X')
        v_input = keras.Input(batch_input_shape=[1,None, 1], name='v')
        
        
        dense11 = keras.layers.Dense(units, use_bias=True, activation='relu')
        dense12 = keras.layers.Dense(units, use_bias=True, activation='relu')
        dense13 = keras.layers.Dense(1, use_bias=True, activation='relu')
        
        dense11.build(input_shape=[1,2])
        dense12.build(input_shape=[1,units])
        dense13.build(input_shape=[1,units])
        
        def sigma_call(inputs):
            y = dense11(inputs)
            y = dense12(y)
            y = dense13(y)
            return y
        
        sigma_call_state_size = 0
        sigma_model_weights = dense11.weights + dense12.weights + dense13.weights
        
        concat = keras.layers.Concatenate()([v_input, F_cs_input])
        
        lugre = LuGre(dt,
                      sigma_call,
                      sigma_call_state_size,
                      sigma_model_weights,
                      alpha=alpha,
                      v_s=v_s,
                      sigma_1=sigma_1*v_std/F_std,
                      sigma_2=sigma_2*v_std/F_std,
                      callv=3,
                      # sigma_1_regularizer=0.02,
                      stateful=True,
                      return_sequences=True,
                      name="lugre")(concat)
        
        model2 = keras.Model(
            inputs = [v_input, F_cs_input],
            outputs = [lugre]
        )
        
        model = model2
        #%% set weights
        s_model = keras.models.load_model("./model_saves/loop experimental models/%s model"%val_earthquake)
        
        dense11.set_weights(s_model.layers[0].get_weights())
        dense12.set_weights(s_model.layers[1].get_weights())
        dense13.set_weights(s_model.layers[2].get_weights())
        #%% harmonic tests
        for i in range(13):
            name = "test" + str(i)
            F_test = np.load("./datasets/dataset-2/F/%s.npy"%name).reshape(-1, 1)
            v_test = np.load("./datasets/dataset-2/v/%s.npy"%name).reshape(-1, 1)
            x_test = np.load("./datasets/dataset-2/x/%s.npy"%name).reshape(-1, 1)
            
            F_test = F_test.reshape(-1, 1)
            v_test = v_test.reshape(1, -1, 1)
            
            F_c_test = np.full(F_test.shape, F_cneg) * (F_test<0) + np.full(F_test.shape, F_cpos) * (F_test>=0)
            F_s_test = np.full(F_test.shape, F_sneg) * (F_test<0) + np.full(F_test.shape, F_spos) * (F_test>=0)
            
            D_test = np.append(F_c_test, F_s_test, axis=-1)
            
            D_test = np.expand_dims(D_test, 0)
            
            # set initial state using rearranged LuGre equation
            F0 = F_test[0,0]
            v0 = v_test[0,0,0]
            F_c0 = D_test[0,0,0]
            F_s0 = D_test[0,0,1]
            y_init = (F0-(sigma_1+sigma_2)*v0)/(1-sigma_1*np.abs(v0/g(F_c0, F_s0,v_s,v0)))
            y_init = np.array([[y_init]])
            # scale inputs
            v_test = v_test/v_std
            D_test = D_test/F_std
            y_init = y_init/F_std
            model.layers[-1].states[0].assign(y_init)
            
            F_pred = model.predict([v_test, D_test])
            F_pred = (F_pred * F_std).reshape(-1, 1)
            
            v_test = (v_test * v_std).reshape(-1, 1)
            
            np.save("./model predictions/loop experimental/%s/harmonic/%s.npy"%(val_earthquake, name), F_pred)
            
            # plot
            # t = np.array([dt*j for j in range(v_test.size)])
            
            # fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(6, 2.5), sharey=True)
            # ax1.set_ylabel("force (1000 N)")
            # ax1.plot(t, F_test/1000, linewidth=1.0)
            # ax1.plot(t, F_pred/1000, linewidth=1.0)
            # ax1.set_xlabel("time (s)")
            # ax2.plot(x_test, F_test/1000, linewidth=1.0)
            # ax2.plot(x_test, F_pred/1000, linewidth=1.0)
            # ax2.set_xlabel("displacement (m)")
            # ax3.plot(v_test, F_test/1000, linewidth=1.0)
            # ax3.plot(v_test, F_pred/1000, linewidth=1.0)
            # ax3.set_xlabel("velocity (m/s)")
            # fig.tight_layout()
            # fig.savefig("./plots/dataset-2/harmonic/%s.png"%name, dpi=500)
            # plt.close()
        #%% predictions for earthquake validation
        for layer in model.layers:
            if(layer.stateful):
                layer.reset_states()
        
        val_names = \
            ['DuzceDBE',
             'DuzceMCE',
             'ImperialValleyDBE',
             'ImperialValleyMCE',
             'KocaeliDBE',
             'KocaeliMCE']
        
        for i, name in enumerate(val_names):
            t_test, F_test, x_test, v_test = np.load("./datasets/dataset-2/validation/%s.npy"%name).T
            
            F_test = F_test.reshape(-1, 1)
            x_test = x_test.reshape(-1, 1)
            v_test = v_test.reshape(-1, 1)
            
            
            F_test = F_test.reshape(-1, 1)
            v_test = v_test.reshape(1, -1, 1)
            
            F_c_test = np.full(F_test.shape, F_cneg) * (F_test<0) + np.full(F_test.shape, F_cpos) * (F_test>=0)
            F_s_test = np.full(F_test.shape, F_sneg) * (F_test<0) + np.full(F_test.shape, F_spos) * (F_test>=0)
            
            D_test = np.append(F_c_test, F_s_test, axis=-1)
            
            D_test = np.expand_dims(D_test, 0)
            
            # set initial state using rearranged LuGre equation
            F0 = F_test[0,0]
            v0 = v_test[0,0,0]
            F_c0 = D_test[0,0,0]
            F_s0 = D_test[0,0,1]
            
            
            r = restricted_indices[i]
            F_test = F_test[r[0]:r[1]]
            D_test = D_test[:,r[0]:r[1],:]
            v_test = v_test[:,r[0]:r[1],:]
            
            
            # set initial state using rearranged LuGre equation
            F0 = F_test[0,0]
            v0 = v_test[0,0,0]
            F_c0 = D_test[0,0,0]
            F_s0 = D_test[0,0,1]
            y_init = (F0-(sigma_1+sigma_2)*v0)/(1-sigma_1*np.abs(v0/g(F_c0, F_s0,v_s,v0)))
            y_init = np.array([[y_init]])
            
            # rescale inputs
            v_test = v_test/v_std
            D_test = D_test/F_std
            y_init = y_init/F_std
            
            
            model.layers[-1].states[0].assign(y_init)
            
            F_pred = model.predict([v_test, D_test])
            F_pred = (F_pred * F_std).reshape(-1, 1)
            
            
            np.save("./model predictions/loop experimental/%s/validation/%s.npy"%(val_earthquake, name), F_pred)

if __name__ == '__main__':
    run_numerical()
    run_experimental()