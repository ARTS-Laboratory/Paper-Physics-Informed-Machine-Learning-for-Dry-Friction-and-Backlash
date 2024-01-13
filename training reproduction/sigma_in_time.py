import math
import numpy as np
import tensorflow.keras as keras
import matplotlib.pyplot as plt
import json

from lugre import LuGre
"""
Creating the sigma_0 in time plots for the journal paper
"""
#%%
plt.rcParams.update({'image.cmap': 'viridis'})
cc = plt.rcParams['axes.prop_cycle'].by_key()['color']
plt.rcParams.update({'font.serif': ['Times New Roman', 'Times', 'DejaVu Serif',
                                    'Bitstream Vera Serif', 'Computer Modern Roman', 'New Century Schoolbook',
                                    'Century Schoolbook L',  'Utopia', 'ITC Bookman', 'Bookman',
                                    'Nimbus Roman No9 L', 'Palatino', 'Charter', 'serif']})
plt.rcParams.update({'font.family': 'serif'})
plt.rcParams.update({'font.size': 10})
plt.rcParams.update({'mathtext.fontset': 'custom'})
plt.rcParams.update({'mathtext.rm': 'serif'})
plt.rcParams.update({'mathtext.it': 'serif:italic'})
plt.rcParams.update({'mathtext.bf': 'serif:bold'})
plt.close('all')
#%% multiphysics
amplitudes = [.02, .05, .1, .2]
frequencies = [.1, .2, .5, 1, 2]
batch_train_time = .2

arr = np.load("./datasets/generateddatav2.npy")

arr = arr[:,:,:,:] # exclude no data

dt = 1/200
t = np.array([dt*i for i in range(arr.shape[2])])

arr = arr.reshape((-1, arr.shape[2], 3))
num_batches = arr.shape[0]

x = arr[:,:,1]
v = arr[:,:,0]
F = arr[:,:,2:]/1000 # output force
D = np.full((x.shape[0], x.shape[1], 2), [1.0, 1.02])

# scale inputs
v_std = 0.7603725298200057

v = (v)/v_std

x = np.expand_dims(x, -1)
v = np.expand_dims(v, -1)
#%%
alpha = 2
v_s=0.01/v_std
sigma_0_star = 24 # doesn't matter
sigma_1=100*v_std/1000 # small sigma_1
sigma_2=0 # zero viscosity assumption
units=50


F_cs_input = keras.Input(batch_input_shape=[num_batches, None, 2], name="F_cs_input")
# X_input = keras.Input(shape=[None, 2], name='X')
v_input = keras.Input(batch_input_shape=[num_batches,None, 1], name='v')


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
              sigma_0_star=sigma_0_star,
              alpha=alpha,
              v_s=v_s,
              sigma_1=sigma_1,
              sigma_2=sigma_2,
              # sigma_1_regularizer=0.02,
              stateful=True,
              return_sequences=True,
              callv=2,
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
s_model = keras.models.load_model("./model_saves/multiphysics sigma model-2")

dense1.set_weights(s_model.layers[0].get_weights())
dense2.set_weights(s_model.layers[1].get_weights())
dense3.set_weights(s_model.layers[2].get_weights())
#%%
d = model.predict([v, D]) / v_std

pred = d.squeeze()

i = 7

s_pred = pred[i]

t1 = 5
t2 = 10

d = np.logical_and(t >= t1, t <= t2)

t_ = t[d] - t[d][0]
s_pred = s_pred[d]

plt.figure(figsize=(5, 2.1))
plt.plot(t_, s_pred)
plt.ylabel("$\sigma_0$ (kN/m)")
plt.xlabel("time (s)")
plt.yticks([10, 20, 30, 40])
plt.xlim((t_[0], t_[-1]))
plt.tight_layout()
plt.savefig("./plots/journal plots/sigma_0 in time/multiphysics sigma_0 in time.png", dpi=500)
plt.savefig("./plots/journal plots/sigma_0 in time/multiphysics sigma_0 in time.svg")








#%% experimental

F_batched = np.load("./datasets/dataset-2/F_batched.npy")
v_batched = np.load("./datasets/dataset-2/v_batched.npy")
x_batched = np.load("./datasets/dataset-2/x_batched.npy")

num_batches = F_batched.shape[0]
# these values were found in preprocess_training.py and would have to be
# changed if the dataset changes

scales = json.load(open('./metrics/experimental_scales.json'))

F_std = scales['F_std']
v_std = scales['v_std']
F_sneg = scales['F_sneg']
F_spos = scales['F_spos']
F_cneg = scales['F_cneg']
F_cpos = scales['F_cpos']
sigma_1 = scales['sigma_1']
sigma_2 = scales['sigma_2']
#%% 
alpha = 2

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
              sigma_0_star=sigma_0_star,
              alpha=alpha,
              v_s=v_s,
              sigma_1=sigma_1*v_std/F_std,
              sigma_2=sigma_2*v_std/F_std,
              callv=2,
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
s_model = keras.models.load_model("./model_saves/loop experimental models/DuzceDBE model")

dense11.set_weights(s_model.layers[0].get_weights())
dense12.set_weights(s_model.layers[1].get_weights())
dense13.set_weights(s_model.layers[2].get_weights())
#%%
i = 1
name = "test" + str(i)
F_test = np.load("./datasets/dataset-2/F/%s.npy"%name).reshape(-1, 1)
v_test = np.load("./datasets/dataset-2/v/%s.npy"%name).reshape(-1, 1)
x_test = np.load("./datasets/dataset-2/x/%s.npy"%name).reshape(-1, 1)

v_test = v_test/v_std

F_test = F_test.reshape(-1, 1)
v_test = v_test.reshape(1, -1, 1)

F_c_test = np.full(F_test.shape, F_cneg) * (F_test<0) + np.full(F_test.shape, F_cpos) * (F_test>=0)
F_s_test = np.full(F_test.shape, F_sneg) * (F_test<0) + np.full(F_test.shape, F_spos) * (F_test>=0)

D_test = np.append(F_c_test, F_s_test, axis=-1)

D_test = np.expand_dims(D_test, 0)/F_std

d = model.predict([v_test, D_test])

s_pred = d.squeeze() * F_std/v_std / 1000

dt = 1/1024
t = np.array([dt*i for i in range(s_pred.size)])

t1 = 10.27
t2 = 14.27
# t1 = 0
# t2 = t[-1]

p = np.logical_and(t >= t1, t <= t2)
# p = np.ones(t.shape, dtype=bool)

t_ = t[p] - t[p][0]
s_pred = s_pred[p]

plt.figure(figsize=(5, 1.9))
plt.plot(t_, s_pred, linewidth=1.0)
plt.ylabel("$\sigma_0$ (kN/m)")
plt.xlabel("time (s)")
plt.xlim((t_[0], t_[-1]))
# plt.xticks([0, 1, 2, 3, 4])
plt.tight_layout()
plt.savefig("./plots/journal plots/sigma_0 in time/experimental sigma_0 in time.png", dpi=500)
plt.savefig("./plots/journal plots/sigma_0 in time/experimental sigma_0 in time.svg")
