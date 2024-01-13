import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib import cm
import tensorflow.keras as keras
import json
"""
state space plot for a single loop experimental model
"""
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

scales = json.load(open('./metrics/experimental_scales.json'))
F_std = scales['F_std']
v_std = scales['v_std']
F_sneg = scales['F_sneg']
F_spos = scales['F_spos']
F_cneg = scales['F_cneg']
F_cpos = scales['F_cpos']

F_batched = np.load("./datasets/dataset-2/F_batched.npy")
v_batched = np.load("./datasets/dataset-2/v_batched.npy")
x_batched = np.load("./datasets/dataset-2/x_batched.npy")

num_batches = F_batched.shape[0]
# these values were found in preprocess_training.py and would have to be
# changed if the dataset changes

# x_scaler = StandardScale(with_mean=False)

sigma_model = keras.models.load_model("./model_saves/loop experimental models/DuzceDBE model")

y_axis = np.linspace(-F_sneg, F_spos, 500)
v_axis = np.linspace(np.min(v_batched), np.max(v_batched), 500)

y_mesh, v_mesh = np.meshgrid(y_axis, v_axis)

coords = np.vstack((v_mesh.reshape(-1)/v_std, y_mesh.reshape(-1)/F_std)).T
coords = np.expand_dims(coords, 1)

sigma_pred = sigma_model.predict(coords) * F_std / v_std / 1000

sigma_pred = sigma_pred.reshape(500, 500)
# make sure the reshape is done correctly
# sigma_pred1 = np.zeros((500, 500))
# c = 0
# for i in range(500):
#     for j in range(500):
#         coord = np.array([[[v_mesh[i,j]/v_scaler.s, y_mesh[i,j]/F_scaler.s]]])
#         sigma_pred1[i,j] = sigma_model.predict(coord) *F_scaler.s/v_scaler.s/1000
#         print(sigma_pred1[i,j])
#         print(sigma_pred[i,j])
#         print('\n')

fig, ax = plt.subplots(figsize=(4, 3), subplot_kw=dict(projection='3d'))

surf = ax.plot_surface(y_mesh/1000, v_mesh, sigma_pred, cmap='viridis', antialiased=False)
ax.set_box_aspect([1, 1, .5])

ax.set_xlabel('$y$ (kN)')
ax.set_ylabel('$v$ (m/s)')
ax.set_zlabel('$\sigma_0$ (kN/m)')
plt.tight_layout()
plt.show()
fig.savefig(
    "./plots/journal plots/state space/experimental state space.png", dpi=500)
fig.savefig("./plots/journal plots/state space/experimental state space.svg")