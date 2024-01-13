import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib import markers
import tensorflow.keras as keras
import json
"""
most journal paper plots - only loop experimental plots
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
#%%
def experimental_harmonic_plot():
    test_index = 1
    
    name = "test" + str(test_index)
    F_test = np.load("./datasets/dataset-2/F/%s.npy" % name).flatten()
    x_test = np.load("./datasets/dataset-2/x/%s.npy" % name).flatten()
    v_test = np.load("./datasets/dataset-2/v/%s.npy" % name).flatten()
    # using model excluding Duzce DBE
    F_pred = np.load(
        "./model predictions/loop experimental/DuzceDBE/harmonic/test%d.npy" % test_index).flatten()
    F_pred_lugre = np.load(
        "./model predictions/dataset-2 standard/harmonic/%s.npy"%name).flatten()
    dt = 1/1024
    t = np.array([dt*i for i in range(F_test.size)])
    
    F_test1 = F_test[F_test.size*2//7:-F_test.size*2//7]/1000
    x_test1 = x_test[x_test.size*2//7:-F_test.size*2//7]
    v_test1 = v_test[v_test.size*2//7:-v_test.size*2//7]
    t1 = t[t.size*2//7:-t.size*2//7]
    t1 = t1 - t1[0]
    F_pred1 = F_pred[F_pred.size*2//7:-F_pred.size*2//7]/1000
    F_pred_lugre1 = F_pred_lugre[F_pred_lugre.size*2//7:-F_pred_lugre.size*2//7]/1000
    
    timeframe=(2.5, 6.5)
    p = np.logical_and(t1>2.5, t1<6.5)
    
    # harmonic test with prediction
    
    fig = plt.figure(figsize=(6,3.6), layout="constrained")
    gs = GridSpec(2, 3, figure=fig)
    ax1 = fig.add_subplot(gs[0,:])
    ax2 = fig.add_subplot(gs[1,0:2])
    # plt.ylim((-3.5, 3.5))
    ax1.set_ylabel("force (kN)")
    ax1.plot(t1, F_test1, linewidth=0.6, label='experimental')
    ax1.plot(t1, F_pred_lugre1, linewidth=0.6, linestyle='-.', label='LuGre model')
    ax1.plot(t1, F_pred1, linewidth=0.6, c=cc[2], linestyle='--', label='physics-ML model')
    ax1.set_xlabel("time (s)")
    ax1.set_xlim((t1[0], t1[-1]))
    ax1b = ax1.twinx()
    ax1b.set_ylabel("error (kN)")
    ax1b.plot(t1, (F_test1 - F_pred_lugre1), c=cc[3], linewidth=.8, linestyle=':')
    ax1b.plot(t1, (F_test1 - F_pred_lugre1), c=cc[3], linewidth=0.3)
    ax1b.plot([],[], marker='s', c=cc[3], label='LuGre error')
    ax1b.plot(t1, (F_test1 - F_pred1), c=cc[4], linewidth=0.8, linestyle=':')
    ax1b.plot(t1, (F_test1 - F_pred1), c=cc[4], linewidth=0.3)
    ax1b.plot([],[], marker='s', c=cc[4], label='physics-ML error')
    
    ax2.set_ylabel("force (kN)")
    ax2.plot(t1, F_test1, linewidth=1.2, label='experimental')
    ax2.plot(t1, F_pred_lugre1, linewidth=1.2, linestyle='-.', label='LuGre model')
    ax2.plot(t1, F_pred1, linewidth=1.2, c=cc[2], linestyle='--', label='physics-ML model')
    ax2.set_xlabel("time (s)")
    ax2.set_xlim((2.9, 4.1))
    ax2b = ax2.twinx()
    ax2b.set_ylabel("error (kN)")
    ax2b.plot(t1, (F_test1 - F_pred_lugre1), c=cc[3], linewidth=1.8, linestyle=':')
    ax2b.plot(t1, (F_test1 - F_pred_lugre1), c=cc[3], linewidth=0.4)
    ax2b.plot([],[], marker='s', c=cc[3], label='LuGre error')
    ax2b.plot(t1, (F_test1 - F_pred1), c=cc[4], linewidth=1.8, linestyle=':')
    ax2b.plot(t1, (F_test1 - F_pred1), c=cc[4], linewidth=0.4)
    ax2b.plot([],[], marker='s', c=cc[4], label='physics-ML error')
    ax2b.set_ylim((-8.27,5.62))
    # legend
    h1, l1 = ax2.get_legend_handles_labels()
    h2, l2 = ax2b.get_legend_handles_labels()
    fig.legend(bbox_to_anchor=(.95,.47), handles=h1+h2, labels=l1+l2)
    plt.tight_layout()
    # plt.savefig("./plots/experimental harmonic.svg")
    plt.savefig("./plots/experimental harmonic.png", dpi=500)
#%%
def experimental_earthquake_plot():
    # plot from earthquake tests (Duzce DBE)
    restricted_indices = [(4608, 12288), (4608, 12288), (3072, 7680), (3072, 7680), (6144, 14336), (6144, 14336)]
    val_names = \
        ['DuzceDBE',
         'DuzceMCE',
         'ImperialValleyDBE',
         'ImperialValleyMCE',
         'KocaeliDBE',
         'KocaeliMCE']
    t_start = [2.5, 2.5, 0.5, 0.5, 0, 0]
    timeframes = [(9.5, 12.5)]
    exp_val = np.load("./model predictions/metrics/expvalidation.npy")
    for i in range(6):
        name = val_names[i]
        r = restricted_indices[i]
        
        t_test, F_test, x_test, v_test = np.load(
            "./datasets/dataset-2/validation/%s.npy" % name).T
        # using model excluding Duzce DBE
        F_pred = np.load(
            "./model predictions/loop experimental/%s/validation/%s.npy"%(name, name)).flatten()
        F_pred_lugre = np.load(
            "./model predictions/dataset-2 standard/validation/%s.npy"%name).flatten()
        
        
        t1 = t_test[r[0]:r[1]]
        F_test1 = F_test[r[0]:r[1]]
        F_pred1 = F_pred
        t1 -= t1[0]
        
        # cut off first part of excitations
        p = t1>t_start[i]
        t1 = t1[p]
        F_test1 = F_test1[p]
        F_pred1 = F_pred1[p]
        F_pred_lugre = F_pred_lugre[p]
        t1 -= t1[0]
        
        # gather metrics
        snr = exp_val[i+1,i,0]
        mae = exp_val[i+1,i,1]
        rmse = exp_val[i+1,i,2]
        rrse = exp_val[i+1,i,3]
        nrmse = exp_val[i+1,i,4]
        trac = exp_val[i+1,i,5]
        
        # textstr = '\n'.join((
        #     'SNR: %.2f'%snr,
        #     'MAE: %.1f N'%mae,
        #     'RMSE: %.1f N'%rmse,
        #     'RRSE: %.4f'%rrse,
        #     'NRMSE: %.4f'%nrmse,
        #     'TRAC: %.4f'%trac
        # ))
        textmetrics = '\n'.join((
            'SNR:                          ',
            'MAE:',
            'RMSE:',
            'RRSE:',
            'NRMSE:',
            'TRAC:'
        ))
        textvalues = '\n'.join((
            '%.2f'%snr,
            '%.1f N'%mae,
            '%.1f N'%rmse,
            '%.4f'%rrse,
            '%.4f'%nrmse,
            '%.4f'%trac
        ))
        props = dict(boxstyle='round', edgecolor='k', facecolor='w', alpha=.2)
        
        fig, (ax, ax_text) = plt.subplots(1, 2, figsize=(6, 2.35), layout='constrained', gridspec_kw={'width_ratios':[2.9, 1]})
        # gs = GridSpec(1, 5, figure=fig)
        # ax = fig.add_subplot(gs[0,0:-1])
        # ax_text = fig.add_subplot(gs[0,-1])
        ax_text.axis('off')
        
        ax.plot(t1, F_test1/1000, label='experiment')
        ax.plot(t1, F_pred_lugre/1000, c=cc[1], linestyle='--', label='LuGre model')
        ax.plot(t1, F_pred1/1000, c=cc[2], linestyle='--', label='physics-ML model')
        ax.set_xlim((t1[0],t1[-1]))
        ax.set_xlabel("time (s)")
        ax.set_ylabel("force (kN)")
        # ax.legend(loc='best')
        handles, labels = ax.get_legend_handles_labels()
        ax_text.text(0.0,0.47, textmetrics, bbox=props)
        ax_text.text(.55,0.47,textvalues)
        fig.legend(bbox_to_anchor=(1.005,0.545),handles=handles, labels=labels)
        fig.tight_layout()
        # fig.savefig("./plots/experimental " + name + ".svg")
        fig.savefig("./plots/experimental " + name + ".png", dpi=500)
#%%
def experimental_earthquake_displacement_plot():
    from scipy.fft import fft, fftfreq
    restricted_indices = [(4608, 12288), (4608, 12288), (3072, 7680), (3072, 7680), (6144, 14336), (6144, 14336)]
    val_names = \
        ['DuzceDBE',
         'DuzceMCE',
         'ImperialValleyDBE',
         'ImperialValleyMCE',
         'KocaeliDBE',
         'KocaeliMCE']
    
    r1 = restricted_indices[0]
    r2 = restricted_indices[2]
    r3 = restricted_indices[4]
    
    t_DuzceDBE, F_DuzceDBE, x_DuzceDBE, v_DuzceDBE = np.load(
        "./datasets/dataset-2/validation/%s.npy" % 'DuzceDBE').T
    t_DuzceMCE, F_DuzceMCE, x_DuzceMCE, v_DuzceMCE = np.load(
        "./datasets/dataset-2/validation/%s.npy" % 'DuzceMCE').T
    t_ImpVDBE, F_ImpVDBE, x_ImpVDBE, v_ImpVDBE = np.load(
        "./datasets/dataset-2/validation/%s.npy" % 'ImperialValleyDBE').T
    t_ImpVMCE, F_ImpVMCE, x_ImpVMCE, v_ImpVMCE = np.load(
        "./datasets/dataset-2/validation/%s.npy" % 'ImperialValleyMCE').T
    t_KocaeliDBE, F_KocaeliDBE, x_KocaeliDBE, v_KocaeliDBE = np.load(
        "./datasets/dataset-2/validation/%s.npy" % 'KocaeliDBE').T
    t_KocaeliMCE, F_KocaeliMCE, x_KocaeliMCE, v_KocaeliMCE = np.load(
        "./datasets/dataset-2/validation/%s.npy" % 'KocaeliMCE').T
    
    t = t_DuzceDBE.copy()
    
    x_DuzceDBE = x_DuzceDBE[r1[0]:]
    x_DuzceMCE = x_DuzceMCE[r1[0]:]
    x_ImpVDBE = x_ImpVDBE[r2[0]:]
    x_ImpVMCE = x_ImpVMCE[r2[0]:]
    x_KocaeliDBE = x_KocaeliDBE[r3[0]:]
    x_KocaeliMCE = x_KocaeliMCE[r3[0]:]
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(4.9, 2.4))
    
    # time series
    ax1.plot(t[:x_DuzceDBE.size], x_DuzceDBE, linewidth=.8, label='DuzceDBE', linestyle='solid')
    ax1.plot(t[:x_DuzceMCE.size], x_DuzceMCE, linewidth=.8, label='DuzceMCE', linestyle='solid')
    ax1.plot(t[:x_ImpVDBE.size], x_ImpVDBE, linewidth=.8, label='Imp.ValleyDBE', linestyle='dashed')
    ax1.plot(t[:x_ImpVMCE.size], x_ImpVMCE, linewidth=.8, label='Imp.ValleyMCE', linestyle='dashed')
    ax1.plot(t[:x_KocaeliDBE.size], x_KocaeliDBE, linewidth=.8, label='KocaeliDBE', linestyle='dotted')
    ax1.plot(t[:x_KocaeliMCE.size], x_KocaeliMCE, linewidth=.8, label='KocaeliMCE', linestyle='dotted')
    ax1.set_xlabel("time (s)")
    ax1.set_ylabel("displacement (m)")
    ax1.set_xlim((0, 10))
    # PSD
    def psd(x):
        N = x.size
        p = np.abs(fft(x)[:N//2])**2/(N/1024)**2
        freq = fftfreq(N, 1/1024)[:N//2]
        return p, freq
    psd_DuzceDBE, freq_DuzceDBE = psd(x_DuzceDBE)
    psd_DuzceMCE, freq_DuzceMCE = psd(x_DuzceMCE)
    psd_ImpVDBE, freq_ImpVDBE = psd(x_ImpVDBE)
    psd_ImpVMCE, freq_ImpVMCE = psd(x_ImpVMCE)
    psd_KocaeliDBE, freq_KocaeliDBE = psd(x_KocaeliDBE)
    psd_KocaeliMCE, freq_KocaeliMCE = psd(x_KocaeliMCE)
    
    ax2.plot(freq_DuzceDBE, psd_DuzceDBE, linewidth=.8, label=r'Duzce$_{\mathregular{DBE}}$', linestyle='solid')
    ax2.plot(freq_DuzceMCE, psd_DuzceMCE, linewidth=.8, label=r'Duzce$_{\mathregular{MCE}}$', linestyle='solid')
    ax2.plot(freq_ImpVDBE, psd_ImpVDBE, linewidth=.8, label=r'Imp.Valley$_{\mathregular{DBE}}$', linestyle='dashed')
    ax2.plot(freq_ImpVMCE, psd_ImpVMCE, linewidth=.8, label=r'Imp.Valley$_{\mathregular{MCE}}$', linestyle='dashed')
    ax2.plot(freq_KocaeliDBE, psd_KocaeliDBE, linewidth=.8, label=r'Kocaeli$_{\mathregular{DBE}}$', linestyle='dotted')
    ax2.plot(freq_KocaeliMCE, psd_KocaeliMCE, linewidth=.8, label=r'Kocalei$_{\mathregular{MCE}}$', linestyle='dotted')
    ax2.set_yscale('log')
    ax2.set_xscale('log')
    ax2.set_xlim(0, 20)
    ax2.set_ylim((10**-9, 10**2.5))
    ax2.set_xlabel('frequency (Hz)')
    ax2.set_ylabel(r'spectral power (m$^2$)')
    ax2.legend(loc=1)
    # ax2.set_ylim(())
    ax2.grid(False)
    # plt.xlim((0, t[x_KocaeliDBE.size]))
    fig.tight_layout()
    # plt.savefig("./plots/earthquake displacement.svg")
    plt.savefig("./plots/earthquake displacement.png", dpi=500)
#%%
def experimental_state_space_plot():
    scales = json.load(open('./model_saves/scaling/exclude DuzceDBE.json'))
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
    # fig.savefig("./plots/experimental state space.svg")
    fig.savefig("./plots/experimental state space.png", dpi=500)
#%%
def experimental_sigma_0_in_time_plot():
    from lugre import LuGre
    F_batched = np.load("./datasets/dataset-2/F_batched.npy")
    v_batched = np.load("./datasets/dataset-2/v_batched.npy")
    x_batched = np.load("./datasets/dataset-2/x_batched.npy")
    
    num_batches = F_batched.shape[0]
    
    scales = json.load(open('./model_saves/scaling/exclude DuzceDBE.json'))
    
    F_std = scales['F_std']
    v_std = scales['v_std']
    F_sneg = scales['F_sneg']
    F_spos = scales['F_spos']
    F_cneg = scales['F_cneg']
    F_cpos = scales['F_cpos']
    sigma_1 = scales['sigma_1']
    sigma_2 = scales['sigma_2']
    
    dt=1/1024
    v_s=0.01
    alpha = 2
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
    
    # callv=2: return sigma_0 prediction
    lugre = LuGre(dt,
                  sigma_call,
                  sigma_call_state_size,
                  sigma_model_weights,
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
    # set weights
    s_model = keras.models.load_model("./model_saves/loop experimental models/DuzceDBE model")

    dense11.set_weights(s_model.layers[0].get_weights())
    dense12.set_weights(s_model.layers[1].get_weights())
    dense13.set_weights(s_model.layers[2].get_weights())
    # load and predict from test
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
    # plt.savefig("./plots/experimental sigma_0 in time.svg")
    plt.savefig("./plots/experimental sigma_0 in time.png", dpi=500)
#%%
def numerical_harmonic_plot():
    arr_true = np.load("./datasets/numerical_generated_data.npy")
    arr_pred = np.load("./model predictions/numerical/predictiondata.npy")
    arr_lugre_pred = np.load(
        "./model predictions/numerical standard/F_pred.npy")
    
    arr_true = arr_true.reshape((-1, arr_true.shape[2], 3))
    num_batches = arr_true.shape[0]
    
    true_tot = np.zeros([0])
    t = np.array([.005*i for i in range(arr_true.shape[1])])
    
    i = 9
    test = arr_true[i]
    
    x_test = test[:,1]
    v_test = test[:,0]
    F_test = test[:,2]
    
    true_tot = np.append(true_tot, F_test)
    
    F_pred = arr_pred[i]
    F_pred_lugre = arr_lugre_pred[i].flatten()*1000
    
    p = np.logical_and(t>1.2, t<7.1)
    # q = np.logical_and(t>.9, t<1.92)
    
    fig = plt.figure(figsize=(6,3.6), layout="constrained")
    gs = GridSpec(2, 3, figure=fig)
    ax1 = fig.add_subplot(gs[0,:])
    ax2 = fig.add_subplot(gs[1,0:2])
    
    ax1.set_ylabel("force (kN)")
    ax1.plot(t[p]-t[p][0], F_test[p]/1000, linewidth=1.0, label='multiphysics model')
    ax1.plot(t[p]-t[p][0], F_pred_lugre[p]/1000, linewidth=1.0, linestyle='-.', label='LuGre model')
    ax1.plot(t[p]-t[p][0], F_pred[p]/1000, linewidth=1.0, c=cc[2], linestyle='--', label='physics-ML model')
    ax1.set_xlim((0, t[p][-1]-t[p][0]))
    ax1.set_xlabel("time (s)")
    ax1b = ax1.twinx()
    ax1b.set_ylabel("error (N)")
    ax1b.plot(t[p]-t[p][0], (F_test - F_pred_lugre)[p], c=cc[3], linewidth=1.5, linestyle=':')
    ax1b.plot(t[p]-t[p][0], (F_test - F_pred_lugre)[p], c=cc[3], linewidth=0.3)
    ax1b.plot([],[], marker='s', c=cc[3], label='LuGre error')
    ax1b.plot(t[p]-t[p][0], (F_test - F_pred)[p], c=cc[4], linewidth=1.5, linestyle=':')
    ax1b.plot(t[p]-t[p][0], (F_test - F_pred)[p], c=cc[4], linewidth=0.3)
    ax1b.plot([],[], marker='s', c=cc[4], label='physics-ML error')
    
    ax2.set_ylabel("force (N)")
    ax2.plot(t[p]-t[p][0], F_test[p]/1000, linewidth=1.2, label='multiphysics model')
    ax2.plot(t[p]-t[p][0], F_pred_lugre[p]/1000, linewidth=1.2, linestyle='-.', label='LuGre model')
    ax2.plot(t[p]-t[p][0], F_pred[p]/1000, linewidth=1.2, c=cc[2], linestyle='--', label='physics-ML model')
    ax2.set_xlim((1.2, 1.9))
    ax2.set_xlabel("time (s)")
    ax2b = ax2.twinx()
    ax2b.set_ylabel("error (N)")
    ax2b.plot(t[p]-t[p][0], (F_test - F_pred_lugre)[p], c=cc[3], linewidth=1.8, linestyle=':')
    ax2b.plot(t[p]-t[p][0], (F_test - F_pred_lugre)[p], c=cc[3], linewidth=0.4)
    ax2b.plot([],[], marker='s', c=cc[3], label='LuGre error')
    ax2b.plot(t[p]-t[p][0], (F_test - F_pred)[p], c=cc[4], linewidth=1.8, linestyle=':')
    ax2b.plot(t[p]-t[p][0], (F_test - F_pred)[p], c=cc[4], linewidth=0.4)
    ax2b.plot([],[], marker='s', c=cc[4], label='physics-ML error')
    
    h1, l1 = ax1.get_legend_handles_labels()
    h2, l2 = ax1b.get_legend_handles_labels()
    fig.legend(bbox_to_anchor=(.95,.47), handles=h1+h2, labels=l1+l2)
    plt.tight_layout()
    # plt.savefig("./plots/numerical harmonic.svg")
    plt.savefig("./plots/numerical harmonic.png", dpi=500)
#%% multiphysics experimental - Duzce DBE
def numerical_earthquake_plot():
    restricted_indices = [(900, 2400), (900, 2400), (600, 1500), (600, 1500), (1200, 2800), (1200, 2800)]
    val_names = \
        ['DuzceDBE',
          'DuzceMCE',
          'ImperialValleyDBE',
          'ImperialValleyMCE',
          'KocaeliDBE',
          'KocaeliMCE']
    
    name = val_names[0]
    r = restricted_indices[0]
    
    F_test = np.load(
        "./datasets/numerical validation/%s.npy" % name)[:,2:]
    F_pred = np.load(
        "./model predictions/numerical/validation/%s.npy" % name).flatten()
    F_pred_lugre = np.load(
        "./model predictions/numerical standard/validation/%s.npy"%name).flatten()
    
    dt = 1/200
    t_test = np.array([i*dt for i in range(F_test.size)])
    
    timeframe = (11, 18)
    
    t1 = t_test[r[0]:r[1]]
    t1 = t1 - t1[0]
    
    F_test1 = F_test[r[0]:r[1]]
    F_pred1 = F_pred
    F_pred_lugre1 = F_pred_lugre
    
    plt.figure(figsize=(5, 2.5))
    plt.plot(t1, F_test1/1000, label='numerical model')
    plt.plot(t1, F_pred_lugre1/1000, c=cc[1], linestyle='--', label='LuGre model')
    plt.plot(t1, F_pred1/1000, c=cc[2], linestyle='--', label='physics-ML model')
    plt.xlim((t1[0], t1[-1]))
    plt.xlabel("time (s)")
    plt.ylabel("force (kN)")
    plt.legend(loc=2)
    plt.tight_layout()
    # plt.savefig("./plots/numerical " + name + ".svg")
    plt.savefig("./plots/numerical " + name + ".png", dpi=500)
#%%
def numerical_state_space_plot():
    # just to get v range and scaling factor
    arr = np.load("./datasets/numerical_generated_data.npy")
    arr = arr.reshape((-1, arr.shape[2], 3))
    
    v = arr[:, :, 0]
    
    # scale inputs
    v_std = 0.7603725298200057
    
    v_max = np.min(v)
    v_min = np.max(v)
    # v_max = -2
    # v_min= 2
    
    sigma_model = keras.models.load_model("./model_saves/numerical model")
    
    y_axis = np.linspace(-1.02, 1.02, 500)
    v_axis = np.linspace(v_min, v_max, 500)
    
    y_mesh, v_mesh = np.meshgrid(y_axis, v_axis)
    
    coords = np.vstack((v_mesh.reshape(-1)/v_std, y_mesh.reshape(-1))).T
    coords = np.expand_dims(coords, 1)
    
    sigma_pred = sigma_model.predict(coords)/v_std
    
    sigma_pred = sigma_pred.reshape(500, 500)
    # make sure the reshape is done correctly
    # sigma_pred1 = np.zeros((500, 500))
    # c = 0
    # for i in range(500):
#     for j in range(500):
#         coord = np.array([[v_mesh[i,j], y_mesh[i,j]]])
#         sigma_pred1[i,j] = sigma_model.predict(coord)
#     print(i)
    
    fig, ax = plt.subplots(figsize=(4, 3), subplot_kw=dict(projection='3d'))
    
    surf = ax.plot_surface(y_mesh, v_mesh, sigma_pred, cmap='viridis', alpha=.85, antialiased=False)
    # ax.plot(y, v, z, marker='.', linewidth=0, markersize=.25, c=cc[0], alpha=1, fillstyle='full')
    style = markers.MarkerStyle(marker='.', fillstyle='full')
    ax.set_box_aspect([1, 1, .5])
    
    ax.set_xlabel('$y$ (kN)')
    ax.set_ylabel('$v$ (m/s)')
    ax.set_zlabel('$\sigma_0$ (kN/m)')
    plt.tight_layout()
    plt.show()
    # fig.savefig("./plots/numerical state space.svg")
    fig.savefig("./plots/numerical state space.png", dpi=500)
#%%
def numerical_sigma_0_in_time_plot():
    from lugre import LuGre
    arr = np.load("./datasets/numerical_generated_data.npy")
    
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
    #
    alpha = 2
    v_s=0.01/v_std
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
    # callv=2: return sigma_0 prediction
    lugre = LuGre(dt,
                  sigma_call,
                  sigma_call_state_size,
                  sigma_model_weights,
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
    # set weights 
    s_model = keras.models.load_model("./model_saves/numerical model")
    
    dense1.set_weights(s_model.layers[0].get_weights())
    dense2.set_weights(s_model.layers[1].get_weights())
    dense3.set_weights(s_model.layers[2].get_weights())
    #
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
    # plt.savefig("./plots/numerical sigma_0 in time.svg")
    plt.savefig("./plots/numerical sigma_0 in time.png", dpi=500)