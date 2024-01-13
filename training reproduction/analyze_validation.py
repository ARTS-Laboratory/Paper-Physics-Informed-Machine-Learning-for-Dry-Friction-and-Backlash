import numpy as np
"""
loop on experimental dataset

metrics:
    SNR
    MAE
    RMSE
    RRSE
    NRMSE
    TRAC
"""
#%% define functions
import math
""" signal to noise ratio """
def snr(sig, noisy_signal, dB=True):
    noise = sig - noisy_signal
    a_sig = math.sqrt(np.mean(np.square(sig)))
    a_noise = math.sqrt(np.mean(np.square(noise)))
    snr = (a_sig/a_noise)**2
    if(not dB):
        return snr
    return 10*math.log(snr, 10)
""" root mean squared error """
def rmse(sig, pred, squared=False):
    error = sig - pred
    num = np.sum(np.square(error))
    denom = np.size(sig)
    e = num/denom
    if(squared):
        return e
    return np.sqrt(e)
""" root relative squared error """
def rrse(sig, pred):
    error = sig - pred
    mean = np.mean(sig)
    num = np.sum(np.square(error))
    denom = np.sum(np.square(sig-mean))
    return np.sqrt(num/denom)
""" normalized root mean squared error """
def nrmse(sig, pred):
    return rmse(sig, pred)/(np.max(sig)-np.min(sig))
""" time response assurance criterion """
def trac(sig, pred):
    num = np.square(np.dot(sig, pred))
    denom = np.dot(sig, sig) * np.dot(pred, pred)
    return num/denom
""" mean absolute error """
def mae(sig, pred):
    return np.sum(np.abs(sig-pred))/sig.size

metrics = [snr, mae, rmse, rrse, nrmse, trac]
#%% experimental -- harmonic tests
def analyze_experimental():
    num_batches = 13
    earthquakes = ['DuzceDBE',
                   'DuzceMCE',
                   'ImperialValleyDBE',
                   'ImperialValleyMCE',
                   'KocaeliDBE',
                   'KocaeliMCE']
    
    # first axis [LuGre, (six physics-ML models)]
    exp_harm = np.zeros([7, num_batches+1, len(metrics)])
    
    true_tot = np.zeros([0])
    pred_paths = ['dataset-2 standard/harmonic/'] +\
        ['loop experimental/%s/harmonic/'%earthquake_name for earthquake_name in earthquakes]
    pred_tots = [np.zeros([0]) for i in range(exp_harm.shape[0])]
    v_max = 0
    
    for i in range(num_batches):
        name = "test" + str(i)
        F_test = np.load("./datasets/dataset-2/F/%s.npy"%name).flatten()
        v_test = np.load("./datasets/dataset-2/v/%s.npy"%name).flatten()
        x_test = np.load("./datasets/dataset-2/x/%s.npy"%name).flatten()
        v_max = max(v_max, np.max(v_test))
        
        true_tot = np.append(true_tot, F_test)
        
        for j, pred_path in enumerate(pred_paths):
            pred = np.load('./model predictions/'+pred_path+'%s.npy'%name).flatten()
            
            
            pred_tots[j] = np.append(pred_tots[j], pred)
            
            for k, metric in enumerate(metrics):
                exp_harm[j, i, k] = metric(F_test, pred)
    # total across all datasets
    for j, pred_tot in enumerate(pred_tots):
        for k, metric in enumerate(metrics):
            exp_harm[j, -1, k] = metric(true_tot, pred_tot)
    #%% experimental -- earthquake tests
    val_names = \
        ['DuzceDBE',
          'DuzceMCE',
          'ImperialValleyDBE',
          'ImperialValleyMCE',
          'KocaeliDBE',
          'KocaeliMCE']
    restricted_indices = [(4608, 12288), (4608, 12288), (3072, 7680), (3072, 7680), (6144, 14336), (6144, 14336)]
    # first axis [LuGre, (six physics-ML models)]
    exp_eq = np.zeros([7, len(val_names), len(metrics)])
    pred_paths = ['dataset-2 standard/validation/'] +\
        ['loop experimental/%s/validation/'%earthquake_name for earthquake_name in earthquakes]
    pred_tots = [np.zeros([0]) for i in range(exp_harm.shape[0])]
    true_tot = np.zeros([0])
    
    for i, name in enumerate(val_names):
        t_test, F_test, x_test, v_test = np.load("./datasets/dataset-2/validation/%s.npy"%name).T
        # F_pred = np.load("./model predictions/dataset-2/validation/%s.npy"%name).flatten()
        
        r = restricted_indices[i]
        F_test = F_test[r[0]:r[1]]
        t_test = t_test[r[0]:r[1]]
        
        # plt.figure(figsize=(6, 3.2))
        # plt.title(name)
        # plt.plot(F_test, label='experimental')
        # plt.plot(F_pred_lugre, label='LuGre')
        # plt.plot(F_pred_mlp, label='physics-ML')
        # plt.legend()
        # plt.tight_layout()
        # plt.savefig('./plots/all tests/experimental validation/%s.png'%name, dpi=300)
        
        true_tot = np.append(true_tot, F_test)
        
        for j, pred_path in enumerate(pred_paths):
            pred = np.load('./model predictions/'+pred_path+'%s.npy'%name).flatten()
            
            pred_tots[j] = np.append(pred_tots[j], pred)
            
            for k, metric in enumerate(metrics):
                exp_eq[j, i, k] = metric(F_test, pred)
    # total across all datasets
    # for j, pred_tot in enumerate(pred_tots):
    #     for k, metric in enumerate(metrics):
    #         exp_eq[j, -1, k] = metric(true_tot, pred_tot)
    #%% save results to .npy
    np.save("./model predictions/metrics/expharmonic.npy", exp_harm)
    np.save("./model predictions/metrics/expvalidation.npy", exp_eq)
def analyze_numerical():
    #%% multiphysics harmonic table
    arr_true = np.load("./datasets/numerical_generated_data.npy")
    arr_pred = np.load("./model predictions/numerical/predictiondata.npy")
    arr_lugre_pred = np.load("./model predictions/numerical standard/F_pred.npy")
    
    arr_true = arr_true.reshape((-1, arr_true.shape[2], 3))
    num_batches = arr_true.shape[0]
    
    true_tot = np.zeros([0])
    
    # [[physics-ML, LuGre], batch + total, metric]
    multi_harm = np.zeros([2, num_batches+1, len(metrics)])
    for i in range(num_batches):
        name = "test" + str(i)
        test = arr_true[i]
        
        x_test = test[:,1]
        v_test = test[:, 0]
        F_test = test[:,2]
        
        true_tot = np.append(true_tot, F_test)
        
        F_pred = arr_pred[i]
        F_pred_lugre = arr_lugre_pred[i].flatten()*1000
        
        # plt.figure(figsize=(7, 3))
        # plt.plot(F_test, label='experimnetal')
        # plt.plot(F_pred_lugre, label='LuGre model')
        # plt.plot(F_pred, label='physics-ML model')
        # plt.legend()
        # plt.tight_layout()
        # plt.savefig('./plots/all tests/multiphysics harmonic/%s.png'%name, dpi=500)
        # plt.close()
        
        for j, metric in enumerate(metrics):
            multi_harm[0, i, j] = metric(F_test, F_pred)
            multi_harm[1, i, j] = metric(F_test, F_pred_lugre)
    
    for j, metric in enumerate(metrics):
        multi_harm[0, -1, j] = metric(true_tot, arr_pred.flatten())
        multi_harm[1, -1, j] = metric(true_tot, arr_lugre_pred.flatten()*1000)
    #%% multiphysics earthquake table
    val_names = \
        ['DuzceDBE',
          'DuzceMCE',
          'ImperialValleyDBE',
          'ImperialValleyMCE',
          'KocaeliDBE',
          'KocaeliMCE']
    # restricted_indices = [(900, 2400), (600, 1500), (1200, 2800)]
    restricted_indices = [(900, 2400), (900, 2400), (600, 1500), (600, 1500), (1200, 2800), (1200, 2800)]
    
    true_tot = np.zeros([0])
    mlp_tot = np.zeros([0])
    lugre_tot = np.zeros([0])
    
    multi_eq = np.zeros([2, len(val_names)+1, len(metrics)])
    for i, name in enumerate(val_names):
        test = np.load("./datasets/numerical validation/%s.npy"%name)
        F_test = test[:,2]
        
        F_pred = np.load("./model predictions/numerical/validation/%s.npy"%name)
        F_pred_lugre = np.load("./model predictions/numerical standard/validation/%s.npy"%name)
        
        r = restricted_indices[i]
        F_test = F_test[r[0]:r[1]]
        
        # F_pred = F_pred[1:]
        # F_pred_lugre = F_pred_lugre[1:]
        
        # plt.figure(figsize=(6, 3.2))
        # plt.title(name)
        # plt.plot(F_test, label='experimental')
        # plt.plot(F_pred_lugre, label='LuGre')
        # plt.plot(F_pred, label='physics-ML')
        # plt.legend()
        # plt.tight_layout()
        # plt.savefig('./plots/all tests/multiphysics validation/%s.png'%name, dpi=300)
        
        # plt.figure()
        # plt.title(name)
        # plt.plot(F_test-F_pred_lugre, label='lugre error')
        # plt.plot(F_test-F_pred, label='physics-ml')
        # plt.legend()
        
        # plt.figure(figsize=(7, 3))
        # plt.plot(F_test, label='experimnetal')
        # plt.plot(F_pred_lugre, label='LuGre model')
        # plt.plot(F_pred, label='physics-ML model')
        # plt.legend()
        # plt.tight_layout()
        # plt.savefig('./plots/all tests/multiphysics validation/%s.png'%name, dpi=500)
        # plt.close()
        
        true_tot = np.append(true_tot, F_test)
        mlp_tot = np.append(mlp_tot, F_pred)
        lugre_tot = np.append(lugre_tot, F_pred_lugre)
        
        for j, metric in enumerate(metrics):
            multi_eq[0, i, j] = metric(F_test, F_pred)
            multi_eq[1, i, j] = metric(F_test, F_pred_lugre)
    
    for j, metric in enumerate(metrics):
        multi_eq[0, -1, j] = metric(true_tot, mlp_tot)
        multi_eq[1, -1, j] = metric(true_tot, lugre_tot)
    #%% save results to .npy
    np.save("./model predictions/metrics/multiharmonic.npy", multi_harm)
    np.save("./model predictions/metrics/multivalidation.npy", multi_eq)

#%%
if __name__ == '__main__':
    analyze_experimental()
    analyze_numerical()
