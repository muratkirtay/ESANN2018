import numpy as np
from mHTM.region import SPRegion
import scipy.io as sio

def main():

    # Parameters to construct cortical structure
    nsamples, nbits, pct_active = 500, 1024, 0.4
    num_of_objs, num_of_imgs = 100, 72
    data_path, sdr_path =  'pixel_binary/', 'sdrs/'

    seed = 123456789
    kargs = {
        'ninputs': nbits, 'ncolumns': 1024, 'nactive': int(nbits * 0.2), 'global_inhibition': True,
        'trim': 1e-4, 'disable_boost': True, 'seed': seed, 'nsynapses': 100, 'seg_th': 10,
        'syn_th': 0.5, 'pinc': 0.001, 'pdec': 0.001, 'pwindow': 0.5, 'random_permanence': True,
        'nepochs': 10 }

    sp = SPRegion(**kargs)

    for j in range(num_of_objs):
        obj_str = 'obj'+str(j+1)
        obj_path = data_path + obj_str + '.mat'
        data_content = extract_mat_content(obj_path, 'obj')
        sdrs = np.zeros((num_of_imgs, nbits), dtype=np.int64)
        for i in range(len(data_content)):
            sp.fit(data_content[i])
            sp_output = sp.predict(data_content[i])  
            outp = sp_output * 1  
            sdrs[i, :] = outp

        sdr_mat = sdr_path + 'sdrObj' + str(j+1) + '.mat'
        sio.savemat(sdr_mat, mdict={'obj': sdrs})

if __name__ == '__main__':
    main()
