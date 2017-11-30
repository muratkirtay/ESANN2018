import recognition_functions as sfuncs
import numpy as np
import scipy.io as sio
import time
import logging as lg
import os

np.random.seed(42)
fname = 'logs/pixel_svm.log'
desc = "Multi class SVM with pixel vectors, 3D Color Histogram and HOG as inputs"
init = time.time()

lg.basicConfig(filename=fname, format='%(message)s', level=lg.INFO)
lg.info('Experiment date: %s', time.ctime())
lg.info('Experiment Description: %s', desc)


def main():

    # The implementation procedure is the same for the different feature extraction methods
    # uncomment below lines for different inputs
    pixel_dpath, pixel_dhead, pixlen = '../DATA/pixel/', 'obj', 3072
    #pixel_dpath, pixel_dhead, pixlen = '../DATA/pixel_hog/', 'obj', 800
    #pixel_dpath, pixel_dhead, pixlen = '../DATA/pixel_histogram_normalized/', 'obj', 1000

    nobjs, nimgs = 100, 72
    tr_val_tst_rates = [[0.05,0.05,0.9], [0.1,0.1,0.8], [0.15,0.15,0.7], [0.2,0.2,0.6], [0.25,0.25,0.5], [0.3,0.3,0.4],[0.35,0.35,0.3], [0.4,0.4,0.2], [0.45,0.45,0.1]]

    for i in range(len(tr_val_tst_rates)):
        tr_rate, val_rate, tst_rate = tr_val_tst_rates[i][0], tr_val_tst_rates[i][1], tr_val_tst_rates[i][2]
        lg.info("Running for training: %.2f validation: %.2f test: %.2f \n" % (tr_rate*100, val_rate*100, tst_rate*100))
        tr_len, val_len =  int(nimgs*tr_rate), int(nimgs*val_rate)
        tst_len = nimgs - (tr_len+val_len)
        lg.info("Running for lengths training:  %d, validation: %d, test: %d \n" % (tr_len, val_len, tst_len))

        Xtr,  Ytr = np.zeros((tr_len*nobjs, pixlen), dtype=np.float32), np.zeros((tr_len*nobjs), dtype=np.int32)
        Xval, Yval = np.zeros((val_len*nobjs, pixlen), dtype=np.float32), np.zeros((val_len*nobjs), dtype=np.int32)
        Xte, Yte = np.zeros((tst_len*nobjs, pixlen), dtype=np.float32), np.zeros((tst_len*nobjs), dtype=np.int32)

        # non repeated image ids
        for ii in range(nobjs):
            head = pixel_dhead + str(ii + 1)
            path = pixel_dpath + head
            obj_mat = sfuncs.extract_mat_content(path, 'obj')
            trids, valids, tstids = sfuncs.generate_random_training_validation_testing_ids(tr_len, val_len, nimgs)

            # Split training, validation and testing sets
            Xtr[ii*tr_len: ii*tr_len+tr_len, :], Ytr[ii*tr_len: ii*tr_len+tr_len] = obj_mat[trids, :], np.repeat(ii, len(trids))
            Xval[ii*val_len: ii*val_len+val_len, :], Yval[ii*val_len: ii*val_len+val_len] = obj_mat[valids, :], np.repeat(ii, len(valids))
            Xte[ii*tst_len: ii*tst_len+tst_len, :], Yte[ii*tst_len: ii*tst_len+tst_len] = obj_mat[tstids, :], np.repeat(ii, len(tstids))

        # Zero-centering the pixel values
        pixel_mean = np.mean(Xtr, axis=0)
        Xtr -= pixel_mean
        Xval -= pixel_mean
        Xte -= pixel_mean

        # Add bias node to each Xs to work on a single X and W
        Xtr_b = np.hstack((Xtr, np.ones((Xtr.shape[0],1.0))))
        Xval_b = np.hstack((Xval, np.ones((Xval.shape[0],1.0))))
        Xte_b = np.hstack((Xte, np.ones((Xte.shape[0],1.0))))

        niters, delta, th_it, th_acc = 10000, 1.0, 20, 0.01
        stepsize, reg = [1e-1, 1e-2, 1e-3, 1e-4, 2e-2, 2e-3, 2e-4 ], [1e-5, 1.5e-5, 2e-5, 2.5e-5, 3e-5, 4e-5, 5e-5]

        for ssize in range(len(stepsize)):
            for regst in range(len(reg)):
                W_b = np.random.randn(pixlen+1, nobjs) * 0.00001
                cv_flag, its = True, 0
                loss_mat, tr_acc, val_acc = [],[],[]
                while cv_flag:
                    loss, grad = sfuncs.extract_loss_and_gradient(W_b, Xtr_b, Ytr, delta, reg[regst])
                    W_b += -1*stepsize[ssize] * grad

                    tr_accuracy, trpred = sfuncs.get_accuracy(Xtr_b, W_b, Ytr)
                    val_accuracy, valpred = sfuncs.get_accuracy(Xval_b, W_b, Yval)

                    loss_mat.append(loss)
                    tr_acc.append(tr_accuracy)
                    val_acc.append(val_accuracy)

                    # early stopping criteria; check last nth iteriation if it doesnt increase by threshold stop the learning
                    if its >= th_it:
                        loss_arr = val_acc[len(val_acc)-th_it: len(val_acc)]
                        if sfuncs.early_stop(loss_arr, th_acc):
                            tst_accuracy = sfuncs.get_accuracy(Xte_b, W_b, Yte)
                            tst_accuracy, Yte_pred = sfuncs.get_accuracy(Xte_b, W_b, Yte)
                            conf_mat = sfuncs.construct_conf_mat(Yte, Yte_pred , nobjs)
                            confname = 'results/pixel/training_'+str(int(tr_rate*100))+'_pixel_stepsize_'+str(ssize)+'_'+str(regst)+'.mat'
                            sio.savemat(confname, mdict={'conf_mat': conf_mat})
                            lg.info("Learning rate: %E Regularization strength: %E, TR acc: %f, Val acc: %f Tst acc: %f" % (stepsize[ssize], reg[regst], tr_accuracy, val_accuracy, tst_accuracy))
                            cv_flag = False
                    its += 1

            lg.info("------------------------------------------------------")


if __name__ == '__main__':
    main()
    lg.info('Experiment Finished Date/Time: %s', time.ctime())
