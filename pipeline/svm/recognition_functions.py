import numpy as np
import scipy.io as sio

def generate_output_vector(nclasses, reps):
    """ Generate Y vector for the objects.
    """
    return np.repeat(np.arange(1, nclasses+1), reps)

def generate_random_training_validation_testing_ids(tr_len, val_len, nimgs):
    """ Create training, validation and testing sets by randomly choosing
        images (ids) from the dataset.
    """
    ids = np.random.permutation(np.arange(0, nimgs))[:]

    trids = ids[0:tr_len]
    valids = ids[tr_len:(2*val_len)]
    tstid = ids[(tr_len+val_len):]

    return trids, valids, tstid

def extract_mat_content(path, header='obj'):
    """ Extract content of a .mat file located in path and has unique header.
    """
    content = sio.loadmat(path)

    return content[header]

def extract_loss_and_gradient(W_b, Xtr_b, Ytr, delta, reg):
      """ Perform gradient descent to optimize Hing-loss.
      """
      loss = 0.0
      dW = np.zeros(W_b.shape)

      ntraining = Xtr_b.shape[0]
      fscores = Xtr_b.dot(W_b)

      ftrue = fscores[np.arange(ntraining), Ytr]
      margins = np.maximum(0, fscores - ftrue[:, np.newaxis] + delta)
      margins[np.arange(ntraining), Ytr] = 0.0

      loss = np.sum(margins) / ntraining
      loss += 0.5 * reg * np.sum(W_b * W_b)

      hold_margins = margins
      hold_margins[margins > 0] = 1

      row_sum = np.sum(hold_margins, axis=1)
      hold_margins[np.arange(ntraining), Ytr] = -1 * row_sum
      dW = Xtr_b.T.dot(hold_margins)
      dW /= ntraining*1.0
      dW += reg * W_b

      return loss, dW

def get_accuracy(X, W, Y):
    """ Extract accuracy values for given actual values output vector.
    """
    scores = X.dot(W)
    predictions = np.argmax(scores, axis=1)
    acc = np.mean(predictions == Y) * 100.0

    return acc, predictions


def construct_conf_mat(Yorig, Ypred, nobjs):
    """ Create the confusion matrix by using actual id and predicted ids.
    """

    csize = nobjs + 1 # eliminate zero as class
    conf_mat = np.zeros((csize, csize))
    for i in range(len(Yorig)):
        conf_mat[Yorig[i], Ypred[i]] += 1

    return conf_mat


def early_stop(arr, threshold):
    """ Monitor learning performance on validation set.
    """
    count = np.count_nonzero(np.diff(arr) <= threshold)
    if count == (len(arr)-1):
        return True
    else:
        return False
