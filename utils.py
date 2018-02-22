import functools
import numpy as np

def lazy_property(function):
    attribute = '_cache_' + function.__name__

    @property
    @functools.wraps(function)
    def decorator(self):
        if not hasattr(self, attribute):
            setattr(self, attribute, function(self))
        return getattr(self, attribute)
    
    return decorator

def get_cm(sess, model, vals, preds):
    return sess.run(model.confusion_matrix, {model.cm_vals: vals, model.cm_preds: preds})

#def calc_cm(vals, preds):
#    vals, preds = vals.astype(int), preds.astype(int)
#    min_val, max_val = min(np.amin(vals), np.amin(preds)), (max(np.amax(vals), np.amax(preds))+1)
#    vals = np.concatenate([np.reshape(vals == i, [1, vals.size]).astype(int)   for i in range(min_val, max_val)], axis = 0)
#    preds = np.concatenate([np.reshape(preds == i, [preds.size, 1]).astype(int) for i in range(min_val, max_val)], axis = 1)
#    return np.matmul(vals, preds)

#from DiagnosisModel import DiagnosisPrediction
def vals_preds(model, data, sess, normalise=False):
    feed_dict = data.feed_dict(model, 1.0, normalise)
    vals = np.reshape(feed_dict[model.targets], [-1, feed_dict[model.targets].shape[-1]])
    preds = sess.run(model.prediction, feed_dict)
    vals, preds = vals.astype(int), preds.astype(int)
    max_val = max(vals.shape[1], np.amax(preds))
    preds = np.concatenate([np.reshape(preds == i, [preds.size, 1]).astype(int) for i in range(max_val)], axis = 1)
    return vals, preds

def calc_cm(vals, preds):
    return np.matmul(vals.T, preds)

def print_cm(cm):
    if cm.dtype == 'float64':
        print('\n'.join(['\t'.join(['{:4f}'.format(item) for item in row])
            for row in cm]))
    else:
        print('\n'.join(['\t'.join(['{:4d}'.format(item) for item in row])
            for row in cm]))

def matthews_cc(cm):
    cm = cm.astype(float)
    mcc_num = np.prod(np.diag(cm)) - np.prod(np.diag(cm[[1, 0]]))
    mcc_denom = np.sqrt(np.prod([np.sum(cm[0]), np.sum(cm[1]), np.sum(cm[:,0]), np.sum(cm[:,1])]))
    if mcc_denom == 0:
        return 0
    else:
        return mcc_num / mcc_denom

def youdens_j(cm):
    cm = cm.astype(float)
    cm = cm / np.reshape(np.sum(cm, 1), [-1, 1])
    return (np.sum(np.diag(cm)) * 2 / float(cm.shape[0])) - 1

def f1(cm):
    cm = cm.astype(float)
    precision = cm[1, 1] / np.sum(cm[:, 1])
    recall = cm[1, 1] / np.sum(cm[1])
    return 2 * precision * recall / (precision + recall)

def clean_tf_output(output, seq_length):
    output = np.squeeze(output)
    return output[:seq_length].astype(float)

def list_mean(list_of_lists):
    sums = [np.sum(list_of_lists[i]) for i in range(len(list_of_lists))]
    nums = [len(list_of_lists[i]) for i in range(len(list_of_lists))]
    return np.sum(sums) / np.sum(nums)

def mean_error(incorrect, diag_pred, lengths = None):
    if diag_pred:
        return np.mean(incorrect)
    else:
        return list_mean([clean_tf_output(incorrect[i], lengths[i]) for i in range(len(incorrect))])

def target_labels(data, file_num, seq_length):
    return [data[file_num][i][1] for i in range(seq_length)]

def cross_cov(x, y):
    cx, cy = x - x.mean(0), y - y.mean(0)
    n = x.shape[0]
    return np.matmul(cx.T, cy)

def mcc(x, y):
    covxy = np.mean(np.diag(cross_cov(x, y)))
    covxx = np.mean(np.diag(cross_cov(x, x)))
    covyy = np.mean(np.diag(cross_cov(y, y)))
    denom = np.sqrt(covxx * covyy)
    if denom == 0:
        return 0
    else:
        return covxy / denom
