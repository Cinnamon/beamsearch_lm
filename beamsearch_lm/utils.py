import os
import json
import numpy as np
import editdistance

def softmax(X, theta = 1.0, axis = None):
    """
    Compute the softmax of each element along an axis of X.

    Parameters
    ----------
    X: ND-Array. Probably should be floats.
    theta (optional): float parameter, used as a multiplier
        prior to exponentiation. Default = 1.0
    axis (optional): axis to compute values along. Default is the
        first non-singleton axis.

    Returns an array the same size as X. The result will sum to 1
    along the specified axis.
    """

    # make X at least 2d
    y = np.atleast_2d(X)

    # find axis
    if axis is None:
        axis = next(j[0] for j in enumerate(y.shape) if j[1] > 1)

    # multiply y against the theta parameter,
    y = y * float(theta)

    # subtract the max for numerical stability
    y = y - np.expand_dims(np.max(y, axis = axis), axis)

    # exponentiate y
    y = np.exp(y)

    # take the sum along the specified axis
    ax_sum = np.expand_dims(np.sum(y, axis = axis), axis)

    # finally: divide elementwise
    p = y / ax_sum

    # flatten if X was 1D
    if len(X.shape) == 1: p = p.flatten()

    return p

def calculate_accuracy(preds, tgts):
    """
    :param pred: (b,w)
    :param tgt: (b,w)
    :return: score
    """

    total_char_error  = 0
    total_field_error = 0

    total_char  = 0 #sum([len(tgt) for tgt in tgts])
    total_field = len(tgts)

    for pred, tgt in zip(preds, tgts):
        # if print_something and pred != tgt:
        #     print ("pred:", pred, "tgt:", tgt)

        n_char_error = editdistance.eval(pred, tgt)
        total_char_error += n_char_error

        if pred != tgt: total_field_error += 1

        total_char += max(len(pred), len(tgt))

    return 1. - total_char_error / float(total_char), 1. - total_field_error / float(total_field)


def load_data_for_training_lm(txt_fn):
    lines = open(txt_fn, 'r', encoding='utf-8').readlines()
    results = []

    for line in lines:
        results += [line.strip()]

    charset = "".join(results).replace(" ","")
    charset = set(charset)

    return results, charset

def load_data_from_ocr(src_folder):
    json_lbl_fn = os.path.join(src_folder, "labels.json")
    json_lbl = json.load(open(json_lbl_fn, 'r'))

    mapping_dct = {}
    for fn, lbl in json_lbl.items():
        full_fn = os.path.join(src_folder, fn)
        mapping_dct[full_fn] = lbl.replace(" ","")

    return mapping_dct

# specific to model
def preprocess_ocr_logit(ctc_logit_matrix, softmax_theta):
    ctc_logit_matrix = softmax(ctc_logit_matrix, theta=softmax_theta, axis=1)

    # move ctc_blank to the end
    ctc_logit_matrix = np.hstack((ctc_logit_matrix[:, 1:], ctc_logit_matrix[:, 0:1]))

    return ctc_logit_matrix


