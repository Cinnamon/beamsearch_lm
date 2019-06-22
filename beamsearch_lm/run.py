import os
import json
import numpy as np
import jaconv
import time
from beam_search import BeamSearch
from lm_models.n_gram_lm import NgramModel as LMModel
from utils import calculate_accuracy, load_data_for_training_lm, load_data_from_ocr, preprocess_ocr_logit

if __name__ == "__main__":
    lm_save_fn = "./save/lm_model_240k_account_name_katakana_field_7_ffg.pkl"

    """
    1. Training Language Model/ or load existing model if found
    """
    if not os.path.exists(lm_save_fn) and True:
        print ("training language model again ...")

        train_data_fn = "./lm_data/normed_kana_names.txt"
        n_gram = 5 #4 # use 3 for context, 1 for token

        lm_model = LMModel(n = n_gram)
        train_sentences, train_charset = load_data_for_training_lm(train_data_fn)

        # training
        lm_model.train(train_sentences)

        # saving
        lm_model.save(lm_save_fn)
    else:
        lm_model = LMModel.load(lm_save_fn)

    """
    2. prepare input, ctc logit matrix extracting from OCR models 
    """
    ocr_output_src_folder = "/home/vanph/Desktop/pets/lm/official/ocr_logit_data/np_logit_field_7_ffg"
    mapping_inout = load_data_from_ocr(ocr_output_src_folder)

    # parameters
    softmax_theta = .2

    beam_width = 4 #2
    lm_factor = 0.4
    topk = 5
    classes = list(
        "()-.・0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ゙゚アイウエオカキクケコサシスセソタチツテトナニヌネノハヒフヘホマミムメモヤユヨ"
        "ラリルレロワンガギグゲゴザジズゼゾダヂヅデドバビブベボヴヷパピプペポ"
    )

    beam_search = BeamSearch(beam_width, lm_factor, topk, classes, lm_model)
    mapping_inout_result = []

    s_time = time.time()
    for input_fn, lbl in mapping_inout.items():
        # prepare input
        ctc_logit_matrix = np.load(input_fn)
        ctc_logit_matrix = preprocess_ocr_logit(ctc_logit_matrix, softmax_theta)

        # search
        predict = beam_search.search(ctc_logit_matrix)

        # post-processing
        predict = jaconv.normalize(predict)
        lbl = jaconv.normalize(lbl)

        # track
        mapping_inout_result += [(predict, lbl)]

    print ("take time:", time.time() - s_time)

    """
    3. Calculating accuracy 
    """
    preds = [elem[0] for elem in mapping_inout_result]
    lbls  = [elem[1] for elem in mapping_inout_result]

    # Debug & Testing
    if False:
        for input_fn, pred, lbl in zip(list(mapping_inout.keys()), preds, lbls):
            print ("input_fn:", os.path.split(input_fn)[-1], ",pred:", pred, ",lbl:", lbl, "is_correct:", pred == lbl)

    acc_char, em = calculate_accuracy(preds, lbls)
    print ("acc by char:", acc_char)
    print ("em:", em)