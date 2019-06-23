import numpy as np
import jaconv
import time
import _pickle as cPickle

from beamsearch_lm.beam_search import BeamSearch
from beamsearch_lm.lm_models.n_gram_lm import NgramModel as LMModel
from sklearn.base import BaseEstimator, RegressorMixin
from beamsearch_lm.utils import calculate_accuracy, load_data_for_training_lm, load_data_from_ocr, preprocess_ocr_logit

default_classes = list(
        "()-.・0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ゙゚アイウエオカキクケコサシスセソタチツテトナニヌネノハヒフヘホマミムメモヤユヨ"
        "ラリルレロワンガギグゲゴザジズゼゾダヂヅデドバビブベボヴヷパピプペポ"
    )

default_train_lm_fn = "/home/vanph/Desktop/pets/lm/official/lm_data/normed_kana_names.txt"
default_predict_score_logit_fn = "/home/vanph/Desktop/pets/lm/official/ocr_logit_data/np_logit_field_7_ffg" # folder

class SklearnAutoCorrectWrapper(BaseEstimator, RegressorMixin):
    def __init__(self, softmax_theta=.2, beam_width=2, lm_factor=.4, topk=5, n_gram=4, classes=default_classes,
                 train_lm_fn=default_train_lm_fn, predict_score_logit_fn=default_predict_score_logit_fn):
        # parameters for pre-processing
        self.softmax_theta = softmax_theta

        # parameters for beam search
        self.beam_width = beam_width
        self.lm_factor = lm_factor
        self.topk = topk

        # parameters for language model training
        self.n_gram = n_gram
        self.train_lm_fn = train_lm_fn

        # general parameters
        self.classes = classes

        # for class
        self.lm_model = None
        self.beam_search = None
        self.predict_score_logit_fn = predict_score_logit_fn

    def fit(self, X, y=None):
        """
        :param X: just dummy data, do not use.
        :param y: y, correct answers
        :return:
        """

        """
        Follow this previous steps:
        1. Training LM again with the newer parameters
        """
        lm_model = LMModel(n=self.n_gram)
        train_sentences, train_charset = load_data_for_training_lm(self.train_lm_fn)

        print ("Number of training sentences:", len(train_sentences))
        print ("Number of unique characters:", len(train_charset))

        lm_model.train(train_sentences)
        self.lm_model = lm_model

        """
        2. Building beam-search with its corresponding parameters
        """
        self.beam_search = BeamSearch(self.beam_width, self.lm_factor, self.topk, self.classes, self.lm_model)

        # compulsory
        return self

    def pre_process(self, ctc_logit_matrix):
        return preprocess_ocr_logit(ctc_logit_matrix, self.softmax_theta)

    def predict(self, X, y=None):
        return self.beam_search.search(self.pre_process(X))

    def save(self, output_fn):
        ge_save_dct = {
            'classes': self.classes,
            'n_gram': self.n_gram,
            'topk': self.topk,
            'lm_factor':self.lm_factor,
            'beam_width':self.beam_width,
            'softmax_theta':self.softmax_theta,
        }

        lm_save_dct = {
            "n": self.lm_model.n,
            "all_ngrams": self.lm_model.all_ngrams,
            "eps": self.lm_model.eps,
        }

        bs_save_dct = {
            'beam_width':self.beam_width,
            'lm_factor': self.lm_factor,
            'topk':self.topk,
            'classes':self.classes,
        }

        save_dct = {
            'general': ge_save_dct,
            'lm': lm_save_dct,
            'bs': bs_save_dct
        }

        cPickle.dump(save_dct, open(output_fn, 'wb'))

    @staticmethod
    def load(saved_fn):
        save_dct = cPickle.load(open(saved_fn, "rb"))

        # restore
        ge_save_dct = save_dct['general']
        ge_save_dct['train_lm_fn'] = ''
        ge_save_dct['predict_score_logit_fn'] = ''

        model = SklearnAutoCorrectWrapper(**ge_save_dct)

        # restore lm
        lm_save_dct = save_dct['lm']
        lm_model = LMModel(lm_save_dct['n'], lm_save_dct['eps'])
        lm_model.all_ngrams = lm_save_dct['all_ngrams']

        model.lm_model = lm_model

        # restore bs
        bs_save_dct = save_dct['bs']
        bs_save_dct['lm_model'] = model.lm_model

        bs = BeamSearch(**bs_save_dct)
        model.beam_search = bs

        return model

    def score(self, X, y=None):
        """
        Follow this previous steps:
        2. Apply beam search with its corresponding parameters
        """
        mapping_inout = load_data_from_ocr(ocr_output_src_folder)

        preds = []
        lbls = []

        for input_fn, lbl in mapping_inout.items():
            ctc_logit_matrix = np.load(input_fn)

            predict = self.predict(ctc_logit_matrix)
            preds += [jaconv.normalize(predict)]
            lbls  += [jaconv.normalize(lbl)]

        """
        3. Return the accuracy by character, exact match
        """
        acc_by_char, em = calculate_accuracy(preds, lbls)

        return em

if __name__ == "__main__":
    # Test using with sklearn.GridSearch
    from sklearn.model_selection import GridSearchCV

    """
    1. Default parameter spaces. 
    """

    tuned_parameters = {
        "softmax_theta": [.2], #[.2, .25],
        "beam_width": [4], #[2, 3, 4 , 5],
        "lm_factor": [.4],#[.4, .35, .31],
        "topk": [5],#[5, 4, 6],
        "n_gram": [5],#[4, 3, 5],
        "classes":[default_classes],
        "train_lm_fn":[default_train_lm_fn],
    }

    """
    2. Search through spaces
    """
    cv =  [([0,0,0,0],[0])] #zip([0] * 10,[1] * 10) # just dummy data

    base_model = SklearnAutoCorrectWrapper()
    gs = GridSearchCV(base_model, param_grid=tuned_parameters, n_jobs=-1, cv=cv, verbose=2, refit=False)

    """
    3. Prepare input&output
    """
    ocr_output_src_folder = "/home/vanph/Desktop/pets/lm/official/ocr_logit_data/np_logit_field_7_ffg"
    mapping_inout = load_data_from_ocr(ocr_output_src_folder)

    X = []
    y = []
    for input_fn, lbl in mapping_inout.items():
        ctc_logit_matrix = np.load(input_fn)

        X += [ctc_logit_matrix]
        y += [jaconv.normalize(lbl)]

    gs.fit(X, y)

    print ("Best parameters:")

    print (gs.best_params_)
    print (gs.best_score_)

    """
    4. Refit and save model
    """
    from sklearn.externals import joblib

    lm_model = SklearnAutoCorrectWrapper(**gs.best_params_)
    lm_model.fit(X, y)

    lm_model.save("/home/vanph/Desktop/pets/lm/official/save/lm_model_wrapper_v2.pkl")

    lm_model = SklearnAutoCorrectWrapper.load("/home/vanph/Desktop/pets/lm/official/save/lm_model_wrapper_v2.pkl")

    mapping_inout = load_data_from_ocr(ocr_output_src_folder)

    preds = []
    lbls = []

    for input_fn, lbl in mapping_inout.items():
        print("processing:", input_fn)
        ctc_logit_matrix = np.load(input_fn)

        predict = lm_model.predict(ctc_logit_matrix)
        preds += [jaconv.normalize(predict)]
        lbls += [jaconv.normalize(lbl)]

    """
    3. Return the accuracy by character, exact match
    """
    acc_by_char, em = calculate_accuracy(preds, lbls)

    print(acc_by_char, em)