import _pickle as cPickle
from math import exp, log

SOS_TOKEN = "<s>"
EOS_TOKEN = "</s>"

CONVERT_TABLE = {
    '゙' : ['゛','ﾞ'],
    '゚' : ['゜','ﾟ']
}

def tokenize(text):
    text = text.replace(" ", "")

    for _to, _froms in CONVERT_TABLE.items():
        for _from in _froms:
            if _from in text: text = text.replace(_from, _to)

    return list(text)

def ngrams(n, tokens):
    all_ngrams = []
    tokens.append(EOS_TOKEN)
    context = tuple(SOS_TOKEN for i in range(n - 1))

    for token in tokens:
        all_ngrams.append((context, token))
        if n != 1:
            context = context[n - (n - 1):] + (token,)

    return all_ngrams

class NgramModel(object):

    def __init__(self, n=2, eps=1e-6):

        self.n = n
        self.all_ngrams = {}
        self.eps = eps

    def update(self, sentence):

        tokens = tokenize(sentence)
        new_ngrams = ngrams(self.n, tokens)

        for context, token in new_ngrams:

            if context in self.all_ngrams:
                if token in self.all_ngrams[context]:
                    self.all_ngrams[context][token] += 1
                else:
                    self.all_ngrams[context][token] = 1
            else:
                self.all_ngrams[context] = {token: 1}

    def train(self, train_sentences):
        for train_sentence in train_sentences:
            self.update(train_sentence)

    def save(self, out_fn):
        cPickle.dump({
            "n" : self.n,
            "all_ngrams" : self.all_ngrams,
            "eps": self.eps
        }, open(out_fn, 'wb'))

    @staticmethod
    def load(save_fn):
        save_dct = cPickle.load(open(save_fn, 'rb'))
        model = NgramModel(save_dct['n'], save_dct['eps'])
        model.all_ngrams = save_dct['all_ngrams']

        return model

    def prob(self, context, token):

        if context not in self.all_ngrams:
            return 0.0

        if token not in self.all_ngrams[context]:
            return 0.0

        count = self.all_ngrams[context][token]
        total = sum(self.all_ngrams[context].values())

        return 1.0 * count / total

    def get_log_prob(self, sentence, next_char):
        token = next_char
        context_len = self.n - 1

        context = sentence[-context_len:]
        context = [SOS_TOKEN] * (context_len - len(context)) + list(context)
        context = tuple(context)

        return log(self.prob(context, token) + self.eps)

    def get_log_prob_of_EOS(self, sentence, eps=1e-6):
        token = EOS_TOKEN
        context_len = self.n - 1

        context = sentence[-context_len:]
        context = [SOS_TOKEN] * (context_len - len(context)) + list(context)
        context = tuple(context)

        return log(self.prob(context, token) + eps)

if __name__ == "__main__":
    """
    Test Tokenize
    """
    result = tokenize("カ)アスノシンタク タ゛イヒヨ")
    print (result)

    #exit()

    """
    Test N-gram
    """
    result = ngrams(4, tokenize("カ"))
    print (result)

    result = ngrams(4, tokenize("カ)アスノシンタク タ゛イヒヨ"))
    print (result)

    exit()

    """
    Test N-gram model
    """
    m = NgramModel(3)
    m.update("a b c d")
    m.update("a b a b")

    print (m.prob(context=('b', 'c'), token='d'))

    result = m.perplexity("a b")
    print (result)


