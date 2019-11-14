from nltk.corpus import gutenberg
from nltk.util import ngrams
from ksn_lm import KneserNeyLM
gut_ngrams = (ngram for sent in gutenberg.sents() for ngram in ngrams(sent, 3,pad_left=True, pad_right=True, right_pad_symbol='</s>', left_pad_symbol='<s>'))
lm = KneserNeyLM(3, gut_ngrams)
score = lm.score_sent(('This', 'is', 'a', 'sample', 'sentence', '.'))
print(score)
sent = lm.unconditioned_text_generate()
print(sent)
sent = lm.conditioned_text_generate(('This','is'))
print(sent)
print(lm.generate_next_word(('This','is'),lm.highest_order_probs()))