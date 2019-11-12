import os,re,sys,dill as pickle, json,itertools
from sklearn.model_selection import train_test_split
from collections import Counter,defaultdict
from nltk.util import pad_sequence,bigrams,ngrams as ng,everygrams
from nltk.lm.preprocessing import pad_both_ends,flatten,padded_everygram_pipeline
from nltk.lm import MLE,Vocabulary,Lidstone,WittenBellInterpolated,KneserNeyInterpolated
from nltk.tokenize.treebank import TreebankWordDetokenizer
from nltk.lm.preprocessing import flatten
text=[["hello","world","my","name","is","khan"],["Please","accept","my","request"]]
test=[["hello","to","all","of","you"],["i","am","musa","hirapurawala"]]
n=[1,2,3]
def get_data(n,text):
	train_ngrams = [ng(t,n,  pad_right=True, pad_left=True, left_pad_symbol="<s>", right_pad_symbol="</s>") for t in text]
	words = [word for sent in text for word in sent]
	words.extend(["<s>", "</s>"])
	train_vocab = Vocabulary(words)
	#print(sorted(train_vocab))
	#train_vocab =flatten(pad_both_ends(sent, n=n) for sent in text)
	return train_ngrams,train_vocab
def train_lm_models(n,text):
	models={}
	discount=0.75
	gamma=0.5
	#train_ngrams,train_vocab = get_data(n,text)

	'''model4 = KneserNeyInterpolated(order=n,discount=discount)
	train_ngrams,train_vocab = get_data(n,text)
	model4.fit(train_ngrams,train_vocab)
	models["4"]=model4
	model3= WittenBellInterpolated(order=n)
	train_ngrams,train_vocab = get_data(n,text)
	model3.fit(train_ngrams,train_vocab)
	models["3"]=model3'''
	model2 = Lidstone(order=n,gamma=gamma)
	train_ngrams,train_vocab = get_data(n,text)
	model2.fit(train_ngrams,train_vocab)
	models["2"]=model2
	model1 = MLE(order=n)
	train_ngrams,train_vocab = get_data(n,text)
	model1.fit(train_ngrams,train_vocab)
	models["1"]=model1
	return models

for n in n:
	print("order: ",n)
	models = train_lm_models(n,text)
	#test_ngrams,_ = get_data(n,test)
	for model_name,model in models.items():
		print("model_name ",model_name)
		test_ngrams,_ = get_data(n,test)
		for ngrams in test_ngrams:
			s= model.perplexity(ngrams)
			print(s)
		#s/=l
		#print(s)
	print("\n\n")
