import argparse
import os,re,sys,dill as pickle, json,itertools,string,random
from sklearn.model_selection import train_test_split
from collections import Counter,defaultdict
from nltk import FreqDist,trigrams,ConditionalFreqDist,MLEProbDist,LidstoneProbDist,LaplaceProbDist,ELEProbDist,WittenBellProbDist,KneserNeyProbDist,ConditionalProbDist
from nltk.util import pad_sequence,bigrams,ngrams,everygrams
from nltk.lm.preprocessing import pad_both_ends,flatten,padded_everygram_pipeline
from nltk.lm import MLE,Vocabulary,Lidstone,WittenBellInterpolated,KneserNeyInterpolated
from nltk.tokenize.treebank import TreebankWordDetokenizer
detokenize = TreebankWordDetokenizer().detokenize
DATA_TYPE="insta"
if DATA_TYPE in ("tw","insta","fb"):
	DATA_PATH = "../data/preprocessed_data/"+DATA_TYPE
	for file in os.listdir(DATA_PATH):
		if file==DATA_TYPE+"_data":
			DATA_PATH=os.path.join(DATA_PATH,file)
	MODEL_PATH = "../lm_models/ngram_models/"+DATA_TYPE
	MODEL_DATA_PATH = "../model_data/"+DATA_TYPE
	for path in [MODEL_PATH,MODEL_DATA_PATH]:
		if not os.path.exists(path):
			os.makedirs(path)

else:
	raise Exception('DATA_TYPE incorrectly defined!')

def save_model(file_path,model):
	file = open(file_path,"wb")
	pickle.dump(model,file)
	file.close()

def load_model(file_path):
	file = open(file_path,"rb")
	model = pickle.load(file)
	file.close()
	return model

def write_in_file(filename,data):
	fd = open(filename,"w",encoding="utf-8")
	fd.write("\n".join(data))
	fd.close()

def get_model_data():
	data={}
	for file in os.listdir(MODEL_DATA_PATH):
		data[file.split(".")[0]]=load_model(os.path.join(MODEL_DATA_PATH,file))
	return data

def get_seed_words(n,cond_prob_dist,context=None):
	if context:
		context = context.split()
		if n==2:
			if len(context)>=1:
				return context[-1]
			else:
				raise Exception("Context length should be atleast 1.")
		elif n==3:
			if len(context)>=2:
				return context[-2],context[-1]
			else:
				raise Exception("Context length should be atleast 2.")
	else:
		if n==2:
			return random.choice(cond_prob_dist.conditions())
		elif n==3:
			return random.choice(cond_prob_dist.conditions())

def text_generation(n,num_words,cond_prob_dist,context=None):
	fl=0
	if context:
		history = context.split()[:-n+1]
		if history:
			str_=" ".join(history)
			str_+=" "
		else:
			fl=1
	else:
		fl=1
	if n==2:
		word = get_seed_words(n,cond_prob_dist,context)
		word = word.strip()
		if fl:
			str_=word
		else:
			str_+=word
		for i in range(num_words):
			word = cond_prob_dist[word].generate()
			str_+=" "+word.strip()
		print(str_)

	elif n==3:
		prev_word,word = get_seed_words(n,cond_prob_dist,context)
		prev_word=prev_word.strip()
		word = word.strip()
		if fl:
			str_=prev_word+" "+word
		else:
			str_+=prev_word+" "+word
		for i in range(num_words):
			next_word = cond_prob_dist[(prev_word,word)].generate()
			str_+=" "+next_word.strip()
			prev_word = word
			word = next_word
		print(str_)


def lm_model(n,train_sents):
	train_words=[]
	for sent in train_sents:
		for word in sent.split():
			train_words.append(word)
	if n==2:
		Ngrams = bigrams(train_words)
	if n==3:
		Ngrams = trigrams(train_words)
		condition_pairs = (((w0, w1), w2) for w0, w1, w2 in Ngrams)
		Ngrams = condition_pairs
	cond_freq_dist = ConditionalFreqDist(Ngrams)
	#p=os.path.join(MODEL_PATH,str(n)+"-cond_freq_dist.pkl")
	#cond_freq_dist = load_model(p)
	#save_model(p,cond_freq_dist)
	#print("Done!")
	cond_prob_dist = ConditionalProbDist(cond_freq_dist,LidstoneProbDist,gamma=0.5)
	p=os.path.join(MODEL_PATH,str(n)+"-cond_prob_dist-lid.pkl")
	save_model(p,cond_prob_dist)
	#cond_prob_dist = ConditionalProbDist(cond_freq_dist,MLEProbDist)
	#p=os.path.join(MODEL_PATH,str(n)+"-cond_prob_dist-mle.pkl")
	#cond_prob_dist = ConditionalProbDist(cond_freq_dist,WittenBellProbDist,bins=100000)
	#p=os.path.join(MODEL_PATH,str(n)+"-cond_prob_dist-wb.pkl")
	#cond_prob_dist = load_model(p)
	#save_model(p,cond_prob_dist)

	if n==4:
		cond_prob_dist = ConditionalProbDist(cond_freq_dist,KneserNeyProbDist,discount=0.75)
		p=os.path.join(MODEL_PATH,"ksn-cond_prob_dist.pkl")
		save_model(p,cond_prob_dist)

	#print("Done!")
	num_words = 1000
	#context="what are"
	context=None
	text_generation(n,num_words,cond_prob_dist,context=context)



def main():
	if os.path.exists(MODEL_DATA_PATH):
		if len(os.listdir(MODEL_DATA_PATH))==0:
			train_sents,test_sents,dev_sents = train_test_dev_split(DATA_PATH,MODEL_DATA_PATH)
		else:
			print("Loading data...")
			model_data = get_model_data()
			train_sents,test_sents,dev_sents = model_data["train_sents"],model_data["test_sents"],model_data["dev_sents"]
	else:
		os.makedirs(MODEL_DATA_PATH)
		train_sents,test_sents,dev_sents = train_test_dev_split(DATA_PATH,MODEL_DATA_PATH)
	print("Training sents: ",len(train_sents))
	print("Testing sents: ",len(test_sents))
	print("Development sents: ",len(dev_sents))
	N=[2]
	for n in N:
		lm_model(n,train_sents)

if __name__ == '__main__':
	main()