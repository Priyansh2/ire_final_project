import argparse
import os,re,sys,dill as pickle, json,itertools,string,random
from sklearn.model_selection import train_test_split
from collections import Counter,defaultdict
from nltk import FreqDist,trigrams,ConditionalFreqDist,MLEProbDist,LidstoneProbDist,LaplaceProbDist,ELEProbDist,WittenBellProbDist,KneserNeyProbDist,ConditionalProbDist
from nltk.util import pad_sequence,bigrams,ngrams,everygrams
from nltk.lm.preprocessing import pad_both_ends,flatten,padded_everygram_pipeline
from nltk.lm import MLE,Vocabulary,Lidstone,WittenBellInterpolated,KneserNeyInterpolated
from nltk.tokenize.treebank import TreebankWordDetokenizer
from ksn_lm import KneserNeyLM

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

def get_data_sents(train_data):
	lines= open(train_data,"r").read().split("\n")
	return [line.strip() for line in lines if line.strip()]

def get_sample_sents(train_data,k=2000000):
	return train_data[:k]


def get_ngram_prob(word,context,model):
	if len(context)==2:
		context = tuple(context)
	else:
		context = context[0]
	return model[context].prob(word)

def get_seq_prob(sent,n,model):
	prob_seq=0
	sent=sent.split()
	for i in range(n-1,len(sent)):
		ngram_prob = get_ngram_prob(sent[i],sent[i-n+1:i],model)
		#if not ngram_prob:
			#return float('-inf')
		try:
			log_prob = math.log(ngram_prob,2)
		except:
			log_prob = 0
		prob_seq+=log_prob

def calculate_perplexity(test_sents,model,n):
	cnt=0
	for sent in test_sents:
		cnt+=len(sent)
	sum_=0
	for sent in test_sents:
		sum_+=get_seq_prob(sent,n,model)/cnt
	return pow(2,-sum_)

def calculate_perplexity_ksn(test_sents,model):
	cnt=0
	for sent in test_sents:
		cnt+=len(sent.split())
	sum_=0
	for sent in test_sents:
		sum_+=model.score_sent(tuple(sent.split()))/cnt
	return pow(2,-sum_)

def main():
	parser = argparse.ArgumentParser()
	parser.add_argument('-m', help='Pre-trained model file (e.g: ../lm_models/ngram_models/tw/<model_file>', dest='model_path',default=None)
	parser.add_argument('-ksn',help='If N-gram model uses Kneserney smoothing then put 1 else 0',dest='ksn_model',default=0,type=int)
	parser.add_argument('-i', help='Preprocessed test data file (eg: ../model_data/tw/<data_in_specific_format>)', dest='te_data',default=None)
	parser.add_argument('-pkl', help='Test data file format: 0 for "text" format and 1 for "pickled" format', dest='te_data_format',default=1,type=int)

	args = parser.parse_args()
	if len(sys.argv) == 1:
		parser.print_help()
		sys.exit()
	if args.model_path and args.te_data:
		model_path = args.model_path
		test_data = args.te_data
	else:
		if not args.model_path:
			print("-m argument missing!")

		if not args.te_data:
			print("-i argument missing!")
		parser.print_help()
		sys.exit()
	if args.te_data_format:
		test_sents = load_model(test_data)
	else:
		test_sents = get_data_sents(test_data)
	print("Testing sents: ",len(test_sents))
	model_name = model_path.split("/")[-1].split(".pkl")[0]
	if args.ksn_model==1:
		model_name = "Interpolated Kneserney"
	if "wb" in model_name:
		model_name = "Interpolated Wittenbell"
	if "lid" in model_name:
		model_name = "MLE Lidstone"
	if "mle" in model_name:
		model_name = "MLE"
	print("Loading model...")
	model = load_model(model_path)
	print("Done!!")
	n=0
	if model_name!= "Interpolated Kneserney":
		try:
			cond = model.conditions()[0]
			if isinstance(cond,tuple):
				n = 3
			else:
				n=2
		except:
			n=1
			context = context.strip().split()
	else:
		n = model.highest_order
	print("Order of LM: ",n)
	print("Model Name: ",model_name)
	if n==2 or n==3:
		if model_name == "Interpolated Kneserney":
			perplexity = calculate_perplexity_ksn(test_sents,model)
		else:
			perplexity = calculate_perplexity(test_sents,model,n)
	print(perplexity)


if __name__ == '__main__':
	main()