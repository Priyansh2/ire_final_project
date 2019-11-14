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
detokenize = TreebankWordDetokenizer().detokenize
MODEL_PATH=""
ngram_map={1:"unigram",2:"bigram",3:"trigram"}

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

def mle_model(n,train_data,forced_save):
	train_ngrams,train_vocab = train_data["ngrams"],train_data["vocab"]
	print("Training MLE model..")
	p = MODEL_PATH+".pkl"
	if forced_save and os.path.exists(p):
		os.remove(p)
	if not os.path.exists(p):
		if n==1:
			model = MLE(order=n)
			model.fit(train_ngrams,train_vocab)
		else:
			cond_freq_dist = ConditionalFreqDist(train_ngrams)
			cond_prob_dist = ConditionalProbDist(cond_freq_dist,MLEProbDist)
			model = cond_prob_dist
		save_model(p,model)


def lid_model(n,train_data,gamma,forced_save):
	train_ngrams,train_vocab = train_data["ngrams"],train_data["vocab"]
	print("Training Lidstone model..")
	p=MODEL_PATH+".pkl"
	if forced_save and os.path.exists(p):
		os.remove(p)
	if not os.path.exists(p):
		if n==1:
			model = Lidstone(order=n,gamma=gamma)
			model.fit(train_ngrams,train_vocab)
		else:
			cond_freq_dist = ConditionalFreqDist(train_ngrams)
			cond_prob_dist = ConditionalProbDist(cond_freq_dist,LidstoneProbDist,gamma=0.5)
			model = cond_prob_dist
		save_model(p,model)


def wb_interp_model(n,train_data,forced_save):
	train_ngrams,train_vocab = train_data["ngrams"],train_data["vocab"]
	print("Training Interpolation model with Wittenbell smoothing..")
	p=MODEL_PATH+".pkl"
	if forced_save and os.path.exists(p):
		os.remove(p)
	if not os.path.exists(p):
		if n==1:
			model = WittenBellInterpolated(order=n)
			model.fit(train_ngrams,train_vocab)
		else:
			cond_freq_dist = ConditionalFreqDist(train_ngrams)
			cond_prob_dist = ConditionalProbDist(cond_freq_dist,WittenBellProbDist,bins=100000)
			model = cond_prob_dist
		save_model(p,model)


def ksn_interp_model(n,train_data,discount,forced_save):
	train_ngrams,train_vocab = train_data["ngrams"],train_data["vocab"]
	print("Training Interpolation model with KneserNey smoothing..")
	p=MODEL_PATH+".pkl"
	if forced_save and os.path.exists(p):
		os.remove(p)
	if not os.path.exists(p):
		if n==1:
			model = KneserNeyInterpolated(order=n,discount=discount)
			model.fit(train_ngrams,train_vocab)
		else:
			model = KneserNeyLM(n, train_ngrams)
		save_model(p,model)

def language_model(train_data,n=3,model_type="mle",smoothing_type=None,params={"discount_factor":0.75,"gamma":0.5},forced_save=False):
	## n = 1,2,3
	## model_type = mle,interp (maximum likelihood estimation, interpolation)
	## smoothing_type = lid,wb,ksn (lidstone,wittenbell, kneserney)
	#NOTE :- lidstone model take 'gamma' param which is in between 0 and 1 but !=0.5. If 0 then model is MLE, if 1 then model is laplace, if 0.5 then ELE (expected liklihood estimation)
	#TODO :- kbo (katz backoff) model and gt (goodturing) smoothing
	if n not in (1,2,3):
		raise Exception('Value of order not in range!')
	if not train_data:
		raise Exception('Missing data!')
	if "ngrams" not in train_data or not train_data["ngrams"]:
		raise Exception('Missing train ngrams!')
	if n==1:
		if "vocab" not in train_data or not train_data["vocab"]:
			raise Exception("Missing train vocabulary!")

	if model_type not in ("mle","interp"):
		raise Exception('Type of model is incorrect!')
	if smoothing_type not in ("lid","wb","ksn",None):
		raise Exception("Type of smoothing is incorrect!")
	if not params:
		raise Exception("Missing Parameter!")
	if model_type=="mle":
		if not smoothing_type:
			mle_model(n,train_data,forced_save)
		elif smoothing_type=="lid":
			if "gamma" not in params:
				raise Exception('Missing Parameter: gamma')
			lid_model(n,train_data,params["gamma"],forced_save)
	elif model_type=="interp":
		if smoothing_type=="wb":
			wb_interp_model(n,train_data,forced_save)
		elif smoothing_type=="ksn":
			if "discount_factor" not in params:
				raise Exception('Missing parameter: discount_factor')
			ksn_interp_model(n,train_data,params["discount_factor"],forced_save)


def train_language_models(N,train_sents,lm_types,params,forced_save=False):
	temp = [sent.split() for sent in train_sents] ## tokenized_train_data
	train_vocab=[]
	for n in N:
		print("Building "+ngram_map[n].upper()+" MODEL...")
		for model_type,smoothing_type in lm_types.items():
			for smoothing in smoothing_type:
				if smoothing=="ksn":
					train_ngrams = (ngram for sent in temp for ngram in ngrams(sent, n,pad_left=True, pad_right=True, right_pad_symbol='</s>', left_pad_symbol='<s>'))
				else:
					if n==1:
						train_ngrams,train_vocab = padded_everygram_pipeline(n,temp)
					else:
						train_ngrams = ngrams([word for sent in temp for word in sent],n) ## conditional pairs of ngrams
						if n==3:
							train_ngrams = (((w0, w1), w2) for w0, w1, w2 in train_ngrams)
				train_data={"ngrams":train_ngrams,"vocab":train_vocab}
				language_model(train_data,n=n,model_type=model_type,smoothing_type=smoothing,params=params,forced_save=forced_save)

def get_data_sents(train_data):
	lines= open(train_data,"r").read().split("\n")
	return [line.strip() for line in lines if line.strip()]

def get_sample_sents(train_data,k=2000000):
	return train_data[:k]

def main():
	global MODEL_PATH
	parser = argparse.ArgumentParser()
	parser.add_argument('-i', help='Preprocessed Training data file (eg: ../model_data/tw/<data_in_specific_format>)', dest='tr_data',default=None)
	parser.add_argument('-pkl', help='Training data file format: 0 for "text" format and 1 for "pickled" format', dest='tr_data_format',default=1,type=int)
	parser.add_argument('-o', help='Output model file (e.g: ../lm_models/ngram_models/tw/<model_name_without_extension>)', dest='model_path',default=None)
	parser.add_argument('-g', help='Lidstone Parameter: 0<=g<=1', dest='g',default=0.5,type=int)
	parser.add_argument('-d', help='Kneserney Discount Factor: 0<=d<=1', dest='d',default=0.75,type=int)
	parser.add_argument('-t', help='Ngram model type (e.g: 1): 1 for MLE, 2 for MLE Lidstone, 3 for Interpolated Wittenbell, 4 for Interpolated Kneserney', dest='model_type',default=4,type=int)
	parser.add_argument('-n', help='Ngram order (e.g: 1): 1 for unigram, 2 for bigram, 3 for trigram', dest='model_order',default=3,type=int)
	args = parser.parse_args()
	if len(sys.argv) == 1:
		parser.print_help()
		sys.exit()
	model_map_={1:"mle",2:"mle",3:"interp",4:"interp"}
	if args.model_path and args.tr_data:
		MODEL_PATH = args.model_path
		train_data = args.tr_data
	else:
		if not args.model_path:
			print("-o argument missing!")

		if not args.tr_data:
			print("-i argument missing!")
		parser.print_help()
		sys.exit()
	smoothing_params = {"discount_factor":args.d,"gamma":args.g}
	N = [args.model_order]
	if args.tr_data_format:
		train_sents = load_model(train_data)
	else:
		train_sents = get_data_sents(train_data)
	if args.model_type==3:
		smoothing_type="wb"
	elif args.model_type==4:
		smoothing_type="ksn"
	elif args.model_type==2:
		if smoothing_params['gamma']!=0:
			smoothing_type="lid"
		else:
			raise Exception('Value of g cant be zero with Lidstone smoothing!')
	else:
		smoothing_type=None

	lm_types ={model_map_[args.model_type]:[smoothing_type]}
	if "tw" in MODEL_PATH:
		train_sents = get_sample_sents(train_sents)
	print("Training sents: ",len(train_sents))
	train_language_models(N,train_sents,lm_types,smoothing_params,forced_save=True)
	print("Done!")

if __name__ == '__main__':
	main()