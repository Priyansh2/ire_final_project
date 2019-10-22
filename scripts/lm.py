import os,re,sys,dill as pickle, json,itertools
from sklearn.model_selection import train_test_split
from collections import Counter,defaultdict
from nltk.util import pad_sequence,bigrams,ngrams,everygrams
from nltk.lm.preprocessing import pad_both_ends,flatten,padded_everygram_pipeline
from nltk.lm import MLE,Vocabulary,Lidstone,WittenBellInterpolated,KneserNeyInterpolated
from nltk.tokenize.treebank import TreebankWordDetokenizer
detokenize = TreebankWordDetokenizer().detokenize
DATA_TYPE="fb"
if DATA_TYPE in ("tw","insta","fb"):
	DATA_PATH = "../data/preprocessed_data/"+DATA_TYPE
	for file in os.listdir(DATA_PATH):
		DATA_PATH=os.path.join(DATA_PATH,file)
	MODEL_PATH = "../lm_models/ngram_models/"+DATA_TYPE
	MODEL_DATA_PATH = "../model_data/"+DATA_TYPE
	for path in [MODEL_PATH,MODEL_DATA_PATH]:
		if not os.path.exists(path):
			os.makedirs(path)

else:
	raise Exception('DATA_TYPE incorrectly defined!')


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

def train_test_dev_split(data_path,output_path):
	print("Splitting data into train,test and dev..")
	data_sents = [line.strip().split() for line in open(data_path).read().split("\n") if line.strip()]
	train_sents, test_sents = train_test_split(data_sents, test_size = 0.2)
	test_sents,dev_sents = train_test_split(test_sents,test_size=0.5)
	p = os.path.join(output_path,"train_sents.pkl")
	if not os.path.exists(p):
		save_model(p,train_sents)
	p = os.path.join(output_path,"test_sents.pkl")
	if not os.path.exists(p):
		save_model(p,test_sents)
	p = os.path.join(output_path,"dev_sents.pkl")
	if not os.path.exists(p):
		save_model(p,dev_sents)
	return train_sents,test_sents,dev_sents

def get_model_data():
	data={}
	for file in os.listdir(MODEL_DATA_PATH):
		data[file.split(".")[0]]=load_model(os.path.join(MODEL_DATA_PATH,file))
	return data


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

def mle_model(n,train_data,forced_save):
	train_ngrams,train_vocab = train_data["ngrams"],train_data["vocab"]
	print("Training MLE model..")
	p=os.path.join(MODEL_PATH,str(n)+"-mle.pkl")
	if forced_save and os.path.exists(p):
		os.remove(p)
	if not os.path.exists(p):
		model = MLE(order=n)
		model.fit(train_ngrams,train_vocab)
		save_model(p,model)


def lid_model(n,train_data,gamma,forced_save):
	train_ngrams,train_vocab = train_data["ngrams"],train_data["vocab"]
	print("Training Lidstone model..")
	p=os.path.join(MODEL_PATH,str(n)+"-mle-lid-"+str(gamma)+".pkl")
	if forced_save and os.path.exists(p):
		os.remove(p)
	if not os.path.exists(p):
		model = Lidstone(order=n,gamma=gamma)
		model.fit(train_ngrams,train_vocab)
		save_model(p,model)


def wb_interp_model(n,train_data,forced_save):
	train_ngrams,train_vocab = train_data["ngrams"],train_data["vocab"]
	print("Training Interpolation model with Wittenbell smoothing..")
	p=os.path.join(MODEL_PATH,str(n)+"-interp-wb.pkl")
	if forced_save and os.path.exists(p):
		os.remove(p)
	if not os.path.exists(p):
		model = WittenBellInterpolated(order=n)
		model.fit(train_ngrams,train_vocab)
		save_model(p,model)


def ksn_interp_model(n,train_data,discount,forced_save):
	train_ngrams,train_vocab = train_data["ngrams"],train_data["vocab"]
	print("Training Interpolation model with KneserNey smoothing..")
	p=os.path.join(MODEL_PATH,str(n)+"-interp-ksn-"+str(discount)+".pkl")
	if forced_save and os.path.exists(p):
		os.remove(p)
	if not os.path.exists(p):
		model = KneserNeyInterpolated(order=n,discount=discount)
		model.fit(train_ngrams,train_vocab)
		save_model(p,model)

def get_models():
	models={"1":{},"2":{},"3":{}}
	for file in os.listdir(MODEL_PATH):
		if os.path.isfile(os.path.join(MODEL_PATH,file)):
			model_name = file.split(".pkl")[0]
			model = load_model(os.path.join(MODEL_PATH,file))
			if "1" in model_name:
				models["1"][model_name]=model
			elif "2" in model_name:
				models["2"][model_name]=model
			else:
				models["3"][model_name]=model
	return models

def train_language_models(N,train_sents,lm_types,forced_save):
	for n in N:
		train_ngrams,train_vocab = padded_everygram_pipeline(n, train_sents)
		train_data={"ngrams":train_ngrams,"vocab":train_vocab} ## train lm on it
		print("Building "+ngram_map[n].upper()+" MODEL...")
		for model_type,smoothing_type in lm_types.items():
			for smoothing in smoothing_type:
				language_model(train_data,n=n,model_type=model_type,smoothing_type=smoothing,forced_save=forced_save)


def generate_sent(num_words, context, model,random_seed=42):
	## for unconditioned_generation context = None
	#context: list of tokens
	content=[]
	for token in model.generate(num_words,context,random_seed=random_seed):
		if token=='<s>':
			continue
		if token=="</s>":
			break
		content.append(token)
	return " ".join(content)



def data_perplexity(n,test_ngrams,model):
	## data :- list of list type in which inner list's elements are tokens
	s=float(0)
	c=0
	for ngrams in test_ngrams:
		s=model.perplexity(ngrams)
		print(s)
		c+=1
	'''try:
		s/=c
	except ZeroDivisionError:
		raise Exception('Float division by zero! Cannot find Perplexity')
	return s'''

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
	lm_types={"mle":[None,"lid"],"interp":["wb","ksn"]} ## (ngram_model,smoothing) pairs
	N=[1,2,3]
	train_language_models(N,train_sents,lm_types,forced_save=True)
	#print()
	'''print("Loading Language Models...\n")
	models = get_models()
	for model_type in models:
		print(ngram_map[int(model_type)].upper()+":")
		print("\n".join(list(models[model_type].keys())))
		print()'''
	'''n=3
	model_name="3-mle.pkl"
	model = load_model(os.path.join(MODEL_PATH,model_name))
	test_sents =[ list(flatten(test_sents))]
	test_ngrams,_ = padded_everygram_pipeline(n,test_sents)
	data_perplexity(n,test_ngrams,model)'''

	'''for model_type in models: ## different model perplexity
		if model_type=="3":
			test_ngrams,test_vocab = padded_everygram_pipeline(int(model_type),test_sents)
			test_data={"ngrams":test_ngrams,"vocab":test_vocab} ## test lm on it
			for model_name in models[model_type]:
				print(model_type,model_name,data_perplexity(int(model_type),test_data["ngrams"],models[model_type][model_name]))'''
if __name__ == '__main__':
	main()