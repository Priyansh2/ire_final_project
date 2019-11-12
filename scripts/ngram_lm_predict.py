import os,re,sys,dill as pickle, json,itertools,string
from sklearn.model_selection import train_test_split
from collections import Counter,defaultdict
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


def get_models():
	print()
	print("Loading Language Models...\n")
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
	for model_type in models:
		print(ngram_map[int(model_type)].upper()+":")
		print("\n".join(list(models[model_type].keys())))
		print()
	return models

def get_model_data():
	data={}
	for file in os.listdir(MODEL_DATA_PATH):
		data[file.split(".")[0]]=load_model(os.path.join(MODEL_DATA_PATH,file))
	return data

def generate_sent(num_words, model,context=None ,random_seed=3):
	## for unconditioned_generation context = None
	#context: list of tokens
	content=[]
	tokens = model.generate(num_words,text_seed=context,random_seed=random_seed)
	print(tokens)
	for token in tokens:
		if token=='<s>':
			continue
		if token=="</s>":
			break
		content.append(token)
	return " ".join(content)


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

	model_name="3-mle-lid-0.5.pkl"
	model = load_model(os.path.join(MODEL_PATH,model_name))
	context=["how","are"]
	print(generate_sent(5,model,context=context))
if __name__=="__main__":
	main()