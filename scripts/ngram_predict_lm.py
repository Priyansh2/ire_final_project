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

def get_seed_words(n,cond_prob_dist,context=None):
	if context:
		context = context.split()
		if n==2:
			if len(context)>=1:
				return context[-1]
			else:
				raise Exception("Context sequence should have atleast 1 token!")
		elif n==3:
			if len(context)>=2:
				return context[-2],context[-1]
			else:
				raise Exception("Context sequence should have atleast 2 tokens!")
	else:
		if n==2:
			return random.choice(cond_prob_dist.conditions())
		elif n==3:
			return random.choice(cond_prob_dist.conditions())

def text_generation(num_words,cond_prob_dist,context=None):
	if isinstance(cond_prob_dist.conditions()[0],tuple):
		n = 3
	else:
		n=2
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

def unigram_text_generate(num_words, model,context=None ,random_seed=3):
	## for unconditioned_generation context = None
	#context: list of tokens
	content=[]
	tokens = model.generate(num_words,text_seed=context,random_seed=random_seed)
	for token in tokens:
		if token=='<s>':
			continue
		if token=="</s>":
			break
		content.append(token)
	print(" ".join(content))



def main():
	parser = argparse.ArgumentParser()
	parser.add_argument('-f',help='Text generation: 0 for unconditioned and 1 conditioned', dest='cflag',default=0,type=int)
	parser.add_argument('-m', help='Pre-trained model file (e.g: ../lm_models/ngram_models/tw/<model_file>', dest='model_path',default=None)
	parser.add_argument('-ksn',help='If N-gram model uses Kneserney smoothing then put 1 else 0',dest='ksn_model',default=0,type=int)
	parser.add_argument('-i', help='Seed/Context Sequence (eg: "my name is" ) of arbitrary length', dest='context_seq', default="",type=str)
	parser.add_argument('-l', help='No. of words to be generated after given seed/context sequence', dest='num_words', default=10,type=int)
	args = parser.parse_args()
	if len(sys.argv) == 1:
		parser.print_help()
		sys.exit()
	if args.model_path:
		model_path = args.model_path
	else:
		print("-m argument missing!")
		parser.print_help()
		sys.exit()
	model_name = model_path.split("/")[-1].split(".pkl")[0]
	if args.ksn_model==1:
		model_name = "Interpolated Kneserney"
	if "wb" in model_name:
		model_name = "Interpolated Wittenbell"
	if "lid" in model_name:
		model_name = "MLE Lidstone"
	if "mle" in model_name:
		model_name = "MLE"

	num_words = args.num_words
	context = args.context_seq.strip()
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
	if isinstance(context,list):
		print("Context Sequence: "," ".join(context))
	else:
		print("Context Sequence: ",context)
	print("Generating Text...\n")
	if args.cflag==1:
		if not context:
			raise Exception("Atleast one word is needed for generating text!")
	else:
		context=None

	if n==1:
		unigram_text_generate(num_words,model,context=context)
	else:
		if model_name == "Interpolated Kneserney":
			if args.cflag==1:
				context = tuple(context)
				if len(context)<n:
					raise Exception("Context sequence should have atleast "+n-1+"token!")
				print(model.conditioned_text_generate(context,num_words))
			else:
				print(model.unconditioned_text_generate(num_words))
		else:
			text_generation(num_words,model,context=context)

if __name__ == '__main__':
	main()