import argparse,time
from subprocess import call
import os,re,sys,dill as pickle, json,itertools,string,random
from sklearn.model_selection import train_test_split
from collections import Counter,defaultdict

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

def punctuations_removal(tokens,type_="str"):
	translator = str.maketrans(string.punctuation+'|'+'—'+'・'+'…'+'¿'+"।",' '*(len(string.punctuation)+6))
	if type_!="str":
		temp=[]
		for token in tokens:
			token = token.strip()
			if token:
				token = token.translate(translator).strip()
				if token:
					if len(list(token))==1:
						if token in ('a','e','i','o','u'):
							temp.append(token)
					else:
						temp.append(token)
		return temp
	else:
		return tokens.translate(translator)


def length_check(token_list):
	if len(token_list)>=10 and len(token_list)<=200:
		return True
	else:
		return False

def train_test_dev_split(data_path,output_path=None,forced_save=False):
	print("Splitting data into train,test and dev..")
	data_sents = []
	l=[]
	for line in open(data_path).read().split("\n"):
		line=line.strip()
		if line:
			tokens =punctuations_removal(line.split(),type_="list")
			l.append(len(tokens))
			data_sents.append(" ".join(tokens))
			#sys.exit()
	train_sents, test_sents = train_test_split(data_sents, test_size = 0.2)
	test_sents,dev_sents = train_test_split(test_sents,test_size=0.5)
	if output_path:
		p = os.path.join(output_path,"train_sents.pkl")
		if forced_save and os.path.exists(p):
			os.remove(p)
		if not os.path.exists(p):
			save_model(p,train_sents)
		p = os.path.join(output_path,"test_sents.pkl")
		if forced_save and os.path.exists(p):
			os.remove(p)
		if not os.path.exists(p):
			save_model(p,test_sents)
		p = os.path.join(output_path,"dev_sents.pkl")
		if forced_save and os.path.exists(p):
			os.remove(p)
		if not os.path.exists(p):
			save_model(p,dev_sents)
	print("Average seq. length: ",sum(l)/len(l)) # 60 for insta , 30 for fb, 14 for tw
	print("Seq. length distribution: ",Counter(l))
	print("Training sents: ",len(train_sents))
	print("Testing sents: ",len(test_sents))
	print("Development sents: ",len(dev_sents))

	return train_sents,test_sents,dev_sents

def main():
	DATA_TYPE="insta"
	overwrite=True
	data_dir= os.path.join("/".join(os.getcwd().split("/")[:-1]),"data/preprocessed_data/"+DATA_TYPE)
	remove_cmd = "cd "+data_dir+";rm -rf "+DATA_TYPE+"_data"
	create_cmd = "cd "+data_dir+";cat "+DATA_TYPE+"_* > "+DATA_TYPE+"_data"
	if DATA_TYPE in ("tw","fb"):
		call(remove_cmd, shell=True)
		call(create_cmd, shell=True)
	time.sleep(10)
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

	if os.path.exists(MODEL_DATA_PATH):
		if len(os.listdir(MODEL_DATA_PATH))==0 or overwrite:
			train_sents,test_sents,dev_sents = train_test_dev_split(DATA_PATH,MODEL_DATA_PATH,forced_save=True)
	else:
		os.makedirs(MODEL_DATA_PATH)
		train_sents,test_sents,dev_sents = train_test_dev_split(DATA_PATH,MODEL_DATA_PATH,forced_save=True)
	if DATA_TYPE in ("tw","fb"):
		call(remove_cmd, shell=True)

if __name__ == '__main__':
	main()