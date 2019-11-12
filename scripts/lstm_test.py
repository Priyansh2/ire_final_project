import argparse
from keras.preprocessing.sequence import pad_sequences
from keras.layers import Embedding, LSTM, Dense, Dropout
from keras.preprocessing.text import Tokenizer
from keras.callbacks import EarlyStopping
from keras.models import Sequential
import keras.utils as ku 
import numpy as np
import pickle
from keras import optimizers
from keras.models import load_model


parser = argparse.ArgumentParser()

parser.add_argument('-f',help='unconditioned[0]/conditioned[1]', dest='flag',default='0')
parser.add_argument('-m', help='Model Path with model name', dest='modelpath')
parser.add_argument('-i', help='Input Sequence', dest='input', default="")
parser.add_argument('-l', help='Number of words to be predicted', dest='length', default=1,type=int)

args = parser.parse_args()

def generate_model(input_words,no_of_next_words,max_sequence_len,model):
    for j in range(no_of_next_words):
        token_list = tokenizer.texts_to_sequences([input_words])[0]
        token_list = pad_sequences([token_list],maxlen = max_sequence_len-1,padding='pre')
        predicted = model.predict_classes(token_list,verbose = 0)
        output_word = ""
        for word,index in tokenizer.word_index.items():
            if(index == predicted):
                output_word = word
                break
        input_words += " "+output_word
    return input_words

tokenizer = Tokenizer()
seq_len_file = args.modelpath+"_len.txt"
tokenizer_file = args.modelpath+".pickle"
model_file = args.modelpath+".h5"

file = open(seq_len_file,"r")
max_sequence_len = int(file.read())
file.close()
with open(tokenizer_file, 'rb') as handle:
    tokenizer = pickle.load(handle)
model = load_model(model_file)
print(generate_model(args.input,args.length,max_sequence_len,model))


