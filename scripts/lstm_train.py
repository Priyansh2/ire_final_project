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
# from glove import Corpus, Glove
# import itertools
tokenizer = Tokenizer()

def training_glove_weights(filepath,modelpath):
    file = open(filepath,"r")
    data = file.readlines()
    corpus = Corpus()
    corpus.fit(data,window=10)
    glove = Glove(no_components=5, learning_rate=0.05)
    glove.fit(corpus.matrix, epochs=30, no_threads=4, verbose=True)
    glove.add_dictionary(corpus.dictionary)
    model_name = modelpath+"_glove.model"
    glove.save(model_name)

def getting_glove_weights(modelpath):
    model_name = modelpath+"_glove.model"
    glove_model=Glove.load(model_name)
    embedding_matrix = dict()
    embedding_matrix = glove_model.dictionary
    return embedding_matrix

def getting_pretrained_glove_weights():
    embeddings_index = dict()
    f = open('glove.6B/glove.6B.100d.txt')
    for line in f:
        values = line.split()
        word = values[0]
        coefs = np.asarray(values[1:], dtype='float32')
        embeddings_index[word] = coefs
    f.close()
    embedding_matrix = np.zeros((vocabulary_size, 100))
    for word, index in tokenizer.word_index.items():
        if index > vocabulary_size - 1:
            break
        else:
            embedding_vector = embeddings_index.get(word)
            if embedding_vector is not None:
                embedding_matrix[index] = embedding_vector
    return embedding_matrix

def create_model_glove(predictors,labels,max_sequence_len,total_words,embedding_matrix,h,n,lrr,dr,b,epochss,modelpath):
    input_len = max_sequence_len - 1
    model = Sequential()
    model.add(Embedding(total_words, 100, input_length = input_len , weights=[embedding_matrix], trainable=False))
    for i in range(h):
    	model.add(LSTM(n))
    model.add(Dropout(dr))
    model.add(Dense(total_words, activation='softmax'))
    opt_adam = optimizers.adam(lr=lrr)
    model.compile(loss='sparse_categorical_crossentropy',
        metrics=['sparse_categorical_accuracy'],optimizer=opt_adam)
    model.fit(predictors,labels,batch_size=b,epochs=epochss,verbose=1)
    print(model.summary)
    model_path = modelpath+".h5"
    model.save(model_path)
    # return model


def create_model(predictors,labels,max_sequence_len,total_words,h,n,lrr,dr,b,epochss,modelpath):
    input_len = max_sequence_len - 1
    print("input length" , input_len)
    model = Sequential()
    model.add(Embedding(total_words, 100 , input_length = input_len))
#     model.add(LSTM(512,input_shape = (None,input_len,100),return_sequences=True))
#     model.add(LSTM(512,input_shape = (None, 512)))
    # model.add(LSTM(150,return_sequences=True))
    for i in range(h):
    	model.add(LSTM(n))
    model.add(Dropout(dr))
    model.add(Dense(total_words, activation='softmax'))
    opt_adam = optimizers.adam(lr=lrr)
    # model.compile(loss='categorical_crossentropy',optimizer='adam')
    model.compile(loss='sparse_categorical_crossentropy',
        metrics=['sparse_categorical_accuracy'],optimizer=opt_adam)
    model.fit(predictors,labels,batch_size=b,epochs=epochss,verbose=1)
    print(model.summary())
    model_path = modelpath+".h5"
    model.save(model_path)
    # return model

def dataset_preparation(filepath,modelpath):
    file = open(filepath,"r")
    corpus = file.readlines()
    # corpus = data
    # corpus = []
    # for line in data:
    #     l = len(line.split(" "))
    #     if(l>150):
    #         line_temp = line.split(" ")
    #         n = l/150
    #         c = 0
    #         while(c!=n):
    #             if(c==n-1):
    #                 corpus.append(" ".join(line_temp[(c*150):]))
    #             else:
    #                 corpus.append(" ".join(line_temp[(c*150):(c*150)+150]))
    #             c += 1
    #     else:
    #         corpus.append(line)


    #     corpus = data.lower().split("\n")
    #     corpus = data.lower()
    tokenizer.fit_on_texts(corpus)
    total_words = len(tokenizer.word_index) + 1
    input_sequences = []
    count = 0
    for line in corpus:
        if(count<5):
            print(line)
        count += 1
        if((count%500)==0):
            print(count)
        token_list = tokenizer.texts_to_sequences([line])[0]
        for i in range(1,len(token_list)):
            n_gram_sequence = token_list[:i+1]
            input_sequences.append(n_gram_sequence)
    max_sequence_len = max([len(x) for x in input_sequences])
    input_sequences = pad_sequences(input_sequences,maxlen = max_sequence_len, padding = 'pre')
    input_sequences = np.array(input_sequences)
    # print("number of input sequences" , len(input_sequences))
    predictors = input_sequences[:,:-1]
    labels = input_sequences[:,-1]
    # labels = ku.to_categorical(labels,num_classes=total_words)
    # saving
    tokenizer_path = modelpath+".pickle"
    with open(tokenizer_path, 'wb') as handle:
        pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)
    return predictors,labels,max_sequence_len,total_words


parser = argparse.ArgumentParser()
parser.add_argument('-hl', help='Number of hidden layers', dest='h', default=1,type=int)
parser.add_argument('-n', help='Number of nodes in each hidden layer', dest='n', default=512, type=int)
parser.add_argument('-lr',help='Learning rate', dest='lr', default=0.001, type=float)
parser.add_argument('-dr',help='Dropout rate', dest='dr', default=0.1, type=float)
parser.add_argument('-e', help='Number of training epochs', dest='epochs', default=100, type=int)
parser.add_argument('-b', help='Batch size', dest='b', default=128, type=int)
parser.add_argument('-i', help='Training data file', dest='filepath')
parser.add_argument('-o', help='Output model file (e.g: /data/model1)', dest='modelpath')
parser.add_argument('-w', help='Word embedding: no/pre-glove/glove', dest='word_embed',default='no')
args = parser.parse_args()


predictors,labels,max_sequence_len,total_words = dataset_preparation(args.filepath,args.modelpath)
seq_len_file = args.modelpath+"_len.txt"
file = open(seq_len_file,"a")
file.write(str(max_sequence_len))
file.close()
if(args.word_embed == 'no'):
	create_model(predictors,labels,max_sequence_len,total_words,args.h,args.n,args.lr,args.dr,args.b,args.epochs,args.modelpath)
elif(args.word_embed == 'pre-glove'):
	embedding_matrix = getting_pretrained_glove_weights()
	create_model_glove(predictors,labels,max_sequence_len,total_words,embedding_matrix,args.h,args.n,args.lr,args.dr,args.b,args.epochs,args.modelpath)
else:
	training_glove_weights(args.filepath,args.modelpath)
	embedding_matrix = getting_glove_weights(args.modelpath)
	create_model_glove(predictors,labels,max_sequence_len,total_words,embedding_matrix,args.h,args.n,args.lr,args.dr,args.b,args.epochs,args.modelpath)

