# Language Modelling

This application is an implementation of both the statistical (Ngram based) and deeplearning (LSTM RNN based) language models built for three social media channels namely, Facebook,Twitter and Instagram. The api has functionality of training , testing models and predicting from them. By 'prediction' it means text generation using trained language model which can be done in ```unconditioned``` and ```conditioned``` ways. Read following sections to know more about it.    

## Requirements

Support for Python 3. Install the package requirements via
```console
pip install -r requirements.txt
```

## Data
 
The preprocessed data for each platforms are splitted into train-test-dev in ratio 8:1:1 using script ```train_test_dev_split.py``` found in ```scripts``` folder. The data is shuffled before making the splits. The data splits for various social platforms can be found under ```model_data/<type>``` where <type> is the directory name for specific social media platform which is as follows: ```tw``` for twitter, ```insta```for instagram and ```fb``` for facebook. 
The statistics for preprocessed data for all three platforms are given below.
```
Twitter: 40.6 Lakhs sequences
Instagram: 1.38 Lakhs sequences including instagram image captions and comments 
FaceBook: 11.4 Lakhs sequences including facebook post and comments
```
The word "sequence" refers to 'tweets' for twitter, it can be post or a comment for facebook and similarly it can be caption or a comment for instagram. The average length of sequences (in round figure) of twitter,facebook and instagram datasets are 14, 30 and 60 words respectively.

## Usage

### 1. Training

Set the various parameter values required for training in ```train.sh```. Default Values for parameters is present in Parameter Section and also set in 'train.sh' file. 
The format of training file can be both "pickled" and "text". If the format is pickled, then it would expect it to be a list of sequences where each sequence is of ```string``` format and if its in text format then it would expect one sequence per line or sequences separated by newline character.  
Moreover, the trianing data should be preprocessed beforehand. You can also use our preprocess script built for the purpose which is located in ```scripts/preprocess.py``` which would take raw text file and outputs preprocessed text file with each sequence in each line.
```
  1.1 For training N-Gram Model, use:
      ./train.sh ngram <Path to train data> <Path to store trained model>
      
    
  
  1.2 For training LSTM Model, use:
      ./train.sh lstm <Path to train data> <Path to store trained model>
     
```
### Parameters
``` 
1. NGram Model
   The following arguments for training are optional:
   -g         Lidstone paramter(Gamma)[0.5]
   -d         Kneserney discounting factor[0.75]
   -i         Training data file
   -pkl       Training data file format: 1-pickled,0-text[1]
   -o         Output model file (Ex: ../lm_models/ngram_models/tw/<model_name>)
   -t         N-gram model type: 1-MLE,2-Lidstone,3-Wittenbell,4-Kneserney[4]
   -n         N-gram model order: 1-unigram,2-bigram,3-trigram[3]


2. LSTM Model
   The following arguments for training are optional:
   -hl        No. of Hidden layers[1]
   -n         No. of Nodes in each hidden layer[500]
   -dr        Dropout Factor[0.1]
   -e         No. of Epochs[100]
   -lr        Learning rate[0.001]
   -b         Batch Size[128]
   -i         Training data file
   -o         Output model file (Ex: ../lm_models/lstm_models/tw/<model_name>)
   -w         Word embedding options:no/pre-glove/glove[no]
```
   
### 2. Predicting (Text Generation)

Set the various parameter values required for text generation in ```predict.sh```.Predicted text will be displayed in console along with the seed text (if provided)

1. N-Gram Model
```
  1.1 For unconditional text generation, use:
      ./predict.sh ngram 0 <no of words> <path of pre-trained model>
      
     
  
  1.2 For conditional text generation, use:
      ./predict.sh ngram 1 <no of words> <path of pre-trained model> text
      
     
```
2. LSTM Model
```
  2.1. For unconditional text generation, use:
       ./predict.sh lstm 0 <no of words> <path of pre-trained model>
    
   
  2.2. For conditional text generation,use:
       ./predict.sh lstm 1 <no of words> <path of pre-trained model> text
       
    

### Parameters

<no of words> No. of words to be generated
1             Flag for Conditional text generation
0             Flag for Unconditional text generation
text          Seed/Context sequence.Format: 'space separated tokens' string
 
```

## Output

Trained models of all three social platforms are stored in ```lm_models/<model_technique>/<type>```. Here '<model_technique>' is the type of technology used to build language model. If model is N-gram based then its value is ```ngram_models``` else ```lstm_models```. '<type>' is the directory name for specific social media platform which is as follows: ```tw``` for twitter, ```insta```for instagram and ```fb``` for facebook.
Each model has its own output format like model inside "ngram_models" are in ```.pkl``` where as models in "lstm_models" are in ```.hdf5``` format.

```NOTE: This git repository don't contain heavy files because of file-size limitation (max 100 MB is allowed) by github and will be available from below mentioned link. All the N-gram based models and training data of of all social platforms are present in "bzip2" compressed form. Originally they all were in pickled form (.pkl)
         

https://iiitaphyd-my.sharepoint.com/:f:/g/personal/priyansh_agrawal_research_iiit_ac_in/EnV74VjDxMNNohD1LJ1QE5oB8eHv39TGAAhSAIXA1nC5mQ?e=CMuQyv
```
