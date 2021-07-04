# Language Modelling
In this project, we have developed Ngram and RNN based language models using NLTK, Spacy, and Keras for three social media channels: Facebook, Twitter, and Instagram. The API provides command line for training, testing and predicting (text generation) from pre-trained models in both unconditioned and conditioned ways. Read following sections to know more about it.    

## Requirements

Support for Python 3. Install the package requirements with
```console
pip install -r requirements.txt
```

## Data

The preprocessed data for each platform are split into train-test-dev in ratio 8:1:1 using script ```train_test_dev_split.py``` present in the ```scripts``` folder. The data is shuffled before doing the splits. Divided for various social platforms are present in ```model_data/<social_type>``` where "<social_type>" is the directory name for a specific social media platform (```tw``` for Twitter, ```insta```for Instagram and ```fb``` for Facebook). 
The statistics for preprocessed data for all three platforms are mentioned below.
```
Twitter: 4.06 Million sequences
Instagram: 138 Thousand sequences including instagram image captions and comments 
FaceBook: 1.14 Million sequences including facebook post and comments
```
The word "sequence" refers to 'tweets' for Twitter, it can be a post or a comment for Facebook, and similarly, it can be a caption or a comment for Instagram. The average length of sequences (in the round figure) of Twitter, Facebook, and Instagram datasets are 14, 30, and 60 words, respectively.

## Usage

### 1. Training

Set the parameter values required for training in ```train.sh```. Default Values for parameters are present in the ```Parameters``` Section. The training data file can be in either "pickle" or "text" format. If the data is pickled, it would expect it to be a list of sequences; and if it is in text format, it would expect one sequence per line or sequences separated by the newline character. 
```Note:``` Before the training, data should be preprocessed. You can also use our preprocess script present in ```scripts/preprocess.py``` This script takes raw text files and outputs preprocessed text files with each sequence in each line. 
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

Set the parameter values required for text generation in ```predict.sh```.Predicted text will be displayed in console along with the seed text (if provided).

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

Trained models of all three social platforms are stored in ```lm_models/<model_technique>/<social_type>```.  Here "<model_technique>" is the type of technology used to build a language model. If the model is N-gram based, its value is ```ngram_models``` else ```lstm_models```. "<social_type>" is the directory name for a specific social media platform (```tw``` for Twitter, ```insta```for Instagram, and ```fb``` for Facebook).
Each model has its output format, such as model inside "ngram_models" are in ```.pkl``` whereas models in "lstm_models" are in ```.hdf5``` format.

```NOTE:```
This git repository doesn't contain any large files because of size limitations (max 100 MB is allowed) by Github and will be available from below mentioned link. All the N-gram-based models and training data of all social platforms are present in "bzip2" compressed form (.bz2). Initially, they all were in pickled form (.pkl)
        
```LINK:``` 
https://iiitaphyd-my.sharepoint.com/:f:/g/personal/priyansh_agrawal_research_iiit_ac_in/EnV74VjDxMNNohD1LJ1QE5oB8eHv39TGAAhSAIXA1nC5mQ?e=CMuQyv

