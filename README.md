# Language Modelling

This application is an implementation of both the statistical (Ngram based) and deeplearning (LSTM RNN based) language models built for three social media channels, Facebook,Twitter and Instagram. The api has functionality of training , testing models and predicting from them. The prediction can be done in ```unconditioned``` and ```conditioned``` ways.    
 

## Requirements

Support for Python 3. Install the package requirements via
```console
pip install -r requirements.txt
```  
 
## Data
 
The preprocessed training data can be found in the ```data/preprocessed``` folder for different social media platforms by name ```tw``` for twitter, ```insta```for instagram and ```fb``` for facebook. The stats for preprocessed data for all three platforms are given below.
```
Twitter: 40.6 Lakhs sequences
Instagram: 1.38 Lakhs sequences including instagram image captions and comments 
FaceBook: 11.4 Lakhs sequences including facebook post and comments
``` 
The word "sequence" refers to 'tweets' for twitter, it can be post or a comment for facebook and similarly it can be caption or a comment for instagram.
 
## Usage

### 1. Training
Set the optional parameter values required for training in ```train.sh``` file.Default Values for parameters is present in Tuning Parameter Section.
```
1.1 N-Gram Model,use
    console
    ./train.sh ngram <Path to data file> <Path to store trained model>
   
1.2 LSTM Model,use:
     console
    ./train.sh lstm <Path to data file> <Path to store trained model>
```
   
### 2.  Text Generation
```
1 N-Gram Model
     1.1 For unconditional text generation, use:
        console
      ./predict.sh ngram -un(unconditional flag) <no of words> <Path to use trained model>
   
     1.2 For conditional text generation, use:
        console
      ./predict.sh ngram -cn(conditional flag) <no of words> <Path to use trained model> text
      
2. LSTM Model
      2.1. For lstm unconditional text generation, use:
         console
         ./predict.sh lstm -un(unconditional flag) <no of words> <Path to use trained model>
   
      2.2. For lstm conditional text generation,use:
         console
         ./predict.sh lstm -cn(conditional flag) <no of words> <Path to use trained model> text
  ```

## Tuning Parameters
```
1. NGram Model
   The following arguments for training are optional:
   -g         Lidstone factor/Gamma[0.5]
   -d         Kneserney discouting factor[0.75]
2. LSTM Model
   The following arguments for training are optional:
   -h         No of Hidden layers[1]
   -n         No of Nodes in each hidden layer[500]
   -d         Dropout Factor[0.1]
   -e         No of Epochs[100]
```
 

## Output
Trained models are stored in form of ```.pkl``` files in the ```model_data/``` folder.
Predicted text will be displayed in console.         

Download necessaary utils from https://iiitaphyd-my.sharepoint.com/:f:/g/personal/priyansh_agrawal_research_iiit_ac_in/EnV74VjDxMNNohD1LJ1QE5oB8eHv39TGAAhSAIXA1nC5mQ?e=CMuQyv
