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

### Training
#### N-Gram Model
For ngram, use:
```console
./train.sh ngram <Path to data file> <Path to store trained model>
```
#### LSTM Model
For lstm, use:
```console
./train.sh lstm <Path to data file> <Path to store trained model>
```

### Text Generation
#### N-Gram Model

For unconditional ngram text generation, use:
```console
./predict.sh ngram <un>(unconditional flag) <no of words> <Path to use trained model>
```

For conditional ngram text generation, use:
```console
./predict.sh ngram <cn>(conditional flag) <no of words> <Path to use trained model> text
```
#### LSTM Model

For lstm unconditional text generation, use:
```console
./predict.sh lstm  <un>(unconditional flag) <no of words> <Path to use trained model>
```
For lstm conditional text generation,use:
``` console
./predict.sh lstm  <cn>(conditional flag) <no of words> <Path to use trained model> text
```

## Tuning Parameters
You can edit the parameters during training by specifying their values in the ```train``` file.

Parameters that can be edited for ngram model:
-  default: 300

Parameters that can be edited for lstm model:
- No of Hidden layers,dafault 1
- No of Nodes in each hidden layer,default 500
- Dropout Factor, 0.1
- No of Epochs, default 100
 

## Output
Trained models are stored in form of ```.pkl``` files in the ```model_data/``` folder.
Predicted text will be displayed in console.         

Download necessaary utils from https://iiitaphyd-my.sharepoint.com/:f:/g/personal/priyansh_agrawal_research_iiit_ac_in/EnV74VjDxMNNohD1LJ1QE5oB8eHv39TGAAhSAIXA1nC5mQ?e=CMuQyv
