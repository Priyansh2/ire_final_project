# Language Modelling

This application is an implementation of both the ngram and lstm models for training and testing data from social media channels(Facebook,Twitter and Instagram).
 

## Requirements

Support for Python 3. Install the package requirements via
```console
pip install -r requirements.txt
```  
 
## Data
 
The training data can be found in the ```data/``` folder.
Dataset Used
```
Twitter: 40Lakhs Tweets
Instagram: 1.38 Lakh Caption
FaceBook: 3.6 Lakh Post
``` 
 
 
## Usage

For training, use the train script.
For ngram, use:
```console
./train ngram
```

For lstm, use:
```console
./train lstm
```

For testing, use the predict script.
For ngram, use:
```console
./predict ngram
```

For lstm, use:
For unconditional text generation:
```console
   ./predict lstm <datatype>(fb/twitter/instagram) <un>(unconditional flag)
```
For conditional text generation:
``` console
    ./predict lstm <datatype>(fb/twitter/instagram) <cn> text
```

## Tuning Parameters
You can edit the parameters during training by specifying their values in the ```train``` file.
Parameters that can be edited for ngram model:
- , default: 300

Parameters that can be edited for lstm model:
- No of Hidden layers,dafault 1
- No of Nodes in each hidden layer,default 500
- Dropout Factor, 0.1
- No of Epochs, default 100
 

## Output
Trained models are stored in form of ```.pkl``` files in the ```model_data/``` folder.
Predicted text will be displayed in console.         

Download necessaary utils from https://iiitaphyd-my.sharepoint.com/:f:/g/personal/priyansh_agrawal_research_iiit_ac_in/EnV74VjDxMNNohD1LJ1QE5oB8eHv39TGAAhSAIXA1nC5mQ?e=CMuQyv
