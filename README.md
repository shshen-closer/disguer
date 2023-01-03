# KeyCoin@USTC solution for Global Knowledge Tracing Challenge @AAAI2023

## Dependencies:

- python >= 3.8  (or >= 3.8.0 )
- tensorflow-gpu == 2.5.0  (or >= 2.5.0 ) 
- numpy
- tqdm
- utils
- pandas
- sklearn
- any other necessary package
## Hardware:
A GPU with more than 10G memory

##Note: the whole training and testing process takes about one and a half hours

## Before

First, put the datasets (i.e., 'keyid2idx.json', 'pykt_test.csv', 'questions.json', 'train_valid_sequences.csv') in the data file.

Then, make the results folder for saving predictions:

`mkdir results`


Then, training the model:

`python train_all.py`

Then, making predictions:

`python test_all.py`

`python meaning.py`

**The output prediction.csv is our result**
