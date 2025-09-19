'''
    Torjan Horse clasification 

    output two 
    1 is torjran
    2 not torjan


    alpha = 0.01
    activation function ReLU = { max(alpha * x, x)}

    output function :

    softmax()= e ^x/e^(sum_i(xi))

'''

import pandas as pd

dataset= pd.read_csv("./Trojan_Detection.csv")

dataset_size=dataset.shape[0]
dataset_feilds=dataset.shape[1]


# print(dataset)

train_size = int(dataset_size * 0.7)
test_size = dataset_size - train_size

x_train = dataset[0:train_size, 0:dataset_feilds - 1]
y_train = dataset[0:train_size, -1:]

x_test = dataset[train_size:, 0:dataset_feilds - 1]
y_test = dataset[train_size:, -1:]


# constants

L=3 #is the no if layer

MAX_ITR=1000

LEARNING_RATE=0.2 #η 

NO_OF_INPUT=[85,43,21,2] #  

NO_OF_OUTPUT=10

BATCH_SIZE=500


