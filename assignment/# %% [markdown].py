
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.datasets import mnist

(x_train, y_train), (x_test, y_test) = mnist.load_data()
L=3 #is the no if layer

MAX_ITR=100000

LEARNING_RATE=0.5 #η 

NO_OF_INPUT=[784,128,64,10] # mem friendly input h1,h2 output

NO_OF_OUTPUT=10

BATCH_SIZE=10


def initialize_parameters(layer_dims, method="xavier"):
    """
    Initializes weights and biases for a fully connected neural network.
    
    Parameters:
    -----------
    layer_dims : list of int
        Sizes of each layer in the network. Example: [784, 128, 64, 10]
    method : str
        Initialization method: "xavier" or "he"
    
    Returns:
    --------
    weights : list of np.ndarray
        Weight matrices for each layer
    biases : list of np.ndarray
        Bias vectors for each layer
    """
    weights = [] #(128,784) , (64,128) ,(10,64)
    biases = [] #(764 x 1 , 128 x 1 , 64 x 1 , 10 x 1)
    
    for i in range(len(layer_dims)-1):
        n_in = layer_dims[i]
        n_out = layer_dims[i+1]
        
        if method == "xavier":
            W = np.random.randn(n_out, n_in) * np.sqrt(1.0 / n_out)
        elif method == "he":
            W = np.random.randn(n_out,n_in) * np.sqrt(2.0 / n_out)
        else:
            raise ValueError("Invalid method. Use 'xavier' or 'he'.")
        
        b = np.zeros((1, n_out))
        
        weights.append(W)
        biases.append(b)
    
    return weights, biases

weight,bias=initialize_parameters(NO_OF_INPUT)

print(weight[0].shape)
print(weight[1].shape)
print(weight[2].shape)
print(bias[0].shape)
print(bias[1].shape)
print(bias[2].shape)




input=[]

output=[]  

# print(output)


def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    s = sigmoid(x)
    return s * (1 - s)

def softmax(x):
    exp_x = np.exp(x - np.max(x))  # stability trick
    return exp_x / np.sum(exp_x, axis=0, keepdims=True)


def loss_fn(y_cal,y_org):
    return -1*np.sum(y_org * np.log(y_cal + 1e-9))

def eigen(output):
    ans=np.eye(10)[output]
    return ans

def forwardpropogation(input): #(10,784)
    activation,preactivation=[input],[]
    #(10,784)  ()
   
    for i in range(L):
        print("a,",activation[i].shape,"w,",weight[i].shape,"b,",bias[i].shape)
        preactivation.append(np.dot(weight[i],activation[-1].T) + bias[i].T)  # w0(128,784) , w1(64,128) ,w2(10,64) a0=(10,784) a1(10,128) a2(10,64)

        #preac (128,10) (64,10) (10,10)
        if i == L - 1:
            A = softmax(preactivation[i])  # output layer 
        else:
            A = sigmoid(preactivation[i])  # hidden layers (128,10)

        activation.append(A.T)

    return activation,preactivation # a0=(10,784) a1(10,128) a2(10,64) a3(10,10) #preac (128,10) (64,10) (10,10)


def backpropogation(activation,preactivation,y):

    dz=activation[-1]-y # foe the last layer  (10,10)
    grad_w=[None] * L
    grad_b=[None] * L
    for i in reversed (range(L)):
        A_prev=activation[i] # a3(10,10) a2(10,64)  a1(10,128) a0=(10,784)
        grad_b.append(dz)
        grad_w.append(np.dot(A_prev.T,dz))

        if i > 0:  # for hidden layers
            dA_prev = np.dot(weight[i].T, dZ)
            dZ = dA_prev * sigmoid_derivative(preactivation[i-1])
        
    return grad_w,grad_b


#---NAIN--
for epoch in range(MAX_ITR):
    idx=np.arange(len(x_train))
    np.random.shuffle(idx)

    x_train=x_train[idx] #(60000,28,28)
    y_train=y_train[idx] #(60000)


    for i in range(0,len(x_train),BATCH_SIZE):
        x_batch=x_train[i : i+BATCH_SIZE]
        x_batch = x_batch.reshape(x_batch.shape[0], -1) #(10,784)

        y_batch=eigen(y_train[i:i+BATCH_SIZE]) #(10,10)

        activation,preactivation=forwardpropogation(x_batch) ##a()  p()
        
        # loss=loss_fn(activation[-1],y_batch)

        dW,db=backpropogation(activation,preactivation,y_batch)

    


