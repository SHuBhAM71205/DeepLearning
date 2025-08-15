import numpy as np 

#global variables 
MAX_ITR=1000
NO_OF_LAYERS=3
NO_OF_OUTPUT=4
NO_OF_INPUT=4

OUTPUT_FUNCTION="SOFTMAX"

#=======================MATRIX OF THE WEIIGHT , BIAS , PREACTIVATION , ACTIVATION ================#

# the assumption is that the for each layer accept the last the no of the neuron is no_of Input
   
weights = [np.random.randn(NO_OF_INPUT, NO_OF_INPUT) * 0.01 for _ in range(NO_OF_LAYERS - 1)]

biases = [np.zeros((NO_OF_INPUT, 1)) for _ in range(NO_OF_LAYERS - 1)]

# Output layer weight: (NO_OF_OUTPUT, NO_OF_INPUT)

weights.append(np.random.randn(NO_OF_OUTPUT, NO_OF_INPUT) * 0.01)

biases.append(np.zeros((NO_OF_OUTPUT, 1)))

activation = []

preactivation = []


#========================FUNCTIONS===========================#
def lossFunction(): 
    """ 
        l(theta)= SUMATION(ACTUAL_DIST * PREDICTED_DIST)

        arguments:

        return:

    """
    
    return 

#----------------------------||---------------------#
def activationFunction(Layer_Num:int) -> list: # activate the waited sunmation given by the preActivation
    """
        activationFunction(x)=g(ai(x))
        g is any function
        the activation function is assumed to be the sigmoid function

        argumet : the layer number for which activation you want to count
                  [a1,a2,a3,a4,...,an]
        
        return : this depend on the function use

                Sigmoid → Output is in range (0,1) for each neuron.
                Tanh → Output is in range (-1,1).
                ReLU → Output is in range [0, ∞).
    """

    pass


def preActivationFunction(Layer_Num:int) -> list: #preactivationfun is use to do the weighted sum of the previous layer output which is activation(prev. layer)
    """
        it is the waited sum of the wait with the previous layer output 

        ai= (Wi*hi-1)  + bi

        argument : layer number
        
        return : vector 
                 [a1,a2,a3,.....,an] where n is the number of the neuron in  layer=Layer_Num
                 where the a1 is for the first neuron in layer
    """
    wL=weights[Layer_Num]
    bL=biases[Layer_Num]

    hL=activation[Layer_Num-1]

    aL=[]
    for x,y,z in zip(wL,bL,hL):
        aL.insert((x*z) + y)
    pass    


#--------------------------------------------------------------------------------------#
#it is use to do the forward calculation ie input give n output
def forwardpropogation()->list:
    for i in range(1,NO_OF_LAYERS):
        preActivationFunction(i)
        activationFunction(i)
    pass

#after predicting the value through the forwardpropogation we want to find the corectness then this is use to calculate the gradients
def backpropogation():
    
    pass



#=================================MAIN===================================================#
INPUT = np.array([[0.5], [0.2], [0.1], [0.7]])  # column vector (4 x 1)

OUTPUT = np.array()

activation.append(INPUT) 

count=0

while( count < MAX_ITR ):
    
    count+=1

