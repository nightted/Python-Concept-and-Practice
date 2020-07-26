# -*- coding: utf-8 -*-

import numpy as np

class Cross_entropy():
    
    @staticmethod
    def forward(activa,y_data):
        
        return sum(np.na_to_num(y_data*np.log(activa)))
    
    @staticmethod
    def backward(activa,y_data):
        
        return y_data/activa  
    
#暴力轉換array to diag法XD
def ArrayToDiag(z):
    
    z = np.reshape(z,(1,-1))
    np.diag(z[0])
    
    return np.diag(z[0])


#Notice that z input is an numpy vector!!
#These function in class below out put a Matrix!!
class sigmoid_Mtx():
    
    @staticmethod
    def forward(z):
    
        sig = 1.0/(1.0 + np.exp(z))
        return sig # sigmoid
    
    @staticmethod
    def backward(z):
        
        sig_d = (1.0-1.0/(1.0 + np.exp(z)))*1.0/(1.0 + np.exp(z))
        return ArrayToDiag(sig_d) # sigmoid * (1-sigmoid)
    

class softmax_Mtx():
    
    @staticmethod
    def forward(z):
        
        soft = np.exp(z)/sum(np.exp(z)) 
        
        return soft
    
    @staticmethod
    def backward(z):   
        
        S = np.exp(z)/sum(np.exp(z))
        S_triangle = np.dot(S,-1.0*S.T)
        S_diag = ArrayToDiag(S) 
        
        S_softmax = S_triangle + S_diag
    
        return S_softmax
    
    
class RELU_Mtx():
    
    @staticmethod
    def forward(z):
        
        z = np.array([k if k[0] >= 0 else [0] for k in z.tolist()])
            
        return z
    
    @staticmethod
    def backward(z):   
        
        z = np.array([[1] if k[0] >= 0 else [0] for k in z.tolist()])
        
        return ArrayToDiag(z)
        



# Neural Net
class Network(object):


    def __init__(self,size,learning_rate,RegulizationConst,act_func = RELU_Mtx , final_act_func = softmax_Mtx ,  loss_func = Cross_entropy ):

        # The num of total layers includes the input neuron layer , so the total activation layer is  sizes - 1
        self.num_layers = len(size)
        self.size = size
        self.learning_rate = learning_rate
        self.RegulizationConst = RegulizationConst
        self.parameter_init()
        
        self.act_func = act_func # RELU
        self.final_act_func = final_act_func # softmax
        self.loss_func = loss_func # cross entropy
        
    def parameter_init(self):
        
        self.bias = [np.random.randn(i,1) for i in self.size[1:]]
        self.weights = [np.random.randn(j,k) for j ,k in zip(self.size[1:],self.size[:-1])]

    #Add layer in fornt of the final layer
    def AddingLayers(self,num_neuron):
        ''''
        code here!!!

        '''

    def feedforward(self, a ) :
        for i , j in zip(self.weights , self.bias):
            a = self.act_func(np.dot(a,i) + j)      
        return a
    
    def backprop(self, x_data , y_data):
        
        #initialize partial w,b matrix
        partial_W = [np.zeros(w.shape) for w in self.weights]
        partial_b = [np.zeros(b.shape) for b in self.bias]
        
        #Feedforward:
        Activation = [x_data] # storage for activation & z value for backprop use
        Zs = []
        a = x_data     
        cnt = 1
        for w , b in zip(self.weights,self.bias):

            #other layer of sigmoid
            if cnt != self.num_layers - 1:
                
                z = np.dot(w,a) + b
                Zs.append(z)
                a = self.act_func.forward(z)
                Activation.append(a)
                
                cnt +=1

            #Last layer of softmax  
            else :
                z = np.dot(w,a) + b
                Zs.append(z)
                a = self.final_act_func.forward(z)
                Activation.append(a)
        
        #Backward:
        partialError_act = self.loss_func.backward(Activation[-1], y_data)  
        delta = np.dot(self.final_act_func.backward(Zs[-1]),partialError_act)  # partial E / partial z vector _ l (delta l)
        partial_W[-1] = np.dot(delta,Activation[-2].T) # partial W = partial E / partial z _ l * (Activation_l-1).T
        partial_b[-1] = delta # partial b = partial E / partial z _ l 
        #print("partial soft:",self.final_act_func.backward(Zs[-1]),"partialError_act:",partialError_act,"delta:",delta,"partial W:",partial_W[-1])
        
        for i in range(2,self.num_layers):

            partialError_act = np.dot(self.weights[-i+1].T,delta) 
            delta = np.dot(self.act_func.backward(Zs[-i]),partialError_act)
            partial_W[-i] = np.dot(delta,Activation[-i-1].T) 
            partial_b[-i] = delta
        
        return partial_W , partial_b


    def mini_batch_update(self,learningRate=None,RegulizationConst=None,n,mini_batch):
        # mini_batch are datasets , EX: [(x1,y1),(x2,y2),(x3,y3)] => lists of tuple

        Total_partial_w = [np.zeros(w.shape) for w in self.weights]
        Total_partial_b = [np.zeros(b.shape) for b in self.bias]

        for mini_x , mini_y in mini_batch:

            partial_w , partial_b = self.backprop(mini_x,mini_y)
            Total_partial_w = [i + j for i,j in zip(Total_partial_w , partial_w)]
            Total_partial_b = [i + j for i,j in zip(Total_partial_b , partial_b)]

        learningRate = self.learning_rate
        RegulizationConst = self.RegulizationConst
        self.weights = [(1-RegulizationConst/n*learningRate)*w - learningRate/len(mini_batch)*par_w for par_w, w in zip(Total_partial_w, self.weights)]
        self.bias = [ b - learningRate/len(mini_batch)*par_b for par_b, b in zip(Total_partial_b, self.bias)]


    def SGD(self,batch_size,epoch,training_data):

        for epoch in range(epoch):

            for i in range(0,len(training_data),batch_size):

                train_batch = training_data[i:i+batch_size]
                self.mini_batch_update(train_batch)







'''         
x= np.array([[1.5],[2.3]])
y= np.array([[1.7],[2.1]])


a = Network([2,5,2,6,2])
W,B = a.backprop(x, y)
print("W:",W,"B:",B)
'''



'''   
@staticmethod
def sigmoid(z) :
    
    def drt(z):
        
        return np.diag((1.0-1.0/(1.0 + np.exp(z)))*1.0/(1.0 + np.exp(z))) # sigmoid * (1-sigmoid)
    
    
    return 1.0/(1.0 + np.exp(z))

@staticmethod
def softmax(z):
    
    def drt(z):
        
        S = z
        S_triangle = np.dot(S,-1.0*S.T)
        S_diag = np.diag(z.T.tolist()[0]) # 暴力轉換array to diag法XD
        S_softmax = S_triangle + S_diag
    
        return S_softmax
    
    return np.exp(z)/sum(np.exp(z)) 

@staticmethod
def RELU(z):
    
    def drt(z):
        
        z = [1 if ele > 0 else 0 for ele in z ]
        
        return 
        
    
    z = [ele if ele > 0 else 0 for ele in z ]
    
    return z 
    
'''      
    
    