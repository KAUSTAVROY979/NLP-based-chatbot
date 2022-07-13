import torch
import torch.nn as nn

#ANN
class NeuralNet(nn.Module):
    #input size is equal to the length of bag of words
    def __init__(self, input_size, hidden_size, num_classes):
        #invocation of parent class is necessary otherwise AttributeError
        #requires _modules attribute so super() required
        super(NeuralNet, self).__init__()
        #biasing is set to true by default
        
        
        #input layer of the module
        self.l1 = nn.Linear(input_size, hidden_size)
        #hidden layer  
        self.l2 = nn.Linear(hidden_size, hidden_size)
        #output layer 
        self.l3 = nn.Linear(hidden_size, num_classes)
        #activation function
        self.relu = nn.ReLU()
        #relu makes -ve values 0
        #value=  0  if x<0 else x
    #forward() provides computation performed at every call of the model
    #should be overridden by subclass
    def forward(self, x):
        out = self.l1(x)
        out = self.relu(out)
        out = self.l2(out)
        out = self.relu(out)
        out = self.l3(out)
        # no activation and no softmax at the end
        return out