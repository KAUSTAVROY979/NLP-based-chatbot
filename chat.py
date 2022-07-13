import random
import json

import torch

from model import NeuralNet
from nltk_utils import bag_of_words, tokenize

device = torch.device('cpu')

with open('C:\\Users\\Kaustav Roy\\Desktop\\mini-project\\intents.json', 'r') as json_data:
    intents = json.load(json_data)

#loading model data from data.pth file 
FILE = "C:\\Users\\Kaustav Roy\\Desktop\\mini-project\\data.pth"
data = torch.load(FILE)
#extraction of  data from the file
input_size = data["input_size"]
hidden_size = data["hidden_size"]
output_size = data["output_size"]
all_words = data['all_words']
tags = data['tags']
model_state = data["model_state"]

#loading the model to device
model = NeuralNet(input_size, hidden_size, output_size).to(device)
#loading model parameters into the model
model.load_state_dict(model_state)
# sets the model to evaluation mode
model.eval()


bot_name = "Roy"

def get_response(msg):
    #tokenizing input msg
    sentence = tokenize(msg)
    #creating bag of words
    X = bag_of_words(sentence, all_words)
    #reshape function of numpy. 
    #reshapes the array into numpy tensor having 1 row and columns = X.shape[0]
    #x.shape[0] returns no. of columns or length of array
    X = X.reshape(1, X.shape[0])
    #loads the numpy tensor into device(cpu)
    X = torch.from_numpy(X).to(device)

    output = model(X)
    #returns the predicted index of the tag
    _, predicted = torch.max(output, dim=1)
    #extracting the tag using the predicted index
    tag = tags[predicted.item()]
    #normalisation of output using softmax function and calculating probability
    probs = torch.softmax(output, dim=1)
    #extracting the probability of the predicted item 
    prob = probs[0][predicted.item()]
    #if probability is > 75% then return response 
    if prob.item() > 0.75:
        for intent in intents['intents']:
            if tag == intent["tag"]:
                return random.choice(intent['responses'])
    
    return "Sorry I can't answer that......."
#print("Welcome To Roy's Kitchen(Type 'exit' or 'quit' to exit)")
#while True:
    # sentence = "do you use credit cards?"
"""  
if __name__ == "__main__":
    x = get_response("hi")
    print(x)
""" 