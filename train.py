import numpy as np
import random
import json

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from nltk_utils import bag_of_words, tokenize, stem
from model import NeuralNet

with open('C:\\Users\\Kaustav Roy\\Desktop\\mini-project\\intents.json', 'r') as f:
    intents = json.load(f)

all_words = []
tags = []
xy = []
# loop through each sentence in our intents patterns
for intent in intents['intents']:
    tag = intent['tag']
    #print(tag)
    # add to tag list
    tags.append(tag)
    for pattern in intent['patterns']:
        # tokenize each word in the sentence
        w = tokenize(pattern)
        # add to our words list of all words in input pattern
        all_words.extend(w)
        # add to xy pair
        #print(f'{w}:{tag}\n')
        xy.append((w, tag))

# stem and lower each word
#list(string.punctuation)
ignore_words = ['!', '"', '#', '$', '%', '&', "'", '(', ')', '*', '+', ',', '-', '.', '/', ':', ';', '<', '=', '>', '?', '@', '[', '\\', ']', '^', '_', '`', '{', '|', '}', '~'] 
all_words = [stem(w) for w in all_words if w not in ignore_words]
# remove duplicates and sort
all_words = sorted(set(all_words))
#print(all_words)
tags = sorted(set(tags))
#print(tags)
"""
print(len(xy), "patterns")
print(len(tags), "tags:", tags)
print(len(all_words), "unique stemmed words:", all_words)
"""
# create training data
X_train = []
y_train = []
#pattern_sentence contains words of pattern field for the corresponding tag
for (pattern_sentence, tag) in xy:
    # X: bag of words for each pattern_sentence
    bag = bag_of_words(pattern_sentence, all_words)
    X_train.append(bag)
    # y: contains the indexes of the labels
    label = tags.index(tag)
    y_train.append(label)

#converting data into a 1-D array
X_train = np.array(X_train)
y_train = np.array(y_train)


# Hyper-parameters for the
num_epochs = 1000
batch_size = 8
learning_rate = 0.001
#length of bag of words
input_size = len(X_train[0])
#no. of neurons
hidden_size = 8
output_size = len(tags)
print(input_size, output_size)

class ChatDataset(Dataset):

    #defining the data for the dataset including length of the data
    def __init__(self):
        self.n_samples = len(X_train)
        self.x_data = X_train
        self.y_data = y_train

    # support indexing such that dataset[i] can be used to get i-th sample
    def __getitem__(self, index):
        return self.x_data[index], self.y_data[index]

    # we can call len(dataset) to return the size
    def __len__(self):
        return self.n_samples

dataset = ChatDataset()
#helps to create pipeline for loading data into the model
#also adds Additional features such as shuffling, Batch size of data etc.
train_loader = DataLoader(dataset=dataset,
                          batch_size=batch_size,
                          shuffle=True,
                          num_workers=0)
#selecting device for the model
device = torch.device('cpu')
#loading the model into the device
model = NeuralNet(input_size, hidden_size, output_size).to(device)

#calling CrossEntropyLoss function to calculate loss
criterion = nn.CrossEntropyLoss()
#implementing optimization by using adam algorithm
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
#for param in model.parameters():
#    print(type(param), param.size())

# Training the model
for epoch in range(num_epochs):
    for (words, labels) in train_loader:
        #loading words and labels into the device
        words = words.to(device)
        labels = labels.to(dtype=torch.long).to(device) #torch.long is equivalent to INT64
        
        #giving input to the model
        outputs = model(words)
        
        #calculating loss using CrossEntropyLoss
        loss = criterion(outputs, labels)
        #print(loss)
        
        #zero_grad() clears the gradient for each optimization
        optimizer.zero_grad()
        #does a backward propagation of the calculated losses
        loss.backward()
        #performing single step optimization on the basis of tthe backward propagation loss
        optimizer.step()
    #adding 1 because of zero based indexing of epoch in loop
    if (epoch+1) % 100 == 0:
        print (f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')


print(f'final loss: {loss.item():.4f}')

#saving the model into data.pth file
data = {
"model_state": model.state_dict(),
"input_size": input_size,
"hidden_size": hidden_size,
"output_size": output_size,
"all_words": all_words,
"tags": tags
}

FILE = "data.pth"
torch.save(data, FILE)

print(f'training complete. file saved to {FILE}')