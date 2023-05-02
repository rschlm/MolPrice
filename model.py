# This is the code where we create the model

import numpy as np
import matplotlib.pyplot as plt
import torch
from torchvision import transforms, datasets
import sklearn
import pandas as pd


def choose_model(p, train_data_loader):
    """
    This function defines the architecture of the neural network depending on the desired width of the hidden layer:

    Arguments:
        - p: width of the hidden layer
        - train_data_loader: input data of the neural network, used to access the size of the input layer

    Returns:
        - neural_network: one-hidden-layer fully-connected neural network model with ReLu activation

    Comments:
        We can change the size of our hidden layer, the number of layers and the activation functions as well
    """

    input_size=train_data_loader.dataset[:][0].shape[1]
    hidden_layer=p
    output_size=1

    neural_network=torch.nn.Sequential(
        torch.nn.Linear(input_size,hidden_layer),
        torch.nn.ReLU(),
        torch.nn.Linear(hidden_layer,output_size)
    )

    return neural_network


def train_loop(train_data_loader, val_data_loader, model, epochs = 1000, learning_rate = 0.01, optimizer = 'Adam', loss = 'MSELoss'):
    """
    This function trains a neural network with SGD and its variants.
    Input: train_data_loader = iterable over the training set;
            val_data_loader = iterable over the validation set;
            model = neural network architecture with weights and biases;
            epochs = number of iterations of the algorithm
            learning_rate = size of gradient updates;
            optimizer = which variant of gradient descent algorithm to use;
            loss = loss function.
    Output: trained neural network model and metrics stored in *.csv file.
            val_loss= final loss on the validation set
    
    Comment: This function returns the validation loss in order to choose the learning rate of the model.
    """

    # select which optimizer and which loss function to use
    if loss == 'MSELoss':
        l = torch.nn.MSELoss()
    if optimizer == 'Adam':
        o = torch.optim.Adam(model.parameters(), lr=learning_rate)

    # extract the size of the training set
    size = len(train_data_loader.dataset)

    # initialize an empty dictionary
    res = {'epoch': [], 'train_loss': []}

    # loop over epochs
    for e in range(epochs):
        print(f"Epoch {e+1}\n-------------------------------")

        # initialize the variables for both train loss and accuracy  
        running_loss = 0
        
        # loop over all batches and over all input-output pair within a batch
        for batch, (X, y) in enumerate(train_data_loader): 

            o.zero_grad() # setting gradient to zeros       
            pred = model(X).squeeze() # get predictions through forward pass
            loss = l(pred, y) # compute the loss
            
            
            loss.backward() # backward propagation        
            
            o.step() # update the gradient to new gradients

        # compute the metrics of interest
            running_loss += loss.item()
            # check how the training is evolving accross the batches
            if batch % 5 == 0:
                current = (batch + 1) * len(X)
                print(f"Processing batch n. {batch+1} ----> running loss: {running_loss/(batch + 1):>7f},  [{current:>5d}/{size:>5d}]")
        
        running_loss = running_loss/(batch + 1)
        print("")
        print("running training loss =  ", running_loss)
        print("")

        res['epoch'].append(e) # populate the dictionary of results
        res['train_loss'].append(running_loss)

    print("Done!")

    res = pd.DataFrame.from_dict(res) # translate the dictionary into a pandas dataframe
    res.to_csv(f"./metrics_over_epochs_{learning_rate}.csv", mode = 'w', index = False) # store the results into a *.csv file
    torch.save(model.state_dict(), f'nn_{learning_rate}.pt')
    
    val_loss=l(model(val_data_loader.dataset[:][0]).squeeze(), val_data_loader.dataset[:][1])

    return val_loss

def train_loop_best_param(train_data_loader, val_data_loader, model, epochs = 1000, learning_rate = 0.001, optimizer = 'Adam', loss = 'MSELoss'):
    """
    This function is the same as previously but includes the validation set during the training.
    Input: train_data_loader = iterable over the training set;
            val_data_loader = iterable over the validation set;
            model = neural network architecture with weights and biases;
            epochs = number of iterations;
            learning_rate = size of gradient updates;
            optimizer = which variant of gradient descent algorithm to use;
            loss = loss function.
    Output: trained neural network model and metrics stored in *.csv file. 

    Comments: This function is used after optimization of the hyperparameters
    """

    # select which optimizer and which loss function to use
    if loss == 'MSELoss':
        l = torch.nn.MSELoss()
    if optimizer == 'Adam':
        o = torch.optim.Adam(model.parameters(), lr=learning_rate)

    # extract the size of the training set
    size = len(train_data_loader.dataset)

    # initialize an empty dictionary
    res = {'epoch': [], 'train_loss': [], 'val_loss': [] }

    # loop over epochs
    for e in range(epochs):
        print(f"Epoch {e+1}\n-------------------------------")

        # initialize the variables for both train loss and accuracy  
        running_loss = 0
        
        # loop over all batches and over all input-output pair within a batch
        for batch, (X, y) in enumerate(train_data_loader): 

            o.zero_grad() # setting gradient to zeros       
            pred = model(X).squeeze() # get predictions through forward pass
            loss = l(pred, y) # compute the loss
            
            
            
            loss.backward() # backward propagation        
            
            o.step() # update the gradient to new gradients

        # compute the metrics of interest
            running_loss += loss.item()
            # check how the training is evolving accross the batches
            if batch % 5 == 0:
                current = (batch + 1) * len(X)
                print(f"Processing batch n. {batch+1} ----> running loss: {running_loss/(batch + 1):>7f},  [{current:>5d}/{size:>5d}]")
        
        running_loss = running_loss/(batch + 1)
        print("")
        print("running training loss =  ", running_loss)
        print("")

        val_loss=float(l(model(val_data_loader.dataset[:][0]).squeeze(), val_data_loader.dataset[:][1])) # compute the loss for the validation set
        

        print(f'the validation loss is {val_loss}')
        
        res['epoch'].append(e) # populate the dictionary of results
        res['train_loss'].append(running_loss)
        res['val_loss'].append(val_loss)

    print("Done!")

    res = pd.DataFrame.from_dict(res) # translate the dictionary into a pandas dataframe
    res.to_csv(f"./metrics_over_epochs_best.csv", mode = 'w', index = False) # store the results into a *.csv file
    torch.save(model.state_dict(), f'nn_best.pt')

    return 