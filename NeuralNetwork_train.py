import pandas as pd
import numpy as np

from csv import reader
from math import exp
from random import random, randrange

import h5py
import matplotlib.pyplot as plt

from keras.layers import Dense
from keras.models import Sequential

from evaluation import acuracycheckl

# load a CSV file
def load_csv(filename):
    dataset = list()
    with open(filename,'r') as file:
        csv_reader = reader(file)
        for row in csv_reader:
            if not row:
                continue
            dataset.append(row)
    return dataset

def title_seperate(dataset):
    datasets = []
    columnnames = []
    columnnames.append(dataset[0])
    for i in range(len(dataset)):
        if(i == 0):
            continue
        datasets.append(dataset[i])
    return datasets,columnnames   

def predictwithann(filename) :
    # load and prepare data
    dataset = load_csv(filename)
    dataset,columnnames = title_seperate(dataset)
    dataset = np.array(dataset)

    # split into input (X) and output (Y) variables
    X = dataset[:,0:11]
    Y = dataset[:,11]

    #create model
    model = Sequential()
    model.add(Dense(22, input_dim=11, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))

    # Compile model
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

    # Fit the model
    history = model.fit(X, Y, epochs=350, batch_size=10)

    #save model
    model.save('models/my_model.h5')
    # evaluate the model
    scores = model.evaluate(X, Y)
    # calculate predictions
    predictions = model.predict(X)
    # round predictions
    rounded = [round(x[0]) for x in predictions]
    rounded = np.array(rounded) 
    rounded = rounded.astype(int)

    return rounded

df = pd.read_csv('datasets/redwine-train.csv')
dataset = np.array(df) #convert opend dataframe df to numpy array
wine_quality = dataset[:,11] #save wine quality result in to seperate list
wine_quality = wine_quality.astype(int)

finalresult = predictwithann('datasets/redwine-train.csv')

acuracycheckl(wine_quality  , finalresult)