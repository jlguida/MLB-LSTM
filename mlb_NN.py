import tensorflow as tf
import numpy as np 
import pandas as pd
import io
import re
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Embedding
from keras.layers import LSTM
from keras.layers import Flatten
from keras.layers import Dropout
from keras.layers import TimeDistributed
from keras.layers import Activation
from keras.models import Model


from io import StringIO
import csv
import urllib.request

# fix random seed for reproducibility
np.random.seed(7)
# define validation size, this equates to number of seasons
VALIDATION_SIZE = 3
HIDDEN_SIZE = 162
BATCH_SIZE = 486
NUM_STEPS = 3


def run():

    ethier=np.zeros([12,162,26])
    for x in range(12):
        year = 2006+x
        print("Loading season " + str(year))
        file='ethier'+str(year)+'.csv'
        ethier[x]=parse_PlayerSeason(file, year)
    
    #Training instances
    X_train=ethier[:-VALIDATION_SIZE-1]
    X_train = X_train.reshape((1296,26))
    X_train = np.nan_to_num(X_train)
    print(X_train[:35])
    V=ethier[-VALIDATION_SIZE:]
    V=V.reshape((486,26))
    V=np.nan_to_num(V)
    Y=ethier[1:-VALIDATION_SIZE]
    Y = Y.reshape((1296,26))
    Y = np.nan_to_num(Y)

    #Testing instances for training
    Y_train=np.zeros([1296])
    i = 0
    for row in Y:
        Y_train[i]=row[3]
        i = i+1
    Y_train = Y_train.reshape((1296,))

    #Validation instances
    V_data=ethier[-VALIDATION_SIZE-1:-1]
    V_data = V_data.reshape((486,26))

    #Validation testing data
    V_test = np.zeros([486])
    for i,row in enumerate(V):
        V_test[i]=row[3]
    V_test = V_test.reshape((486,))
    print(X_train.shape)
    print(Y_train.shape)
    print(V_data.shape)
    print(V_test.shape)
    #Build model !!!
    model=Sequential()
    model.add(Dense(50, input_dim=26, activation='relu'))
    model.add(Dense(12, activation='relu'))
    model.add(Dense(1, activation='linear'))
    layer_name = 0
    
    #model.add(LSTM(units=HIDDEN_SIZE, unroll=True, input_shape=(162,50,)))
    #model.add(LSTM(units=HIDDEN_SIZE, unroll=True, dropout=0.5,input_shape=(162,50,)))
    #model.add(TimeDistributed(Dense(HIDDEN_SIZE)))
    #model.add(Activation('softmax'))

    #Train the model
    model.compile(loss='mse', optimizer='adam', metrics=['mean_squared_error'])
    print(model.layers[0].output)
    print(model.layers[1].output)
    print(model.layers[2].output)
    model.fit(X_train, Y_train, epochs=150, batch_size=BATCH_SIZE)
    intermediate_layer_model = Model(inputs=model.input,
                                 outputs=model.layers[layer_name].output)
    intermediate_output = intermediate_layer_model.predict(V_data)
    #print (intermediate_output)
    intermediate_layer_model = Model(inputs=model.input,
                                 outputs=model.layers[1].output)
    intermediate_output = intermediate_layer_model.predict(V_data)
    #print (intermediate_output)
    intermediate_layer_model = Model(inputs=model.input,
                                 outputs=model.layers[2].output)
    intermediate_output = intermediate_layer_model.predict(V_data)
    #print (intermediate_output)
    print(model.summary())
    scores = model.evaluate(V_data, V_test)
    print("\n%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))
    #predictions = model.predict(V_data)
    #print(predictions)
    




##############################################################################################
#INPUT: CSV filename for a single season of a single player
#OUTPUT: numpy array (162,26) of stats for that season
#Column labels 2006 Batting Game Log: 
# Gcar,PA,AB,R,H,2B,3B,HR,RBI,BB,IBB,SO,HBP,SH,SF,GDP,SB,CS,BA,OBP,SLG,OPS,BOP,aLI,WPA,RE24
##############################################################################################
def parse_PlayerSeason(player_season, year):
    temp=np.array([])
    temp=np.genfromtxt(player_season, delimiter=',')                #read in tampered CSV file
    if(year>2014):
        temp=np.delete(temp, [0,3,4,5,6,7,8,23,35,36,37],1)         #remove stats that we are not using
    else:
        temp=np.delete(temp, [0,3,4,5,6,7,8,23,35],1)           
    temp[:,[2]] = np.floor(temp[:,[2]])                           #turns third column's values into its row number in the (162, 27) return matrix
    ret=np.zeros([162,27])                                          #init return matrix
    for row in temp:                                         #enumerate temp matrix
        ret[row[1].astype(int)-1]=row                               #assign game rows to return matrix, leaving games unplayed as 0 vectors
    ret=np.delete(ret,[1],1)
    return ret

##############################################################################################
#INPUT: name of a player as a tuple (first name, last name)
#OUTPUT: outputs nothing, saves player stats to a local file
#Data scraped from BaseballReference.com
#Only ran once for training
##############################################################################################
def scrape_PlayerData(name):
    with urllib.request.urlopen('https://www.baseball-reference.com/players/gl.fcgi?id=ethiean01&t=b&year=2008') as response:
        html = response.read()
    print(html)











if __name__ == "__main__":
	run()
## fix random seed for reproducibility
    #numpy.random.seed(7)
    # load pima indians dataset
    #dataset = numpy.loadtxt("pima-indians-diabetes.csv", delimiter=",")
    # split into input (X) and output (Y) variables
    #X = dataset[:,0:8]
    #Y = dataset[:,8]
    # create model
    #model = Sequential()
    #model.add(Dense(12, input_dim=8, activation='relu'))
    #model.add(Dense(8, activation='relu'))
    #model.add(Dense(1, activation='sigmoid'))
    # Compile model
    #model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    # Fit the model
    #model.fit(X, Y, epochs=150, batch_size=10)
    # evaluate the model
    #scores = model.evaluate(X, Y)
    #print("\n%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))