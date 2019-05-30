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
from keras.layers import Lambda
from keras.models import Model
from io import StringIO
import csv
import urllib.request



# define validation size, this equates to number of games left out of training
VALIDATION_SIZE = 486

#Hidden size equates to the number of gamnes in each time step
TIME_STEP = 8
NUM_STATS = 9
BATCH_SIZE = 75
GAMES_IN_A_SEASON = 162
NUM_STEPS = 3
TOTAL_GAMES = 0
VALIDATION_SEASON = 3 
NUM_CATEGORIES = 3


def run():
    # fix random seed for reproducibility
    np.random.seed(7)

    #Create the Andre Ethier stat matrix ()
    ethier=np.zeros([12,162,NUM_STATS])
    for x in range(12):
        year = 2006+x
        file='ethier'+str(year)+'.csv'
        ethier[x]=parse_PlayerSeason(file, year)
    ethier=ethier.reshape((1944,NUM_STATS))
    ethier=np.delete(ethier,[1942,1941,1940,1939,1938,1937,1936],0)
    TOTAL_GAMES = ethier.shape[0]
    print(ethier.shape)


    #Training instances
    ethier = np.nan_to_num(ethier)
    X_train = np.zeros([TOTAL_GAMES-VALIDATION_SIZE,NUM_STATS])
    Y_train = np.zeros([TOTAL_GAMES-VALIDATION_SIZE,NUM_CATEGORIES])
    for i,row in enumerate(ethier):
        if(i < TOTAL_GAMES-VALIDATION_SIZE):
            if(i < GAMES_IN_A_SEASON*VALIDATION_SEASON | i > GAMES_IN_A_SEASON * VALIDATION_SEASON + VALIDATION_SIZE):
                category=np.zeros(NUM_CATEGORIES)
                scores = 0
                if(ethier[i+1][1] == 1):
                    scores = 1
                if(ethier[i+1][1] > 1):
                    scores = 2
                category[scores] = 1
                Y_train[i] = category
                X_train[i] = row

    Y_train = np.delete(Y_train,[1450,1449,1448],0)
    print(Y_train.shape)
    Y_train = Y_train.reshape((181,TIME_STEP,NUM_CATEGORIES))
    print(Y_train.shape)
    X_train = np.delete(X_train,[1450,1449,1448],0)
    X_train = X_train.reshape((181,TIME_STEP,NUM_STATS))
    print("X_train shape: ", X_train.shape)
    print("Y_train shape: ", Y_train.shape)


    
    V=ethier[GAMES_IN_A_SEASON*VALIDATION_SEASON+1:GAMES_IN_A_SEASON*VALIDATION_SEASON+VALIDATION_SIZE+1]
    V=V.reshape((486,NUM_STATS))
    V=np.nan_to_num(V)
    
    

    #Validation instances
    V_data=ethier[GAMES_IN_A_SEASON*3:GAMES_IN_A_SEASON*3+VALIDATION_SIZE]
    V_data = V_data.reshape((486,NUM_STATS))

    #Validation testing data
    V_test = np.zeros([486,NUM_CATEGORIES])
    for i,row in enumerate(V):
        category=np.zeros(NUM_CATEGORIES)
        scores = 0
        if(row[1] == 1):
            scores = 1
        if(row[1] > 1):
            scores = 2
        category[scores]=1
        V_test[i]=category
    V_test = V_test.reshape((486,NUM_CATEGORIES))
    
    V_data = np.delete(V_data,[479,480,481,482,483,484],0)
    V_test = np.delete(V_test,[479,480,481,482,483,484],0)
    V_data = V_data.reshape((60,TIME_STEP,NUM_STATS))
    V_test = V_test.reshape((60,TIME_STEP,NUM_CATEGORIES))
    print(" \n X_train: First Game of 1st season (32 games): \n ", X_train[:4])
    print("\n V_data: 1st game of 4th season (8 games): \n", V_data[0])
    print("\n Y_train: Second Game of 1st season (32 games): \n", Y_train[:4])
    print("\n V_test: 2nd Game of 4th Season (8 games): \n", V_test[0])
 
    #Build model !!!
    model=Sequential()
    #model.add(LSTM(units=25, input_shape=(TIME_STEP,NUM_STATS)))
    #model.add(Dropout(.2))
    #model.add(Dense(3, activation='softmax'))
    model.add(LSTM(units=25, input_shape=(TIME_STEP,NUM_STATS), return_sequences=True))
    model.add(LSTM(units=25, input_shape=(TIME_STEP,NUM_STATS), return_sequences=True))
    #model.add(Dropout(.2))
    model.add(TimeDistributed(Dense(25, activation='relu')))
    model.add(TimeDistributed(Dense(3, activation='softmax')))
    model.add(TimeDistributed(Dense(3, activation='softmax')))



    #Train the model
    model.compile(loss='binary_crossentropy', optimizer='adagrad',  metrics=['accuracy'])   
    model.fit(X_train, Y_train, epochs=1, batch_size=BATCH_SIZE)

    #Testing tensor output
    for i in range(4):
        intermediate_layer_model = Model(inputs=model.input,outputs=model.layers[i].output)
        intermediate_output = intermediate_layer_model.predict(V_data)
        #print ("Layer " + str(i) + " output: ", intermediate_output)
    print(model.summary())
    #scores = model.evaluate(V_data, V_test)
    #print("\n%s: %.2f%%" % ('accuracy', scores[1]*100))
    predictions = model.predict(V_data)
    print("\n\n\n Predictions for 4th season (30 games): \n\n\n")
    print (predictions)
    correct = 0
    incorrect = 0
    predicted_1 = 0
    predicted_2 = 0
    predicted_false = 0
    total_games_with_hits = 0
    total_games_without_hits = 0
    twohits = 0

    
    for i in range(0,58):
        for j in range(8):
            most_likely = np.argmax(predictions[i][j])
            if(np.argmax(V_test[i][j]) == 1):
                total_games_with_hits += 1
            if(np.argmax(V_test[i][j]) == 2):
                twohits += 1
            if(np.argmax(V_test[i][j]) == 0):
                total_games_without_hits += 1
            if (most_likely == np.argmax(V_test[i][j])):
                correct = correct + 1
                if(most_likely == 1):
                    predicted_1 += 1
                    #print(" \n Predicted: ", predictions[i][j]," Runs: ",most_likely," Actual: ", np.argmax(V_test[i][j]), "\n")
                if(most_likely == 2):
                    predicted_2 += 1
                else:
                    predicted_false += 1
            else:
                print(" \n Predicted: ", predictions[i][j]," Runs: ",most_likely," Actual: ", np.argmax(V_test[i][j]), "\n")
                incorrect = incorrect + 1

    print(" \n Correct: \n", correct, correct/(58*8))
    print(" \n Incorrect: \n", incorrect, incorrect/(58*8))
    print(total_games_with_hits,twohits,total_games_without_hits)
    print(" \n Predicted 1 hit: \n", predicted_1, predicted_1/total_games_with_hits)
    print(" \n Predicted 2 hit: \n", predicted_2, predicted_2/twohits)
    print(" \n Predicted 0 hit: \n", predicted_false, predicted_false/total_games_without_hits)
    print(" \n Percent 0 hit games: \n", total_games_without_hits/(58*8))
    print(" \n Percent 1 hit games: \n", total_games_with_hits/(58*8))
    print(" \n Percent 1+ hit games: \n", twohits/(58*8))





    
def hardmax(x):
    max = np.argmax(x)
    result = np.zeros(5)
    result[max] = 1
    print(x.shape)
    return (x * result)



##############################################################################################
#INPUT: CSV filename for a single season of a single player
#OUTPUT: numpy array (162,NUM_STATS) of stats for that season
#Column labels 2006 Batting Game Log: 
#PA,AB,R,H,2B,3B,HR,RBI,BB,IBB,SO,HBP,SH,SF,GDP,SB,CS,BA,OBP,SLG,OPS,BOP,aLI,WPA,RE24
##############################################################################################
def parse_PlayerSeason(player_season, year):
    temp=np.array([])
    temp=np.genfromtxt(player_season, delimiter=',')                #read in tampered CSV file
    if(year>2014):
        temp=np.delete(temp, [0,1,3,4,5,6,7,8,9,14,18,20,21,22,23,24,25,26,27,28,29,30,32,33,34,35,36,37],1)         #remove stats that we are not using
    else:
        temp=np.delete(temp, [0,1,3,4,5,6,7,8,9,14,18,20,21,22,23,24,25,26,27,28,29,30,32,33,34,35],1)           
    temp[:,[0]] = np.floor(temp[:,[0]])                           #turns third column's values into its row number in the (162, 27) return matrix
    ret=np.zeros([162,10])                                          #init return matrix
    for i,row in enumerate(temp):                                         #enumerate temp matrix
        ret[row[0].astype(int)-1]=row           
    ret=np.delete(ret,0,1)                    #assign game rows to return matrix, leaving games unplayed as 0 vectors
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
