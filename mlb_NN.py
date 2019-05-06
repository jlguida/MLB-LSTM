import tensorflow as tf
import numpy as np 
import pandas as pd
import io
import re
from keras.models import Sequential
from keras.layers import Dense
from io import StringIO
import csv
import urllib.request

# fix random seed for reproducibility
np.random.seed(7)


def run():
    ethier2006=parse_PlayerSeason('ethier2006.csv')
    scrape_PlayerData(2)
    print(ethier2006[0:30])  





##############################################################################################
#INPUT: CSV filename for a single season of a single player
#OUTPUT: numpy array (162,28) of stats for that season
#Column labels 2006 Batting Game Log: 
# Gcar,PA,AB,R,H,2B,3B,HR,RBI,BB,IBB,SO,HBP,SH,SF,GDP,SB,CS,BA,OBP,SLG,OPS,BOP,aLI,WPA,RE24
##############################################################################################
def parse_PlayerSeason(player_season):
    temp=np.array([])
    temp=np.genfromtxt(player_season, delimiter=',')        #read in tampered CSV file
    temp=np.delete(temp, [0,3,4,5,6,7,8,23,35],1)            #remove stats that we are not using
    temp[:,[2]] = np.floor(temp[:,[2]])                     #turns third column's values into its row number in the (162, 27) return matrix
    ret=np.zeros([162,27])                                 #init return matrix
    for row in temp:                                        #enumerate temp matrix
        ret[row[2].astype(int)-1]=row                       #assign game rows to return matrix, leaving games unplayed as 0 vectors
    ret=np.delete(ret,[2],1)
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