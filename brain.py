#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov  5 20:25:32 2019

@author: fhall, spell, jhardy
"""
# IMPORTS
#from botsettings import API_TOKEN # imports api token for jarvis
import csv                        # csv parsing
import json                       # allow parsing of json strings
import matplotlib.pyplot as plt; plt.rcdefaults()
import numpy as np     
import matplotlib.pyplot as plt           
import pickle                     # pickle the brain
import re
import sqlite3 as sq              # to access database
import os

# scikit-learn imports
from sklearn.pipeline import Pipeline
from sklearn import preprocessing
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.linear_model import SGDClassifier

# =============================================================================
# FUNCTIONS START

def clean_txt(text):
    """ Clean the text by changing to lowercase, removing non-ASCII characters
    """
    text = text.lower()    
    #replace consecutive non-ASCII characters with apostrophe
    text = re.sub(r'[^\x00-\x7F]+',"'", text)
    
    return text


def action_check(action,text,filename):
    # check out regular expressions --> re.search(r'\bis\b', your_string)
    """ Check to see if the ACTION matches the TEXT
    """
    old_action = action
    # order matters!
    action_words = [
            {'GREET':{'hello', 'hola','whats up','what\'s up', 'hi jarvis', 'hey there', 'hi!', 'hey!', 'hiya', 'hey.'}},
            {'TIME':{'the time', 'sun set', 'sun rise', 'what time','clock','hour', 'watch say'}},
            {'PIZZA':{'pizza','topping','takeout',' pie','order','food', 'pepperoni'}},
            {'WEATHER':{'weather',' rain',' wind','temperature','forecast','sleet','snow'}},
            {'JOKE':{'joke', 'funny','funniest','cheer','laugh','hilarious','knock', 'humor'}}
            ]
    
    for action_dict in action_words:
#        if words in text_words and action != action_header:
        for new_action, word_set in action_dict.items():
            # issue of "hey" in "they"...
            if any(word in text for word in word_set):
                # check length to resolve hi, hey issue...
                action = new_action

    ### !!! CLEAN THIS UP BEFORE SUBMITTING !!! ###
    if action != old_action:
#        print('changed {}'.format(old_action).ljust(16),'to {}'.format(action).ljust(10), text, sep='\t') 
        # counter
        global change_count
        change_count += 1
        # catch bad data files
        if filename not in bad_data_files:
            bad_data_files[filename] = [text]
        else:
            bad_data_files[filename].append(text)
        
    return action

# FUNCTIONS END
# =============================================================================

# Read in external msg_txt,action data
test_data = {}

X = []
Y = []

data_directory = 'new_clean_data'
i = 0

counter = 0
change_count = 0
bad_data_files = {}

directory = os.fsencode(data_directory)  # establish directory
for file in os.listdir(directory):

        filename = os.fsdecode(file)
        filename = data_directory + '/' + filename
        # don't parse .DS_Store file
        if filename != "{}/.DS_Store".format(data_directory):
            try:
                # try parsing as json
                f = open(filename)
                for row in f:
                    data = json.loads(row)
                    text = clean_txt(data['TXT'])
                    action = action_check(data['ACTION'], text, filename)
                    # make into lists
                    X.append(text)
                    Y.append(action)
                f.close()
                      
            except:
                f = open(filename, 'r')
                reader = csv.reader(f)
                for row in reader:
                    text = clean_txt(row[0])
                    action = action_check(row[-1], text, filename)
                    # make into lists
                    X.append(text)
                    Y.append(action)
                f.close()

# cleaning external data
print('total changes:', change_count)


# add data from jarbis.db
try:
    database = sq.connect('jarvis.db')
    conn = database.cursor()

except ValueError as err:
    print(err)

read_table = "SELECT * FROM training_data"
conn.execute(read_table)
table_training_text = conn.fetchall()

for row in table_training_text:
    X.append(row[0])
    Y.append(row[1])


# =============================================================================
# Model Classifier Part:
    
# define 3 best performing models:
bayes_list = []
lin_list = []
sgd_list = []

for i in range(25):
    nb_clf = Pipeline([
        ('vect', CountVectorizer(stop_words='english')),
        ('tfidf', TfidfTransformer()),
        ('clf', MultinomialNB())])
    
    lin_clf = Pipeline([
        ('vect', CountVectorizer(stop_words='english')),
        ('tfidf', TfidfTransformer()),
        ('clf', LinearSVC())])
    
    sgd_clf = Pipeline([
        ('vect', CountVectorizer()),
        ('tfidf', TfidfTransformer()),
        ('clf', SGDClassifier())])
    
    # SPLIT OUT TEST & TRAIN SETS
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
    
    # CREATE MODEL & TEST
    sgd_model = sgd_clf.fit(X_train, Y_train)
    
    sgd_test = sgd_model.predict(X_test)

# Pickle the brain
filename = 'jarvis_UNCANNYHORSE.pkl'
pickle.dump(sgd_model, open(filename, 'wb'))