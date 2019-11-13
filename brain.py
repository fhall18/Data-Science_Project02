#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov  5 20:25:32 2019
@author: fhall, spell1, jhardy
"""
# IMPORTS
#from botsettings import API_TOKEN # imports api token for jarvis
import csv                        # csv parsing
import json                       # allow parsing of json strings
#import numpy as np                
import pickle                     # pickle the brain
import re
import sqlite3 as sq              # to access database
import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

# scikit-learn imports
from sklearn.pipeline import Pipeline
from sklearn import preprocessing
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.linear_model import SGDClassifier
from sklearn.decomposition import PCA
from sklearn.metrics import confusion_matrix
from sklearn.metrics import f1_score
# =============================================================================
# FUNCTIONS START

def findWholeWord(word, txt):
    return re.compile(r'\b({0})\b'.format(word), flags=re.IGNORECASE).search(txt)

def clean_txt(text):
    """ Clean the text by changing to lowercase, removing non-ASCII characters
    """
    text = text.lower()    
    #replace consecutive non-ASCII characters with apostrophe
    text = re.sub(r'[^\x00-\x7F]+',"'", text)
    
    return text


def action_check(action,text):
    """ Check to see if the ACTION matches the TEXT
    """
    error_ct = 0
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
        for new_action, word_set in action_dict.items():
            if any(word in text for word in word_set):
                action = new_action
                
    if findWholeWord('sun', text):
        action = 'WEATHER'
            
    if findWholeWord('hi', text):
        action = 'GREET'
        
    if findWholeWord('hey', text):
        action = 'GREET'
        
    if action != old_action:
        error_ct = 1
        
    return action
    
# FUNCTIONS END
# =============================================================================


# Read in external msg_txt,action data
test_data = {}

X = []
Y = []

data_directory = 'data'
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
                    action = action_check(data['ACTION'], text)
                    # make into lists
                    X.append(text)
                    Y.append(action)
                f.close()
                      
            except:
                f = open(filename, 'r')
                reader = csv.reader(f)
                for row in reader:
                    text = clean_txt(row[0])
                    if '}' in row[-1]:
                        w = row[-1]
                        action = action_check(w[12:].rstrip('"}'), text)
                    else:
                        action = action_check(row[-1], text)
                    # make into lists
                    if '{' in text:
                        text = text[9:].rstrip('"')
                    
                    X.append(text)
                    Y.append(action)
                    
                f.close()

## =============================================================================
## Model Classifier Part:
## define best performing model:
lin_clf = Pipeline([
    ('vect', CountVectorizer()),
    ('tfidf', TfidfTransformer()),
    ('clf', LinearSVC())])

lin_model = lin_clf.fit(X, Y)

# Pickle the brain
filename = 'jarvis_UNCANNYHORSE.pkl'
pickle.dump(lin_model, open(filename, 'wb'))

## =============================================================================
