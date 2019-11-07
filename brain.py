#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov  5 20:25:32 2019

@author: fhall
"""
# IMPORTS
#from botsettings import API_TOKEN # imports api token for jarvis
import csv                        # csv parsing
import json                       # allow parsing of json strings
#import numpy as np                
import pickle                     # pickle the brain
import re
#import sqlite3 as sq              # to access database
import os
#import time                       # timers

# scikit-learn imports
from sklearn.pipeline import Pipeline
from sklearn import preprocessing
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
#from sklearn.svm import SVC
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
#from sklearn.model_selection import cross_val_score
#from sklearn.tree import DecisionTreeClassifier
#from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import SGDClassifier
#from sklearn.model_selection import GridSearchCV

# =============================================================================
# FUNCTIONS START

def clean_txt(text):
    """ Clean the text by changing to lowercase, removing non-ASCII characters
    """
    text = text.lower()    
    #replace consecutive non-ASCII characters with apostrophe
    text = re.sub(r'[^\x00-\x7F]+',"'", text)
    
    return text


def classification_check(action,text,filename):
    # check out regular expressions --> re.search(r'\bis\b', your_string)
    """ Check to see if the ACTION matches the TEXT
    """
    old_action = action
    # order matters!
    action_words = [
            {'GREET':{'hello', 'hola','whats up','what\'s up', 'hi jarvis', 'hey there', 'hi!', 'hey!', 'hiya', 'hey.'}},
            {'TIME':{'the time', 'sun set', 'sun rise', 'what time','clock','hour', 'watch say'}},
            {'PIZZA':{'pizza','dinner','topping','takeout',' pie','order','food','cheese', 'pepperoni'}},
            {'WEATHER':{'weather',' sun ',' rain',' wind','temperature','forecast','sleet','snow'}},
            {'JOKE':{'joke', 'funny','funniest','cheer','laugh','hilarious','knock', 'humor'}}
            ]
    
    for action_dict in action_words:
#        if words in text_words and action != action_header:
        for new_action, word_set in action_dict.items():
            # issue of "hey" in "they"...
            if any(word in text for word in word_set):
                # check length to resolve hi, hey issue...
                action = new_action

    if action != old_action:
        print('changed {}'.format(old_action).ljust(16),'to {}'.format(action).ljust(10), text, sep='\t') 
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
#                    text = data['TXT']
                    action = classification_check(data['ACTION'], text, filename)
#                    action = data['ACTION']
                    # make into lists
                    X.append(text)
                    Y.append(action)
                f.close()
                      
            except:
                f = open(filename, 'r')
                reader = csv.reader(f)
                for row in reader:
                    text = clean_txt(row[0])
#                    text = row[0]
                    action = classification_check(row[-1], text, filename)
#                    action = row[-1]
                    # make into lists
                    X.append(text)
                    Y.append(action)
                f.close()

# cleaning external data
print('total changes:', change_count)


# =============================================================================
# Model Classifier Part:

le = preprocessing.LabelEncoder()
ct_vec = CountVectorizer()
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


X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)

# CREATE MODELS
bayes_model = nb_clf.fit(X_train, Y_train)
lin_model = lin_clf.fit(X_train, Y_train)
sgd_model = sgd_clf.fit(X_train, Y_train)

bayes = nb_clf.predict(X_test)
lin = lin_model.predict(X_test)
sgd = sgd_model.predict(X_test)

# ACCURACY SCORES FOR PREDICTION
bayes_acc = accuracy_score(Y_test, bayes)
lin_acc = accuracy_score(Y_test, lin)
sgd_acc = accuracy_score(Y_test, sgd)

# PRINT OUT ACCURACY SCORES FOR ALL MODELS
print("BAYES: ", bayes_acc * 100)
print("LIN: ", lin_acc * 100)
print("SGD: ", sgd_acc * 100)

#parameters = {'vect__ngram_range': [(1, 1), (1, 2)], 'tfidf__use_idf': (True, False), 'clf__alpha': (1e-2, 1e-3), }
#gs_clf = GridSearchCV(lin_clf, parameters, cv=5, iid=False, n_jobs=-1)
#gs_clf = gs_clf.fit(X_train, Y_train)
#print(gs_clf.best_score_)
#print(gs_clf.best_params_)


# Pickle the brain
filename = 'jarvis_UNCANNYHORSE.pkl'
pickle.dump(sgd_model, open(filename, 'wb'))

# Load the pickled model 
knn_from_pickle = pickle.load(open(filename,'rb')) 
  
# Use the loaded pickled model to make predictions 
print(knn_from_pickle.predict(['whats going on']))


