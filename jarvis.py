#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 24 22:02:47 2019
@author: spell, jhardy, fhall
"""
# =============================================================================
# IMPORTS
# from botsettings import API_TOKEN  # imports api token for jarvis
import csv  # csv parsing
import json  # allow parsing of json strings
import pickle

import requests  # api get/post writing
import re
import sqlite3 as sq  # to access database
# from string import punctuation    # to remove punctuation from text
import os
import time  # timers
import websocket
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn import preprocessing
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import GridSearchCV

try:
    import thread
except ImportError:
    import _thread as thread


# =============================================================================
# DATABASE START
# =============================================================================
# try:
#    database = sq.connect('jarvis.db')
#    conn = database.cursor()
#
# except ValueError as err:
#    print(err)
#
#
# def insert_query(training_txt, action_txt):
#    create_table = """CREATE TABLE IF NOT EXISTS training_data(
#                            txt TEXT,
#                            label TEXT);
#        """
#
#    conn.execute(create_table)
#    # store the new action in here
#    read_table = "SELECT * FROM training_data WHERE label = '{c_action}'".format(c_action=action_txt)
#    conn.execute(read_table)
#
#    table_training_text = conn.fetchall()
#
#    # store all training text under certain action into this list
#    training_txt_lst = []
#    for i in table_training_text:
#        training_txt_lst.append(i[0])
#    # check if the training text already exist, if not add
#    if training_txt not in training_txt_lst:
#        conn.execute("""INSERT INTO training_data(txt, label) VALUES(?,?)""",
#                     (training_txt, action_txt))
#
#    database.commit()
#
#
# def read_query(training_txt, action_txt):
#    read_table = "SELECT * FROM training_data"
#    conn.execute(read_table)
#
#    table_training_text = conn.fetchall()
#
#    # store all training text under certain action into this list
#    training_txt_lst = []
#    for i in table_training_text:
#        training_txt_lst.append(i[1].lower())
#        # check if the training text already exist, if yes return action
#        if training_txt in training_txt_lst:
#            action_txt = i[0]
#            return action_txt


# =============================================================================
# DATABASE END
# =============================================================================

# FUNCTIONS START
# =============================================================================

def clean_txt(text):
    """ Clean the text by changing to lowercase, removing non-ASCII characters
    """
    text = text.lower()
    # replace consecutive non-ASCII characters with apostrophe
    text = re.sub(r'[^\x00-\x7F]+', "'", text)

    return text


change_count = 0


def classification_check(action, text):
    """ Check to see if the ACTION matches the TEXT
    """
    action_words = {'GREET': ('hello'),
                    'TIME': ('time'),
                    'PIZZA': ('pizza'),
                    'JOKE': ('joke'),
                    'WEATHER': ('weather')}
    for action_header, words in action_words.items():
        if words in text and action != action_header:
            new_action = action_header
            #            print('changed action', text, action, new_action, sep='\t')
            global change_count
            change_count += 1
    return action


# Read in external msg_txt,action data
test_data = {}

X = [];
Y = []
reports_directory = 'data'
i = 0
counter = 0
# Loop over all of the files in reports
directory = os.fsencode(reports_directory)  # establish directory

for file in os.listdir(directory):

    filename = os.fsdecode(file)
    filename = reports_directory + '/' + filename
    # don't parse .DS_Store file
    if filename != "data/.DS_Store":
        try:
            # try parsing as json
            f = open(filename)
            for row in f:
                data = json.loads(row)
                text = clean_txt(data['TXT'])
                action = classification_check(data['ACTION'], text)
                #                    try:
                #                        # if action already exists, append text
                #                        test_data[action].append(text)
                #                    except KeyError:
                #                        # if new action, create list of text
                #                        test_data[action] = [text]
                X.append(text)
                Y.append(action)
            f.close()

        except:
            f = open(filename, 'r')
            reader = csv.reader(f)
            for row in reader:
                text = clean_txt(row[0])
                action = classification_check(row[1], text)
                #                    try:
                #                        # if action already exists, append text
                #                        test_data[action].append(text)
                #                    except KeyError:
                #                        # if new action, create list of text
                #                        test_data[action] = [text]
                X.append(text)
                Y.append(action)
            f.close()

# cleaning external data
print('total changes:', change_count)

# run through key-value and analyse distribution

# ML Part:

le = preprocessing.LabelEncoder()
ct_vec = CountVectorizer()
nb_clf = Pipeline([
    ('vect', CountVectorizer(stop_words='english')),
    ('tfidf', TfidfTransformer()),
    ('clf', MultinomialNB())])
lin_clf = Pipeline([
    ('vect', CountVectorizer(stop_words='english')),
    ('tfidf', TfidfTransformer()),
    ('clf', SVC(kernel='linear'))])
# poly_clf = Pipeline([
#        ('vect', CountVectorizer()),
#        ('tfidf', TfidfTransformer()),
#        ('clf', SVC(kernel = 'poly'))])
# rbf_clf = Pipeline([
#        ('vect', CountVectorizer()),
#        ('tfidf', TfidfTransformer()),
#        ('clf', SVC(kernel = 'rbf'))])
# sig_clf = Pipeline([
#        ('vect', CountVectorizer()),
#        ('tfidf', TfidfTransformer()),
#        ('clf', SVC(kernel = 'sigmoid'))])
# tree_clf = Pipeline([
#        ('vect', CountVectorizer()),
#        ('tfidf', TfidfTransformer()),
#        ('clf', DecisionTreeClassifier())])
# forest_clf = Pipeline([
#        ('vect', CountVectorizer()),
#        ('tfidf', TfidfTransformer()),
#        ('clf', RandomForestClassifier())])
sgd_clf = Pipeline([
    ('vect', CountVectorizer(stop_words='english')),
    ('tfidf', TfidfTransformer()),
    ('clf', SGDClassifier())])

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)

bayes_model = nb_clf.fit(X_train, Y_train)
lin_model = lin_clf.fit(X_train, Y_train)
# poly_model = poly_clf.fit(X_train, Y_train)
# rbf_model = rbf_clf.fit(X_train, Y_train)
# sig_model = sig_clf.fit(X_train, Y_train)
# tree_model = tree_clf.fit(X_train, Y_train)
# forest_model = forest_clf.fit(X_train, Y_train)
sgd_model = sgd_clf.fit(X_train, Y_train)

# CREATE MODELS
bayes = nb_clf.predict(X_test)
lin = lin_model.predict(X_test)
# poly = poly_model.predict(X_test)
# rbf = rbf_model.predict(X_test)
# sig = sig_model.predict(X_test)
# tree = tree_model.predict(X_test)
# forest = forest_model.predict(X_test)
sgd = sgd_model.predict(X_test)

# ACCURACY SCORES FOR PREDICTION
bayes_acc = accuracy_score(Y_test, bayes)
lin_acc = accuracy_score(Y_test, lin)
# poly_acc = accuracy_score(Y_test, poly)
# rbf_acc = accuracy_score(Y_test, rbf)
# sig_acc = accuracy_score(Y_test, sig)
# tree_acc = accuracy_score(Y_test, tree)
# forest_acc = accuracy_score(Y_test, forest)
sgd_acc = accuracy_score(Y_test, sgd)

# PRINT OUT ACCURACY SCORES FOR ALL MODELS
print("BAYES: ", bayes_acc * 100)
print("LIN: ", lin_acc * 100)
# print("POLY: ", poly_acc*100)
# print("RBF: ", rbf_acc*100)
# print("SIGMOID: ", sig_acc*100)
# print("TREE: ", tree_acc*100)
# print("RAND FOREST: ", forest_acc*100)
print("SGD: ", sgd_acc * 100)

parameters = {'vect__ngram_range': [(1, 1), (1, 2)], 'tfidf__use_idf': (True, False), 'clf__alpha': (1e-2, 1e-3), }
gs_clf = GridSearchCV(sgd_model, parameters, n_jobs=-1)
gs_clf = gs_clf.fit(X_train, Y_train)
print(gs_clf.best_score_)
print(gs_clf.best_params_)

filename = 'jarvis_UNCANNYHORSE.pkl'
pickle.dump(sgd_model, open(filename, 'wb'))














