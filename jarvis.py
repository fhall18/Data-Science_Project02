#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 24 22:02:47 2019
@author: spell, jhardy, fhall
"""
# =============================================================================
# IMPORTS
from botsettings import API_TOKEN  # imports api token for jarvis
import csv  # csv parsing
import json  # allow parsing of json strings
import requests  # api get/post writing
import re
import sqlite3 as sq  # to access database
# from string import punctuation    # to remove punctuation from text
import os
import time  # timers
import websocket
from sklearn import preprocessing
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score

try:
    import thread
except ImportError:
    import _thread as thread

# =============================================================================
# DATABASE START
# =============================================================================
try:
    database = sq.connect('jarvis.db')
    conn = database.cursor()

except ValueError as err:
    print(err)


def insert_query(training_txt, action_txt):
    create_table = """CREATE TABLE IF NOT EXISTS training_data(
                            txt TEXT,
                            label TEXT);
        """

    conn.execute(create_table)
    # store the new action in here
    read_table = "SELECT * FROM training_data WHERE label = '{c_action}'".format(c_action=action_txt)
    conn.execute(read_table)

    table_training_text = conn.fetchall()

    # store all training text under certain action into this list
    training_txt_lst = []
    for i in table_training_text:
        training_txt_lst.append(i[0])
    # check if the training text already exist, if not add
    if training_txt not in training_txt_lst:
        conn.execute("""INSERT INTO training_data(txt, label) VALUES(?,?)""",
                     (training_txt, action_txt))

    database.commit()


def read_query(training_txt, action_txt):
    read_table = "SELECT * FROM training_data"
    conn.execute(read_table)

    table_training_text = conn.fetchall()

    # store all training text under certain action into this list
    training_txt_lst = []
    for i in table_training_text:
        training_txt_lst.append(i[1].lower())
        # check if the training text already exist, if yes return action
        if training_txt in training_txt_lst:
            action_txt = i[0]
            return action_txt


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

ct = 0
word_ct = {}
vectorized = {}

X = ct_vec.fit_transform(X)
tf_transformer = TfidfTransformer(use_idf=False).fit(X)
X = tf_transformer.transform(X)
# standardized_X = preprocessing.scale(X)

Y = le.fit_transform(Y)
print(X)
clf = MultinomialNB().fit(X, Y)
print(clf)
#
# bayes_model = MultinomialNB().fit(X, Y)
# lin_model = SVC(kernel = 'linear').fit(standardized_X, y)
# poly_model = SVC(kernel = 'poly').fit(standardized_X, y)
# rbf_model = SVC(kernel = 'rbf').fit(standardized_X, y)
# sig_model = SVC(kernel = 'sigmoid').fit(standardized_X, y)
#
# bayes = bayes_model.predict(standardized_X)
# lin = lin_model.predict(standardized_X)
# poly = poly_model.predict(standardized_X)
# rbf = rbf_model.predict(standardized_X)
# sig = sig_model.predict(standardized_X)
#
#
##CROSS VALIDATION SCORES
# bayes_scores = cross_val_score(bayes_model, standardized_X, y, cv=k)
# lin_scores = cross_val_score(lin_model, standardized_X, y, cv=k)
# poly_scores = cross_val_score(poly_model, standardized_X, y, cv=k)
# rbf_scores = cross_val_score(rbf_model, standardized_X, y, cv=k)
# sig_scores = cross_val_score(sig_model, standardized_X, y, cv=k)
#
#
##ACCURACY SCORES FOR PREDICTION
# X_train, X_test, y_train, y_test = train_test_split(standardized_X, Y)
# bayes_acc = accuracy_score(y_test, bayes)
# lin_acc = accuracy_score(y_test, lin)
# poly_acc = accuracy_score(y_test, poly)
# rbf_acc = accuracy_score(y_test, rbf)
# sig_acc = accuracy_score(y_test, sig)
#
#
# print("bayes: ", np.mean(bayes_scores)*100)
# print("linear: ", np.mean(lin_scores)*100)
# print("poly: ", np.mean(poly_scores)*100)
# print("rbf: ", np.mean(rbf_scores)*100)
# print("sigmoid: ", np.mean(sig_scores)*100)
#
#
#
# print("accuracy: ", bayes_acc)
# print("accuracy: ", lin_acc)
# print("accuracy: ", poly_acc)
# print("accuracy: ", rbf_acc)
# print("accuracy: ", sig_acc)


# what model do we want to use?
# DecisionTreeClassifier()
# multinomialNB()
# SVM
# RandomForest



















