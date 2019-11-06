#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 24 22:02:47 2019
@author: spell, jhardy, fhall
"""
# =============================================================================
# IMPORTS
from botsettings import API_TOKEN # imports api token for jarvis
import csv                        # csv parsing
import json                       # allow parsing of json strings
import numpy as np                
import pickle                     # pickle the brain
import requests                   # api get/post writing
import re
import sqlite3 as sq              # to access database
import os
import time                       # timers
import websocket                  # 
try:
    import thread
except ImportError:
    import _thread as thread
    
# scikit-learn
from sklearn.pipeline import Pipeline
from sklearn import preprocessing
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import GridSearchCV

# =============================================================================
# DATABASE START
# =============================================================================
try:
    database = sq.connect('jarvis.db')
    conn = database.cursor()

except ValueError as err:
    print(err)
    
def insert_query(training_txt, action_txt):   
    create_table  = """CREATE TABLE IF NOT EXISTS training_data(
                            txt TEXT,
                            label TEXT);
        """
        
    conn.execute(create_table)
    #store the new action in here
    read_table = "SELECT * FROM training_data WHERE label = '{c_action}'".format(c_action = action_txt)
    conn.execute(read_table)
    
    table_training_text = conn.fetchall()
    
    #store all training text under certain action into this list
    training_txt_lst = []
    for i in table_training_text:
        training_txt_lst.append(i[0])
    #check if the training text already exist, if not add
    if training_txt not in training_txt_lst:
        conn.execute("""INSERT INTO training_data(txt, label) VALUES(?,?)""",
                         (training_txt, action_txt))
            
    database.commit()
    
def read_query(training_txt, action_txt):
    read_table = "SELECT * FROM training_data"
    conn.execute(read_table)
    
    table_training_text = conn.fetchall()
    
    #store all training text under certain action into this list
    training_txt_lst = []
    for i in table_training_text:
        training_txt_lst.append(i[1].lower())
    #check if the training text already exist, if yes return action
        if training_txt in training_txt_lst:
            action_txt = i[0]
            return action_txt
# =============================================================================
# DATABASE END

# =============================================================================
# FUNCTIONS START

def clean_txt(text):
    """ Clean the text by changing to lowercase, removing non-ASCII characters
    """
    text = text.lower()    
    #replace consecutive non-ASCII characters with apostrophe
    text = re.sub(r'[^\x00-\x7F]+',"'", text)
    
    return text

change_count = 0

def classification_check(action,text):
    # check out regular expressions --> re.search(r'\bis\b', your_string)
    """ Check to see if the ACTION matches the TEXT
    """
    old_action = action
    # order matters!
    action_words = [
            {'PIZZA':{'pizza','dinner','topping','takeout',' pie','order','food','cheese', 'pepperoni'}},
            {'WEATHER':{'weather',' sun',' rain',' wind','temperature','forecast','sleet','snow'}},
            {'TIME':{'the time', 'sun set', 'sun rise', 'what time','clock','hour', 'watch say'}},
            {'JOKE':{'joke', 'funny','funniest','cheer','laugh','hilarious','knock', 'humor'}},
            {'GREET':{'hello', 'hola','whats up','what\'s up', 'hi jarvis', 'hey there', 'hi!', 'hey!', 'hiya', 'hey.'}}
            ]
    
    for action_dict in action_words:
#        if words in text_words and action != action_header:
        for new_action, word_set in action_dict.items():
            # issue of "hey" in "they"...
            if any(word in text for word in word_set):
                # check length to resolve hi, hey issue...
                action = new_action

    if action != old_action:
#        print('changed {}'.format(old_action).ljust(16),'to {}'.format(action).ljust(10), text, sep='\t') 
        # counter
        global change_count
        change_count += 1
    
#        if action == 'GREET':
#            print(old_action.ljust(10), action.ljust(10), text)
#    if ('hi' in text or 'hey' in text) and len(text) < 5:
#        print(old_action, action, text, sep='\t')
        
    return action


# FUNCTIONS END
# =============================================================================

# Read in external msg_txt,action data
test_data = {}
X = []
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
#                    text = clean_txt(data['TXT'])
#                    action = classification_check(data['ACTION'], text)
                    action = data['ACTION']
                    text = data['TXT']
                    # make into lists
                    X.append(text)
                    Y.append(action)
#                    try:
#                        # if action already exists, append text
#                        test_data[action].append(text)
#                    except KeyError:
#                        # if new action, create list of text
#                        test_data[action] = [text]
                f.close()
                      
            except:
                f = open(filename, 'r')
                reader = csv.reader(f)
                for row in reader:
#                    text = clean_txt(row[0])
#                    action = classification_check(row[-1], text)
                    text = row[0]
                    action = row[-1]
                    X.append(text)
                    Y.append(action)
#                    try:
#                        # if action already exists, append text
#                        test_data[action].append(text)
#                    except KeyError:
#                        # if new action, create list of text
#                        test_data[action] = [text]
                f.close()
# cleaning external data
print('total changes:', change_count)


# ML Part:

# what model do we want to use?
    # DecisionTreeClassifier()
    # multinomialNB()
    # SVM
    # RandomForest

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

filename = 'jarvis_UNCANNYHORSE.pkl'
pickle.dump(sgd_model, open(filename, 'wb'))

# Load the pickled model 
knn_from_pickle = pickle.load(open(filename,'rb')) 
  
# Use the loaded pickled model to make predictions 
print(knn_from_pickle.predict(['whats going on']))


# =============================================================================
# RUN JARVIS
# =============================================================================

## establish boolean variables
#training_bool = False
#action_bool = False
#action = ''
#msg_txt = ''
#
## authenticate and build url to pass to websocket
#token = API_TOKEN
#def start_rtm():
#    url = "https://slack.com/api/rtm.start"
#    querystring = {"token":token}
#    # execute the request
#    response = requests.request("GET", url, params=querystring)
#    r_text = json.loads(response.text)
#    return r_text["url"]
#
#
## sends an answer from jarvis
#def answer(text, channel):
#    url = "https://slack.com/api/chat.postMessage"
#    querystring = {"token":token,"channel":channel,"text":text,"as_user":True}
#    requests.request("POST", url, params=querystring)
#   
#
## creating wss for websocket object
#r = start_rtm()
#
#
## block to handle messaging
#def on_message(ws, message):
#    # parse message into a dictionary
#    message = json.loads(message)
#    # make sure message is not from a bot
#    if 'bot_id' not in message and message['type'] == 'message':
#        text = message['text'].lower() # message to lower case
#        channel = message['channel']    
#        
#        ### DELETE BEFORE SUBMITTING ###
##        username = message['user']
##        print('USERNAME:', username)
##        print('MESSAGE:', text)
#        
#        # jarvis response to training
#        if 'training time' in text:
#            bot_text = 'OK, I\'m ready for training. What NAME should this ACTION be?'
#            answer(bot_text,channel)
#            # set training to True  
#            global training_bool
#            training_bool = True
#            global action_bool
#            action_bool = False
#        
#        # exit if done
#        elif text == 'done' and training_bool == True:
#            global msg_txt
#            msg_txt = ''
#            training_bool = False
#            action_bool = False
#            answer('finished training',channel)
#            
#        # Action is given -> start training
#        elif training_bool == True and action_bool == False:
#            # Pick up declared acation
#            bot_text = 'Ok, let\'s call this action `{}`. Now give me some text!'.format(text)
#            answer(bot_text,channel)
#            # set action to True
#            action_bool = True
#            global action
#            action = text         
#        
#        # Ask for next text to train on?
#        elif  message != 'done' and action_bool == True:
#            # add text training text with action to db
#            answer('OK, what\'s next?',channel)
#            # assign training text
#            msg_txt = text
#        
#        # insert action and training into table
#        if(action != '' and msg_txt != ''):
#            insert_query(msg_txt,action)
#        
#        
#        # outputs action for a given training text
#        if training_bool == False and action_bool == False:
#            msg_txt = text.lower()
#            action = 'hello'
#            read = read_query(msg_txt, action)
#            print('Action',read)
#        
#        ### DELETE BEFORE SUBMITTING ###
#        # variable check (prints for user and bot messages)
##        print('training:', training, 'action:',action)
#        
#
## handles errors
#def on_error(ws, error):
#    print(error)
#
## handles closing ws
#def on_close(ws):
#    print("### closed ###")
#
#def on_open(ws):
#    def run(*args):
#        for i in range(3):
#            time.sleep(1)
#            ws.send("Hello %d" % i)
#        time.sleep(1)
#        ws.close()
#        print("thread terminating...")
#    thread.start_new_thread(run, ())
#
#
## create websocket object 
#if __name__ == "__main__":
#    websocket.enableTrace(True)
#    ws = websocket.WebSocketApp(r,
#                                  on_message = lambda ws, msg: on_message(ws, msg),
#                                  on_error = lambda ws, msg: on_error(ws, msg),
#                                  on_close = lambda ws: on_close(ws))
#    
#    ws.run_forever()