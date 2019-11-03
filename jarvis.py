#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 24 22:02:47 2019
@author: fhall
"""
# =============================================================================
# IMPORTS
from botsettings import API_TOKEN # imports api token for jarvis
import csv                        # csv parsing
import json                       # allow parsing of json strings
import requests                   # api get/post writing
import sqlite3 as sq              # to access database
import sys, os
import time                       # timers
import websocket                  # 
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

# Read in external msg_txt,action data
test_data = {}
reports_directory = 'data'
i = 0
counter = 0
# Loop over all of the files in reports
directory = os.fsencode(reports_directory)  # establish directory

for file in os.listdir(directory):
        i+=1
        print(i,file)
        filename = os.fsdecode(file)
        filename = reports_directory + '/' + filename

        if filename != "data/.DS_Store":
            try:    
                # try parsing as json
                f = open(filename)
                for row in f:
                    data = json.loads(row)
                    try:
                        # if it already exists
                        test_data[data['ACTION']].append(data['TXT'])
                    except KeyError:
                        # if new
                        test_data[data['ACTION']] = [data['TXT']]
                    counter +=1
                f.close()
                      
            except:
                f = open(filename, 'r')
                reader = csv.reader(f)
                for row in reader:
                    try:
                        # if it already exists
                        test_data[row[1]].append(row[0])
                    except KeyError:
                        # if new
                        test_data[row[1]] = [row[0]]
                    counter +=1
                f.close()

# cleaning external data
# general cleaning (lowercase, quotations, etc.)

# run through key-value and analyse distribution


# establish boolean variables
training_bool = False
action_bool = False
action = ''
msg_txt = ''

# authenticate and build url to pass to websocket
token = API_TOKEN
def start_rtm():
    url = "https://slack.com/api/rtm.start"
    querystring = {"token":token}
    # execute the request
    response = requests.request("GET", url, params=querystring)
    r_text = json.loads(response.text)
    return r_text["url"]


# sends an answer from jarvis
def answer(text, channel):
    url = "https://slack.com/api/chat.postMessage"
    querystring = {"token":token,"channel":channel,"text":text,"as_user":True}
    requests.request("POST", url, params=querystring)
   

# creating wss for websocket object
r = start_rtm()


# block to handle messaging
def on_message(ws, message):
    # parse message into a dictionary
    message = json.loads(message)
    # make sure message is not from a bot
    if 'bot_id' not in message and message['type'] == 'message':
        text = message['text'].lower() # message to lower case
        channel = message['channel']    
        
        ### DELETE BEFORE SUBMITTING ###
#        username = message['user']
#        print('USERNAME:', username)
#        print('MESSAGE:', text)
        
        # jarvis response to training
        if 'training time' in text:
            bot_text = 'OK, I\'m ready for training. What NAME should this ACTION be?'
            answer(bot_text,channel)
            # set training to True  
            global training_bool
            training_bool = True
            global action_bool
            action_bool = False
        
        # exit if done
        elif text == 'done' and training_bool == True:
            global msg_txt
            msg_txt = ''
            training_bool = False
            action_bool = False
            answer('finished training',channel)
            
        # Action is given -> start training
        elif training_bool == True and action_bool == False:
            # Pick up declared acation
            bot_text = 'Ok, let\'s call this action `{}`. Now give me some text!'.format(text)
            answer(bot_text,channel)
            # set action to True
            action_bool = True
            global action
            action = text         
        
        # Ask for next text to train on?
        elif  message != 'done' and action_bool == True:
            # add text training text with action to db
            answer('OK, what\'s next?',channel)
            # assign training text
            msg_txt = text
        
        # insert action and training into table
        if(action != '' and msg_txt != ''):
            insert_query(msg_txt,action)
        
        
        # outputs action for a given training text
        if training_bool == False and action_bool == False:
            msg_txt = text.lower()
            action = 'hello'
            read = read_query(msg_txt, action)
            print('Action',read)
        
        ### DELETE BEFORE SUBMITTING ###
        # variable check (prints for user and bot messages)
#        print('training:', training, 'action:',action)
        

# handles errors
def on_error(ws, error):
    print(error)

# handles closing ws
def on_close(ws):
    print("### closed ###")

def on_open(ws):
    def run(*args):
        for i in range(3):
            time.sleep(1)
            ws.send("Hello %d" % i)
        time.sleep(1)
        ws.close()
        print("thread terminating...")
    thread.start_new_thread(run, ())


# create websocket object 
if __name__ == "__main__":
    websocket.enableTrace(True)
    ws = websocket.WebSocketApp(r,
                                  on_message = lambda ws, msg: on_message(ws, msg),
                                  on_error = lambda ws, msg: on_error(ws, msg),
                                  on_close = lambda ws: on_close(ws))
    
    ws.run_forever()
