#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 24 22:02:47 2019
@author: spell, jhardy, fhall
"""
# =============================================================================
# IMPORTS
from botsettings import API_TOKEN    # imports api token for jarvis
import json                          # allow parsing of json strings         
import pickle                        # pickle the brain
import requests                      # api get/post writing
import sqlite3 as sq                 # to access database
import time                          # timers
import websocket                     # connect to RTM and slack API 
from weather import weather_forecast # weather function to plot
from weather import legit_city       # check if city/town is valid

# Competition 3: File upload
try:
    from slacker import Slacker       # Competition 3: for posting images
    slack = Slacker(API_TOKEN)
except:
    print('slacker not loaded')

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
                            text TEXT,
                            action TEXT);
        """
        
    conn.execute(create_table)
    #store the new action in here
    read_table = "SELECT * FROM training_data WHERE action = '{c_action}'".format(c_action = action_txt)
    conn.execute(read_table)
    
    table_training_text = conn.fetchall()
    
    #store all training text under certain action into this list
    training_txt_lst = []
    for i in table_training_text:
        training_txt_lst.append(i[0])
    #check if the training text already exist, if not add
    if training_txt not in training_txt_lst:
        conn.execute("""INSERT INTO training_data(text, action) VALUES(?,?)""",
                         (training_txt, action_txt))
            
    database.commit()
        
# revised function: takes in TEXT and spits out ACTION
def read_query(training_txt):
    read_table = "SELECT action FROM training_data WHERE text = '{}'".format(training_txt)
    conn.execute(read_table)
    
    table_training_text = conn.fetchall()
    try: 
        action_txt = table_training_text[0][0]
    except:
        action_txt = 'no action for this text'
        
    return action_txt

# DATABASE END
# =============================================================================
# RUN JARVIS
# =============================================================================


# establish boolean variables
training_bool = False
action_bool = False
testing_bool = False
wf_bool = False
username = None
user = None
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

def upload_file(channel, file):
    # upload image using Slacker
    slack.files.upload(file, channels=channel)

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
        username = message['user']
        
        # !!! WEATHER TIME !!!
        if 'forecast time' in text:
            bot_text1 = "Looks like you want to know what the weather will be!"
            bot_text2 = "Just give me the name of a `CITY` or `TOWN` in the US and I\'ll show you the 5-day weather forecast."
            answer(bot_text1,channel)
            answer(bot_text2,channel)
            global wf_bool 
            global user
            wf_bool = True
            # tracker to ensure username is consistant
            user = message['user']
            
        elif text == 'done' and wf_bool == True:
            wf_bool = False
            answer('OK, let me know if you want to know the weather again!',channel)
            
        elif text == 'help' and wf_bool == True:
            answer('This is a weather service that I humbly provide. All you need to do is provide a valid city in the United States and I will show you a 5-day forecast of the temperature.',channel)
            
        elif wf_bool == True and username == user:

            # check valid city in US
            lc = legit_city(text)
            if lc == 'OK, let me find that 5-day forecast!':
                answer(lc,channel)
                # run the weather for that city
                weather_forecast(text)
                # upload plot
                upload_file(channel,'plots/forecast.png')
                answer('Is there another forecast that I can get for you?', channel)
            else:
                # try again message
                answer(lc,channel)
        
        
        # !!! TESTING TIME !!!
        if 'testing time' in text:
            bot_text1 = "I\'m training my brain with the data you\'ve already given me..."
            # update pickled brain
            import brain
            bot_text2 = "OK, I\'m ready for testing. Write me something and I'll try to figure it out."
            answer(bot_text1,channel)
            answer(bot_text2,channel)
            # update testing boolean
            global testing_bool
            testing_bool = True
        
        # exit if done
        elif text == 'done' and testing_bool == True:
            testing_bool = False
            answer('OK, I\'m finished testing',channel)
        
        # testing response
        elif testing_bool == True:
            brain = pickle.load(open("jarvis_UNCANNYHORSE.pkl", 'rb'))
            result = brain.predict([text])[0]
            bot_text1 = 'Ok, I think the action you mean is `{}`...'.format(result)
            bot_text2 = "Write me something else and I\'ll try to figure it out."
            answer(bot_text1,channel)
            answer(bot_text2,channel)
            
            
        # !!! TRAINING TIME !!!
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
            # run brain.py
            
            
        # Action is given -> start training
        elif training_bool == True and action_bool == False:
            # Pick up declared acation
            bot_text = 'Ok, let\'s call this action `{}`. Now give me some text!'.format(text.upper())
            answer(bot_text,channel)
            # set action to True
            action_bool = True
            global action
            action = text         
        
        # Ask for next text to train on?
        elif  message != 'done' and action_bool == True:
            # add text training text with action to db
            insert_query(text,action)
            # reply what's next
            answer('OK, what\'s next?',channel)
        
        # outputs action for a given training text
#        if training_bool == False and action_bool == False:
#            msg_txt = text.lower()
#            read = read_query(msg_txt)
#            print('Action',read)


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