#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov  6 20:51:07 2019
@author: fhall, spell1, jhardy
"""

# =============================================================================
# IMPORTS
import csv                        # csv parsing
import json                       # allow parsing of json strings
import re
import os

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
        
    return error_ct
    

# FUNCTIONS END
# =============================================================================

data_directory = 'data1'
errors = []
ct=0
directory = os.fsencode(data_directory)  # establish directory
for file in os.listdir(directory):
        filename = os.fsdecode(file)
        filename = data_directory + '/' + filename
        # don't parse .DS_Store file
        if filename != "{}/.DS_Store".format(data_directory):
            
            try:
                # try parsing as json
                error_ct = 0
                sent_ct = 0
                f = open(filename)
                for row in f:
                    data = json.loads(row)
                    error_ct += action_check(data['TXT'],data['ACTION'])
                    sent_ct += 1
                    
                f.close()
                errors.append(error_ct/sent_ct)
#                print(error_ct/sent_ct)
                      
            except:
                error_ct = 0
                sent_ct = 0
                f = open(filename, 'r')
                reader = csv.reader(f)
                for row in reader:
                    text = row[0]
                    if '{' in text:
                        text = clean_txt(text[9:].rstrip('"'))
                    else:
                        text = clean_txt(row[0])
                    
                    if '}' in row[-1]:
                        w = row[-1]
                        error_ct += action_check(w[12:].rstrip('"}'), text)
                    else:
                        error_ct += action_check(row[-1], text)
                        
                    sent_ct += 1
                    
                f.close()
                
                errors.append(error_ct/sent_ct)
#                print(error_ct/sent_ct)
                
            if(error_ct/sent_ct >= .01):
                print('BAD')
                ct+=1
            else:
                print('GOOD')
                ct+=1
#print(ct)
#
#print(errors)
