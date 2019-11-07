#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov  6 20:51:07 2019

@author: fhall, spell, jhardy
"""

# =============================================================================
# IMPORTS
import csv                        # csv parsing
import json                       # allow parsing of json strings
import numpy as np                
import re
import sqlite3 as sq              # to access database
import os
import time                       # timers

# =============================================================================
# FUNCTIONS START

def bad_data(action,text,filename):
    pass

# FUNCTIONS END
# =============================================================================

data_directory = 'data'

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
                    bad_data(data['TXT'],data['ACTION'])
                f.close()
                      
            except:
                f = open(filename, 'r')
                reader = csv.reader(f)
                for row in reader:
                    bad_data(row[0],row[-1])
                f.close()

