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
import re
import os
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

# scikit-learn imports
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import confusion_matrix
# =============================================================================
# FUNCTIONS START
def plt_con_mat(cm, title, f):
    labels = ['WEATHER','JOKE','PIZZA','GREET','TIME']
    ax= plt.subplot()
    sns.heatmap(cm, annot=True, ax = ax, fmt='g'); #annot=True to annotate cells
    # labels, title and ticks
    ax.set_xlabel('Predicted labels');ax.set_ylabel('True labels'); 
    ax.set_title(title); 
    ax.xaxis.set_ticklabels(labels); ax.yaxis.set_ticklabels(labels)
    plt.savefig(f)
    plt.clf()

def plt_hist(data, title, f):
    plt.hist(data)
    plt.title(title)
    plt.xlabel('Accuracies')
    plt.ylabel('# Models')
    plt.savefig(f)
    plt.clf()

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
        error_ct += 1
        
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
                        action = action_check(row[-1], text, filename)
                    
                    if '{' in text:
                        text = text[9:].rstrip('"')
                    
                    X.append(text)
                    Y.append(action)
                f.close()
                


# =============================================================================
# Model Classifier Part:
    
# define 3 best performing models for analysis:
nb_clf = Pipeline([
    ('vect', CountVectorizer()),
    ('tfidf', TfidfTransformer()),
    ('clf', MultinomialNB())])

lin_clf = Pipeline([
    ('vect', CountVectorizer()),
    ('tfidf', TfidfTransformer()),
    ('clf', LinearSVC())])

sgd_clf = Pipeline([
    ('vect', CountVectorizer()),
    ('tfidf', TfidfTransformer()),
    ('clf', SGDClassifier())])

  #=============================================================================
#External data  
num_tests = 100
b_ct = 0
l_ct = 0
s_ct = 0
b_accs = []
l_accs = []
s_accs = []
for n in range(num_tests):
# SPLIT OUT TEST & TRAIN SETS
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
    
    # CREATE MODEL & TEST
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
        
    b_ct += bayes_acc*100
    l_ct += lin_acc*100
    s_ct += sgd_acc*100
    
    b_accs.append(bayes_acc*100)
    l_accs.append(lin_acc*100)
    s_accs.append(sgd_acc*100)
    
# PRINT OUT AVERAGED ACCURACY SCORES
print("External Data")
print("BAYES: ", b_ct/num_tests)
print("LIN: ", l_ct/num_tests)
print("SGD: ", s_ct/num_tests)


#=============================================================================
#Database data
X2 = ['What time is it?', 'What is the current time?', 'Time?', 'Can you tell me the time right now?', 'Do you know what time it is?', 'Please tell me the time', 'Time, please', 'Do you know the time?', 'Tell me the current time', 'Could you please tell me the time', 'Do you happen to have the time?', 'What\'s the time?', 'Get me a pizza', 'Order me a pizza', 'I want pizza', 'Can you get me some pizza', 'Pizza, please', 'Pizza?', 'I need some pizza', 'Can I get some pizza?', 'I\'m in the mood for pizza', 'I could go for some pizza right now', 'I\'m craving pizza', 'How are you?', 'What are you doing?', 'I hope you are doing well', 'How is everything?', 'How do you do?', 'What\'s up?', 'What\'s cracking?', 'What\'s on your mind?', 'What\'s new with you?', 'Good day', 'It\'s nice to speak with you', 'What is the temperature?', 'Is it hot out?', 'What is it doing outside?', 'What is the weather like right now?', 'Is it raining?', 'Weather?', 'Tell me about the weather', 'What\'s the forecast right now?', 'What are the conditions outside?', 'How\'s the weather?', 'Tell me a joke', 'Tell me something funny', 'Joke?', 'Do you have any jokes to tell me?', 'I want to hear a joke', 'Do you know any good jokes?', 'Crack a joke', 'Give me your best joke', 'Make me laugh', 'Got any wisecracks?']
Y2 = ['Time', 'Time', 'Time', 'Time', 'Time', 'Time', 'Time', 'Time', 'Time', 'Time', 'Time', 'Time', 'Pizza', 'Pizza', 'Pizza', 'Pizza', 'Pizza', 'Pizza', 'Pizza', 'Pizza', 'Pizza', 'Pizza', 'Pizza', 'Greet', 'Greet', 'Greet', 'Greet', 'Greet', 'Greet', 'Greet', 'Greet', 'Greet', 'Greet', 'Greet', 'Weather', 'Weather', 'Weather', 'Weather', 'Weather', 'Weather', 'Weather', 'Weather', 'Weather', 'Weather', 'Joke', 'Joke', 'Joke', 'Joke', 'Joke', 'Joke', 'Joke', 'Joke', 'Joke', 'Joke']
for y in Y2:
    y = y.upper()

b_ct2 = 0
l_ct2 = 0
s_ct2 = 0
b_accs2 = []
l_accs2 = []
s_accs2 = []

for n in range(num_tests):
# SPLIT OUT TEST & TRAIN SETS
    X_train2, X_test2, Y_train2, Y_test2 = train_test_split(X2, Y2, test_size=0.2)
    
    # CREATE MODEL & TEST
    bayes_model2 = nb_clf.fit(X_train2, Y_train2)
    lin_model2 = lin_clf.fit(X_train2, Y_train2)
    sgd_model2 = sgd_clf.fit(X_train2, Y_train2)
    
    bayes2 = bayes_model2.predict(X_test2)
    lin2 = lin_model.predict(X_test2)
    sgd2 = sgd_model.predict(X_test2)
    
    # ACCURACY SCORES FOR PREDICTION
    bayes_acc2 = accuracy_score(Y_test2, bayes2)
    lin_acc2 = accuracy_score(Y_test2, lin2)
    sgd_acc2 = accuracy_score(Y_test2, sgd2)
        
    b_ct2 += bayes_acc2*100
    l_ct2 += lin_acc2*100
    s_ct2 += sgd_acc2*100
    
    b_accs2.append(bayes_acc2*100)
    l_accs2.append(lin_acc2*100)
    s_accs2.append(sgd_acc2*100)
    
# PRINT OUT ACCURACY SCORES
print('\nDatabase Data')
print("BAYES: ", b_ct2/num_tests)
print("LIN: ", l_ct2/num_tests)
print("SGD: ", s_ct2/num_tests)

#t-test
print("\nT-test between external and database data")
print(stats.ttest_ind(b_accs, b_accs2, equal_var = False))
print(stats.ttest_ind(l_accs, l_accs2, equal_var = False))
print(stats.ttest_ind(s_accs, b_accs2, equal_var = False))


##=============================================================================
##Combined data
b_ct3 = 0
l_ct3 = 0
s_ct3 = 0
b_accs3 = []
l_accs3 = []
s_accs3 = []
X3 = []
for i in X:
    X3.append(i)
for j in X2:
    X3.append(j)
Y3 = []
for a in Y:
    Y3.append(a)
for b in Y2:
    Y3.append(b.upper())


for n in range(num_tests):
# SPLIT OUT TEST & TRAIN SETS
    X_train3, X_test3, Y_train3, Y_test3 = train_test_split(X3, Y3, test_size=0.2)
    
    # CREATE MODEL & TEST
    bayes_model3 = nb_clf.fit(X_train3, Y_train3)
    lin_model3 = lin_clf.fit(X_train3, Y_train3)
    sgd_model3 = sgd_clf.fit(X_train3, Y_train3)
    
    bayes3 = bayes_model3.predict(X_test3)
    lin3 = lin_model3.predict(X_test3)
    sgd3 = sgd_model3.predict(X_test3)
    
    # ACCURACY SCORES FOR PREDICTION
    bayes_acc3 = accuracy_score(Y_test3, bayes3)
    lin_acc3 = accuracy_score(Y_test3, lin3)
    sgd_acc3 = accuracy_score(Y_test3, sgd3)
        
    b_ct3 += bayes_acc3*100
    l_ct3 += lin_acc3*100
    s_ct3 += sgd_acc3*100
    
    b_accs3.append(bayes_acc3*100)
    l_accs3.append(lin_acc3*100)
    s_accs3.append(sgd_acc3*100)
    
# PRINT OUT ACCURACY SCORES
print('\nCombined Data--80/20')
print("BAYES: ", b_ct3/num_tests)
print("LIN: ", l_ct3/num_tests)
print("SGD: ", s_ct3/num_tests)

##t-test

print("\nT-test between external and combined")
print(stats.ttest_ind(b_accs, b_accs3, equal_var = False))
print(stats.ttest_ind(l_accs, l_accs3, equal_var = False))
print(stats.ttest_ind(s_accs, s_accs3, equal_var = False))
print("\nT-test between database and combined")
print(stats.ttest_ind(b_accs3, b_accs2, equal_var = False))
print(stats.ttest_ind(l_accs3, l_accs2, equal_var = False))
print(stats.ttest_ind(s_accs3, s_accs2, equal_var = False))

#=============================================================================
#Combined data split 50/50
b_ct4 = 0
l_ct4 = 0
s_ct4 = 0
b_accs4 = []
l_accs4 = []
s_accs4 = []


for n in range(num_tests):
# SPLIT OUT TEST & TRAIN SETS
    X_train4, X_test4, Y_train4, Y_test4 = train_test_split(X3, Y3, test_size=0.5)
    
    # CREATE MODEL & TEST
    bayes_model4 = nb_clf.fit(X_train4, Y_train4)
    lin_model4 = lin_clf.fit(X_train4, Y_train4)
    sgd_model4 = sgd_clf.fit(X_train4, Y_train4)
    
    bayes4 = bayes_model4.predict(X_test4)
    lin4 = lin_model4.predict(X_test4)
    sgd4 = sgd_model4.predict(X_test4)
    
    # ACCURACY SCORES FOR PREDICTION
    bayes_acc4 = accuracy_score(Y_test4, bayes4)
    lin_acc4 = accuracy_score(Y_test4, lin4)
    sgd_acc4 = accuracy_score(Y_test4, sgd4)
        
    b_ct4 += bayes_acc4*100
    l_ct4 += lin_acc4*100
    s_ct4 += sgd_acc4*100
    
    b_accs4.append(bayes_acc4*100)
    l_accs4.append(lin_acc4*100)
    s_accs4.append(sgd_acc4*100)
    
# PRINT OUT ACCURACY SCORES
print('\nCombined Data--50/50')
print("BAYES: ", b_ct4/num_tests)
print("LIN: ", l_ct4/num_tests)
print("SGD: ", s_ct4/num_tests)


#=============================================================================
#Combined data split 20/80

b_ct5 = 0
l_ct5 = 0
s_ct5 = 0
b_accs5 = []
l_accs5 = []
s_accs5 = []

for n in range(num_tests):
# SPLIT OUT TEST & TRAIN SETS
    X_train5, X_test5, Y_train5, Y_test5 = train_test_split(X3, Y3, test_size=0.8)
    
    # CREATE MODEL & TEST
    bayes_model5 = nb_clf.fit(X_train5, Y_train5)
    lin_model5 = lin_clf.fit(X_train5, Y_train5)
    sgd_model5 = sgd_clf.fit(X_train5, Y_train5)
    
    bayes5 = bayes_model5.predict(X_test5)
    lin5 = lin_model5.predict(X_test5)
    sgd5 = sgd_model5.predict(X_test5)
    
    # ACCURACY SCORES FOR PREDICTION
    bayes_acc5 = accuracy_score(Y_test5, bayes5)
    lin_acc5 = accuracy_score(Y_test5, lin5)
    sgd_acc5 = accuracy_score(Y_test5, sgd5)
        
#    print(bayes_acc)
#    print(lin_acc)
#    print(sgd_acc)
    b_ct5 += bayes_acc5*100
    l_ct5 += lin_acc5*100
    s_ct5 += sgd_acc5*100
    
    b_accs5.append(bayes_acc5*100)
    l_accs5.append(lin_acc5*100)
    s_accs5.append(sgd_acc5*100)
    
# PRINT OUT ACCURACY SCORES
print('\nCombined Data--20/80')
print("BAYES: ", b_ct5/num_tests)
print("LIN: ", l_ct5/num_tests)
print("SGD: ", s_ct5/num_tests)


#=============================================================================
#Confusion Matrix:

b_cm1 = confusion_matrix(Y_test, bayes)
l_cm1 = confusion_matrix(Y_test, lin)
s_cm1 = confusion_matrix(Y_test, sgd)
plt_con_mat(b_cm1, 'Confusion Matrix for Naive Bayes, External Data', 'nb_cm_ext.png')
plt_con_mat(l_cm1, 'Confusion Matrix for Linear SVC, External Data', 'lin_cm_ext.png')
plt_con_mat(s_cm1, 'Confusion Matrix for SGD, External Data', 'sgd_cm_ext.png')
#
b_cm2 = confusion_matrix(Y_test2, bayes2)
l_cm2 = confusion_matrix(Y_test2, lin2)
s_cm2 = confusion_matrix(Y_test2, sgd2)
plt_con_mat(b_cm2, 'Confusion Matrix for Naive Bayes, Database Data', 'nb_cm_dat.png')
plt_con_mat(l_cm2, 'Confusion Matrix for Linear SVC, Database Data', 'lin_cm_dat.png')
plt_con_mat(s_cm2, 'Confusion Matrix for SGD, Database Data', 'sgd_cm_dat.png')

b_cm3 = confusion_matrix(Y_test3, bayes3)
l_cm3 = confusion_matrix(Y_test3, lin3)
s_cm3 = confusion_matrix(Y_test3, sgd3)
plt_con_mat(b_cm3, 'Confusion Matrix for Naive Bayes, Combo Data', 'nb_cm_comb.png')
plt_con_mat(l_cm3, 'Confusion Matrix for Linear SVC, Combo Data', 'lin_cm_comb.png')
plt_con_mat(s_cm3, 'Confusion Matrix for SGD, Combo Data', 'sgd_cm_comb.png')

#=============================================================================
#Histograms:

plt_hist(b_accs, 'Naive Bayes Accuracy, External Data', 'nb_acc_ext.png')
plt_hist(l_accs, 'Linear SVC Accuracy, External Data', 'lin_acc_ext.png')
plt_hist(s_accs, 'SGD Accuracy, External Data', 'sgd_acc_ext.png')

plt_hist(b_accs2, 'Naive Bayes Accuracy, Database Data', 'nb_acc_dat.png')
plt_hist(l_accs2, 'Linear SVC Accuracy, Database Data', 'lin_acc_dat.png')
plt_hist(s_accs2, 'SGD Accuracy, Database Data', 'sgd_acc_dat.png')

plt_hist(b_accs3, 'Naive Bayes Accuracy, Combo Data', 'nb_acc_comb.png')
plt_hist(l_accs3, 'Linear SVC Accuracy, Combo Data', 'lin_acc_comb.png')
plt_hist(s_accs3, 'SGD Accuracy, Combo Data', 'sgd_acc_comb.png')

# =============================================================================
