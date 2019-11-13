##################################### EDA ####################################

#METHODS######################################################################
# Removing punctuation from text
from string import punctuation
def clean_text(text):
    exclude = set(punctuation)
    list_letters_noPunct = [ char for char in text if char not in exclude]
    text = "".join(list_letters_noPunct)
    return text

## Counts of words for each ACTION for the external data 
def WordCounter(lst):
    c = Counter()
    for x in lst:
        for y in x:
            c[y] += 1
    print(c.most_common(21))
# Splitting external data
X1 = []
for txt in X:
    X1.append(clean_text(txt))

X1 = [i.split() for i in X1]

# Zipping together Actions and Cleaned text from external data
act_text = list(zip(Y, X1))

# 
group_data = table_training_text

#  Group data separation by ACTION
PizzaGrp = [t for t in group_data if t[0].startswith('Pizza')]
GreetGrp = [t for t in group_data if t[0].startswith('Greet')]
JokeGrp = [t for t in group_data if t[0].startswith('Joke')]
WeatherGrp = [t for t in group_data if t[0].startswith('Weather')]
TimeGrp = [t for t in group_data if t[0].startswith('Time')]

# External data separation by ACTION
Pizza = [t for t in act_text if t[0].startswith('PIZZA')]
Greet = [t for t in act_text if t[0].startswith('GREET')]
Joke = [t for t in act_text if t[0].startswith('JOKE')]
Weather = [t for t in act_text if t[0].startswith('WEATHER')]
Time = [t for t in act_text if t[0].startswith('TIME')]

###### REPEAT FOR ALL ACTIONS
PizzaWords = [[i for i, j in Pizza],
              [j for i, j in Pizza]]
GreetWords = [[i for i, j in Greet],
              [j for i, j in Greet]]
JokeWords = [[i for i, j in Joke],
              [j for i, j in Joke]]
WeatherWords =  [[i for i, j in Weather],
              [j for i, j in Weather]]
TimeWords = [[i for i, j in Time],
              [j for i, j in Time]]
PizzaWordsGrp = [[i for i, j in PizzaGrp],
              [j for i, j in PizzaGrp]]
GreetWordsGrp = [[i for i, j in GreetGrp],
              [j for i, j in GreetGrp]]
JokeWordsGrp = [[i for i, j in JokeGrp],
              [j for i, j in JokeGrp]]
WeatherWordsGrp =  [[i for i, j in WeatherGrp],
              [j for i, j in WeatherGrp]]
TimeWordsGrp = [[i for i, j in TimeGrp],
              [j for i, j in TimeGrp]]



WordCounter(PizzaWords)
WordCounter(GreetWords)
WordCounter(JokeWords)
WordCounter(WeatherWords)
WordCounter(TimeWords)

### Sentiment Analysis
# Remove ACTION from tuple so we just have a list of the text
PizzaWords = PizzaWords[-1]
GreetWords = GreetWords[-1]
JokeWords = JokeWords[-1]
WeatherWords = WeatherWords[-1]
TimeWords = TimeWords[-1]
PizzaWordsGrp = PizzaWordsGrp[-1]
GreetWordsGrp = GreetWordsGrp[-1]
JokeWordsGrp = JokeWordsGrp[-1]
WeatherWordsGrp = WeatherWordsGrp[-1]
TimeWordsGrp = TimeWordsGrp[-1]

# Split the group data 
PWG = []
for txt in PizzaWordsGrp:
    PWG.append(clean_text(txt))

PWG = [i.split() for i in PWG]

GWG = []
for txt in GreetWordsGrp:
    GWG.append(clean_text(txt))

GWG = [i.split() for i in GWG]

JWG = []
for txt in JokeWordsGrp:
    JWG.append(clean_text(txt))

JWG = [i.split() for i in JWG]

WWG = []
for txt in WeatherWordsGrp:
    WWG.append(clean_text(txt))

WWG = [i.split() for i in WWG]

TWG = []
for txt in TimeWordsGrp:
    TWG.append(clean_text(txt))

TWG = [i.split() for i in TWG]

WordCounter(PWG)
WordCounter(GWG)
WordCounter(JWG)
WordCounter(WWG)
WordCounter(TWG)




# Create list of strings so we can run Sentiment Analyzer
PizzaString = list(map(' '.join,PizzaWords))
GreetString = list(map(' '.join,GreetWords))
JokeString = list(map(' '.join,JokeWords))
WeatherString = list(map(' '.join,WeatherWords))
TimeString = list(map(' '.join,TimeWords))
PizzaStringGrp = list(map(' '.join,PWG))
GreetStringGrp = list(map(' '.join,GWG))
JokeStringGrp = list(map(' '.join,JWG))
WeatherStringGrp = list(map(' '.join,WWG))
TimeStringGrp = list(map(' '.join,TWG))

### initialize the analyzer
analyzer = SentimentIntensityAnalyzer()

# Append the polarity scores to a list for each action
# Calculate the mean for each of the components for each ACTION
PizzaResults = []
for x in PizzaString:
    result1 = analyzer.polarity_scores(x)
    PizzaResults.append(result1)
PizzaMeanNeg = sum(d['neg'] for d in PizzaResults) / len(PizzaResults)
PizzaMeanNeu = sum(d['neu'] for d in PizzaResults) / len(PizzaResults)
PizzaMeanPos = sum(d['pos'] for d in PizzaResults) / len(PizzaResults)
PizzaMeanCom = sum(d['compound'] for d in PizzaResults) / len(PizzaResults)

PizzaResultsGrp = []
for x in PizzaStringGrp:
    result1 = analyzer.polarity_scores(x)
    PizzaResultsGrp.append(result1)
PGrpMeanNeg = sum(d['neg'] for d in PizzaResultsGrp) / len(PizzaResultsGrp)
PGrpMeanNeu = sum(d['neu'] for d in PizzaResultsGrp) / len(PizzaResultsGrp)
PGrpMeanPos = sum(d['pos'] for d in PizzaResultsGrp) / len(PizzaResultsGrp)
PGrpMeanCom = sum(d['compound'] for d in PizzaResultsGrp) / len(PizzaResultsGrp)


GreetResults = []
for x in GreetString:
    result1 = analyzer.polarity_scores(x)
    GreetResults.append(result1)
GreetMeanNeg = sum(d['neg'] for d in GreetResults) / len(GreetResults)
GreetMeanNeu = sum(d['neu'] for d in GreetResults) / len(GreetResults)
GreetMeanPos = sum(d['pos'] for d in GreetResults) / len(GreetResults)
GreetMeanCom = sum(d['compound'] for d in GreetResults) / len(GreetResults)

GreetResultsGrp = []
for x in GreetStringGrp:
    result1 = analyzer.polarity_scores(x)
    GreetResultsGrp.append(result1)
GGrpMeanNeg = sum(d['neg'] for d in GreetResultsGrp) / len(GreetResultsGrp)
GGrpMeanNeu = sum(d['neu'] for d in GreetResultsGrp) / len(GreetResultsGrp)
GGrpMeanPos = sum(d['pos'] for d in GreetResultsGrp) / len(GreetResultsGrp)
GGrpMeanCom = sum(d['compound'] for d in GreetResultsGrp) / len(GreetResultsGrp)


JokeResults = []
for x in JokeString:
    result1 = analyzer.polarity_scores(x)
    JokeResults.append(result1)
JokeMeanNeg = sum(d['neg'] for d in JokeResults) / len(JokeResults)
JokeMeanNeu = sum(d['neu'] for d in JokeResults) / len(JokeResults)
JokeMeanPos = sum(d['pos'] for d in JokeResults) / len(JokeResults)
JokeMeanCom = sum(d['compound'] for d in JokeResults) / len(JokeResults)

JokeResultsGrp = []
for x in JokeStringGrp:
    result1 = analyzer.polarity_scores(x)
    JokeResultsGrp.append(result1)
JGrpMeanNeg = sum(d['neg'] for d in JokeResultsGrp) / len(JokeResultsGrp)
JGrpMeanNeu = sum(d['neu'] for d in JokeResultsGrp) / len(JokeResultsGrp)
JGrpMeanPos = sum(d['pos'] for d in JokeResultsGrp) / len(JokeResultsGrp)
JGrpMeanCom = sum(d['compound'] for d in JokeResultsGrp) / len(JokeResultsGrp)


WeatherResults = []
for x in WeatherString:
    result1 = analyzer.polarity_scores(x)
    WeatherResults.append(result1)
WeatherMeanNeg = sum(d['neg'] for d in WeatherResults) / len(WeatherResults)
WeatherMeanNeu = sum(d['neu'] for d in WeatherResults) / len(WeatherResults)
WeatherMeanPos = sum(d['pos'] for d in WeatherResults) / len(WeatherResults)
WeatherMeanCom = sum(d['compound'] for d in WeatherResults) / len(WeatherResults)

WeatherResultsGrp = []
for x in WeatherStringGrp:
    result1 = analyzer.polarity_scores(x)
    WeatherResultsGrp.append(result1)
WGrpMeanNeg = sum(d['neg'] for d in WeatherResultsGrp) / len(WeatherResultsGrp)
WGrpMeanNeu = sum(d['neu'] for d in WeatherResultsGrp) / len(WeatherResultsGrp)
WGrpMeanPos = sum(d['pos'] for d in WeatherResultsGrp) / len(WeatherResultsGrp)
WGrpMeanCom = sum(d['compound'] for d in WeatherResultsGrp) / len(WeatherResultsGrp)


TimeResults = []
for x in TimeString:
    result1 = analyzer.polarity_scores(x)
    TimeResults.append(result1)
TimeMeanNeg = sum(d['neg'] for d in TimeResults) / len(TimeResults)
TimeMeanNeu = sum(d['neu'] for d in TimeResults) / len(TimeResults)
TimeMeanPos = sum(d['pos'] for d in TimeResults) / len(TimeResults)
TimeMeanCom = sum(d['compound'] for d in TimeResults) / len(TimeResults)

TimeResultsGrp = []
for x in TimeStringGrp:
    result1 = analyzer.polarity_scores(x)
    TimeResultsGrp.append(result1)
TGrpMeanNeg = sum(d['neg'] for d in TimeResultsGrp) / len(TimeResultsGrp)
TGrpMeanNeu = sum(d['neu'] for d in TimeResultsGrp) / len(TimeResultsGrp)
TGrpMeanPos = sum(d['pos'] for d in TimeResultsGrp) / len(TimeResultsGrp)
TGrpMeanCom = sum(d['compound'] for d in TimeResultsGrp) / len(TimeResultsGrp)



####### Visualizations for EDA ########
## Bring the means for the different external ACTIONs together into a list
POSlst = [PizzaMeanPos, GreetMeanPos, JokeMeanPos, WeatherMeanPos, TimeMeanPos]
NEGlst = [PizzaMeanNeg, GreetMeanNeg, JokeMeanNeg, WeatherMeanNeg, TimeMeanNeg]
NEUlst = [PizzaMeanNeu, GreetMeanNeu, JokeMeanNeu, WeatherMeanNeu, TimeMeanNeu]
COMlst = [PizzaMeanCom, GreetMeanCom, JokeMeanCom, WeatherMeanCom, TimeMeanCom]
y_lst = ['PIZZA', 'GREET', 'JOKE', 'WEATHER', 'TIME']

plt.figure()
plt.bar(y_lst, POSlst, align='center', alpha=0.5)
plt.xticks(y_lst)
plt.ylabel('Mean POS Sentiment')
plt.title('Mean POS Sentiment for External Data')
plt.show()

plt.figure()
plt.bar(y_lst, NEGlst, align='center', alpha=0.5)
plt.xticks(y_lst)
plt.ylabel('Mean NEG Sentiment')
plt.title('Mean NEG Sentiment for External Data')
plt.show()

plt.figure()
plt.bar(y_lst, NEUlst, align='center', alpha=0.5)
plt.xticks(y_lst)
plt.ylabel('Mean NEU Sentiment')
plt.title('Mean NEU Sentiment for External Data')
plt.show()

plt.figure()
plt.bar(y_lst, COMlst, align='center', alpha=0.5)
plt.xticks(y_lst)
plt.ylabel('Mean Compound Sentiment')
plt.title('Mean Compound Score for External Data')
plt.show()


# comparing the sentiment of group text and external text on PIZZA
labels = ['Pos', 'Neg', 'Neu', 'Compound']
ExtMeans = [PizzaMeanPos, PizzaMeanNeg, PizzaMeanNeu, PizzaMeanCom]
GrpMeans = [PGrpMeanPos, PGrpMeanNeg, PGrpMeanNeu, PGrpMeanCom]

x = np.arange(len(labels))
width = 0.35

fig, ax = plt.subplots()
rects1 = ax.bar(x - width/2, ExtMeans, width, label='External')
rects2 = ax.bar(x + width/2, GrpMeans, width, label='Group')

ax.set_ylabel('Mean')
ax.set_title('Mean Sentiment for External and Group Pizza Data')
ax.set_xticks(x)
ax.set_xticklabels(labels)
ax.legend()
fig.tight_layout()
plt.show()

# comparing the sentiment of group text and external text on GREET
labels = ['Pos', 'Neg', 'Neu', 'Compound']
ExtMeans = [GreetMeanPos, GreetMeanNeg, GreetMeanNeu, GreetMeanCom]
GrpMeans = [GGrpMeanPos, GGrpMeanNeg, GGrpMeanNeu, GGrpMeanCom]

x = np.arange(len(labels))
width = 0.35

fig, ax = plt.subplots()
rects1 = ax.bar(x - width/2, ExtMeans, width, label='External')
rects2 = ax.bar(x + width/2, GrpMeans, width, label='Group')

ax.set_ylabel('Mean')
ax.set_title('Mean Sentiment for External and Group Greet Data')
ax.set_xticks(x)
ax.set_xticklabels(labels)
ax.legend()
fig.tight_layout()
plt.show()

# comparing the sentiment of group text and external text on JOKE
labels = ['Pos', 'Neg', 'Neu', 'Compound']
ExtMeans = [JokeMeanPos, JokeMeanNeg, JokeMeanNeu, JokeMeanCom]
GrpMeans = [JGrpMeanPos, JGrpMeanNeg, JGrpMeanNeu, JGrpMeanCom]

x = np.arange(len(labels))
width = 0.35

fig, ax = plt.subplots()
rects1 = ax.bar(x - width/2, ExtMeans, width, label='External')
rects2 = ax.bar(x + width/2, GrpMeans, width, label='Group')

ax.set_ylabel('Mean')
ax.set_title('Mean Sentiment for External and Group Joke Data')
ax.set_xticks(x)
ax.set_xticklabels(labels)
ax.legend()
fig.tight_layout()
plt.show()

# comparing the sentiment of group text and external text on WEATHER
labels = ['Pos', 'Neg', 'Neu', 'Compound']
ExtMeans = [WeatherMeanPos, WeatherMeanNeg, WeatherMeanNeu, WeatherMeanCom]
GrpMeans = [WGrpMeanPos, WGrpMeanNeg, WGrpMeanNeu, WGrpMeanCom]

x = np.arange(len(labels))
width = 0.35

fig, ax = plt.subplots()
rects1 = ax.bar(x - width/2, ExtMeans, width, label='External')
rects2 = ax.bar(x + width/2, GrpMeans, width, label='Group')

ax.set_ylabel('Mean')
ax.set_title('Mean Sentiment for External and Group Weather Data')
ax.set_xticks(x)
ax.set_xticklabels(labels)
ax.legend()
fig.tight_layout()
plt.show()

# comparing the sentiment of group text and external text on TIME
labels = ['Pos', 'Neg', 'Neu', 'Compound']
ExtMeans = [TimeMeanPos, TimeMeanNeg, TimeMeanNeu, TimeMeanCom]
GrpMeans = [TGrpMeanPos, TGrpMeanNeg, TGrpMeanNeu, TGrpMeanCom]

x = np.arange(len(labels))
width = 0.35

fig, ax = plt.subplots()
rects1 = ax.bar(x - width/2, ExtMeans, width, label='External')
rects2 = ax.bar(x + width/2, GrpMeans, width, label='Group')

ax.set_ylabel('Mean')
ax.set_title('Mean Sentiment for External and Group Time Data')
ax.set_xticks(x)
ax.set_xticklabels(labels)
ax.legend()
fig.tight_layout()
plt.show()
