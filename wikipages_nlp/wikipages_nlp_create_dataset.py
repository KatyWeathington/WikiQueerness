#

'''
Analyzer Harness for Wikipages

This is a great script. It will need some things installed:
pip install --upgrade nltk
pip install beautifulsoup4
pip install twython
pip install progressbar2
pip install pandas
** probably some others that Jed has forgotten
'''

__author__ = 'whatknows'
from analyzer import Analyzer
from bs4 import BeautifulSoup
import pandas as pd
from nltk.corpus import stopwords
import re, os
import pprint
import nltk
from tqdm import tqdm
import progressbar
nltk.download('vader_lexicon')
nltk.download('punkt')
nltk.download('stopwords')

# Global Variables for Script
pb_widgets=[
    ' [',
    progressbar.Counter(), ' | ',
    progressbar.Percentage(), ' | ',
    progressbar.Timer(),
    '] ',
    progressbar.Bar(),
    ' (', progressbar.AdaptiveETA(), ') ',
    ]



# Type your sample message here
msg = 'Hi! We love all the text. :D'

# Start analyzing the text with the different techniques

### FEATURE GENERATORS ###

# Empath
# @TODO: Add link to Empath here.
analyzer = Analyzer('Empath')
headers = analyzer.headers
metrics = analyzer.analyze(msg)
pprint.pprint("-- Empath Test--")
pprint.pprint(msg)
pprint.pprint(metrics)

# LIWC
# @TODO: Add link to LIWC here.
analyzer = Analyzer('LIWC')
headers = analyzer.headers
metrics = analyzer.analyze(msg)
pprint.pprint("-- LIWC --")
pprint.pprint(msg)
pprint.pprint(metrics)


import liwc
import re
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from empath import Empath

parse, liwc_category_names = liwc.load_token_parser('LIWC2007_English100131.dic')
headers = ['funct', 'pronoun', 'ppron', 'i', 'we', 'you',
                'shehe', 'they', 'ipron', 'article', 'verb',
                'auxverb', 'past', 'present', 'future', 'adverb',
                'preps', 'conj', 'negate', 'quant', 'number', 'swear',
                'social', 'family', 'friend', 'humans', 'affect',
                'posemo', 'negemo', 'anx', 'anger', 'sad', 'cogmech',
                'insight', 'cause', 'discrep', 'tentat', 'certain',
                'inhib', 'incl', 'excl', 'percept', 'see', 'hear',
                'feel', 'bio', 'body', 'health', 'sexual', 'ingest',
                'relativ', 'motion', 'space', 'time', 'work', 'achieve',
                'leisure', 'home', 'money', 'relig', 'death', 'assent',
                'nonfl', 'filler']

tokens = msg.lower().split()

wc = len(tokens)
from collections import Counter
counts = Counter(category for token in tokens for category in parse(token))
print(counts)

results = {}
for item in headers:
    results[item] = 0

results['wc'] = wc

for item in counts:
    results[item] = counts[item]/wc



print(counts)

# VADER
# @TODO: Add link to VADER here.
analyzer = Analyzer('Vader')
headers = analyzer.headers
metrics = analyzer.analyze(msg)
pprint.pprint("-- Vader Test--")
pprint.pprint(msg)
pprint.pprint(metrics)

# SPAM DETECTION
# spam_detection = Spam_detect()
# result = spam_detection.is_spam(msg)
# pprint.pprint(msg)
# pprint.pprint(result)

# BAG OF WORDS
# review_text = BeautifulSoup(msg, "html.parser").get_text()
# letters_only = re.sub("[^a-zA-Z]", " ", review_text) #removes non-letters
# words = letters_only.lower().split() #spliting into words
# stops = set(stopwords.words("english"))
# meaningful_words = [w for w in words if not w in stops] #remove stop words
# print(" ".join(meaningful_words))
# TODO: Turn this into a class.
# Make this be like other classes in this repo.
# It should accept some text.
# It should return words with the counts.
# Optionally limit the amount or responses?

# KYLO
# Link: https://github.com/cu-idlab/kylo
## This is complicated enough taht you should refer to the Kylo Harness in
## the kylo repository.


## UTILITY FUNCTIONS
# Function to remove tags
def remove_tags(html):

    # parse html content
    soup = BeautifulSoup(html, "html.parser")

    for data in soup(['style', 'script']):
        # Remove tags
        data.decompose()

    # return data by retrieving the tag content
    return ' '.join(soup.stripped_strings)


def parse_to_text(input,is_json=False,parse_text=True):
    if is_json:
        page_html = input['parse']['text']#['*']
    else:
        page_html = input

    # Parse the HTML into Beautiful Soup
    soup = BeautifulSoup(page_html,'lxml')

    # Remove sections at end
    bad_sections = ['See_also','Notes','References','Bibliography','External_links']
    sections = soup.find_all('h2')
    for section in sections:
        if section.span['id'] in bad_sections:

            # Clean out the divs
            div_siblings = section.find_next_siblings('div')
            for sibling in div_siblings:
                sibling.clear()

            # Clean out the ULs
            ul_siblings = section.find_next_siblings('ul')
            for sibling in ul_siblings:
                sibling.clear()

    # Get all the paragraphs
    paras = soup.find_all('p')

    text_list = []

    for para in paras:
        if parse_text:
            _s = para.text
            # Remove the citations
            _s = re.sub(r'\[[0-9]+\]','',_s)
            text_list.append(_s)
        else:
            text_list.append(str(para))

    return '\n'.join(text_list)

# STEPS FOR PIPELINE CREATION

# 0. CREATE DATAFRAME TO HOLD RESULTS

# Create header lists
print('-- Add Headers to DF --')
analyzer = Analyzer('Vader')
headers = analyzer.headers
vader_headers = []
for i in headers:
    vader_headers.append('vader_' + i)

analyzer = Analyzer('Empath')
headers = analyzer.headers
empath_headers = []
for i in headers:
    empath_headers.append('vader_' + i)

# analyzer = Analyzer('LIWC')
# headers = analyzer.headers
# liwc_headers = []
# for i in headers:
#     liwc_headers.append('vader_' + i)


# 1. OPEN REFERENCE DATASET
print("--OPEN REFERENCE DATASET--")
names_df = pd.read_csv('../peopleByBirthYears.csv')
# // Drop a bunch of rows for testing
# names_df = names_df.iloc[-10: , :]
names_df['text'] = ''
# columns
# Index(['Name', 'Year', 'inTransWomen', 'inTransMen', 'inNB', 'inTrans',
#        'inWomen', 'inMen', 'inLGBT', 'inBisexual_people', 'inGay_men',
#        'inLesbians', 'inQueer', 'links', 'categories', 'QID'],
#       dtype='object')


# GRAB FILES FROM FOLDER
print("--GENERATE TEXT FILES --")
print("--RAW TEXT--")
output_folder = '../wikipages/'
# for i, row in names_df.iterrows():
for i, row in progressbar.progressbar(names_df.iterrows(), max_value=names_df.shape[0], widgets=pb_widgets):
    name = row['Name']
    name = name.replace('/','')
    filename = str(name)+'.html'

    # 2. REMOVE HTML CONTENT
    # If the source HTML file exists
    if os.path.exists(os.path.join(output_folder, filename)):
        # But there isn't a raw text version of it, create it.
        if not os.path.exists(os.path.join(output_folder, filename+'_raw')):
            file = open(os.path.join(output_folder, filename),'r',encoding='utf-8')
            html = file.read()
            raw_text = remove_tags(html)
            # print(raw_text)
            with open(os.path.join(output_folder, filename+'_raw'),'w',encoding='utf-8') as w:
                w.write(raw_text)
            file.close()
        else:
            # Otherwise, just read the raw text
            file = open(os.path.join(output_folder, filename+'_raw'),'r',encoding='utf-8')
            raw_text = file.read()
            file.close()
        names_df.at[i,'text'] = raw_text
        # else:
        #     print('-',end='')


# GRAB FILES FROM FOLDER
# print("--PARSE TO TEXT--")
# output_folder = '../wikipages/'
# # for i, row in names_df.iterrows():
# for i, row in progressbar.progressbar(names_df.iterrows(), max_value=names_df.shape[0], widgets=pb_widgets):
#     name = row['Name']
#     name = name.replace('/','')
#     filename = str(name)+'.html'
#
#     # 2. REMOVE HTML CONTENT
#     if os.path.exists(os.path.join(output_folder, filename)):
#         if not os.path.exists(os.path.join(output_folder, filename+'_wikifunc')):
#             file = open(os.path.join(output_folder, filename),'r',encoding='utf-8')
#             html = file.read()
#             raw_text = parse_to_text(html)
#             with open(os.path.join(output_folder, filename+'_wikifunc'),'w',encoding='utf-8') as w:
#                 w.write(raw_text)
#             file.close()
#             # names_df.at[i,'text'] = raw_text
#         # else:
#         #     print('-',end='')


# 3. RUN EACH OF THE NLP LIBRARIES

# WORD Count
print("--WORD COUNT--")
names_df['word_count'] = ''

for i, row in progressbar.progressbar(names_df.iterrows(), max_value=names_df.shape[0], widgets=pb_widgets):
    tokens = nltk.word_tokenize(row['text'])
    count = len(nltk.Text(tokens))
    # print(count)
    names_df.at[i,'word_count'] = count


# VADER
print("--VADER--")
print("--add headers--")

analyzer = Analyzer('Vader')
headers = analyzer.headers
for col in headers:
    names_df['vader_'+col] = ''

print("--parse--")
for i, row in progressbar.progressbar(names_df.iterrows(), max_value=names_df.shape[0], widgets=pb_widgets):
    metrics = analyzer.analyze(row['text'])
    # print(metrics)
    for col in headers:
        names_df.at[i,'vader_'+col] = metrics[col]


# EMPATH
print("--EMPATH--")
print("--add headers--")
analyzer = Analyzer('Empath')
headers = analyzer.headers
for col in headers:
    names_df['empath_'+col] = ''

print("--parse--")
for i, row in progressbar.progressbar(names_df.iterrows(), max_value=names_df.shape[0], widgets=pb_widgets):
    metrics = analyzer.analyze(row['text'])
    for col in headers:
        names_df.at[i,'empath_'+col] = metrics[col]

print("-- CLEAN UP DF --")
names_df.drop(columns=['text'])

print("-- SAVE THE FILE --")
names_df.to_csv('wikipages_nlp.csv')
