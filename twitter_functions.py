#----------------------------------------------
# Load dependencies
#----------------------------------------------
import streamlit as st
from streamlit_metrics import metric, metric_row
from PIL import Image
import pandas as pd
import datetime as dt
import base64
import tweepy as tw
import pandas as pd
import yaml
import re
import unicodedata
import nltk
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from textblob import TextBlob
nltk.download('stopwords')
import string
from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator
import matplotlib.pyplot as plt

#----------------------------------------------
# DEFINE VARIABLES
#----------------------------------------------

# English stopwords
stopwords_en = nltk.corpus.stopwords.words('english')

# French stopwords
stopwords_fr = nltk.corpus.stopwords.words('french')
    
# words


#----------------------------------------------
# DEFINE FUNCTIONS
#----------------------------------------------

# Function 1
#-----------------
def get_table_download_link(df):
    # Reference: https://discuss.streamlit.io/t/how-to-download-file-in-streamlit/1806
    """
    Generates a link allowing the data in a given panda dataframe to be downloaded
    in:  dataframe
    out: href string
    """
    csv = df.to_csv(index=False)
    b64 = base64.b64encode(csv.encode()).decode()  # some strings <-> bytes conversions necessary here
    href = f'<a href="data:file/csv;base64,{b64}" download="tweets.csv">Download CSV file</a>'
    return href

# Function 2
#-----------------
def feature_extract(df):
    #TODO: add emoticons and emojis to this! and other punctuation

    # Create pre-clean character count feature
    df['character_ct'] = df.full_text.apply(lambda x: len(x))
    # Create stopword count features (english and french)
    df['stopword_en_ct'] = df.full_text.apply(lambda x: len([x for x in x.split() if x in stopwords_en]))
    df['stopword_fr_ct'] = df.full_text.apply(lambda x: len([x for x in x.split() if x in stopwords_fr]))
    # Create hashtag count feature
    df['hashtag_ct'] = df.full_text.apply(lambda x: len([x for x in x.split() if x.startswith('#')]))
    # Create link count feature
    df['link_ct'] = df.full_text.apply(lambda x: len([x for x in x.split() if x.startswith('https')]))
    # Create @ sign count feature
    df['atsign_ct'] = df.full_text.apply(lambda x: len([x for x in x.split() if x.startswith('@')]))
    # Create numeric count feature
    df['numeric_ct'] = df.full_text.apply(lambda x: len([x for x in x.split() if x.isdigit()]))
    # Create an uppercase count feature
    df['uppercase_ct'] = df.full_text.apply(lambda x: len([x for x in x.split() if x.isupper()]))
    return df

# Function 3a
#-------------
def round1_text_clean(text):
    emoji_pattern = re.compile("["
                               u"\U0001F600-\U0001F64F"  # emoticons
                               u"\U0001F300-\U0001F5FF"  # symbols & pictographs
                               u"\U0001F680-\U0001F6FF"  # transport & map symbols
                               u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
                               u"\U00002500-\U00002BEF"  # chinese char
                               u"\U00002702-\U000027B0"
                               u"\U00002702-\U000027B0"
                               u"\U000024C2-\U0001F251"
                               u"\U0001f926-\U0001f937"
                               u"\U00010000-\U0010ffff"
                               u"\u2640-\u2642"
                               u"\u2600-\u2B55"
                               u"\u200d"
                               u"\u23cf"
                               u"\u23e9"
                               u"\u231a"
                               u"\ufe0f"  # dingbats
                               u"\u3030"
                               "]+", flags=re.UNICODE)
    text = emoji_pattern.sub(r'', text) # remove emoji
    text = ' ' + text # added space because there was some weirdness for first word (strip later)
    text = text.lower() # convert all text to lowercase
    text = re.sub(r'(\s)@\w+', '', text) # remove whole word if starts with @
    text = re.sub(r'(\s)\w*\d\w*\w+', '', text) # remove whole word if starts with number
    text = re.sub(r'https\:\/\/t\.co\/*\w*', '', text) # remove https links
    text = re.sub('[%s]' % re.escape(string.punctuation), '', text) # removes punctuation
    text = re.sub('\[.*?\]', '', text) # removes text in square brackets
    #text = re.sub('\w*\d\w*', '', text) # remove whole word if starts with number
    #text = re.sub(r'(\s)#\w+', '', text) # remove whole word if starts with #
    text = text.strip() # strip text
    return text

# Function 3b
#-------------
text_clean_round1 = lambda x: round1_text_clean(x)

# Function 4
#-------------
def text_clean_round2(text):
    """
    A simple function to clean up the data. All the words that
    are not designated as a stop word is then lemmatized after
    encoding and basic regex parsing are performed.
    """
    wnl = nltk.stem.WordNetLemmatizer()
    stopwords = nltk.corpus.stopwords.words('english')
    text = (unicodedata.normalize('NFKD', text)
    .encode('ascii', 'ignore')
    .decode('utf-8', 'ignore'))
    words = re.sub(r'[^\w\s]', '', text).split()
    return [wnl.lemmatize(word) for word in words if word not in stopwords]

# Function 5
#-------------
def text_clean_round3(text):
    #TODO: add emoticons and emojis to this!
    # Load in stopwords
    stopwords_en = nltk.corpus.stopwords.words('english')
    stopwords_fr = nltk.corpus.stopwords.words('french')
    stopwords = stopwords_en + stopwords_fr
    # Create pre-clean character count feature
    text = text.apply(lambda x: " ".join(x for x in x.split() if x not in stopwords))
    return text

# Function 6
#-----------------
def tweets_ngrams(n, top_n, df):
    """
    Generates series of top ngrams
    n: number of words in the ngram
    top_n: number of ngrams with highest frequencies
    """
    text = df.clean_text
    words = text_clean_round2(''.join(str(text.tolist())))
    result = (pd.Series(nltk.ngrams(words, n)).value_counts())[:top_n]
    return result

# Function 7
#----------------

# Credit: https://jackmckew.dev/sentiment-analysis-text-cleaning-in-python-with-vader.html
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

sid_analyzer = SentimentIntensityAnalyzer()

# Get sentiment
def get_sentiment(text:str, analyser,desired_type:str='pos'):
    # Get sentiment from text
    sentiment_score = analyser.polarity_scores(text)
    return sentiment_score[desired_type]

# Get Sentiment scores
def get_sentiment_scores(df, data_column):
    df[f'positive_score'] = df[data_column].astype(str).apply(lambda x: get_sentiment(x,sid_analyzer,'pos'))
    df[f'negative_score'] = df[data_column].astype(str).apply(lambda x: get_sentiment(x,sid_analyzer,'neg'))
    df[f'neutral_score'] = df[data_column].astype(str).apply(lambda x: get_sentiment(x,sid_analyzer,'neu'))
    df[f'compound_score'] = df[data_column].astype(str).apply(lambda x: get_sentiment(x,sid_analyzer,'compound'))
    return df


# Function 8
#----------------

# Credit: https://www.dataquest.io/blog/tutorial-add-column-pandas-dataframe-based-on-if-else-condition/

# classify based on VADER readme rules
def sentiment_classifier(df, data_column):

    # create a list of our conditions
    conditions = [
        (df[data_column] >= 0.05),
        (df[data_column] > -0.05) & (df[data_column] < 0.05),
        (df[data_column] <= -0.05),
        ]

    # create a list of the values we want to assign for each condition
    values = ['Positive', 'Neutral', 'Negative']
    
    # apply
    df['sentiment'] = np.select(conditions, values)
    return df

# Function 9
#----------------

# Credit: https://jackmckew.dev/sentiment-analysis-text-cleaning-in-python-with-vader.html

def print_top_n_tweets(df, sent_type, num_rows):
    text = 'full_text'
    top = df.nlargest(num_rows, sent_type)
    top_tweets = top[[sent_type,text]].reset_index()
    top_tweets = top_tweets.drop(columns = 'index')
    top_tweets.index = top_tweets.index + 1 
    return top_tweets

# Function 10
#----------------
# Function to convert  
def word_cloud_all(df, wordcloud_words): 
    # convert text_cleaned to word
    text = df.clean_text
    word_list = text_clean_round2(''.join(str(text.tolist())))
    # initialize an empty string
    str1 = " " 
    # return string  
    str2 = str1.join(word_list)
    # generate word cloud
    wordcloud = WordCloud(max_font_size=100, max_words=wordcloud_words, background_color="white").generate(str2)
    return wordcloud

# Function 11
#----------------
# Function to convert  
def word_cloud_sentiment(df, sent_type, num_rows, wordcloud_words): 
    # slice df to top n rows for sentiment type selected
    sliced_df = df.nlargest(num_rows, sent_type)
    # convert text_cleaned to word
    text = sliced_df.clean_text
    word_list = text_clean_round2(''.join(str(text.tolist())))
    # initialize an empty string
    str1 = " " 
    # return string  
    str2 = str1.join(word_list)
    # generate word cloud
    wordcloud = WordCloud(max_font_size=100, max_words=wordcloud_words, background_color="white").generate(str2)
    return wordcloud