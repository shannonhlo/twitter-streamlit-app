#----------------------------------------------
# PART 1: LOAD DEPENDENCIES
#----------------------------------------------

from nltk.featstruct import _default_fs_class
import twitter_functions as tf # custom functions file
import streamlit as st
from streamlit_metrics import metric, metric_row
from PIL import Image
import pandas as pd
import datetime as dt
import base64
import tweepy as tw
import pandas as pd
import yaml
import string
import re
import unicodedata
import nltk
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from textblob import TextBlob
from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator
import matplotlib.pyplot as plt
import altair as alt


#----------------------------------------------
# PART 2: DEFINE VARIABLES & FUNCTIONS
#----------------------------------------------

import twitter_functions as tf # custom functions file

#------------------------------------#
# 1) APP TITLE, DESCRIPTION & LAYOUT
## Page expands to full width
## Layout... Divide page to 3 columns (col1 = sidebar, col2 and col3 = page contents)
#------------------------------------#
st.set_page_config(layout="wide")

# Title
#------------------------------------#

image = Image.open('twitter_logo.png')

st.image(image, width = 100)

st.title('Twitter Data App')
st.markdown("""
This app provides insights on tweets from the past week that contain a specific hashtag or keyword.
""")


# About
#------------------------------------#

expander_bar = st.beta_expander("About")
expander_bar.markdown("""
* **Creators:** [Shannon Lo](https://shannonhlo.github.io/) & [Domenic Fayad](https://www.fullstaxx.com/)
* **Python libraries:** base64, pandas, streamlit, tweepy, numpy, matplotlib, seaborn, BeautifulSoup, requests, json, time, yaml
""")


# Layout
#------------------------------------#

col1 = st.sidebar
col2, col3 = st.beta_columns((2,1)) # col1 is 2x greater than col2


#-----------------------------------#
# 2) SIDEBAR, USER INPUT
## User to specify keyword or hashtag
#-----------------------------------#

## Sidebar title
st.sidebar.header('User Inputs')

## User input: specify hashtag or keyword used to search for relevant tweets
user_word = st.sidebar.text_input("Enter a hashtag or keyword", "#covidcanada")

## User input: select language
select_language = st.sidebar.radio('Tweet language', ('All', 'English', 'French'))
if select_language == 'English':
    language = 'en'
if select_language == 'French':
    language = 'fr'

## User input: include retweets or not
#TODO: understand what retweets actually entails
include_retweets = st.sidebar.checkbox('Include retweets in data')

## User input: number of tweets to return
#TODO: set a cap
num_of_tweets = st.sidebar.number_input('Maximum number of tweets', 15)


#-----------------------------------#
# 3) GET DATA FROM TWITTER API
#-----------------------------------#

## Set up Twitter API access
# Reference: https://gist.github.com/radcliff/47af9f6238c95f6ae239
# Load yml file to dictionary
credentials = yaml.load(open('./credentials.yml'), Loader=yaml.FullLoader)

# Define access keys and tokens
consumer_key = credentials['twitter_api']['consumer_key']
consumer_secret = credentials['twitter_api']['consumer_secret']
access_token = credentials['twitter_api']['access_token']
access_token_secret = credentials['twitter_api']['access_token_secret']

auth = tw.OAuthHandler(consumer_key, consumer_secret)
auth.set_access_token(access_token, access_token_secret)
api = tw.API(auth, wait_on_rate_limit = True)

## Get tweets and store as dataframe
# Reference: https://www.earthdatascience.org/courses/use-data-open-source-python/intro-to-apis/twitter-data-in-python/
# define parameters for API request

if include_retweets == False:
    user_word = user_word + ' -filter:retweets'

# Scenario 1: All languages
if select_language == 'All':
    tweets = tw.Cursor(api.search,
                        q=user_word,
                        tweet_mode = "extended").items(num_of_tweets)

# Scenario 2: Specific language (English or French)
if select_language != 'All':
    tweets = tw.Cursor(api.search,
                        q=user_word,
                        tweet_mode = "extended",
                        lang=language).items(num_of_tweets)

# Store as dataframe
tweet_metadata = [[tweet.created_at, tweet.id, tweet.full_text, tweet.user.screen_name, tweet.retweet_count, tweet.favorite_count] for tweet in tweets]
df_tweets = pd.DataFrame(data=tweet_metadata, columns=['created_at', 'id', 'full_text', 'user', 'rt_count', 'fav_count'])

# Add a new data variable
df_tweets['created_dt'] = df_tweets['created_at'].dt.date

# Add a new time variable
df_tweets['created_time'] = df_tweets['created_at'].dt.time

# Create a new text variable to do manipulations on 
df_tweets['clean_text'] = df_tweets.full_text

# Run function #2: Feature extraction
df_tweets = tf.feature_extract(df_tweets)

# Run function #3: Round 1 text cleaning (convert to lower, remove numbers, @, punctuation, numbers. etc.)
df_tweets['clean_text'] = df_tweets.clean_text.apply(tf.text_clean_round1)

# Run function #4: Round 2 text cleaning (create list of tokenized words)
#TODO NOT RUNNING -- FIX?
#df_tweets.clean_text  = tf.text_clean_round2(df_tweets.clean_text)

## Run function #5: Round 3 text cleaning (remove stop words)
df_tweets.clean_text  = tf.text_clean_round3(df_tweets.clean_text)

# Create list of words
words2 = df_tweets.clean_text.tolist()

# Cleaned up dataframe
df_new = df_tweets[["created_dt", "created_time", "full_text", "user", "rt_count", "fav_count"]]
df_new = df_new.rename(columns = {"created_dt": "Date", 
                                 "created_time": "Time", 
                                  "full_text": "Tweet", 
                                  "user": "Username", 
                                  "rt_count": "Retweets",  
                                  "fav_count": "Favourites"})
#-----------------------------------#
# 4) MAINPANEL, VISUALS
#-----------------------------------#

## KPI CARDS
#----------------------------------------------------------
total_tweets = len(df_tweets['full_text'])
highest_retweets = max(df_tweets['rt_count'])
highest_likes = max(df_tweets['fav_count'])

st.subheader('Summary')
metric_row(
    {
        "Number of tweets": total_tweets,
        "Highest number of retweets": highest_retweets,
        "Highest number of likes": highest_likes,
    }
)

## RAW DATA TABLE
#----------------------------------------------------------
if st.checkbox('Show raw Tweets data'):
    st.subheader('Raw data')
    st.write(df_new)
#st.write(center_info_data)

#st.subheader('Raw data')
#st.write(df_new)

## DOWNLOADABLE DATA
#----------------------------------------------------------
st.markdown(tf.get_table_download_link(df_tweets), unsafe_allow_html=True)


## TWEETS PER DAY BARCHART
#----------------------------------------------------------

# Subtitle
st.subheader('Number of Tweets by Day')

# Create dataframe with count of unique tweets by date
tweets_by_day = df_tweets[['created_dt', 'id']].groupby(['created_dt']).agg(['nunique']).reset_index()
tweets_by_day.columns = ['created_dt', 'id']

st.bar_chart(tweets_by_day.set_index('created_dt'))

## FEATURE EXTRACTION COUNTS BARCHART? TODO!
#----------------------------------------------------------

# Subtitle
st.subheader('Feature Extractions Counts')

# Bar chart: count features
df_count = df_tweets[['stopword_en_ct', 'stopword_fr_ct', 'hashtag_ct', 'atsign_ct', 'link_ct', 'numeric_ct', 'uppercase_ct']]

st.bar_chart(df_count)

## SENTIMENT ANALYSIS
#----------------------------------------------------------

# Subtitle
st.subheader('Sentiment Analysis')

# Get sentiment scores on raw tweets
text_sentiment = tf.get_sentiment_scores(df_tweets, 'full_text')

# Add sentiment classification
text_sentiment = tf.sentiment_classifier(df_tweets, 'compound_score')

# Select columns to output
df_sentiment = df_tweets[['created_dt', 'full_text', 'sentiment', 'positive_score', 'negative_score', 'neutral_score', 'compound_score']]

# Write dataframe
st.write(df_sentiment)

## NGRAM WORD COUNTS
#----------------------------------------------------------

# Subtitle
st.subheader('Word Frequency and Ngrams')

# User selections
ngram_option = st.selectbox(
                'Select the number of ngrams',
                ('Single', 'Bigram', 'Trigram'))

# Scenarios
# Scenario 1: Single ngram
if ngram_option == 'Single':
    ngram_num = 1
    ngram_nm = 'Single Word Frequencies'

# Scenario 2: Bigrams
if ngram_option == 'Bigram':
    ngram_num = 2
    ngram_nm = 'Bigram Word Frequencies'

# Scenario 3: Trigrams
if ngram_option == 'Trigram':
    ngram_num = 3
    ngram_nm = 'Trigram Word Frequencies'

# Display ngram based on selection
ngram_visual = tf.tweets_ngrams(ngram_num, 15, df_tweets)

# Conditional subtitle
st.write(ngram_nm)

# Write word frequencies
st.write(ngram_visual)

## WORDCLOUD
#----------------------------------------------------------

# Subtitle
st.subheader('Word Cloud')

# Number of words
wordcloud_words = st.number_input('Choose the max number of words for the word cloud', 15, key = 2)

# Run word cloud function
wordcloud = tf.word_cloud(df_tweets, wordcloud_words)

# Display the generated image:
st.set_option('deprecation.showPyplotGlobalUse', False)
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis("off")
plt.show()
st.write('Word Cloud Generator')
st.pyplot()

# https://towardsdatascience.com/add-animated-charts-to-your-dashboards-with-streamlit-python-f41863f1ef7c