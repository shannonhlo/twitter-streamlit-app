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
num_of_tweets = st.sidebar.number_input('Maximum number of tweets', 100)


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

#----------------------------------------------------------
## SECTION 1: DESCRIPTIVE ANALYSIS
#----------------------------------------------------------

st.header('Descriptive Analysis')

## 1.1: KPI CARDS
#----------------------------
total_tweets = len(df_tweets['full_text'])
highest_retweets = max(df_tweets['rt_count'])
highest_likes = max(df_tweets['fav_count'])

st.subheader('Tweet Summary')
metric_row(
    {
        "Number of tweets": total_tweets,
        "Highest number of retweets": highest_retweets,
        "Highest number of likes": highest_likes,
    }
)

## 1.2: RAW & DOWNLOADABLE DATA TABLE
#----------------------------

# Show raw data if selected
if st.checkbox('Show raw data'):
    st.subheader('Raw data')
    st.write(df_new)

# Click to download raw data as CSV
st.markdown(tf.get_table_download_link(df_tweets), unsafe_allow_html=True)


## 1.3: TWEETS BY DAY BAR CHART
#----------------------------

# Subtitle
st.subheader('Number of Tweets by Day')

# Altair chart: number of total tweets by day
tweets_bar = alt.Chart(df_tweets).mark_bar().encode(
                    x = alt.X('monthdate(created_at):O', axis = alt.Axis(title = 'Month Date')),
                    y = alt.Y('count(id):Q', axis = alt.Axis(title = 'Number of Total Tweets'))#,
                    #tooltip = [alt.Tooltip('sentiment', title = 'Sentiment Group'), 'count(id):Q', alt.Tooltip('average(compound_score)', title = 'Avg Compound Score'), alt.Tooltip('median(compound_score)', title = 'Median Compound Score')] ,
                ).properties(
                    height = 350
                ).interactive()

st.altair_chart(tweets_bar, use_container_width=True)


## 1.4: FEATURE EXTRACTION BAR CHART #TODO FIX THIS
#----------------------------

# Subtitle
#st.subheader('Feature Extractions Counts')

# Bar chart: count features
#df_count = df_tweets[['stopword_en_ct', 'stopword_fr_ct', 'hashtag_ct', 'atsign_ct', 'link_ct', 'numeric_ct', 'uppercase_ct']]

#st.bar_chart(df_count)


## 1.5: NGRAM WORD COUNTS
#----------------------------

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
ngram_visual = tf.tweets_ngrams(ngram_num, 10, df_tweets)
ngram_visual['ngram'] = ngram_visual.index

# Conditional subtitle
st.write(ngram_nm)

# Altair chart: ngram word frequencies
ngram_bar = alt.Chart(ngram_visual).mark_bar().encode(
                    x = alt.X('frequency', axis = alt.Axis(title = 'Word Frequency')),
                    y = alt.Y('ngram', axis = alt.Axis(title = 'Ngram'), sort = '-x'),
                    tooltip = [alt.Tooltip('frequency', title = 'Ngram Frequency')],#,  alt.Tooltip('Ngram', title = 'Ngram Word(s)')] ,
                ).properties(
                    height = 350
                )

st.altair_chart(ngram_bar, use_container_width=True)

## 2.0 SENTIMENT ANALYSIS
#----------------------------------------------------------

# Subtitle
st.header('Sentiment Analysis')

# Expander for Methodology
expander_bar = st.beta_expander("Methodology")
expander_bar.markdown("""
* Applying the [VADER Sentiment](https://github.com/cjhutto/vaderSentiment) library to our text data
* [VADER](https://github.com/cjhutto/vaderSentiment#vader-sentiment-analysis) (**V**alence **A**ware **D**ictionary and s**E**ntiment **R**easoner) = lexicon and rule-based sentiment analysis tool, specifically attuned to sentiments expressed in social media
* [Compound score](https://github.com/cjhutto/vaderSentiment#about-the-scoring) = computed by summing the valence scores of each word in the lexicon, adjusted according to the rules, and then normalized to be between -1 (most extreme negative) and +1 (most extreme positive)
* Positive sentiment: compound score >= 0.05
* Neutral sentiment: (compound score > -0.05) and (compound score < 0.05)
* Negative sentiment: compound score <= -0.05
""")

# Get sentiment scores on raw tweets
text_sentiment = tf.get_sentiment_scores(df_tweets, 'full_text')

# Add sentiment classification
text_sentiment = tf.sentiment_classifier(df_tweets, 'compound_score')

# Select columns to output
df_sentiment = df_tweets[['created_at', 'full_text', 'sentiment', 'positive_score', 'negative_score', 'neutral_score', 'compound_score']]


## 2.1: SUMMARY CARDS
#----------------------------

# Show raw data if selected
sentiment_group = df_sentiment.groupby('sentiment').agg({'sentiment': 'count'}).transpose()

# Click to download raw data as CSV :)
st.subheader('Summary')
metric_row(
    {
        "% 😃 Positive Tweets": "{:.0%}".format(max(sentiment_group.Positive)/total_tweets),
        "% 😐 Neutral Tweets": "{:.0%}".format(max(sentiment_group.Neutral)/total_tweets),
        "% 😡 Negative Tweets": "{:.0%}".format(max(sentiment_group.Negative)/total_tweets),
    }
)

## 2.2: RAW & DOWNLOADABLE DATA TABLE
#----------------------------
if st.checkbox('Show VADER results for each Tweet'):
    st.subheader('Raw data')
    st.write(df_sentiment)

st.markdown(tf.get_table_download_link(df_sentiment), unsafe_allow_html=True)


## 2.3: SENTIMENT BY DAY BAR CHART
#----------------------------
import altair as alt

sentiment_bar = alt.Chart(df_sentiment).mark_bar().encode(
                    x = alt.X('count(id):Q', stack="normalize", axis = alt.Axis(title = 'Percent of Total Tweets', format='%')),
                    y = alt.Y('monthdate(created_at):O', axis = alt.Axis(title = 'Month Date')),
                    tooltip = [alt.Tooltip('sentiment', title = 'Sentiment Group'), 'count(id):Q', alt.Tooltip('average(compound_score)', title = 'Avg Compound Score'), alt.Tooltip('median(compound_score)', title = 'Median Compound Score')] ,
                   # y  = alt.Y('sentiment', sort = '-x'),
                    color=alt.Color('sentiment',
                        scale=alt.Scale(
                        domain=['Positive', 'Neutral', 'Negative'],
                        range=['forestgreen', 'lightgray', 'indianred']))
                ).properties(
                    height = 400
                ).interactive()

# Write the chart
st.subheader('Classifying Tweet Sentiment by Day')
st.altair_chart(sentiment_bar, use_container_width=True)


## 2.4: ANALYZING TOP TWEETS (wordcloud + top tweets)
#----------------------------
st.subheader('Sentiment Wordcloud')
st.write('''*Note: Wordcloud will run on all tweets if sentiment type is ALL*''')

with st.form('Form1'):
    score_type = st.selectbox('Select sentiment', ['All', 'Positive', 'Neutral', 'Negative'], key=1)
    wordcloud_words = st.number_input('Choose the max number of words for the word cloud', 15, key = 3)
    num_tweets =  st.number_input('Choose the top number of tweets *', 5, key = 2)
    submitted1 = st.form_submit_button('Regenerate Wordcloud')

# Scenarios

# Scenario 1: All
if score_type == 'All':
    score_type_nm= 'compound_score'

# Scenario 2: Positive
if score_type == 'Positive':
    score_type_nm= 'positive_score'

# Scenario 3: Neutral
if score_type == 'Neutral':
    score_type_nm = 'neutral_score'

# Scenario 4: Negative
if score_type == 'Negative':
    score_type_nm = 'negative_score'

# Run wordlcloud for top n tweets
if score_type == 'All':         
    wordcloud = tf.word_cloud_all(text_sentiment, wordcloud_words)
else:
    wordcloud = tf.word_cloud_sentiment(text_sentiment, score_type_nm, num_tweets, wordcloud_words)


# Display the generated wordcloud image:
st.set_option('deprecation.showPyplotGlobalUse', False)
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis("off")
plt.show()
st.write('Word Cloud Generator')
st.pyplot()

# Run the top n tweets
top_tweets_res = tf.print_top_n_tweets(df_sentiment, score_type_nm, num_tweets)

# Show resuts as a streamlit table
st.write('Show the top tweets!')
st.table(top_tweets_res)

## 2.5: COMPOUND SCORE HISTOGRAM
#----------------------------
sentiment_histo= alt.Chart(df_sentiment).mark_bar().encode(
                    x = alt.X('compound_score:O', axis = alt.Axis(title = 'VADER Compound Score (Binned)'), bin=alt.Bin(extent=[-1, 1], step=0.25)),
                    y = alt.Y('count(id):Q', axis = alt.Axis(title = 'Number of Tweets')),
                    tooltip = [alt.Tooltip('sentiment', title = 'Sentiment Group'), 'count(id):Q', alt.Tooltip('average(compound_score)', title = 'Average Compound Score'), alt.Tooltip('median(compound_score)', title = 'Median Compound Score')] ,
                   # y  = alt.Y('sentiment', sort = '-x'),
                    color=alt.Color('sentiment',
                        scale=alt.Scale(
                        domain=['Positive', 'Neutral', 'Negative'],
                        range=['forestgreen', 'lightgray', 'indianred']))
                ).properties(
                    height = 400
                ).interactive()

# Write the chart
st.subheader('Checking Sentiment Skewness')
st.write('VADER Compound Scores Histogram')
st.altair_chart(sentiment_histo, use_container_width=True)    