#----------------------------------------------
# PART 1: LOAD DEPENDENCIES
#----------------------------------------------

from nltk.featstruct import _default_fs_class
import twitter_functions as tf # custom functions file
import streamlit as st
from streamlit_metrics import metric, metric_row
import streamlit.components.v1 as components
from PIL import Image
import pandas as pd
#import datetime as dt
#import base64
import tweepy as tw
import yaml
#import string
#import re
#import unicodedata
#import nltk
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation as LDA
#import numpy as np
import matplotlib.pyplot as plt
#import seaborn as sns
#from textblob import TextBlob
#from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator
import matplotlib.pyplot as plt
import altair as alt

# Once merged
#TODO - collapsable sections
#TODO - show to friends
#TODO - code refactor
#TODO - create gif
#TODO - post


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

st.image(image, width = 50)

st.title('Twitter Data App')
st.markdown("""
Search a Twitter hashtag to run the text analyzer!
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

with st.form(key ='Form1'):
    with st.sidebar:
        user_word = st.text_input('Enter a keyword', 'covidcanada')    
        select_language = st.radio('Tweet language', ('All', 'English', 'French'))
        include_retweets = st.checkbox('Include retweets in data') # what does this mean?
        num_of_tweets = st.number_input('Maximum number of tweets', min_value=1, max_value=10000, value=100)
        submitted1 = st.form_submit_button(label = 'Search Twitter 🔎')


# About
#------------------------------------#

## Sidebar title
st.sidebar.text("") # spacing
st.sidebar.header('About the App')
expander_bar = st.sidebar.beta_expander("About")
expander_bar.markdown("""
* **Creators:** [Shannon Lo](https://shannonhlo.github.io/) & [Domenic Fayad](https://www.fullstaxx.com/)
* **References:**
  * https://discuss.streamlit.io/t/how-to-download-file-in-streamlit/1806
  * https://jackmckew.dev/sentiment-analysis-text-cleaning-in-python-with-vader.html
  * https://www.dataquest.io/blog/tutorial-add-column-pandas-dataframe-based-on-if-else-condition/
  * https://ourcodingclub.github.io/tutorials/topic-modelling-python/
""")


# Social
#------------------------------------#
st.sidebar.text("") # spacing
st.sidebar.header('Developer Contact')
st.sidebar.write("[![Star](https://img.shields.io/github/stars/shannonhlo/twitter-streamlit-app.svg?logo=github&style=social)](https://github.com/shannonhlo/twitter-streamlit-app/branches)")
st.sidebar.write("[![Follow](https://img.shields.io/twitter/follow/shannonhlo26?style=social)](https://twitter.com/shannonhlo26)")
st.sidebar.write("[![Follow](https://img.shields.io/twitter/follow/DomenicFayad?style=social)](https://twitter.com/DomenicFayad)")


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

if select_language == 'English':
    language = 'en'
if select_language == 'French':
    language = 'fr'

if include_retweets == False:
    user_word = '#' + user_word + ' -filter:retweets'

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


## CREATE EXPANDER FOR DESCRIPTIVE ANALYSIS
descriptive_expander = st.beta_expander('Expand to see more descriptive analysis', expanded=False)

## 1.2: TWEETS BY DAY LINE CHART
#----------------------------

# Subtitle
descriptive_expander.subheader('Number of Tweets by Day')

# Altair chart: number of total tweets by day
tweets_bar = alt.Chart(df_tweets).mark_line().encode(
                    x = alt.X('monthdate(created_at):O', axis = alt.Axis(title = 'Month Date')),
                    y = alt.Y('count(id):Q', axis = alt.Axis(title = 'Number of Total Tweets'))
                    #tooltip = [alt.Tooltip('sentiment', title = 'Sentiment Group'), 'count(id):Q', alt.Tooltip('average(compound_score)', title = 'Avg Compound Score'), alt.Tooltip('median(compound_score)', title = 'Median Compound Score')] ,
                ).properties(
                    height = 350
                ).interactive()

descriptive_expander.altair_chart(tweets_bar, use_container_width=True)


## 1.3: RAW & DOWNLOADABLE DATA TABLE
#----------------------------

# Show raw data if selected
if descriptive_expander.checkbox('Show raw data'):
    descriptive_expander.subheader('Raw data')
    descriptive_expander.write(df_new)

# Click to download raw data as CSV
descriptive_expander.markdown(tf.get_table_download_link(df_tweets), unsafe_allow_html=True)


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
descriptive_expander.subheader('Word Frequency and Ngrams')

# User selections
ngram_option = descriptive_expander.selectbox(
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
descriptive_expander.write(ngram_nm)

# Altair chart: ngram word frequencies
ngram_bar = alt.Chart(ngram_visual).mark_bar().encode(
                    x = alt.X('frequency', axis = alt.Axis(title = 'Word Frequency')),
                    y = alt.Y('ngram', axis = alt.Axis(title = 'Ngram'), sort = '-x'),
                    tooltip = [alt.Tooltip('frequency', title = 'Ngram Frequency')],#,  alt.Tooltip('Ngram', title = 'Ngram Word(s)')] ,
                ).properties(
                    height = 350
                )

descriptive_expander.altair_chart(ngram_bar, use_container_width=True)

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
        "% 😡 Negative Tweets": "{:.0%}".format(max(sentiment_group.Negative)/total_tweets),
        "% 😐 Neutral Tweets": "{:.0%}".format(max(sentiment_group.Neutral)/total_tweets),
        "% 😃 Positive Tweets": "{:.0%}".format(max(sentiment_group.Positive)/total_tweets),   
    }
)

## 2.2: RAW & DOWNLOADABLE DATA TABLE
#----------------------------
if st.checkbox('Show VADER results for each Tweet'):
    st.subheader('Raw data')
    st.write(df_sentiment)

st.markdown(tf.get_table_download_link(df_sentiment), unsafe_allow_html=True)


## CREATE EXPANDER FOR SENTIMENT ANALYSIS
sentiment_expander = st.beta_expander('Expand to see more sentiment analysis', expanded=False)

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
sentiment_expander.subheader('Classifying Tweet Sentiment by Day')
sentiment_expander.altair_chart(sentiment_bar, use_container_width=True)


## 2.4: ANALYZING TOP TWEETS (wordcloud + top tweets)
#----------------------------
sentiment_expander.subheader('Sentiment Wordcloud')
sentiment_expander.write('''*Note: Wordcloud will run on all tweets if sentiment type is ALL*''')

with sentiment_expander.form('Form2'):
    score_type = st.selectbox('Select sentiment', ['All', 'Positive', 'Neutral', 'Negative'], key=1)
    wordcloud_words = st.number_input('Choose the max number of words for the word cloud', 15, key = 3)
    num_tweets =  st.number_input('Choose the top number of tweets *', 5, key = 2)
    submitted2 = st.form_submit_button('Regenerate Wordcloud')

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
sentiment_expander.write('Word Cloud Generator')
sentiment_expander.pyplot()

# Run the top n tweets
top_tweets_res = tf.print_top_n_tweets(df_sentiment, score_type_nm, num_tweets)

# Show resuts as a streamlit table
sentiment_expander.write('Show the top tweets!')
for i in range(num_tweets):
    i = i + 1
    sentiment_expander.info('**Tweet #**' + str(i) + '**:** ' + top_tweets_res['full_text'][i] + '  \n **Compound Score:** ' + str(top_tweets_res['compound_score'][i]))

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
sentiment_expander.subheader('Checking Sentiment Skewness')
sentiment_expander.write('VADER Compound Scores Histogram')
sentiment_expander.altair_chart(sentiment_histo, use_container_width=True)      

#----------------------------------------------------------
## SECTION 3: TOPIC MODEL
#----------------------------------------------------------

## 3.1: TOPIC MODELLING TABLE
#----------------------------
data = df_tweets['clean_text']

st.header('Major Topics')
with st.form('Form2'):
    number_of_topics = st.number_input('Choose the number of topics. Start with a larger number and decrease if you see topics that are similar.',min_value=1, value=10)
    no_top_words = st.number_input('Choose the number of words in each topic you want to see.',min_value=1, value=10)
    #TODO: modify user inputs for min_df and max_df to be a radio button (eg. do you want to remove spam/anomalies)
    min_df = st.number_input('Ignore words that appear less than the specified proportion (decimal number between 0 and 1).',min_value=0.0, max_value=1.0, value=0.1)
    #TODO: hard code max_df
    max_df = st.number_input('Ignore words that appear more than the specified proportion (decimal number between 0 and 1).',min_value=0.0, max_value=1.0, value=0.9)
    submitted2 = st.form_submit_button('Regenerate topics')

#TODO: radio button to show with weights (analyst view because analyst would be interested in belongingness but avg user might not)
st.write(tf.lda_topics(data, number_of_topics, no_top_words, min_df, max_df))

# st.write(tf.LDA_viz(df_tweets['clean_text'])) 
# html_string = tf.LDA_viz(df_tweets['clean_text'])
# components.v1.html(html_string, width=1300, height=800)
