#------------------------------------#
# 0) Load dependencies
## Libraries (8)
#------------------------------------#

# Load dependencies
import streamlit as st
from streamlit_metrics import metric, metric_row
from PIL import Image
import pandas as pd
import datetime as dt
import base64
import tweepy as tw
import yaml

# Define functions
def get_table_download_link(df):
    # Reference: https://discuss.streamlit.io/t/how-to-download-file-in-streamlit/1806
    """Generates a link allowing the data in a given pandas dataframe to be downloaded
    in:  dataframe
    out: href string
    """
    csv = df.to_csv(index=False)
    b64 = base64.b64encode(csv.encode()).decode()  # some strings <-> bytes conversions necessary here
    href = f'<a href="data:file/csv;base64,{b64}" download="tweets.csv">Download CSV file</a>'
    return href



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



#-----------------------------------#
# 4) MAINPANEL, VISUALS
#-----------------------------------#

## KPI cards
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

## Raw data table
st.subheader('Raw data')
st.write(df_tweets)
st.markdown(get_table_download_link(df_tweets), unsafe_allow_html=True)