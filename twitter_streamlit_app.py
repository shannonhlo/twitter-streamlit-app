# Load dependencies
import streamlit as st
from streamlit_metrics import metric, metric_row
from PIL import Image
import pandas as pd
import datetime as dt
import base64
#import matplotlib.pyplot as plt
#from bs4 import BeautifulSoup
#import requests
#import json

import os
import tweepy as tw
import pandas as pd
import yaml


# Define functions
def get_table_download_link(df):
    # Reference: https://discuss.streamlit.io/t/how-to-download-file-in-streamlit/1806
    """Generates a link allowing the data in a given panda dataframe to be downloaded
    in:  dataframe
    out: href string
    """
    csv = df.to_csv(index=False)
    b64 = base64.b64encode(csv.encode()).decode()  # some strings <-> bytes conversions necessary here
    href = f'<a href="data:file/csv;base64,{b64}" download="tweets.csv">Download csv file</a>'
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
* **Python libraries:** base64, pandas, streamlit, numpy, matplotlib, seaborn, BeautifulSoup, requests, json, time, yaml
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

## User input: include retweets or not
include_retweets = st.sidebar.checkbox('Include retweets in data')

## User input: include retweets or not
select_language = st.sidebar.button('English')

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
num_of_tweets = 5
language = "en"

if include_retweets == False:
    user_word = user_word + " -filter:retweets"

tweets = tw.Cursor(api.search,
                    q=user_word,
                    tweet_mode = 'extended',
                    lang=language).items(num_of_tweets)

tweet_metadata = [[tweet.created_at, tweet.id, tweet.full_text, tweet.user.screen_name, tweet.retweet_count, tweet.favorite_count] for tweet in tweets]
df_tweets = pd.DataFrame(data=tweet_metadata, columns=['created_at', 'id', 'full_text', 'user', 'rt_count', 'fav_count'])

#-----------------------------------#
# 4) MAINPANEL, VISUALS
#-----------------------------------#

## Raw data table
st.subheader('Raw data')
st.write(df_tweets)
st.markdown(get_table_download_link(df_tweets), unsafe_allow_html=True)


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



# expander_bar2.markdown("""
# * Each row must be a unique listing
# * Required columns: `bath, bed, city, days_on_site, price, property_age, property_type, scrape_date, sqft`
# """
#                        )
# uploaded_file = expander_bar2.file_uploader("Upload your file below", type=["csv"])

# ## Expandable sidebar 1: Example CSV input file download
# expander_bar1 = st.sidebar.beta_expander("See Example Data")
# expander_bar1.markdown("""
# * Right click [Example CSV input file](https://raw.githubusercontent.com/dmf95/realtor_streamlit_app/master/dummy_df.csv)  
# * Hit `Save link as..`
# * Change file from `.txt` to `.csv`
# """)

# ## Conditional dataset creation
# if uploaded_file is not None:
#     df = pd.read_csv(uploaded_file)
# else:
#     df = pd.read_csv('dummy_df.csv')

# ## Change data types
# df['scrape_date'] = pd.to_datetime(df['scrape_date'])
# df['bed'] = pd.to_numeric(df['bed'])
# df['bath'] = pd.to_numeric(df['bath'])
# df['price'] = pd.to_numeric(df['price'])
# df['strata_fee'] = pd.to_numeric(df['strata_fee'])



# ## Scenario 1 (user uploads data): read in csv if it exists
# if uploaded_file is not None:
#     input_df = pd.read_csv(uploaded_file)

# ## Scenario 2: (user has not uploaded data yet): read in default values from input selection in the app
# else:
#     def user_input_features():

#         ### date stuff
#         today = dt.datetime.now()
#         year = today.year
#         d = df.scrape_date
#         d1 = pd.DatetimeIndex(d).month.astype(str) + "/" + pd.DatetimeIndex(d).year.astype(str)
#         d2= sorted(d1.unique())
#         scrape_dt = st.sidebar.selectbox('Date Retrieved', d2)
#         ###
#         sorted_city = sorted(df.city.unique())
#         city = st.sidebar.selectbox('City', sorted_city)

#         #TODO figure out to to add neighbourhood such that it filters based on city selection above
#         #TODO see if there is a way to search through description and to find all listings that match

#         sorted_beds = sorted(df.bed.unique())
#         beds = st.sidebar.multiselect('Bedrooms', sorted_beds, sorted_beds)
#         sorted_baths = sorted(df.bath.unique())
#         baths = st.sidebar.multiselect('Bathrooms', sorted_baths, sorted_baths)
#         sorted_property_type = sorted(df.property_type.unique())
#         property_type = st.sidebar.multiselect('Property Type', sorted_property_type, sorted_property_type)
#         price = st.sidebar.slider('Listing Price', min_value=200000, max_value=5000000, value=(200000,5000000),step=100000)
#         sqft = st.sidebar.slider('Square Feet', min_value=300, max_value=10000, value=(300,10000),step=50)
#         property_age = st.sidebar.slider('Property Age', min_value=1900, max_value=year, value=(1900,year),step=1)
#         days = max(df.days_on_site)
#         days_on_site = st.sidebar.slider('Days on Site', min_value=0, max_value=days, value=(0,days),step=1)

#     input_df = user_input_features()

# # TODO: MAIN PANEL SHIT @shannons part :)

# #-----------------------------------#
# # 3) MAINPANEL, VISUALS
# #-----------------------------------#

# ## Inspect the raw data
# st.subheader('Listing prices data')
# st.write(df)

# '''
# ## Step 1: Create KPI cards
# '''
# # Calculate KPIs
# listings_count = len(pd.unique(df['listing_id']))
# # TODO: calculate average days in market
# avg_market_days = 'placeholder'


# # Create visuals
# st.subheader('Vancouver Housing Inventory')
# metric_row(
#     {
#         "Number of listings": listings_count,
#         "Average days on market": avg_market_days
#     }
# )

# '''
# ## Step 3: Create graphs
# '''
# # Create dataframe with count of unique listings by date
# df_inventory = df[['scrape_date', 'listing_id']].groupby(['scrape_date']).agg(['nunique']).reset_index()
# df_inventory.columns = ['scrape_date', 'listings_count']

# st.subheader('Number of listings over time')
# st.line_chart(df_inventory.set_index('scrape_date'))

# '''
# ## Step 4: Create tables
# '''
# # Create dataframe with count of listings by property type and number of bedrooms
# df_prop_bed = df[['listing_id', 'property_type', 'bed']].groupby(['property_type', 'bed']).agg(['nunique']).reset_index()
# df_prop_bed.columns = ['property_type', 'bed', 'listings_count']
# df_prop_bed['bed'] = df_prop_bed['bed'].astype(str) + ' Bedroom(s)'

# # Create separate dataframes for each property type
# df_apt_bed = df_prop_bed[df_prop_bed['property_type'] == 'Apt/Condo']
# df_duplex_bed = df_prop_bed[df_prop_bed['property_type'] == 'Duplex']
# df_house_bed = df_prop_bed[df_prop_bed['property_type'] == 'House']
# df_rec_bed = df_prop_bed[df_prop_bed['property_type'] == 'Recreational']
# df_town_bed = df_prop_bed[df_prop_bed['property_type'] == 'Townhouse']

# # TODO: create separate tabs for the 5 tables, potentially using Bokeh (https://discuss.streamlit.io/t/bokeh-can-provide-layouts-tabs-advanced-tables-and-js-callbacks-in-streamlit/1108)
# st.write(df_apt_bed)
# st.write(df_duplex_bed)
# st.write(df_house_bed)
# st.write(df_rec_bed)
# st.write(df_town_bed)