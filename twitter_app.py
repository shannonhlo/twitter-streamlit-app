#--------------------------------------------------
# PART 1: LOAD DEPENDENCIES & TODO
#--------------------------------------------------
# - 1.1: Load libraries
# - 1.2: Load custom library
# - 1.3: TODO items
#--------------------------------------------------

#~-=~-=~-=~-=~-=~-=~-=~-=~-=~-=~-=~-=~-=~-=~-=~-=~-=~-=~-=~-=

# 1.1: Load libraries
#------------------------------------#
from nltk.featstruct import _default_fs_class
from numpy import e
import streamlit as st
from streamlit_metrics import metric, metric_row
import streamlit.components.v1 as components
from PIL import Image
import pandas as pd
import tweepy as tw
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation as LDA
import matplotlib.pyplot as plt
import altair as alt
import time


#~-=~-=~-=~-=~-=~-=~-=~-=~-=~-=~-=~-=~-=~-=~-=~-=~-=~-=~-=~-=

# 1.2: Load custom library
#------------------------------------#
import twitter_functions as tf # custom functions file


#~-=~-=~-=~-=~-=~-=~-=~-=~-=~-=~-=~-=~-=~-=~-=~-=~-=~-=~-=~-=

# 1.3: TODO items
#------------------------------------#
#TODO - create gif
#TODO - post

#--------------------------------------------------
# PART 2: APP UI SETUP
#--------------------------------------------------
# - 2.1: Main panel setup 
# - 2.2: Sidebar setup
#--------------------------------------------------

#~-=~-=~-=~-=~-=~-=~-=~-=~-=~-=~-=~-=~-=~-=~-=~-=~-=~-=~-=~-=

# 2.1: Main Panel Setup
#------------------------------------#

## 2.1.1: Main Layout
##----------------------------------##
st.set_page_config(layout="wide") # page expands to full width
col1 = st.sidebar
col2, col3 = st.beta_columns((2,1)) # col1 is 2x greater than col2

## 2.1.2: Main Logo
##----------------------------------##
image = Image.open('twitter_logo2.png') #logo
st.image(image, width = 350) #logo width

## 2.1.3: Main Title
##----------------------------------##
#st.title('Tweet Analyzer') #
st.markdown("""
Search a Twitter hashtag in the sidebar to run the tweet analyzer!
""")

#~-=~-=~-=~-=~-=~-=~-=~-=~-=~-=~-=~-=~-=~-=~-=~-=~-=~-=~-=~-=

# 2.2: Sidebar Setup
#------------------------------------#

## 2.1.1: Sidebar Title
##----------------------------------##
st.sidebar.header('Choose Search Inputs') #sidebar title

## 2.2.2: Sidebar Input Fields
##----------------------------------##
with st.form(key ='form_1'):
    with st.sidebar:
        user_word_entry = st.text_input("1. Enter one keyword", "stanleycup", help='Ensure that keyword does not contain spaces')    
        select_hashtag_keyword = st.radio('2. Search hashtags, or all keywords?', ('Hashtag', 'Keyword'), help='Searching only hashtags will return fewer results')
        select_language = st.radio('3. Tweet language', ('All', 'English', 'French'), help = 'Select the language you want the Analyzer to search Twitter for')
        num_of_tweets = st.number_input('4. Maximum number of tweets', min_value=100, max_value=10000, value = 150, step = 50, help = 'Returns the most recent tweets within the last 7 days')
        st.sidebar.text("") # spacing
        submitted1 = st.form_submit_button(label = 'Run Tweet Analyzer 🚀', help = 'Re-run analyzer with the current inputs')

## 2.2.3: Sidebar About Expanders
##----------------------------------##

## About the app title
st.sidebar.text("") # spacing
st.sidebar.header('About the App')

# General expander section
about_expander = st.sidebar.beta_expander("General")
about_expander.markdown("""
* **Creators:** [Shannon Lo](https://shannonhlo.github.io/) & [Domenic Fayad](https://www.fullstaxx.com/)
* **References:**
  * https://discuss.streamlit.io/t/how-to-download-file-in-streamlit/1806
  * https://jackmckew.dev/sentiment-analysis-text-cleaning-in-python-with-vader.html
  * https://www.dataquest.io/blog/tutorial-add-column-pandas-dataframe-based-on-if-else-condition/
  * https://ourcodingclub.github.io/tutorials/topic-modelling-python/
""")

# Methodology expander section
method_expander= st.sidebar.beta_expander("Methodology")
method_expander.markdown("""
* Applying the [VADER Sentiment](https://github.com/cjhutto/vaderSentiment) library to our text data
* [VADER](https://github.com/cjhutto/vaderSentiment#vader-sentiment-analysis) (**V**alence **A**ware **D**ictionary and s**E**ntiment **R**easoner) = lexicon and rule-based sentiment analysis tool, specifically attuned to sentiments expressed in social media
* [Compound score](https://github.com/cjhutto/vaderSentiment#about-the-scoring) = computed by summing the valence scores of each word in the lexicon, adjusted according to the rules, and then normalized to be between -1 (most extreme negative) and +1 (most extreme positive)
* Positive sentiment: compound score >= 0.05
* Neutral sentiment: (compound score > -0.05) and (compound score < 0.05)
* Negative sentiment: compound score <= -0.05
""")


## 2.2.4: Sidebar Social
##----------------------------------##
st.sidebar.text("") # spacing
st.sidebar.header('Developer Contact')
st.sidebar.write("[![Star](https://img.shields.io/github/stars/shannonhlo/twitter-streamlit-app.svg?logo=github&style=social)](https://github.com/shannonhlo/twitter-streamlit-app/branches)")
st.sidebar.write("[![Follow](https://img.shields.io/twitter/follow/shannonhlo26?style=social)](https://twitter.com/shannonhlo26)")
st.sidebar.write("[![Follow](https://img.shields.io/twitter/follow/DomenicFayad?style=social)](https://twitter.com/DomenicFayad)")


#--------------------------------------------------
# PART 3: APP DATA SETUP
#--------------------------------------------------
# - 3.1: Twitter data ETL
# - 3.2: Define key variables
#--------------------------------------------------

#~-=~-=~-=~-=~-=~-=~-=~-=~-=~-=~-=~-=~-=~-=~-=~-=~-=~-=~-=~-=

# 3.1: Twitter Data ETL

# Layout
#------------------------------------#

# Run function 2: Get twitter data 
df_tweets, df_new = tf.twitter_get(select_hashtag_keyword, select_language, user_word_entry, num_of_tweets)

# Run function #3: Feature extraction
df_tweets = tf.feature_extract(df_tweets)

# Run function #4: Round 1 text cleaning (convert to lower, remove numbers, @, punctuation, numbers. etc.)
df_tweets['clean_text'] = df_tweets.clean_text.apply(tf.text_clean_round1)

## Run function #6: Round 3 text cleaning (remove stop words)
df_tweets.clean_text  = tf.text_clean_round3(df_tweets.clean_text)


#~-=~-=~-=~-=~-=~-=~-=~-=~-=~-=~-=~-=~-=~-=~-=~-=~-=~-=~-=~-=


# 3.2: Define Key Variables
#------------------------------------#
user_num_tweets =str(num_of_tweets)
total_tweets = len(df_tweets['full_text'])
highest_retweets = max(df_tweets['rt_count'])
highest_likes = max(df_tweets['fav_count'])


#--------------------------------------------------
# PART 4: APP DATA & VISUALIZATIONS
#--------------------------------------------------
# - 4.1: UX messaging
# - 4.2: Sentiment analysis
# - 4.3: Descriptive analysis
# - 4.4: Topic model analysis
#--------------------------------------------------

#~-=~-=~-=~-=~-=~-=~-=~-=~-=~-=~-=~-=~-=~-=~-=~-=~-=~-=~-=~-=

# 4.1: UX Messaging
#------------------------------------#

# Loading message for users
with st.spinner('Getting data from Twitter...'):
    time.sleep(5)
    # Keyword or hashtag
    if select_hashtag_keyword == 'Hashtag':
        st.success('🎈Done! You searched for the last ' + 
            user_num_tweets + 
            ' tweets that used #' + 
            user_word_entry)

    else:
        st.success('🎈Done! You searched for the last ' + 
            user_num_tweets + 
            ' tweets that used they keyword ' + 
            user_word_entry)

#~-=~-=~-=~-=~-=~-=~-=~-=~-=~-=~-=~-=~-=~-=~-=~-=~-=~-=~-=~-=

# 4.2: Sentiment Analysis
#------------------------------------#

# Subtitle
st.header('❤️ Sentiment Analysis')

# Get sentiment scores on raw tweets
text_sentiment = tf.get_sentiment_scores(df_tweets, 'full_text')

# Add sentiment classification
text_sentiment = tf.sentiment_classifier(df_tweets, 'compound_score')

# Select columns to output
df_sentiment = df_tweets[['created_at', 'full_text', 'sentiment', 'positive_score', 'negative_score', 'neutral_score', 'compound_score']]

# Sentiment group dataframe
sentiment_group = df_sentiment.groupby('sentiment').agg({'sentiment': 'count'}).transpose()

## 4.2.1: Summary Card Metrics
##----------------------------------##

# KPI Cards for sentiment summary
st.subheader('Summary')
metric_row(
    {
        "% 😡 Negative Tweets": "{:.0%}".format(max(sentiment_group.Negative)/total_tweets),
        "% 😑 Neutral Tweets": "{:.0%}".format(max(sentiment_group.Neutral)/total_tweets),
        "% 😃 Positive Tweets": "{:.0%}".format(max(sentiment_group.Positive)/total_tweets),   
    }
)

## 4.2.2: Sentiment Expander Bar
##----------------------------------##
sentiment_expander = st.beta_expander('Expand to see more sentiment analysis', expanded=False)


## 4.2.3: Sentiment by day bar chart
##----------------------------------##

# Altair chart: sentiment bart chart by day
sentiment_bar = alt.Chart(df_sentiment).mark_bar().encode(
                    x = alt.X('count(id):Q', stack="normalize", axis = alt.Axis(title = 'Percent of Total Tweets', format='%')),
                    y = alt.Y('monthdate(created_at):O', axis = alt.Axis(title = 'Date')),
                    tooltip = [alt.Tooltip('sentiment', title = 'Sentiment Group'), 'count(id):Q', alt.Tooltip('average(compound_score)', title = 'Avg Compound Score'), alt.Tooltip('median(compound_score)', title = 'Median Compound Score')],
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


## 4.2.4: Compound Score Histogram
##----------------------------------##

# Histogram for VADER compound score
sentiment_histo= alt.Chart(df_sentiment).mark_bar().encode(
                    x = alt.X('compound_score:O', axis = alt.Axis(title = 'VADER Compound Score (Binned)'), bin=alt.Bin(extent=[-1, 1], step=0.25)),
                    y = alt.Y('count(id):Q', axis = alt.Axis(title = 'Number of Tweets')),
                    tooltip = [alt.Tooltip('sentiment', title = 'Sentiment Group'), 'count(id):Q', alt.Tooltip('average(compound_score)', title = 'Average Compound Score'), alt.Tooltip('median(compound_score)', title = 'Median Compound Score')] ,
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


## 4.2.5: Download raw sentiment data
##----------------------------------##

# Show raw data if selected
if sentiment_expander.checkbox('Show VADER results for each Tweet'):
    sentiment_expander.subheader('Raw data')
    sentiment_expander.write(df_sentiment)

# Click to download raw data as CSV
sentiment_expander.markdown(tf.get_table_download_link(df_sentiment), unsafe_allow_html=True)

#~-=~-=~-=~-=~-=~-=~-=~-=~-=~-=~-=~-=~-=~-=~-=~-=~-=~-=~-=~-=

# 4.3: Wordclouds & Tweets
#------------------------------------#

# Subtitle
st.header('☁️🔝 Wordcloud & Top Tweets')


## 4.3.1: Sentiment Expander Bar
##----------------------------------##

# Setup expander
wordcloud_expander = st.beta_expander('Expand to customize wordcloud & top tweets', expanded=False)

# Sentiment Wordcloud subheader & note
wordcloud_expander.subheader('Advanced Settings')


## 4.3.2: Wordcloud expander submit form
##----------------------------------##

# Sentiment expander form submit for the wordcloud & top tweets
with wordcloud_expander.form('form_2'):    
     score_type = st.selectbox('Select sentiment', ['All', 'Positive', 'Neutral', 'Negative'], key=1)
     wordcloud_words = st.number_input('Choose the max number of words for the word cloud', 15, key = 3)
     top_n_tweets =  st.number_input('Choose the top number of tweets *', 3, key = 2)
     submitted2 = st.form_submit_button('Regenerate Wordcloud', help = 'Re-run the Wordcloud with the current inputs')


## 4.3.3: Plot wordcloud
##----------------------------------##
tf.plot_wordcloud(submitted2, score_type, text_sentiment, wordcloud_words, top_n_tweets)


## 4.3.4: Plot top tweets
##----------------------------------##

# Scenarios

# Scenario 1: All
if score_type == 'All':
    score_type_nm = 'compound_score'
    score_nickname = 'All'

# Scenario 2: Positive
if score_type == 'Positive':
    score_type_nm = 'positive_score'
    score_nickname = 'Positive'

# Scenario 3: Neutral
if score_type == 'Neutral':
    score_type_nm = 'neutral_score'
    score_nickname = 'Neutral'

# Scenario 4: Negative
if score_type == 'Negative':
    score_type_nm = 'negative_score'
    score_nickname = 'Negative'

# Run the top n tweets
top_tweets_res = tf.print_top_n_tweets(df_sentiment, score_type_nm, top_n_tweets)

# Conditional title
str_num_tweets = str(top_n_tweets)
show_top = str('Showing top ' + 
                str_num_tweets + 
                ' ' +
                score_nickname + 
                ' tweets ranked by '+ 
                score_type_nm)

# Write conditional
st.write(show_top)

# Show top n tweets
for i in range(top_n_tweets):
    i = i + 1
    st.info('**Tweet #**' + str(i) + '**:** ' + top_tweets_res['full_text'][i] + '  \n **Score:** ' + str(top_tweets_res[score_type_nm][i]))

#~-=~-=~-=~-=~-=~-=~-=~-=~-=~-=~-=~-=~-=~-=~-=~-=~-=~-=~-=~-=

# 4.4: Descriptive Analysis
#------------------------------------#

# Subtitle
st.header('📊 Descriptive Analysis')


## 4.4.1: Summary Metric Cards
##----------------------------------##

# KPI Cards for descriptive summary
st.subheader('Tweet Summary')
metric_row(
    {
        "Number of tweets": total_tweets,
        "Most Retweets on 1 post": highest_retweets,
        "Most Likes on 1 post": highest_likes,
    }
)

most_retweets_index = df_tweets['rt_count'].idxmax()
most_likes_index = df_tweets['fav_count'].idxmax()

if most_retweets_index == most_likes_index:
    st.info('**Same tweet had the most retweets and likes: **' + df_tweets['full_text'][most_retweets_index])
else:
    st.info('**Tweet with most retweets: **' + df_tweets['full_text'][most_retweets_index])
    st.info('**Tweet with most likes: **' + df_tweets['full_text'][most_likes_index])


## 4.4.2: Descriptive Expander Bar
##----------------------------------##
descriptive_expander = st.beta_expander('Expand to see more descriptive analysis', 
                                        expanded=False)


## 4.4.3: Tweets by day bar chart
##----------------------------------##

# Subtitle
descriptive_expander.subheader('Number of Tweets by Day')

# Altair chart: number of total tweets by day
#TODO: declutter x-axis. Unreadable when there are multiple dates
line = alt.Chart(df_tweets).mark_line(interpolate = 'basis').encode(
                    x = alt.X('monthdatehours(created_at):O', axis = alt.Axis(title = 'Date')),
                    y = alt.Y('count(id):Q', axis = alt.Axis(title = 'Number of Total Tweets', tickMinStep=1)),
                    color = "count(id):Q"
                   # tooltip = [alt.Tooltip('monthdatehours(created_at):O', title = 'Tweet Date'), alt.Tooltip('count(id):Q', title = 'Number of Tweets')]
                ).properties(
                    height = 350
                ).interactive()  

# Plot with altair
descriptive_expander.altair_chart(line, use_container_width=True)


## 4.4.4: Ngram Word Counts
##----------------------------------##

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


## 4.4.5: Download raw descriptive data
##----------------------------------##

# Show raw data if selected
if descriptive_expander.checkbox('Show raw data'):
    descriptive_expander.subheader('Raw data')
    descriptive_expander.write(df_new)

# Click to download raw data as CSV
descriptive_expander.markdown(tf.get_table_download_link(df_tweets), unsafe_allow_html=True)

#~-=~-=~-=~-=~-=~-=~-=~-=~-=~-=~-=~-=~-=~-=~-=~-=~-=~-=~-=~-=

# 4.5: Topic Modeling
#------------------------------------#

# Subtitle
st.header('🧐 Top Themes')

## 4.5.1: Topic Expander Bar
##----------------------------------##
topic_expander = st.beta_expander('Expand to see more topic modeling options', 
                                        expanded=False)

## 4.5.2: Topic Model table
##----------------------------------##

# Define data variable
data = df_tweets['clean_text']

topic_view_option = topic_expander.radio('Choose display options', ('Default view', 'Analyst view (advanced options)'))



if topic_view_option == 'Default view':
    # Topic model expander form submit for topic model table & visual
    with topic_expander.form('form_3'):
        number_of_topics = st.number_input('Choose the number of topics. Start with a larger number and decrease if you see topics that are similar.',min_value=1, value=5)
        no_top_words = st.number_input('Choose the number of words in each topic you want to see.',min_value=1, value=5)
        submitted2 = st.form_submit_button('Regenerate topics', help = 'Re-run topic model analysis with the current inputs')
    df_lda = tf.lda_topics(data, number_of_topics, no_top_words, 0.1, 0.9)
    tf.print_lda_keywords(df_lda, number_of_topics)
else:
    with topic_expander.form('form_3'):
        number_of_topics = st.number_input('Choose the maximum number of topics. Start with a larger number and decrease if you see topics that are similar.',min_value=1, value=5)
        no_top_words = st.number_input('Choose the maximum number of words in each topic you want to see.',min_value=1, value=5)
        min_df = st.number_input('Ignore words that appear less than the specified proportion (decimal number between 0 and 1).',min_value=0.0, max_value=1.0, value=0.1)
        max_df = st.number_input('Ignore words that appear more than the specified proportion (decimal number between 0 and 1).',min_value=0.0, max_value=1.0, value=0.9)
        submitted2 = st.form_submit_button('Regenerate topics', help = 'Re-run topic model analysis with the current inputs')
    df_lda = tf.lda_topics(data, number_of_topics, no_top_words, min_df, max_df)
    st.write('Weights shown in brackets represent how important the word is to each topic')
    tf.print_lda_keywords_weight(df_lda, number_of_topics)
