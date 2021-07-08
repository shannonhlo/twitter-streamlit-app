#----------------------------------------------
# Load dependencies
#----------------------------------------------
import pandas as pd
import base64
import tweepy as tw
import re
import numpy as np
import string
import unicodedata
import nltk
import gensim
import pyLDAvis.gensim_models
import pickle 
import pyLDAvis
import os
import matplotlib.pyplot as plt
import gensim.corpora as corpora
import streamlit as st
from pprint import pprint
from nltk.util import bigrams
from nltk.corpus import stopwords
from gensim.utils import simple_preprocess
from streamlit_metrics import metric, metric_row
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation as LDA
from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

nltk.download('stopwords')


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

# Function 2: 
#----------------
# Hit twitter api & add basic features & output 2 dataframes
# @st.cache(suppress_st_warning=True,allow_output_mutation=True)
def twitter_get(select_hashtag_keyword, select_language, user_word_entry, num_of_tweets):  
    
    # Set up Twitter API access
    # Define access keys and tokens
    consumer_key = st.secrets['consumer_key']
    consumer_secret = st.secrets['consumer_secret']
    access_token = st.secrets['access_token']
    access_token_secret = st.secrets['access_token_secret']

    auth = tw.OAuthHandler(consumer_key, consumer_secret)
    auth.set_access_token(access_token, access_token_secret)
    api = tw.API(auth, wait_on_rate_limit = True)
    
    # Keyword or hashtag
    if select_hashtag_keyword == 'Hashtag':
        user_word = '#' + user_word_entry
    else:
        user_word = user_word_entry

    # Retweets (assumes yes)
    user_word = user_word + ' -filter:retweets'
    # The following is based on user language selection

    # ...English Language
    if select_language == 'English':
        language = 'en'

    # ...French Language
    if select_language == 'French':
        language = 'fr'

    # Retweets (assumes yes)
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


    df_new = df_tweets[["created_dt", "created_time", "full_text", "user", "rt_count", "fav_count"]]
    df_new = df_new.rename(columns = {"created_dt": "Date", 
                                 "created_time": "Time", 
                                  "full_text": "Tweet", 
                                  "user": "Username", 
                                  "rt_count": "Retweets",  
                                  "fav_count": "Favourites"})

    return df_tweets, df_new

# Function 3
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

# Function 4a
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

# Function 4b
#-------------
text_clean_round1 = lambda x: round1_text_clean(x)

# Function 5
#-------------
def text_clean_round2(text):
    """
    A simple function to clean up the data. All the words that
    are not designated as a stop word is then lemmatized after
    encoding and basic regex parsing are performed.
    """
    nltk.download('wordnet')
    wnl = nltk.stem.WordNetLemmatizer()
    stopwords = nltk.corpus.stopwords.words('english')
    text = (unicodedata.normalize('NFKD', text)
    .encode('ascii', 'ignore')
    .decode('utf-8', 'ignore'))
    words = re.sub(r'[^\w\s]', '', text).split()
    return [wnl.lemmatize(word) for word in words if word not in stopwords]

# Function 6
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

# Function 7a
#-----------------
def tweets_ngrams(n, top_n, df):
    """
    Generates series of top ngrams
    n: number of words in the ngram
    top_n: number of ngrams with highest frequencies
    """
    text = df.clean_text
    words = text_clean_round2(''.join(str(text.tolist())))
    result = (pd.Series(data = nltk.ngrams(words, n), name = 'frequency').value_counts())[:top_n]
    return result.to_frame()

# Function 7b
#-----------------
def all_ngrams(top_n, df):
    text = df.clean_text
    words = text_clean_round2(''.join(str(text.tolist())))
    unigram = ((pd.Series(data = nltk.ngrams(words, 1), name = 'freq').value_counts())[:top_n]).to_frame()
    unigram['ngram'] = 'unigram'
    bigram = ((pd.Series(data = nltk.ngrams(words, 2), name = 'freq').value_counts())[:top_n]).to_frame()
    bigram['ngram'] = 'bigram'
    trigram = ((pd.Series(data = nltk.ngrams(words, 3), name = 'freq').value_counts())[:top_n]).to_frame()
    trigram['ngram'] = 'trigram'
    result = unigram.append([bigram, trigram])
    result['ngram_nm'] = result.index
    return result

# Function 8
#----------------

# Credit: https://jackmckew.dev/sentiment-analysis-text-cleaning-in-python-with-vader.html
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

# Function 9
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
    df['sentiment'] = np.select(condlist = conditions, choicelist = values)
    return df

# Function 10
#----------------

# Credit: https://ourcodingclub.github.io/tutorials/topic-modelling-python/

def lda_topics(data, number_of_topics, no_top_words, min_df, max_df):
    # the vectorizer object will be used to transform text to vector form
    vectorizer = CountVectorizer(max_df=max_df, min_df=min_df, token_pattern='\w+|\$[\d\.]+|\S+')

    # apply transformation
    tf = vectorizer.fit_transform(data).toarray()

    # tf_feature_names tells us what word each column in the matrix represents
    tf_feature_names = vectorizer.get_feature_names()

    model = LDA(n_components=number_of_topics, random_state=0)

    model.fit(tf)

    topic_dict = {}
    for topic_idx, topic in enumerate(model.components_):
        topic_dict["Topic %d words" % (topic_idx)]= ['{}'.format(tf_feature_names[i])
                        for i in topic.argsort()[:-no_top_words - 1:-1]]
        topic_dict["Topic %d weights" % (topic_idx)]= ['{:.1f}'.format(topic[i])
                        for i in topic.argsort()[:-no_top_words - 1:-1]]

    topic_df = pd.DataFrame(topic_dict)

    return pd.DataFrame(topic_df)


# Function 11
#----------------
# Credit: https://ourcodingclub.github.io/tutorials/topic-modelling-python/

def display_topics(model, feature_names, no_top_words):
    topic_dict = {}
    for topic_idx, topic in enumerate(model.components_):
        topic_dict["Topic %d words" % (topic_idx)]= ['{}'.format(feature_names[i])
                        for i in topic.argsort()[:-no_top_words - 1:-1]]
        topic_dict["Topic %d weights" % (topic_idx)]= ['{:.1f}'.format(topic[i])
                        for i in topic.argsort()[:-no_top_words - 1:-1]]
    return pd.DataFrame(topic_dict)


# Function 12
#---------------
def sent_to_words(sentences):
    for sentence in sentences:
        # deacc=True removes punctuations
        yield(gensim.utils.simple_preprocess(str(sentence), deacc=True))

#TODO: This function does not work in the streamlit app. Returns "BrokenPipeError: [Errno 32] Broken pipe"
def LDA_viz(data):
    data_words = list(sent_to_words(data))

    # Create Dictionary
    id2word = corpora.Dictionary(data_words)
    # Create Corpus
    texts = data_words
    # Term Document Frequency
    corpus = [id2word.doc2bow(text) for text in texts]

    # number of topics
    num_topics = 10
    # Build LDA model
    lda_model = gensim.models.LdaMulticore(corpus=corpus,
                                        id2word=id2word,
                                        num_topics=num_topics)
    # Print the Keyword in the 10 topics
    # pprint(lda_model.print_topics())
    doc_lda = lda_model[corpus]

    # Visualize the topics
    pyLDAvis.enable_notebook()
    LDAvis_data_filepath = os.path.join('./results/ldavis_prepared_'+str(num_topics))
    # # this is a bit time consuming - make the if statement True
    # # if you want to execute visualization prep yourself
    if 1 == 1:
        LDAvis_prepared = pyLDAvis.gensim_models.prepare(lda_model, corpus, id2word)
        with open(LDAvis_data_filepath, 'wb') as f:
            pickle.dump(LDAvis_prepared, f)
    # load the pre-prepared pyLDAvis data from disk
    with open(LDAvis_data_filepath, 'rb') as f:
        LDAvis_prepared = pickle.load(f)
    pyLDAvis.save_html(LDAvis_prepared, './results/ldavis_prepared_'+ str(num_topics) +'.html')
    
    return LDAvis_prepared

# Function 13
#----------------
# Credit: https://jackmckew.dev/sentiment-analysis-text-cleaning-in-python-with-vader.html

def print_top_n_tweets(df, sent_type, num_rows):
    text = 'full_text'
    top = df.nlargest(num_rows, sent_type)
    top_tweets = top[[sent_type,text]].reset_index()
    top_tweets = top_tweets.drop(columns = 'index')
    top_tweets.index = top_tweets.index + 1 
    return top_tweets

# Function 14
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
    wordcloud = WordCloud(max_font_size=80, max_words=wordcloud_words, background_color="white", height=100).generate(str2)
    return wordcloud

# Function 15
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

# Function 16
#----------------
# Function to plot default wordcloud
def default_wordcloud(text_sentiment):

    # Hard coded default wordcloud (for show)
    score_type = 'All'
    score_type_nm = 'compound_score'
    wordcloud_words = 15
    top_n_tweets = 5
   
    # Run wordlcloud for top n tweets
    if score_type == 'All':         
        wordcloud = word_cloud_all(text_sentiment, wordcloud_words)
    else:
        wordcloud = word_cloud_sentiment(text_sentiment, score_type_nm, top_n_tweets, wordcloud_words)

    # Display the generated wordcloud image:
    st.set_option('deprecation.showPyplotGlobalUse', False)
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis("off")
    plt.show()

    # Write word cloud, need to call it still
    st.write('Word Cloud Generator')
    st.pyplot()
    
    return 

# Function 16
#----------------
# Function to plot wordcloud that changes
def plot_wordcloud(submitted2, score_type, text_sentiment, wordcloud_words, top_n_tweets):
    
    # Scenarios
    # Scenario 1: No input (default)
    if submitted2 is not True:
        score_type = 'All'
        score_type_nm = 'compound_score'
        
    # Scenario 2: All
    if score_type == 'All':
        score_type_nm = 'compound_score'

    # Scenario 3: Positive
    if score_type == 'Positive':
        score_type_nm = 'positive_score'

    # Scenario 4: Neutral
    if score_type == 'Neutral':
        score_type_nm = 'neutral_score'

    # Scenario 5: Negative
    if score_type == 'Negative':
        score_type_nm = 'negative_score'

    # Run wordlcloud for top n tweets
    if score_type == 'All':         
        wordcloud = word_cloud_all(text_sentiment, wordcloud_words)
    else:
        wordcloud = word_cloud_sentiment(text_sentiment, score_type_nm, top_n_tweets, wordcloud_words)

    # Display the generated wordcloud image:
    st.set_option('deprecation.showPyplotGlobalUse', False)
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis("off")
    plt.show()

    # Write word cloud, need to call it still
    st.write('Word Cloud Generator')
    st.pyplot()
    
    return 

# Function 17a
#----------------
# Function to display topics and related keywords
def print_lda_keywords(data, number_of_topics):
    # sets initial topic number to 1 to be used in output
    topic_num = 1
    # takes user input number of topics and loops over topic
    for n in range(0, number_of_topics*2, 2):
        # joins keywords for each topic, separated by comma
        list_topic_words = ", ".join(data.iloc[:, n])
        # prints output
        st.warning('**Theme #**' + str(topic_num) + '**:** ' + list_topic_words)
        # increments topic number by 1 so that each theme printed out will have a new number
        topic_num += 1

# Function 17b
#----------------
# Function to display topics, related keywords, and weights
def print_lda_keywords_weight(data, number_of_topics):
    # sets initial topic number to 1 to be used in output
    topic_num = 1
    # takes user input number of topics and loops over topic and weight
    for n in range(0, number_of_topics*2, 2):
        w = n + 1
        list_topic_words_weight = ", ".join(data.iloc[:, n] + ' (' + data.iloc[:, w] + ')')
        # prints output
        st.warning('**Theme #**' + str(topic_num) + '**:** ' + list_topic_words_weight)
        # increments topic number by 1 so that each theme printed out will have a new number
        topic_num += 1