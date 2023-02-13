import re
import csv
import string
from langdetect import detect

import nltk
#download NLTK files
nltk.download('stopwords')
nltk.download('wordnet')
from nltk.corpus import stopwords
#from nltk.stem.porter import PorterStemmer
from nltk.stem.wordnet import WordNetLemmatizer

def Tweet_Preprocess(tweetlist):

    #Setting up CSV file
    csvfile=csv.writer(open('tweet_preprocessing/tweets.csv','w'))
    csvfields=['Raw Tweets','Preprocessed Tweets']
    csvfile.writerow(csvfields)
    csvrows=[]

    #Setting up Stopwords and Lemmetizer
    stopwords_list=set(stopwords.words('english'))
    wnl=WordNetLemmatizer()

    for rawtweet in tweetlist:

        #Checking if the tweet is written in english language
        if detect(rawtweet)=='en':

            #Converting to lowercase
            clean_tweet=rawtweet.lower()
            
            #Remove hyperlinks
            clean_tweet=re.sub(r'https?://\S+|www\.\S+','',clean_tweet)

            #Remove HTML tags
            clean_tweet=re.sub(r'<.*?>','',clean_tweet)

            #Remove user mentions
            clean_tweet=" ".join([word for word in clean_tweet.split() if not word.startswith('@')])

            #Remove punctuation marks
            clean_tweet=clean_tweet.translate(str.maketrans('','',string.punctuation))

            #Remove emojis and other characters
            clean_tweet=re.sub('[^a-zA-Z0-9]'," ",clean_tweet)

            #Replace combinations of tabs and spaces with single white space
            clean_tweet=re.sub('\s+',' ',clean_tweet)

            #Remove stop words
            clean_tweet=' '.join([word for word in clean_tweet.split(' ') if word not in stopwords_list])
            
            #Perform words lemmetization
            clean_tweet=' '.join([wnl.lemmatize(word) for word in clean_tweet.split(' ')])
            
            #Append pre-processed tweet to csvrorws list
            csvrows.append([rawtweet,clean_tweet])
    
    #write csvrows to csv file
    csvfile.writerows(csvrows)
