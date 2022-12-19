import csv
import pandas as pd
import string
from nltk.corpus import stopwords
import re
from nltk.stem.porter import PorterStemmer

def Tweet_Preprocess(tweetlist):
    #tweets mined which is comes as a list of strings tweetlist
    #limit is the number of tweets

    dict={'tweet':tweetlist}

    #convert list of strings to csv file
    tf=pd.DataFrame(dict)
    df.to_csv('tweet.csv')

    #convert to lower case
    tf['clean_txt']=tf['tweet'].str.lower()

    #remove punctuations
    def remove_punctuations(text):
        punctuations=string.punctuation
        return text.translate(str.maketrans('','',punctuations))

    #tf['clean_txt']=tf['clean_txt'].apply(lambda x:remove_punctuations(x))

    #remove stopwords
    STOPWORDS=set(stopwords.words('english'))
    def remove_stopwords(text):
        return ' '.join([word for word in text.spilt() if word not in STOPWORDS])

    tf['clean_txt']=tf['clean_txt'].apply(lambda x:remove_stopwords(x))

    #can remove frequent words and rare words if needed
    #remove special characters and punctuations
    def remove_spl_chars(text):
        text=re.sub('[^a-zA-Z)-9]'," ",text)
        text=re.sub('\s+',' ',text)
        return text

    tf['clean_txt']=tf['clean_txt'].apply(lambda x:remove_spl_chars(x))

    #stemming
    ps=PorterStemmer()
    def stem_words(text):
        return " ".join([ps.stem(word) for word in text.split()])

    tf['stemmed_txt']=tf['clean_txt'].apply(lambda x:stem_words(x))

    #remove URLs
    def remove_url(text):
        return re.sub(r'https?://\S+|www\.\S+', text)

    tf['clean_txt']=tf['clean_txt'].apply(lambda x:remove_url(x))

    #remove html tags
    def remove_html_tags(text):
        returnre.sub(r'<.*?>','',text)

    tf['clean_txt']=tf['clean_txt'].apply(lambda x:remove_html_tags(x))

    #can perform spelling correction
    # remove twitter handles
    def remove_pattern(input_txt,pattern):
        r=re.findall(pattern, input_txt)
        for word in r:
            input_txt=re.sub(word, "",input_txt)
        return input_txt

    tf['clean_txt']=np.vectorsize(remove_pattern)(tf['clean_txt'],"@[\w]*")

    #can do tokenization
