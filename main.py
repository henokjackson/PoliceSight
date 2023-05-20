from tweet_mining.miner import TweetMiner
from complaint_detection.lstm import load_lstm
from complaint_detection.bert import load_bert
from tweet_preprocessing.preprocessor import Tweet_Preprocess
from complaint_detection.Complaint_Detection import Detect_Complaint

if __name__=='__main__':
	limit=int(input("Number of Tweets : "))
	tweetlist=TweetMiner(limit)
	Tweet_Preprocess(tweetlist)
	bert_tokenizer,bert_model=load_bert()
	lstm_tokenizer,lstm_model=load_lstm()
	for tweet in tweetlist:
		Detect_Complaint(tweet,bert_tokenizer,bert_model,lstm_tokenizer,lstm_model)