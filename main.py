from complaint_detection.LSTM import load_lstm
from complaint_detection.BERT import load_bert
from complaint_detection.tweet_mining.Miner import TweetMiner
from complaint_detection.Complaint_Detection import Detect_Complaint
from complaint_detection.tweet_preprocessing.TweetPreprocessor import Tweet_Preprocess

if __name__=='__main__':
	limit=int(input("Number of Tweets : "))
	tweetlist=TweetMiner(limit)
	Tweet_Preprocess(tweetlist)
	bert_tokenizer,bert_model=load_bert()
	lstm_tokenizer,lstm_model=load_lstm()
	for tweet in tweetlist:
		Detect_Complaint(tweet,bert_tokenizer,bert_model,lstm_tokenizer,lstm_model)
