import base64
import requests
from complaint_detection.LSTM import load_lstm
from complaint_detection.BERT import load_bert
from complaint_detection.tweet_mining.Miner import TweetMiner
from complaint_detection.ComplaintDetection import Detect_Complaint
from complaint_detection.tweet_preprocessing.TweetPreprocessor import Tweet_Preprocess

if __name__=='__main__':
	limit=int(input("Number of Tweets : "))
	#tweetlist=TweetMiner(limit)
	tweetlist=[]
	'''
	tweetlist=["Whenever it rains, the absence of effective drainage systems leads in waterlogging on the roads. Hey @PoliceSight, please do something.","Confusion among drivers is being brought on by the lack of sufficient signs on the highways. @PoliceSight, kindly post appropriate signs.",
	"@PoliceSight, pedestrians are suffering as a result of the walkways' poor maintenance. Please make them good.","Many accidents occur at night as a result of inadequate street lighting. @PoliceSight, kindly add additional lighting.",
	"@PoliceSight, the potholes in the road are resulting in collisions and destruction to automobiles. Please do something."]
	'''
	for i in range(limit):
		data=input("Enter The Tweet : ")
		tweetlist.append(data)
	rawtweetlist=tweetlist.copy()
	#Tweet_Preprocess(tweetlist)
	bert_tokenizer,bert_model=load_bert()
	lstm_tokenizer,lstm_model=load_lstm()
	for i,tweet in enumerate(tweetlist,start=0):
		if Detect_Complaint(tweet,bert_tokenizer,bert_model,lstm_tokenizer,lstm_model)=='Complaint':
			data={'type':'Tweet','tweet':rawtweetlist[i],'incident_frame':base64.b64encode(open('flask/default.jpeg', 'rb').read()).decode(),'incident_type':'None','timestamp':''}
			host='http://127.0.0.1:5000/update'
			response=requests.post(url=host,json=data)
			print(response)