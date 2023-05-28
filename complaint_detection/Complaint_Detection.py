from complaint_detection.bert import bert_run
from complaint_detection.lstm import lstm_run
import requests
def Ensemble(confbert1,confbert0, conflstm1, conflstm0):
  comp=((confbert0*0.51)+(conflstm0*0.49))/2
  notcomp=((confbert1*0.51)+(conflstm1*0.49))/2
  if(notcomp>comp):
    label="NOTComplaint"
  else:
    label="Complaint"
  return label

def Detect_Complaint(tweet,bert_tokenizer,bert_model,lstm_tokenizer,lstm_model):
  confbert1,confbert0=bert_run(tweet,bert_tokenizer,bert_model)
  conflstm1,conflstm0=lstm_run(tweet,lstm_tokenizer,lstm_model)
  label=Ensemble(confbert1,confbert0, conflstm1, conflstm0)
  print(label)