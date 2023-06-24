import os
import torch
from transformers import BertTokenizer, BertForSequenceClassification
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
def load_bert():
    # Load the model and tokenizer
    model_path = "./complaint_detection/model/"
    tokenizer = BertTokenizer.from_pretrained(model_path)
    model = BertForSequenceClassification.from_pretrained(model_path)
    return tokenizer,model

def bert_run(tweet,tokenizer,model):
    # Encode the tweet
    encoded_input = tokenizer(tweet, truncation=True, padding=True, return_tensors='pt')

    # Get the model's output (logits)
    output = model(**encoded_input)

    # The output is a tuple, we're interested in the logits
    logits = output.logits

    # Apply softmax to the logits to get probabilities
    probabilities = torch.nn.functional.softmax(logits, dim=-1)

    # Get the confidence scores for both classes
    confidence_score_class_0 = probabilities[0, 0].item()
    confidence_score_class_1 = probabilities[0, 1].item()

    #print(f'Confidence score for class 0: {confidence_score_class_0}')
    #print(f'Confidence score for class 1: {confidence_score_class_1}')

    return probabilities[0, 1].item(),probabilities[0, 0].item()

