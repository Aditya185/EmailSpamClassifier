# -*- coding: utf-8 -*-
"""
Created on Fri Oct  2 14:49:13 2020

@author: Aditya Prakash
"""

import numpy as np
import pandas as pd
from nltk.corpus import stopwords
import string

def text_process(mess):
    """
    Takes in a string of text, then performs the following:
    1. Remove all punctuation
    2. Remove all stopwords
    3. Returns a list of the cleaned text
    """
    # Check characters to see if they are in punctuation
    nopunc = [char for char in mess if char not in string.punctuation]

    # Join the characters again to form the string.
    nopunc = ''.join(nopunc)
    
    # Now just remove any stopwords
    return [word for word in nopunc.split() if word.lower() not in stopwords.words('english')]


from sklearn.feature_extraction.text import CountVectorizer

messages = pd.read_csv('spam.csv',encoding='latin1')
messages = messages.iloc[:,[0,1]]
messages.columns = ["label", "message"]
messages.head()

messages['length'] = messages['message'].apply(len)
messages.head()

bow_transformer = CountVectorizer(analyzer=text_process).fit(messages['message'])
messages_bow = bow_transformer.transform(messages['message'])

from sklearn.feature_extraction.text import TfidfTransformer

tfidf_transformer = TfidfTransformer().fit(messages_bow)
messages_tfidf = tfidf_transformer.transform(messages_bow)

from sklearn.naive_bayes import MultinomialNB
spam_detect_model = MultinomialNB().fit(messages_tfidf, messages['label'])

from sklearn.model_selection import train_test_split

msg_train, msg_test, label_train, label_test = \
train_test_split(messages['message'], messages['label'], test_size=0.2)

from sklearn.pipeline import Pipeline

pipeline = Pipeline([
    ('bow', CountVectorizer(analyzer=text_process)),
    ('tfidf', TfidfTransformer()),
    ('classifier', MultinomialNB()),
])
pipeline.fit(msg_train,label_train)
# Save your model
import joblib
joblib.dump(pipeline, 'model.pkl')
print("Model dumped!")
#predictions = pipeline.predict(msg_test)
#print("msg: ",pipeline.predict(["message765	UR awarded a City Break and could WIN a å£200 Summer Shopping spree every WK. Txt STORE to 88039 . SkilGme. TsCs087147403231Winawk!Age16 å£1.50perWKsub"])[0])

# Load the model that you just saved
lr = joblib.load('model.pkl')

# Saving the data columns from training
model_columns = list("message")
joblib.dump(model_columns, 'model_columns.pkl')
print("Models columns dumped!")