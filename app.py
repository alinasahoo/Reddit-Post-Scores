from flask import Flask, render_template
import flask
import pickle
import re
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
import pandas as pd
import numpy as np
import praw

app = Flask(__name__)

# text pre processing
def text_preprocess(text):
    text = re.sub(r'[^\w\s]', '', text) 
    l_text = " ".join(word for word in text.lower().split() if word not in ENGLISH_STOP_WORDS)

    return l_text

# loading all models
with open('one_hot.pkl', "rb") as f:
    enc = pickle.load(f)
    
with open('rf.pkl', "rb") as f:
    rf = pickle.load(f)
    
with open('senti.pkl', "rb") as f:
    sid = pickle.load(f)

# extract information for given Reddit url
def get_data(url):
    data = {}
    reddit = praw.Reddit(client_id='rGGcUZbUNTiCFw',
                   client_secret='hD5kq4AUUN4qhvLLrqO77B0FvAGseQ', 
                   user_agent='Reddit WebScrapping')

    sub_data = reddit.submission(url=str(url))
    
    data['Title'] = [str(sub_data.title)]
    data['Gilded'] = [sub_data.gilded]
    data['Over_18'] = [sub_data.over_18]
    data['Number_of_Comments'] = [sub_data.num_comments]
    scores = sid.polarity_scores(sub_data.title)
    compound = scores['compound']
    
    if (compound >= 0.5):
        data['Predicted_value'] = ['positive']
    
    elif (compound >= 0) & (compound <= 0.5):
        data['Predicted_value'] = ['neutral']

    elif (compound <= 0):
        data['Predicted_value'] = ['negative']
        
    df = pd.DataFrame(data)
        
    return df
    
            
@app.route('/')
def home():
   return render_template('Index.html')
            
@app.route('/predict', methods=['POST'])
def predict():
   url = str(flask.request.form['url'])
   
   # get data from url and store it in the form of DataFrame
   data = get_data(url)
   
   # text pre processing
   title = text_preprocess(data['Title'][0])
   
   # converting text to numerics
   df_tokens = pd.read_csv('tokens.csv')
   test_title = []
   for word in title.split():
       if word in df_tokens.columns:
           test_title.append(df_tokens[word])

   # padding with maxlength as 300
   maxlen = 300
   test_title = test_title + [0] * (maxlen - len(test_title))
   
   # using embedding_matrix to convert words to vetors  
   embedding_matrix = np.array(pd.read_csv('embedding_matrix.csv', sep=' '))
   vectors = []
   for n in test_title:
       vectors.append(embedding_matrix[n])
       
   # calculate mean of vectors to get one vector for the text 
   vectors = [item for sublist in vectors for item in sublist]
   arr = np.array(vectors)
   final_vector = np.mean(arr, axis=0)
   final_vector = pd.DataFrame(np.array(final_vector)).T
   
   # one hot encoding with column names
   categories = ['Over_18', 'Predicted_value']
   test_encoded = enc.transform(data[categories])
   
   # drop unnecessary columns
   data.drop(["Title", 'Over_18', 'Predicted_value'], axis=1, inplace=True)
   data.reset_index(inplace=True, drop=True)
   
   # concatenate everything
   col_names = [False, True, 'negative', 'neutral', 'positive']
   test = pd.DataFrame(test_encoded.todense(), columns=col_names)
   
   X_test = pd.concat([data, final_vector, test], axis=1)
   
   # prection
   results = int(rf.predict(X_test))
   
   return render_template('Index.html', results='Predicted score for the post is: {}'.format(results))

if __name__ == "__main__":
   app.run(debug=True)
