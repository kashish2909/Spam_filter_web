import _pickle as cPickle
import nltk
import pandas as pd
from nltk.corpus import stopwords
from joblib import load
from nltk.tokenize import word_tokenize
from flask import Flask
from flask import request, jsonify, redirect, url_for
from flask import Flask, render_template, request,Response


stop_words = set(stopwords.words('english'))

ps = nltk.PorterStemmer()

with open('word_features.pkl', 'rb') as fid1:
    word_features = cPickle.load(fid1)


def find_features(message):
    words = word_tokenize(message)
    features = {}
    for word in word_features:
        features[word] = (word in words)

    return features

nltk_ensemble=load('spam_filter.joblib')

app = Flask(__name__)

@app.route('/')
def spam_filter():
   return render_template('index.html')

@app.route('/result',methods=['POST'])
def hello_world():
    #text_ch="Free entry in 2 a wkly comp to win FA Cup final tkts 21st May 2005. Text FA to 87121 to receive entry question(std txt rate)T&C's apply 08452810075over18's"
    text_ch=request.form["input_text"]
    text=pd.Series(text_ch)

    processed_str = text.str.replace(r'^.+@[^\.].*\.[a-z]{2,}$',
                                    'emailaddress')

    # Replace URLs with 'webaddress'
    processed_str = processed_str.str.replace(r'^http\://[a-zA-Z0-9\-\.]+\.[a-zA-Z]{2,3}(/\S*)?$',
                                    'webaddress')

    # Replace money symbols with 'moneysymb' (£ can by typed with ALT key + 156)
    processed_str = processed_str.str.replace(r'£|\$', 'moneysymb')
        
    # Replace 10 digit phone numbers (formats include paranthesis, spaces, no spaces, dashes) with 'phonenumber'
    processed_str = processed_str.str.replace(r'^\(?[\d]{3}\)?[\s-]?[\d]{3}[\s-]?[\d]{4}$',
                                    'phonenumbr')
        
    # Replace numbers with 'numbr'
    processed_str = processed_str.str.replace(r'\d+(\.\d+)?', 'numbr')

    processed_str = processed_str.str.replace(r'[^\w\d\s]', ' ')

    # Replace whitespace between terms with a single space
    processed_str = processed_str.str.replace(r'\s+', ' ')

    # Remove leading and trailing whitespace
    processed_str = processed_str.str.replace(r'^\s+|\s+?$', '')
    processed_str = processed_str.str.lower()


    processed_str = processed_str.apply(lambda x: ' '.join(
        term for term in x.split() if term not in stop_words))

    processed_str = processed_str.apply(lambda x: ' '.join(
        ps.stem(term) for term in x.split()))

    #print(processed)

    processed_str=find_features(str(processed_str))

    #print(processed)
    res=nltk_ensemble.classify(processed_str)

    ans=""
    if(res==1):
        ans="spam"
        print("spam")
    else:
        ans="ham"
        print("ham")
    return render_template("index.html",ans=ans)

if __name__ == '__main__':
#    print("predicting") 
   app.debug=True
   app.run()
   app.run(debug=True)