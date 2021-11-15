import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd
import numpy as np
import re 
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.stem import SnowballStemmer
from nltk.corpus import wordnet
from nltk.stem.wordnet import WordNetLemmatizer
import nltk

class Recommendation:
    
    def __init__(self):
        nltk.data.path.append('./nltk_data/')
        nltk.download('stopwords')
        nltk.download('punkt')
        nltk.download('averaged_perceptron_tagger')
        nltk.download('wordnet')
        self.data = pickle.load(open('data.pkl','rb'))
        self.user_final_rating = pickle.load(open('user_final_rating.pkl','rb'))
        self.model = pickle.load(open('logistic_regression.pkl','rb'))
        self.raw_data = pd.read_csv("sample30.csv")
        self.data = pd.concat([self.raw_data[['id','name','brand','categories','manufacturer']],self.data], axis=1)
        
    def getTopProductsHTML(self, user):
        items = self.user_final_rating.loc[user].sort_values(ascending=False)[0:20].index
        tfs=pd.read_pickle('tfidf.pkl')
        temp=self.data[self.data.id.isin(items)]
        X = tfs.transform(temp['Review'].values.astype(str))
        temp=temp[['id']]
        temp['prediction'] = self.model.predict(X)
        temp['prediction'] = temp['prediction'].map({'Postive':1,'Negative':0})
        temp=temp.groupby('id').sum()
        temp['positive_percent']=temp.apply(lambda x: x['prediction']/sum(x), axis=1)
        final_list=temp.sort_values('positive_percent', ascending=False).iloc[:5,:].index
        return self.data[self.data.id.isin(final_list)][['id', 'brand',
                              'categories', 'manufacturer', 'name']].drop_duplicates().to_html(index=False)

    def getTopProductsJSON(self, user):
        items = self.user_final_rating.loc[user].sort_values(ascending=False)[0:20].index
        tfs=pd.read_pickle('tfidf.pkl')
        temp=self.data[self.data.id.isin(items)]
        X = tfs.transform(temp['Review'].values.astype(str))
        temp=temp[['id']]
        temp['prediction'] = self.model.predict(X)
        temp['prediction'] = temp['prediction'].map({'Postive':1,'Negative':0})
        temp=temp.groupby('id').sum()
        temp['positive_percent']=temp.apply(lambda x: x['prediction']/sum(x), axis=1)
        final_list=temp.sort_values('positive_percent', ascending=False).iloc[:5,:].index
        return self.data[self.data.id.isin(final_list)][['id', 'brand',
                              'categories', 'manufacturer', 'name']].drop_duplicates().to_json(orient="table")

    def getUsers(self):
        s= np.array(self.user_final_rating.index).tolist()
        return ''.join(e+',' for e in s) 

    def nltk_tag_to_wordnet_tag(self,nltk_tag):
        #POS tagging
        if nltk_tag.startswith('J'):
            return wordnet.ADJ
        elif nltk_tag.startswith('V'):
            return wordnet.VERB
        elif nltk_tag.startswith('N'):
            return wordnet.NOUN
        elif nltk_tag.startswith('R'):
            return wordnet.ADV
        else:
            return None

    def lemmatize_sentence(self,sentence):
        #Tokenize the sentence and fetch POS tags
        snow = SnowballStemmer('english') 
        lemmatizer = nltk.stem.WordNetLemmatizer()
        wordnet_lemmatizer = WordNetLemmatizer()
        nltk_tagged = nltk.pos_tag(nltk.word_tokenize(sentence))
        #Tuple zipping (token, wordnet_tag)
        wordnet_tagged = map(lambda x: (x[0], self.nltk_tag_to_wordnet_tag(x[1])), nltk_tagged)
        lemmatized_sentence = []
        for word, tag in wordnet_tagged:
            if tag is None:
                #If there is no available tag, append the token as it is
                lemmatized_sentence.append(snow.stem(word)) #Stem the word if no lemma is obtained
            else:
                #Else use the tag to lemmatize the token
                lemmatized_sentence.append(lemmatizer.lemmatize(word, tag))
        return " ".join(lemmatized_sentence)

    def analyiseSentiment(self,text):
            tfs=pd.read_pickle('tfidf.pkl')
            mdl=pd.read_pickle('logistic_regression.pkl')
            #Text preprocessing

            #Remove HTML tags
            p = re.compile('<.*?>')
            text=p.sub('',text)

            #Remove punctuations
            p = re.compile(r'[?|!|\'|"|#|.|,|)|(|\|/|~|%|*]')
            text=p.sub('',text)

            #Stopwords
            stop = stopwords.words('english')

            #Lemmatization
            text=self.lemmatize_sentence(text)

            #TF-IDF vectorisation
            sent_T=tfs.transform([text])

            #Predict using our chosen top text classification model
            return mdl.predict(sent_T)[0];