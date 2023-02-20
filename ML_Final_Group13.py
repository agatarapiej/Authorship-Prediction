#!/usr/bin/env python
# coding: utf-8

# In[39]:


import json
import numpy as np
import pandas as pd
import dask.bag as db
import re
import nltk

from collections import OrderedDict
from nltk.stem import WordNetLemmatizer
from sklearn.linear_model import SGDClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score
from sklearn.svm import SVC, LinearSVC
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from nltk.tokenize import word_tokenize
from sklearn.linear_model import LogisticRegression


# In[40]:


import logging
logger = logging.getLogger("spacy")
logger.setLevel(logging.ERROR)


# In[41]:


nltk.download('wordnet')


# In[42]:


# Opening JSON file
f = open('train.json',)
   
# returns JSON object as 
# a dictionary
data = json.load(f)


# In[43]:


print(data)


# In[44]:


data = pd.DataFrame(data)
print(data)


# In[45]:


# Opening JSON file
g = open('test.json',)
   
# returns JSON object as 
# a dictionary
test = json.load(g)
test = pd.DataFrame(test)


# In[46]:


print(test.shape)
def unique(list1):
    x = np.array(list1)
    y = np.unique(x)
    print(y.shape)
    
unique(data['abstract']) 
unique(data['authorId'])
unique(data['paperId']) 
unique(data['venue'])
unique(data['title']) 
unique(data['authorName'])
unique(data['year'])

unique(test['abstract']) 
unique(test['paperId']) 
unique(test['venue'])
unique(test['title']) 
unique(test['year'])

# In[47]:


lemmatizer = WordNetLemmatizer()

def lemma(text):
    return [lemmatizer.lemmatize(word)for word in text ]


# In[48]:


def preprocess(data, columns):
    for col in columns:
        data[col+"_clean"]= data[col].apply(lambda x: re.sub(r'[^A-Za-z]+',' ', x))
        data[col+"_clean"]= data[col+"_clean"].str.lower()
        data[col+"_clean"]= data[col+"_clean"].apply(lambda x: x.strip())
        data[col+"_clean"]= data[col+"_clean"].apply(word_tokenize)
        data[col+"_clean"]= data[col+"_clean"].apply(lemma)
    return data 


# In[49]:


columns = ['title', 'abstract', 'venue']
data = preprocess(data, columns)


# In[50]:


print(data)


# In[51]:


data['year'] = data[ "year"].astype('str').tolist()
data['year']= data["year"].apply(word_tokenize)


# In[52]:


print(data)


# In[53]:


X = data["title_clean"] + data['abstract_clean'] + data['venue_clean'] + data['year']
y = data['authorId']


# In[54]:


print(X[0])


# In[55]:


X_train, X_val, y_train, y_val = train_test_split( X, y, test_size=0.2, random_state=42)


# In[56]:


vectorizer = TfidfVectorizer(preprocessor=' '.join, analyzer = 'word')
vectorizer.fit(X_train)

X_train = vectorizer.transform(X_train)
X_val = vectorizer.transform(X_val)


# In[57]:


#Predicting using SGDClassifier
model = SGDClassifier()
model.fit(X_train, y_train)
pred = model.predict(X_val)
print(accuracy_score(y_val, pred))


# In[172]:


#Predicting using SupportVectorClassifier
svm_model = SVC()
svm_model.fit(X_train, y_train)
pred = svm_model.predict(X_val)
print("*"* 30)
score=accuracy_score(y_val,pred)
print("SVCaccuracy is :{}".format(score))


# In[173]:


#Predicting using DecisionTreeClassifier
Decision=DecisionTreeClassifier()
Decision.fit(X_train,y_train)
pred=Decision.predict(X_val)
print("*"* 30)
score=accuracy_score(y_val,pred)
print("DecisionTreeClassifier accuracy is :{}".format(score))


# In[174]:


#Predicting using KNN
KNN=KNeighborsClassifier(n_neighbors=3)
KNN.fit(X_train,y_train)
pred=KNN.predict(X_val)
print("*"* 30)
score=accuracy_score(y_val,pred)
print("KNeighborsClassifier accuracy is :{}".format(score))


# In[371]:


#Predicting using LinearSVC
svm_linear = LinearSVC()
svm_linear.fit(X_train, y_train)
pred = svm_linear.predict(X_val)
print("*"* 30)
score=accuracy_score(y_val,pred)
print("SVCLinear accuracy is :{}".format(score))


# In[ ]:


#Hyperparameter Tuning 


# In[183]:


param_grid = {'loss': ["hinge", "squared_hinge"], 
              'penalty': ['l1', 'l2'],
              'C' : [0.1,1, 10, 100]} 
  
grid = GridSearchCV(LinearSVC(), param_grid, refit = True, verbose = 3)
  
# fitting the model for grid search
grid.fit(X_train, y_train)


# In[20]:


#Predicting using LinearSVC
svm_linear = LinearSVC(C=10, loss='squared_hinge', penalty='l2')
svm_linear.fit(X_train, y_train)
pred = svm_linear.predict(X_val)
print("*"* 30)
score=accuracy_score(y_val,pred)
print("SVCLinear accuracy is :{}".format(score))


# In[ ]:
print(classification_report(y_val, pred))

#preprocess test data


# In[58]:


columns = ['title', 'abstract', 'venue']
test = preprocess(test, columns)


# In[59]:


test['year'] = test["year"].astype('str').tolist()
test['year']= test["year"].apply(word_tokenize)


# In[60]:


X_test = test["title_clean"] + test['abstract_clean'] + test['venue_clean'] + test['year']


# In[61]:


vectorizer = TfidfVectorizer(preprocessor=' '.join, analyzer = 'word')
vectorizer.fit(X)

X_test = vectorizer.transform(X_test)


# In[239]:


#Vectorize whole training set 


# In[62]:


vectorizer = TfidfVectorizer(preprocessor=' '.join, analyzer = 'word')
vectorizer.fit(X)

X = vectorizer.transform(X)


# In[ ]:


#Predictions


# In[63]:


svm_linear = LinearSVC(C=10, loss='squared_hinge', penalty='l2')
svm_linear.fit(X, y)
predictions = svm_linear.predict(X_test)


# In[64]:


tmp = pd.read_json ("test.json")
predicted = pd.DataFrame({'paperId' : [], 'authorId' : []})
predicted["paperId"], predicted["authorId"] = tmp["paperId"], predictions

print(predicted.head(5))

l = predicted.to_dict(orient = "records", into = OrderedDict())


with open("predicted.json", "w") as file:
    json.dump(l, file, indent =2)

