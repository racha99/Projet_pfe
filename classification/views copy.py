from enum import auto
from django.http.response import JsonResponse
from django.shortcuts import render, redirect
from django.http import HttpResponse, HttpResponseRedirect
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn import svm
from django.views.generic.edit import UpdateView, CreateView, DeleteView
from django.views.generic.detail import DetailView
from django import forms
from crispy_forms.helper import FormHelper
from crispy_forms.layout import Submit, Button, Layout, Div, Field, Fieldset, HTML, ButtonHolder, LayoutObject
from django.urls import reverse_lazy
import json as simplejson
from django.conf import settings
import nltk
nltk.download('punkt') 
nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk import word_tokenize
from nltk.tokenize import sent_tokenize
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
import string
import re
import pickle
import spacy 
import fr_core_news_sm
from django.core.files.storage import FileSystemStorage
import os
import urllib.parse
from urllib.parse import parse_qs
from django.contrib import messages
import pandas as pd
from django.shortcuts import render
import json
from django.contrib.staticfiles.utils import get_files
from django.contrib.staticfiles.storage import StaticFilesStorage
import glob


from skmultilearn.problem_transform import BinaryRelevance
from sklearn.ensemble import RandomForestClassifier
from skmultilearn.problem_transform import LabelPowerset

from sklearn import preprocessing 
from contextlib import ContextDecorator
from typing import ContextManager
from django.shortcuts import render,redirect
from numpy import append
import pandas as pd
import io
from django.http import HttpResponse
import seaborn as sns
import matplotlib.pyplot as plt
from django.core.files.storage import FileSystemStorage
import os
from django.contrib import messages
from collections import Counter
from mlxtend.frequent_patterns import apriori
from mlxtend.frequent_patterns import association_rules
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_samples, silhouette_score
import prince
from kmodes.kmodes import KModes
import shutil

from skmultilearn.adapt import MLkNN
from sklearn.model_selection import GridSearchCV
import numpy as np
from scipy.sparse import lil_matrix
import sys
from wordcloud import WordCloud

def is_ajax(request):
    return request.META.get('HTTP_X_REQUESTED_WITH') == 'XMLHttpRequest'



def Home(request):
    context = {}
    global attribute

    if request.method == 'POST':

        uploaded_file = request.FILES['document']
        # attribute = request.POST.get('attributeid')

        # print(attribute)

        #check if this file ends with csv
        if uploaded_file.name.endswith('.csv'):
            savefile = FileSystemStorage()

            name = savefile.save(uploaded_file.name, uploaded_file) #gets the name of the file
          

            #we need to save the file somewhere in the project, MEDIA
            #now lets do the savings

            d = os.getcwd() # how we get the current dorectory
            file_directory = d+'\media\\'+name #saving the file in the media directory
            print(file_directory)
            readfile(file_directory)

            # request.session['attribute'] = attribute

            # if attribute not in data.axes[1]:
            #     messages.warning(request, 'Please write the column name correctly')
            # else:
            #     print(attribute)
            return redirect(Construction)

        else:
            messages.warning(request, 'File was not uploaded. Please use .csv file extension!')


    return  render(request, 'Home.html', context)
    #return HttpResponse(df_html)
    

def readfile(filename):

    #we have to create those in order to be able to access it around
    # use panda to read the file because i can use DATAFRAME to read the file
    #column;culumn2;column
    global rows,columns,data,my_file,missing_values,df_html
     #read the missing data - checking if there is a null
    missingvalue = ['?', '--']

    my_file = pd.read_csv(filename, sep=None,na_values=missingvalue, engine='python',encoding='utf-8', error_bad_lines=False)
    data = pd.DataFrame(data=my_file, index=None)
    print(my_file)
    #data = Commentpre(data)
    df_html = data.to_html(index=False)
    #print(data)

    rows = len(data.axes[0])
    columns = len(data.axes[1])


    null_data = data[data.isnull().any(axis=1)] # find where is the missing data #na null =['x1','x13']
    missing_values = len(null_data)

def Construction(request): 
    
#     my_file = pd.read_csv("media/Corpus.csv", sep=None, engine='python',encoding='utf-8')
#     data = pd.DataFrame(data=my_file, index=None)
#     Commentaire = [] 
#     for x in data['Commentaire']:
#         Commentaire.append(x)
    
#     global nlp 
#     nlp = fr_core_news_sm.load()
#     table= []
#     for i in data['Commentaire']:
#          a =Preprocess(i)
#          table.append(a)
    
#     data=data.assign(CommentairePrep=table)
#     json_records = data.reset_index().to_json(orient ='records')
#     dataa = []
#     dataa = json.loads(json_records)
#     d,vectors=ngram_tfidf(data,1,1)
#     rows1 = len(d.axes[0])
#     columns1 = len(d.axes[1])
#     print(rows1,columns1)
#     context = {'d': dataa}
#     y=data['Polarité']
#     X_train, X_test, y_train, y_test = train_test_split(d, y, test_size = 0.30, random_state = 42, stratify=y)
#     knn = KNeighborsClassifier(n_neighbors = 8, metric='euclidean')
#     knn.fit(X_train,y_train)
#     prediction=(knn.predict(X_test))
# # Save the trained model as a pickle string.
#     filename = 'polarité_kpp_11.p'
#     path = os.path.join(settings.MODELS,filename)
#     pickle.dump(knn, open(path, 'wb'))
# # Use the loaded pickled model to make predictions
#     loaded_model = pickle.load(open(path, 'rb'))
#     result = loaded_model.score(X_test, y_test)
#     print(result)

    context = {'d': 0}
    return render(request, 'Construction.html', context) 
    #return HttpResponse(df_html)   

def Lemma_Data(x):
  doc = nlp(x) 
  lemmatized_sentence = " ".join([token.lemma_ for token in doc])
  return lemmatized_sentence


def Stop_Wordfr(x):
  operators = set(('pas', 'et', 'ou','mais'))
  stopwords_fr = set(stopwords.words('french'))-operators
  stopfr_filter =  lambda text: [token for token in text if token.lower() not in  stopwords_fr]
  txt=" ".join(stopfr_filter(word_tokenize(x, language="french")))
  return txt


def Data_Filter(x):
  x = re.sub(r'https*\S+', ' ', x) #Supprimer les urls
  x = re.sub(r'@\S+', ' ', x) #Supprimer les tags
  x = re.sub(r'#\S+', ' ', x) #Supprimer les hashtags
  x = re.sub('[%s]' % re.escape(string.punctuation), ' ', x) #Supprimer la ponctuation
  x = re.sub(r'\w*\d+\w*', '', x) #Supprimer les nombres
  x = re.sub(r'\s{2,}', ' ', x) #supprimer double espace vide
  return x

def Preprocess(x):
    x=Data_Filter(x)
    x=Lemma_Data(x)
    x=Stop_Wordfr(x)
    return x

#zzz
def Niveau(x):
    if (x == 'p'):
        x="Polarité"
    else:
        if (x == 's' ):
          x="Sentiment"
        if (x =='a'): 
          x= "Aspect" 
        if (x=='at'):
          x = "Attitude"   
        if (x=='vv'):
          x = "Valeur"   
        if (x=='e'):
          x = "Attente" 
        if (x=='t'):
          x = "testvalue" 
        if (x=='ts'):
          x = "testsentiments" 
        if (x=='ta'):
          x = "testaspects"   

    #add ta3 tous
    return x

def Pretraitement(x,f,l,v):
   if (f == 'true' ) :
        x=Data_Filter(x)
   if(l =='true')  :  
            
            x=Lemma_Data(x) 
   if(v =='true')  :  
            x=Stop_Wordfr(x)            
   return x

def listToString(s):  
    str1 = "" 
    return (str1.join(str(s)))
def ngram_tfidf(data,n1,n2):
    """ représentation vectorielle avec les ngrammes pondérés par la TF/IDF """
    corpus = []
    for index, row in data.iterrows():
        corpus.append(listToString(row['CommentairePrep']))

    vectorizer = TfidfVectorizer(sublinear_tf=True, ngram_range=(n1, n2), min_df=5, max_df=0.7, norm='l2')
    vectors = vectorizer.fit(corpus)
    feature_names = vectorizer.get_feature_names()
    res = vectors.transform(corpus)
    dense = res.todense()
    denselist = dense.tolist()
    df = pd.DataFrame(denselist, columns=feature_names)
    return df, vectors

#zzz
def train_polarite(clas,rep):  
    x=0
    if(clas=="rna"):
        x= MLPClassifier()
    else :
        if(clas=="svm"):
            if(rep=="11"): 
               x=svm.SVC(C=10, kernel='rbf', degree=3, gamma=1) 
            if(rep=="22"): 
                x=svm.SVC(C=1.0, kernel='rbf', degree=3, gamma=1) 
            if(rep=="12"):
                x=svm.SVC(C=10, kernel='rbf', degree=3, gamma=1)  
            if(rep=="33"):
                x=svm.SVC(C=10, kernel='rbf', degree=3, gamma=0.1)  
            if(rep=="13"):
                x=svm.SVC(C=1.0, kernel='linear', degree=3, gamma='auto') 
        else:
            if(rep=="11"):  
              x= KNeighborsClassifier(n_neighbors = 8, metric='euclidean')
            if(rep=="22"): 
              x= KNeighborsClassifier(n_neighbors = 5, metric='euclidean')  
            if(rep=="12"): 
               x= KNeighborsClassifier(n_neighbors = 17, metric='euclidean')
            if(rep=="33"): 
               x= KNeighborsClassifier(n_neighbors = 5, weights='distance') 
            if(rep=="13"):
               x= KNeighborsClassifier(n_neighbors = 21, metric='euclidean')                           
    return (x)

def train_aspect(clas,rep):  
    x=0
    if(clas=="rna"):
        x= MLPClassifier()
    else :
        if(clas=="svm"):
            if(rep=="11"): 
                x=svm.SVC(C=1.0, kernel='linear', degree=3, gamma=1) 
            if(rep=="22"): 
                x=svm.SVC(C=1.0, kernel='sigmoid', degree=3, gamma=1) 
            if(rep=="12"):
                x=svm.SVC(C=1.0, kernel='sigmoid', degree=3, gamma=1)  
            if(rep=="33"):
                x=svm.SVC(C=1.0, kernel='rbf', degree=3, gamma=1)  
            if(rep=="13"):
                x=svm.SVC(C=10, kernel='rbf', degree=3, gamma=0.1) 
        else:
            if(rep=="11"):  
                x= KNeighborsClassifier(n_neighbors = 26, metric='euclidean')
            if(rep=="22"):
                x= KNeighborsClassifier(n_neighbors = 1, metric='euclidean')  
            if(rep=="12"): 
                x= KNeighborsClassifier(n_neighbors = 26, metric='euclidean')
            if(rep=="33"): 
                x= KNeighborsClassifier(n_neighbors = 6, metric='euclidean') 
            if(rep=="13"):
                x= KNeighborsClassifier(n_neighbors = 29, metric='euclidean')                           
    return (x)

def train_sentiment(clas,rep):  
    x=0
    if(clas=="rna"):
        x= MLPClassifier()
    else :
        if(clas=="svm"):
            if(rep=="11"): 
                x=svm.SVC(C=1.0, kernel='linear', degree=3, gamma='auto') 
            if(rep=="22"): 
                x=svm.SVC(C=10, kernel='rbf', degree=3, gamma=0.1) 
            if(rep=="12"):
                x=svm.SVC(C=100, kernel='rbf', degree=3, gamma=0.01)  
            if(rep=="33"):
                x=svm.SVC(C=10, kernel='rbf', degree=3, gamma=0.1)  
            if(rep=="13"):
                x=svm.SVC(C=10, kernel='rbf', degree=3, gamma=0.1) 
        else:
            if(rep=="11"):  
                x= KNeighborsClassifier(n_neighbors = 5, metric='euclidean')
            if(rep=="22"):
                x= KNeighborsClassifier(n_neighbors = 4, metric='euclidean')  
            if(rep=="12"): 
                x= KNeighborsClassifier(n_neighbors = 8, metric='euclidean')
            if(rep=="33"): 
                x= KNeighborsClassifier(n_neighbors = 8, metric='euclidean') 
            if(rep=="13"):
                x= KNeighborsClassifier(n_neighbors = 8, metric='euclidean')                           
    return (x)

def train_attitude(clas,rep):  
    x=0
    if(clas=="rna"):
        x= MLPClassifier()
    else :
        if(clas=="svm"):
            if(rep=="11"): 
                x=svm.SVC(C=1.0, kernel='linear', degree=3, gamma=1) 
            if(rep=="22"): 
                x=svm.SVC(C=1.0, kernel='sigmoid', degree=3, gamma=1) 
            if(rep=="12"):
                x=svm.SVC(C=100, kernel='rbf', degree=3, gamma=0.01)  
            if(rep=="33"):
                x=svm.SVC(C=0.1, kernel='rbf', degree=3, gamma=1)  
            if(rep=="13"):
                x=svm.SVC(C=100, kernel='rbf', degree=3, gamma=0.01) 
        else:
            if(rep=="11"):  
                x= KNeighborsClassifier(n_neighbors = 10, metric='euclidean')
            if(rep=="22"):
                x= KNeighborsClassifier(n_neighbors = 7, metric='euclidean')  
            if(rep=="12"): 
                x= KNeighborsClassifier(n_neighbors = 8, metric='euclidean')
            if(rep=="33"): 
                x= KNeighborsClassifier(n_neighbors = 12, metric='manhttan') 
            if(rep=="13"):
                x= KNeighborsClassifier(n_neighbors = 30, metric='euclidean')                           
    return (x)

def train_valeur(clas,rep):
    x=0
    if(clas=="rna"):
        x= MLPClassifier()
    else :
        if(clas=="svm"):
            if(rep=="11"): 
                x=svm.SVC(C=1.0, kernel='linear', degree=3, gamma=1) 
            if(rep=="22"): 
                x=svm.SVC(C=1.0, kernel='sigmoid', degree=3, gamma=1) 
            if(rep=="12"):
                x=svm.SVC(C=1.0, kernel='sigmoid', degree=3, gamma=1)  
            if(rep=="33"):
                x=svm.SVC(C=1.0, kernel='rbf', degree=3, gamma=1)  
            if(rep=="13"):
                x=svm.SVC(C=10, kernel='rbf', degree=3, gamma=0.1) 
        else:
            if(rep=="11"):  
                x= KNeighborsClassifier(n_neighbors = 26, metric='euclidean')
            if(rep=="22"):
                x= KNeighborsClassifier(n_neighbors = 1, metric='euclidean')  
            if(rep=="12"): 
                x= KNeighborsClassifier(n_neighbors = 26, metric='euclidean')
            if(rep=="33"): 
                x= KNeighborsClassifier(n_neighbors = 6, metric='euclidean') 
            if(rep=="13"):
                x= KNeighborsClassifier(n_neighbors = 29, metric='euclidean')                           
    return (x)

def train_attente(clas,rep):  
    x=0
    if(clas=="rna"):
        x= MLPClassifier()
    else :
        if(clas=="svm"):
            if(rep=="11"): 
                x=svm.SVC(C=1.0, kernel='linear', degree=3, gamma=1) 
            if(rep=="22"): 
                x=svm.SVC(C=1.0, kernel='sigmoid', degree=3, gamma=1) 
            if(rep=="12"):
                x=svm.SVC(C=1.0, kernel='sigmoid', degree=3, gamma=1)  
            if(rep=="33"):
                x=svm.SVC(C=1.0, kernel='rbf', degree=3, gamma=1)  
            if(rep=="13"):
                x=svm.SVC(C=10, kernel='rbf', degree=3, gamma=0.1) 
        else:
            if(rep=="11"):  
                x= KNeighborsClassifier(n_neighbors = 26, metric='euclidean')
            if(rep=="22"):
                x= KNeighborsClassifier(n_neighbors = 1, metric='euclidean')  
            if(rep=="12"): 
                x= KNeighborsClassifier(n_neighbors = 26, metric='euclidean')
            if(rep=="33"): 
                x= KNeighborsClassifier(n_neighbors = 6, metric='euclidean') 
            if(rep=="13"):
                x= KNeighborsClassifier(n_neighbors = 29, metric='euclidean')                           
    return (x)

def train_testvalue(clas,rep): 
    x=0
     #parameters = {'k': range(1,3), 's': [0.5, 0.7, 1.0]}
     #score = 'f1_micro'
    if(clas=="mlknn"):
        x = MLkNN(k=26)
    if(clas=="brrf"):  
        x = BinaryRelevance(classifier = RandomForestClassifier(), require_dense = [False, True])
    if(clas=="brsvm"):  
        x = BinaryRelevance(classifier = svm.SVC(C=1.0, kernel='linear', degree=3, gamma='auto'), require_dense = [False, True])
    if(clas=="brknn"):  
        x = BinaryRelevance(classifier = KNeighborsClassifier(n_neighbors = 17,metric='euclidean'), require_dense = [False, True])
    if(clas=="brmlp"):  
        x = BinaryRelevance(classifier = MLPClassifier(), require_dense = [False, True])

    if(clas=="lprf"):  
        x = LabelPowerset(classifier = RandomForestClassifier(), require_dense = [False, True])
    if(clas=="lpsvm"):  
        x = LabelPowerset(classifier = svm.SVC(C=1.0, kernel='linear', degree=3, gamma='auto'), require_dense = [False, True])
    if(clas=="lpknn"):  
        x = LabelPowerset(classifier = KNeighborsClassifier(n_neighbors = 17,metric='euclidean'), require_dense = [False, True])
    if(clas=="lpmlp"):  
        x = LabelPowerset(classifier = MLPClassifier(), require_dense = [False, True])

        # x = GridSearchCV(MLkNN(), parameters, scoring=score)
        # print('best parameters :', x.best_params_, 'best score: ', clf.best_score_)
    return (x) 

def train_testsentiments(clas,rep): 
    x=0
     #parameters = {'k': range(1,3), 's': [0.5, 0.7, 1.0]}
     #score = 'f1_micro'
    if(clas=="mlknn"):
        x = MLkNN(k=26)
        # x = GridSearchCV(MLkNN(), parameters, scoring=score)
        # print('best parameters :', x.best_params_, 'best score: ', clf.best_score_)

    if(clas=="brrf"):  
        x = BinaryRelevance(classifier = RandomForestClassifier(), require_dense = [False, True])
    if(clas=="brsvm"):  
        x = BinaryRelevance(classifier = svm.SVC(C=1.0, kernel='linear', degree=3, gamma='auto'), require_dense = [False, True])
    if(clas=="brknn"):  
        x = BinaryRelevance(classifier = KNeighborsClassifier(n_neighbors = 17,metric='euclidean'), require_dense = [False, True])
    if(clas=="brmlp"):  
        x = BinaryRelevance(classifier = MLPClassifier(), require_dense = [False, True])

    if(clas=="lprf"):  
        x = LabelPowerset(classifier = RandomForestClassifier(), require_dense = [False, True])
    if(clas=="lpsvm"):  
        x = LabelPowerset(classifier = svm.SVC(C=1.0, kernel='linear', degree=3, gamma='auto'), require_dense = [False, True])
    if(clas=="lpknn"):  
        x = LabelPowerset(classifier = KNeighborsClassifier(n_neighbors = 17,metric='euclidean'), require_dense = [False, True])
    if(clas=="lpmlp"):  
        x = LabelPowerset(classifier = MLPClassifier(), require_dense = [False, True])

    return (x)

def train_testaspects(clas,rep): 
    x=0
     #parameters = {'k': range(1,3), 's': [0.5, 0.7, 1.0]}
     #score = 'f1_micro'
    if(clas=="mlknn"):
        x = MLkNN(k=26)
    
    if(clas=="brrf"):  
        x = BinaryRelevance(classifier = RandomForestClassifier(), require_dense = [False, True])
    if(clas=="brsvm"):  
        x = BinaryRelevance(classifier = svm.SVC(C=1.0, kernel='linear', degree=3, gamma='auto'), require_dense = [False, True])
    if(clas=="brknn"):  
        x = BinaryRelevance(classifier = KNeighborsClassifier(n_neighbors = 17,metric='euclidean'), require_dense = [False, True])
    if(clas=="brmlp"):  
        x = BinaryRelevance(classifier = MLPClassifier(), require_dense = [False, True])

    if(clas=="lprf"):  
        x = LabelPowerset(classifier = RandomForestClassifier(), require_dense = [False, True])
    if(clas=="lpsvm"):  
        x = LabelPowerset(classifier = svm.SVC(C=1.0, kernel='linear', degree=3, gamma='auto'), require_dense = [False, True])
    if(clas=="lpknn"):  
        x = LabelPowerset(classifier = KNeighborsClassifier(n_neighbors = 17,metric='euclidean'), require_dense = [False, True])
    if(clas=="lpmlp"):  
        x = LabelPowerset(classifier = MLPClassifier(), require_dense = [False, True])

        # x = GridSearchCV(MLkNN(), parameters, scoring=score)
        # print('best parameters :', x.best_params_, 'best score: ', clf.best_score_)
    return (x) 

#zzz
def classification_model(nv,classi,rep):
    x=0
    if(nv=="p"):
        x=train_polarite(classi,rep)
    if (nv=="s"):
        x=train_sentiment(classi,rep)
    if(nv=="a"):
        x = train_aspect(classi,rep)
    if(nv=="at"):
        x= train_attitude(classi,rep)
    if(nv=="vv"):
        x=train_valeur(classi,rep)
    if(nv=="e"):
        x=train_attente(classi,rep)
    if(nv=="t"):
        x=train_testvalue(classi,rep) 
    if(nv=="ts"):
        x=train_testsentiments(classi,rep)
    if(nv=="ta"):
        x=train_testaspects(classi,rep) 
    return x                 

def corpus_pretraitement(request):
 
 if is_ajax(request):
        filt = request.POST.get('filtrage', None) # getting data from first_name input 
        lem = request.POST.get('lemmatisation', None)
        vide = request.POST.get('mots', None)
        print(lem,filt,vide)
        my_file = pd.read_csv("media/Neww.csv", sep=None, engine='python',encoding='utf-8', error_bad_lines=False)
        data = pd.DataFrame(data=my_file, index=None)
        Commentaire = [] 
        for x in data['Commentaire']:
            Commentaire.append(x)
        
        global nlp 
        nlp = fr_core_news_sm.load()
        table= []
        for i in data['Commentaire']:
            a =Pretraitement(i,filt,lem,vide)
            table.append(a)
        
        data=data.assign(CommentairePrep=table)
        json_records = data.reset_index().to_json(orient ='records')
        
        return JsonResponse(json_records,safe=False) # return response as JSON
  

def modele_creation(request):
 
 if is_ajax(request):
        data = json.loads(request.POST.get('donnee',None))
        for some_variable in data:
          if some_variable['name'] == 'F':
            filt=some_variable['value']
          if some_variable['name'] == 'L':
            lem=some_variable['value']  
          if some_variable['name'] == 'V': 
            vide=some_variable['value']  
          if some_variable['name'] == 'classif': 
            classi=some_variable['value'] 
          if some_variable['name'] == 'Repres': 
            repr=some_variable['value']  
          if some_variable['name'] == 'quantity': 
             teste=some_variable['value']
          if some_variable['name'] == 'nvclass': 
             niveau=some_variable['value']      
        
        print(filt,lem,vide,classi,repr,teste) 
            
        my_file = pd.read_csv("media/Neww.csv", sep=None, engine='python',encoding='utf-8', error_bad_lines=False)
        data = pd.DataFrame(data=my_file, index=None)
        Commentaire = [] 
        for x in data['Commentaire']:
            Commentaire.append(x)
        
        global nlp 
        nlp = fr_core_news_sm.load()
        table= []
        for i in data['Commentaire']:
            a =Pretraitement(i,filt,lem,vide)
            table.append(a)
        
        data=data.assign(CommentairePrep=table)
        #json_records = data.reset_index().to_json(orient ='records')
        #print(json_records)
        #dataa = []
        #dataa = json.loads(json_records)
        d,vectors=ngram_tfidf(data,int(repr[0]),int(repr[1]))
        rows1 = len(d.axes[0])
        columns1 = len(d.axes[1])
        print(rows1,columns1)
        # context = {'d': dataa}
        
        if (niveau == "t" or niveau == "ts" or niveau == "ta"): 
          if (niveau == "t"): 
             y= data[['Utilité', 'Intrinsèque', 'Accomplissement', 'Cout']]
             label_names=["Utilité","Intrinsèque","Accomplissement","Cout"]
          elif (niveau == "ts"):
             y= data[['Anxiété', 'Colère', 'Ennui', 'Joie', 'Mécontentement', 'Confusion']]
             label_names=["Anxiété","Colère","Ennui","Joie","Mécontentement","Confusion"]
          else :
             y= data[['Présentation', 'Contenu', 'Design', 'Général', 'Communication', 'Structure']]
             label_names=["Présentation","Contenu","Design","Général","Communication","Structure"]

          X_train, X_test, y_train, y_test = train_test_split(d, y, test_size = 0.33, random_state = 42)
          classification = classification_model(niveau,classi,repr)

          X_train = lil_matrix(X_train).toarray()
          y_train = lil_matrix(y_train).toarray()
          X_test = lil_matrix(X_test).toarray()

          classification.fit(X_train, y_train)
          prediction=(classification.predict(X_test))
          print(classification_report(y_test, prediction,target_names=label_names))
          prediction = prediction.toarray()
          # print(prediction)
          
        else :
          y=data[Niveau(niveau)]
          X_train, X_test, y_train, y_test = train_test_split(d, y, test_size = 0.33, random_state = 42)
          classification = classification_model(niveau,classi,repr)
          classification.fit(X_train,y_train)
          prediction=(classification.predict(X_test))

       #  X_train, X_test, y_train, y_test = train_test_split(d, y, test_size = (int(teste)/100), random_state = 42, stratify=y)
        
        
        treatment =""
        if (filt== "true"): treatment ="F"
        if (lem== "true"):  treatment = treatment + "L"
        if (vide== "true"):  treatment = treatment + "V"
    # Save the trained model as a pickle string.

        filename = treatment+"_"+repr+"_"+classi+"_"+teste+"_"+niveau
        path = os.path.join(settings.MODELS,filename)
        if os.path.exists(path):
              os.remove(path)
        pickl = {'classificat':classification,'prediction' :prediction,'x_test' :X_test,'y_test':y_test, 'vectors':vectors}
        pickle.dump(pickl, open(path, 'wb'))
    # Use the loaded pickled model to make predictions
        with open(path,'rb') as p:
            loaded_model = pickle.load(p)
        test=loaded_model['y_test']
        classificatio=loaded_model['classificat']
        result = classificatio.score(X_test, test)
        print(result)

       
        response = {
                          'msg':"Votre modele : "+treatment+repr+classi+teste+niveau+ "  a été créé avec succès" # response message
               }
        return JsonResponse(response) # return response as JSON
       

def Classification(request):     
    path="C:/Users/ASUS/Desktop/PFE_sentiment_analysis-master/classification/models"  # insert the path to your directory   
    files_list =os.listdir(path)
    context={}
    context['files'] = files_list
            
    files = os.listdir(path)
    paths = [os.path.join(path, basename) for basename in files]
    latest_file = max(paths, key=os.path.getctime)
    ar = latest_file.split("\\")
    
    for word in ar:
       nom=word
    
    context['last'] = nom
    pathfile = os.path.join(settings.MODELS,nom)
    with open(pathfile,'rb') as p:
             loaded_model = pickle.load(p)
    x_test=loaded_model['x_test']
    test=loaded_model['y_test']
    classificatio=loaded_model['classificat']

    array = nom.split("_")
    s=[]
    for word in array:
       print(word)
 
    
    for i in range(len(array)):
      s.append(array[i])

    result = classification_report(test,classificatio.predict(x_test))
    
    
    context['test'] = s[3]
    context['represent'] = s[1]
    context['classifieur'] = s[2]
    performances = result.split()

    #zzz
    if (s[4]=="p"):
        context['niveau'] = s[4]
        negative = performances[5],performances[6],performances[7]
        neutre = performances[10],performances[11],performances[12]
        positive = performances[15],performances[16],performances[17]
        context['positive'] = positive
        context['negative'] = negative
        context['neutre'] = neutre

    if(s[4]=="a"):
        context['niveau'] = s[4]
        contenu = performances[5],performances[6],performances[7]
        communication = performances[10],performances[11],performances[12]
        design = performances[15],performances[16],performances[17]
        general = performances[20],performances[21],performances[22]
        presentation = performances[25],performances[26],performances[27]
        structure = performances[30],performances[31],performances[32]
        context['contenu'] = contenu
        context['communication'] = communication
        context['design'] = design
        context['structure'] = structure
        context['presentation'] = presentation
        context['general'] = general

    if(s[4]=="s"):
        context['niveau'] = s[4]
        anxiete = performances[5],performances[6],performances[7]
        colere = performances[10],performances[11],performances[12]
        ennui = performances[15],performances[16],performances[17]
        confusion = performances[20],performances[21],performances[22]
        joie = performances[25],performances[26],performances[27]
        mecontentement = performances[30],performances[31],performances[32]
        neutre = performances[35],performances[36],performances[37]
        context['anxiete'] = anxiete
        context['colere'] = colere
        context['ennui'] = ennui
        context['confusion'] = confusion
        context['joie'] = joie
        context['mecontentement'] = mecontentement
        context['neutre'] = neutre

    if (s[4]=="at"):
        context['niveau'] = s[4]
        amical = performances[5],performances[6],performances[7]
        hostile = performances[10],performances[11],performances[12]
        neutre = performances[15],performances[16],performances[17]
        context['amical'] = amical
        context['hostile'] = hostile
        context['neutre'] = neutre
    
    if (s[4]=="vv"):
        context['niveau'] = s[4]
        intrinseque = performances[5],performances[6],performances[7]
        utilite = performances[10],performances[11],performances[12]
        accomplissement = performances[15],performances[16],performances[17]
        cout = performances[20],performances[21],performances[22]
        neutre = performances[25],performances[26],performances[27]
        context['neutre'] = neutre
        context['intrinseque'] = intrinseque
        context['utilite'] = utilite
        context['accomplissement'] = accomplissement
        context['cout'] = cout

    if(s[4]=="e"):
        context['niveau'] = s[4]
        negative = performances[5],performances[6],performances[7]
        neutre = performances[10],performances[11],performances[12]
        positive = performances[15],performances[16],performances[17]
        context['positive'] = positive
        context['negative'] = negative
        context['neutre'] = neutre
       
    
    print(nom)

    return render(request, 'Classification.html', context)


def change_modele(request):     
    nom = request.POST.get('Nom', None)   
    # text = request.GET.get('text')

    #zzz
    path="C:/Users/ASUS/Desktop/PFE_sentiment_analysis-master/classification/models"  # insert the path to your directory   
    files_list =os.listdir(path)
    pathfile = os.path.join(settings.MODELS,nom)
    with open(pathfile,'rb') as p:
             loaded_model = pickle.load(p)
    x_test=loaded_model['x_test']
    test=loaded_model['y_test']
    classificatio=loaded_model['classificat']

    array = nom.split("_")
    s=[]
    for word in array:
       print(word)
 
    
    for i in range(len(array)):
      s.append(array[i])

    result = classification_report(test,classificatio.predict(x_test))
    context={}
    
    context['test'] = s[3]
    context['represent'] = s[1]
    context['classifieur'] = s[2]
    performances = result.split()

    #zzz
    if (s[4]=="p"):
        context['niveau'] = s[4]
        negative = performances[5],performances[6],performances[7]
        neutre = performances[10],performances[11],performances[12]
        positive = performances[15],performances[16],performances[17]
        context['positive'] = positive
        context['negative'] = negative
        context['neutre'] = neutre

    if(s[4]=="a"):
        context['niveau'] = s[4]
        contenu = performances[5],performances[6],performances[7]
        communication = performances[10],performances[11],performances[12]
        design = performances[15],performances[16],performances[17]
        general = performances[20],performances[21],performances[22]
        presentation = performances[25],performances[26],performances[27]
        structure = performances[30],performances[31],performances[32]
        context['contenu'] = contenu
        context['communication'] = communication
        context['design'] = design
        context['structure'] = structure
        context['presentation'] = presentation
        context['general'] = general

    if(s[4]=="s"):
        context['niveau'] = s[4]
        anxiete = performances[5],performances[6],performances[7]
        colere = performances[10],performances[11],performances[12]
        ennui = performances[15],performances[16],performances[17]
        confusion = performances[20],performances[21],performances[22]
        joie = performances[25],performances[26],performances[27]
        mecontentement = performances[30],performances[31],performances[32]
        neutre = performances[35],performances[36],performances[37]
        context['anxiete'] = anxiete
        context['colere'] = colere
        context['ennui'] = ennui
        context['confusion'] = confusion
        context['joie'] = joie
        context['mecontentement'] = mecontentement
        context['neutre'] = neutre

    if (s[4]=="at"):
        context['niveau'] = s[4]
        amical = performances[5],performances[6],performances[7]
        hostile = performances[10],performances[11],performances[12]
        neutre = performances[15],performances[16],performances[17]
        context['amical'] = amical
        context['hostile'] = hostile
        context['neutre'] = neutre

    if (s[4]=="vv"):
        context['niveau'] = s[4]
        intrinseque = performances[5],performances[6],performances[7]
        utilite = performances[10],performances[11],performances[12]
        accomplissement = performances[15],performances[16],performances[17]
        cout = performances[20],performances[21],performances[22]
        neutre = performances[25],performances[26],performances[27]
        context['neutre'] = neutre
        context['intrinseque'] = intrinseque
        context['utilite'] = utilite
        context['accomplissement'] = accomplissement
        context['cout'] = cout

    if(s[4]=="e"):
        context['niveau'] = s[4]
        negative = performances[5],performances[6],performances[7]
        neutre = performances[10],performances[11],performances[12]
        positive = performances[15],performances[16],performances[17]
        context['positive'] = positive
        context['negative'] = negative
        context['neutre'] = neutre  
    
    print("last"+context['niveau'])


    
    
    return JsonResponse(context)

#zzz
def polarite_prediction(a):
    x=""
    if (a=="P"):
        x= "Votre commentaire est POSITIF"
    if (a=="N"):
        x= "Votre commentaire est NEGATIF"
    if (a=="NE"):
        x= "Votre commentaire est NEUTRE"
    return x
def aspect_prediction(a):
    x=""
    if (a=="S"):
        x= "L'aspect du commentaire est Structure"
    if (a=="Pr"):
        x= "L'aspect du commentaire est Présentation"
    if (a=="C"):
        x= "L'aspect du commentaire est Contenu"
    if (a=="CO"):
        x= "L'aspect du commentaire est Communication"
    if (a=="D"):
        x= "L'aspect du commentaire est Design"  
    if (a=="G"):
        x= "L'aspect du commentaire est Général"      
    return x

def sentiment_prediction(a):
    x=""
    if (a=="J"):
        x= "Le sentiment du commentaire est Joie"
    if (a=="M"):
        x= "Le sentiment du commentaire est Mecontentement"
    if (a=="C"):
        x= "Le sentiment du commentaire est Colere"
    if (a=="F"):
        x= "Le sentiment du commentaire est Confusion"
    if (a=="A"):
        x= "Le sentiment du commentaire est Anxieté"  
    if (a=="E"):
        x= "Le sentiment du commentaire est Ennui"   
    if (a=="NE"):
        x= "Le sentiment du commentaire est Neutre"   
    return x

def attitude_prediction(a):
    x=""
    if (a=="A"):
        x= "L'attitude du commentaire est Amical"
    if (a=="H"):
        x= "L'attitude du commentaire est Hostile"
    if (a=="NE"):
        x= "L'attitude du commentaire est Neutre"  
    return x    

def valeur_prediction(a):
    x=""
    if (a=="I"):
        x= "La valeur du commentaire est Intrinsèque"
    if (a=="U"):
        x= "La valeur du commentaire est Utilité"
    if (a=="A"):
        x= "La valeur du commentaire est Accomplissemnt" 
    if (a=="C"):
        x= "La valeur du commentaire est Cout" 
    if (a=="NE"):
        x= "La valeur du commentaire est Neutre" 
    return x  
def attente_prediction(a):
    x=""
    if (a=="P"):
        x= "L'attente exprimée est POSITIVE"
    if (a=="N"):
        x= "L'attente exprimée est NEGATIVE"
    if (a=="NE"):
        x= "L'attente exprimée est NEUTRE" 

def ajax_posting(request):
 
 if is_ajax(request):
        response = {'msg':""}
        commentaire = request.POST.get('commentaire', None) # getting data 
        modele = request.POST.get('modele', None) # getting data

        array = modele.split("_")
        s=[]
        for i in range(len(array)):
          s.append(array[i])
        
        pathfile = os.path.join(settings.MODELS,modele)
        with open(pathfile,'rb') as p:
                 loaded_model = pickle.load(p)
        classificatio=loaded_model['classificat']
        vectors=loaded_model['vectors']

        if commentaire : #cheking if comment have value
            vector = vectors.transform([commentaire]).todense()
            result = classificatio.predict((vector[0]))

            if (s[4]=="t" or s[4]=="ts" or s[4]=="ta" ):
                 result =result.toarray()
                 print(result)
            
            #zzz
            if(s[4]=="p"):
              response = {'msg':polarite_prediction(result)}
            else:
                if(s[4]=="s"):
                  response = {'msg':sentiment_prediction(result)}
                else:
                    if(s[4]=="a"):
                      response = {'msg':aspect_prediction(result)} # response message
                    else:
                        if(s[4]=="at"):  
                          response = {'msg':attitude_prediction(result)}  # response message
                        else:
                            if(s[4]=="vv"):  
                             response = {'msg':valeur_prediction(result)}
                            else:
                                if(s[4]=="e"):  
                                  response = {'msg':attente_prediction(result)}
                                else:
                                    if(s[4]=="t"): 
                                      c = ["Utilité", "Intrinsèque", "Accomplissement", "Cout"]
                                      msg = ""
                                      for i in range(4):
                                        if result[0,i]==1 :
                                          if msg == "" :
                                            msg = c[i]
                                          else :
                                            msg = msg + " et " + c[i]
                                      response = {'msg' : "Votre commentaire exprime :" +msg}

                                    else:
                                      if(s[4]=="ts"): 
                                        c = ['Anxiété', 'Colère', 'Ennui', 'Joie', 'Mécontentement', 'Confusion']
                                        msg = ""
                                        for i in range(6):
                                          if result[0,i]==1 :
                                            if msg == "" :
                                               msg = c[i]
                                            else :
                                               msg = msg + " et " + c[i]   

                                        response = {'msg' : "Votre commentaire exprime :" +msg}

                                      else:
                                            if(s[4]=="ta"): 
                                              c = ['Présentation', 'Contenu', 'Design', 'Général', 'Communication', 'Structure']
                                              msg = ""
                                              for i in range(6):
                                                if result[0,i]==1 :
                                                  if msg == "" :
                                                     msg = c[i]
                                                  else :
                                                     msg = msg + " et " + c[i]  

                                              response = {'msg' : "Votre commentaire exprime :" +msg}
              
            print(response)
            return JsonResponse(response) # return response as JSON

#zzz
def Modeles(request):        
    # text = request.GET.get('text')
    vector = ClassificationConfig.vectorizer.transform(['je aime le cours']).todense()
    p = ClassificationConfig.model.predict(vector[0])
    y = ClassificationConfig.y
    x = ClassificationConfig.x
    X_train, X_test, y_train, y_test = train_test_split(x, y, test_size = 0.30, random_state = 42, stratify=y)
    prediction=ClassificationConfig.model.predict(X_test)
    result = classification_report(y_test,prediction)
    context={}
    performances = result.split()
    head = performances[0],performances[1],performances[2]
    negative = performances[5],performances[6],performances[7]
    neutre = performances[10],performances[11],performances[12]
    positive = performances[15],performances[16],performances[17]
    context['polarite'] = p
    context['head'] = head
    context['positives'] = positive
    context['negatives'] = negative
    context['neutres'] = neutre

    RSN = MLPClassifier()
    RSN.fit(X_train,y_train)
    predictionr=RSN.predict(X_test)
    resultr = classification_report(y_test,predictionr)
    performances = resultr.split()
    head = performances[0],performances[1],performances[2]
    negativer = performances[5],performances[6],performances[7]
    neutrer = performances[10],performances[11],performances[12]
    positiver = performances[15],performances[16],performances[17]
    context['positiver'] = positiver
    context['negativer'] = negativer
    context['neutrer'] = neutrer


    knn = KNeighborsClassifier(n_neighbors = 17,metric='euclidean')
    knn.fit(X_train,y_train)
    predictionk=(knn.predict(X_test))
    resultk = classification_report(y_test,predictionk)
    performances = resultk.split()
    head = performances[0],performances[1],performances[2]
    negativek = performances[5],performances[6],performances[7]
    neutrek = performances[10],performances[11],performances[12]
    positivek = performances[15],performances[16],performances[17]
    context['positivek'] = positivek
    context['negativek'] = negativek
    context['neutrek'] = neutrek

    ya = ClassificationConfig.aspect
    Xa_train, Xa_test, ya_train, ya_test = train_test_split(x, ya, test_size = 0.30, random_state = 42, stratify=y)
    SVM = svm.SVC(C=1.0, kernel='linear', degree=3, gamma='auto')
    SVM.fit(Xa_train, ya_train)
    predictionas=SVM.predict(Xa_test)
    resultas = classification_report(ya_test,predictionas)
    performances = resultas.split()
    contenu = performances[5],performances[6],performances[7]
    communication = performances[10],performances[11],performances[12]
    design = performances[15],performances[16],performances[17]
    general = performances[20],performances[21],performances[22]
    presentation = performances[25],performances[26],performances[27]
    structure = performances[30],performances[31],performances[32]
    context['contenu'] = contenu
    context['communication'] = communication
    context['design'] = design
    context['structure'] = structure
    context['presentation'] = presentation
    context['general'] = general

    RSN = MLPClassifier()
    RSN.fit(Xa_train,ya_train)
    predictionar=RSN.predict(Xa_test)
    resultar = classification_report(ya_test,predictionar)
    performances = resultar.split()
    contenuar = performances[5],performances[6],performances[7]
    communicationar = performances[10],performances[11],performances[12]
    designar = performances[15],performances[16],performances[17]
    generalar = performances[20],performances[21],performances[22]
    presentationar = performances[25],performances[26],performances[27]
    structurear = performances[30],performances[31],performances[32]
    context['contenuar'] = contenuar
    context['communicationar'] = communicationar
    context['designar'] = designar
    context['structurear'] = structurear
    context['presentationar'] = presentationar
    context['generalar'] = generalar


    knn = KNeighborsClassifier(n_neighbors = 17,metric='euclidean')
    knn.fit(Xa_train,ya_train)
    predictionak=(knn.predict(Xa_test))
    resultak = classification_report(ya_test,predictionak)
    performances = resultak.split()
    contenuak = performances[5],performances[6],performances[7]
    communicationak = performances[10],performances[11],performances[12]
    designak = performances[15],performances[16],performances[17]
    generalak = performances[20],performances[21],performances[22]
    presentationak = performances[25],performances[26],performances[27]
    structureak = performances[30],performances[31],performances[32]
    context['contenuak'] = contenuak
    context['communicationak'] = communicationak
    context['designak'] = designak
    context['structureak'] = structureak
    context['presentationak'] = presentationak
    context['generalak'] = generalak


    return render(request, 'Modeles.html', context)

#  else:
#     # text = request.GET.get('text')
#     vector = ClassificationConfig.vectorizer.transform(['je aime le cours']).todense()
#     p = ClassificationConfig.model.predict(vector[0])
#     y = ClassificationConfig.y
#     x = ClassificationConfig.x
#     X_train, X_test, y_train, y_test = train_test_split(x, y, test_size = 0.30, random_state = 42, stratify=y)
#     prediction=ClassificationConfig.model.predict(X_test)
#     result = classification_report(y_test,prediction)
#     context={}
#     performances = result.split()
#     head = performances[0],performances[1],performances[2]
#     negative = performances[5],performances[6],performances[7]
#     neutre = performances[10],performances[11],performances[12]
#     positive = performances[15],performances[16],performances[17]
#     context['polarite'] = p
#     context['head'] = head
#     context['positive'] = positive
#     context['negative'] = negative
#     context['neutre'] = neutre
#     return render(request, 'Classification.html', context)
       

    

# def index(request):
#     if request.method == 'GET' :
#             text = request.GET.get('text')
#             vector = ClassificationConfig.vectorizer.transform([text]).todense()
#             p = ClassificationConfig.model.predict(vector[0])
#             y = ClassificationConfig.y
#             x = ClassificationConfig.x
#             X_train, X_test, y_train, y_test = train_test_split(x, y, test_size = 0.30, random_state = 42, stratify=y)
#             prediction=ClassificationConfig.model.predict(X_test)
#             result = classification_report(y_test,prediction)
#             return HttpResponse(p+result)

def index(request):
    
    context = {}

    if request.method == 'POST':

        uploaded_file = request.FILES['document']
        #check if this file ends with csv
        if uploaded_file.name.endswith('.csv'):
            savefile = FileSystemStorage()

            name = savefile.save(uploaded_file.name, uploaded_file) #gets the name of the file
           


            #we need to save the file somewhere in the project, MEDIA
            #now lets do the savings

            d = os.getcwd() # how we get the current dorectory
            file_directory = d+'\media\\'+name #saving the file in the media directory
            readfile(file_directory)
            return redirect(results)


    return  render(request, 'emotion/Acceuil.html', context)


            #project_data.csv



def results(request):
    # prepare the visualization
                                #12                          
    message = 'Nous avons trouvé ' + str(rows) + ' lignes et ' + str(columns) + ' colonnes. Données manquantes: ' + str(missing_values)
    messages.warning(request, message)
    headers=[col for col in data.columns]
    out = data.values.tolist()
    totalCom=len(data.index)
    
 #WordCloud 
    # Iterating through data
    exclure_mots = ['d', 'du', 'de', 'la', 'des', 'le', 'et', 'est', 'elle', 'une', 'en', 'que', 'aux', 'qui', 'ces', 'les', 'dans', 'sur', 'l', 'un', 'pour', 'par', 'il', 'ou', 'à', 'ce', 'a', 'sont', 'cas', 'plus', 'leur', 'se', 's', 'vous', 'au', 'c', 'aussi', 'toutes', 'autre', 'comme', 'mais', 'pas', 'ou']

    comment_words = " "
    for i in data['Commentaire']: 
      i = str(i) 
      separate = i.split() 
      for j in range(len(separate)): 
        separate[j] = separate[j].lower() 
      
      comment_words += " ".join(separate)+" " 
    
    # Creating the Word Cloud
    final_wordcloud = WordCloud(width = 800, height = 800, 
                background_color ='black',
                stopwords = exclure_mots,  
                min_font_size = 5).generate(comment_words)

    # Displaying the WordCloud                    
    plt.figure(figsize = (10, 10), facecolor = None) 
    plt.imshow(final_wordcloud) 
    plt.axis("off") 
    plt.tight_layout(pad = 0) 
        
    #plt.show()
    # Save the image in the img folder:
    final_wordcloud.to_file("static/images/wordcloud.png")


    module=[]
    for x in data["Module"]:
        module.append(x)

    polarité = [] 
    for x in data["Polarité"]:
        polarité.append(x)

    valeur = [] 
    for x in data["Utilité"]:
        if x==1 : valeur.append("U")
    for x in data["Intrinsèque"]:
        if x==1 : valeur.append("I") 
    for x in data["Accomplissement"]:
        if x==1 : valeur.append("A")
    for x in data["Cout"]:
        if x==1 : valeur.append("C")
    

    aspect = [] 
    for x in data["Aspect"]:
        aspect.append(x)

    
    sentiment = [] 
    for x in data["Sentiment"]:
        sentiment.append(x)

    attitude = [] 
    for x in data["Attitude"]:
        attitude.append(x)



    attente = [] 
    for x in data["Attente"]:
        attente.append(x)

    module_stat=dict(Counter(module))
    polarité_stat = dict(Counter(polarité))

    valeur_stat = dict(Counter(valeur))

    aspect_stat = dict(Counter(aspect)) 
    sentiment_stat = dict(Counter(sentiment)) 
    attitude_stat = dict(Counter(attitude)) 

    attente_stat = dict(Counter(attente)) 
       
    
  #zzz
    g=data.groupby(["Aspect","Polarité"],as_index=False)['Commentaire'].count()

    app= data.groupby(["Présentation","Polarité"],as_index=False)['Commentaire'].count()
    app = app.rename(columns={'Polarité':'Polaritép'})

    apc= data.groupby(["Contenu","Polarité"],as_index=False)['Commentaire'].count()
    apc = apc.rename(columns={'Polarité':'Polaritéc'})

    apd= data.groupby(["Design","Polarité"],as_index=False)['Commentaire'].count()
    apd = apd.rename(columns={'Polarité':'Polaritéd'})

    apg= data.groupby(["Général","Polarité"],as_index=False)['Commentaire'].count()
    apg = apg.rename(columns={'Polarité':'Polaritég'})

    apco= data.groupby(["Communication","Polarité"],as_index=False)['Commentaire'].count()
    apco = apco.rename(columns={'Polarité':'Polaritéco'})

    aps= data.groupby(["Structure","Polarité"],as_index=False)['Commentaire'].count()
    aps = aps.rename(columns={'Polarité':'Polarités'})


 
    

    gb_sent= data.groupby(["Aspect","Sentiment"],as_index=False)['Commentaire'].count()
    

    gb_atti= data.groupby(["Aspect","Attitude"],as_index=False)['Commentaire'].count()
 



    aspect_g = [] 
    for x in g["Aspect"]:
        aspect_g.append(x)

    aspect_pol = [] 
    for x in app["Présentation"]:
        if x == 1 : aspect_pol.append('Pr')
    for x in apc["Contenu"]:
        if x == 1 : aspect_pol.append('C')
    for x in apd["Design"]:
        if x == 1 : aspect_pol.append('D')
    for x in apg["Général"]:
        if x == 1 : aspect_pol.append('G')
    for x in aps["Structure"]:
        if x == 1 : aspect_pol.append('S')
    for x in apco["Communication"]:
        if x == 1 : aspect_pol.append('Co')

     
    listkeys_AgStat=dict(Counter(aspect_g)).keys()

  
    

    listkeys_Ag=[]
    for x in listkeys_AgStat:
        listkeys_Ag.append(x)
    print (listkeys_Ag)



    list_P1= app[app["Polaritép"]=="P"]
    list_P1= list_P1[list_P1["Présentation"]==1]
    print ('list_P1 : ',list_P1)
    list_P2= apc[apc["Polaritéc"]=="P"]
    list_P2= list_P2[list_P2["Contenu"]==1]
    print ('list_P2 : ',list_P2)
    list_P3= aps[aps["Polarités"]=="P"]
    list_P3= list_P3[list_P3["Structure"]==1]
    print ('list_P3 : ',list_P3)
    list_P4= apco[apco["Polaritéco"]=="P"]
    list_P4= list_P4[list_P4["Communication"]==1]
    print ('list_P4 : ',list_P4)
    list_P5= apd[apd["Polaritéd"]=="P"]
    list_P5= list_P5[list_P5["Design"]==1]
    print ('list_P5 : ',list_P5)
    list_P6= apg[apg["Polaritég"]=="P"]
    list_P6= list_P6[list_P6["Général"]==1]
    print ('list_P6 : ',list_P6)

    listvaluespa_P = [] 
    for x in list_P2["Commentaire"]:
      listvaluespa_P.append(x)
    for x in list_P4["Commentaire"]:
      listvaluespa_P.append(x)
    for x in list_P5["Commentaire"]:
      listvaluespa_P.append(x)
    for x in list_P6["Commentaire"]:
      listvaluespa_P.append(x)
    for x in list_P1["Commentaire"]:
      listvaluespa_P.append(x)
    for x in list_P3["Commentaire"]:
      listvaluespa_P.append(x)
    print ('listvaluespa_P : ', listvaluespa_P)


    list_N1= app[app["Polaritép"]=="N"]
    list_N1= list_N1[list_N1["Présentation"]==1]
    print ('list_N1 : ',list_N1)
    list_N2= apc[apc["Polaritéc"]=="N"]
    list_N2= list_N2[list_N2["Contenu"]==1]
    print ('list_N2 : ',list_N2)
    list_N3= aps[aps["Polarités"]=="N"]
    list_N3= list_N3[list_N3["Structure"]==1]
    print ('list_N3 : ',list_N3)
    list_N4= apco[apco["Polaritéco"]=="N"]
    list_N4= list_N4[list_N4["Communication"]==1]
    print ('list_N4 : ',list_N4)
    list_N5= apd[apd["Polaritéd"]=="N"]
    list_N5= list_N5[list_N5["Design"]==1]
    print ('list_N5 : ',list_N5)
    list_N6= apg[apg["Polaritég"]=="N"]
    list_N6= list_N6[list_N6["Général"]==1]
    print ('list_N6 : ',list_N6)

    listvaluespa_N = [] 
    for x in list_N2["Commentaire"]:
      listvaluespa_N.append(x)
    for x in list_N4["Commentaire"]:
      listvaluespa_N.append(x)
    for x in list_N5["Commentaire"]:
      listvaluespa_N.append(x)
    for x in list_N6["Commentaire"]:
      listvaluespa_N.append(x)
    for x in list_N1["Commentaire"]:
      listvaluespa_N.append(x)
    for x in list_N3["Commentaire"]:
      listvaluespa_N.append(x)
    print ('listvaluespa_N : ',listvaluespa_N)

    list_NE1= app[app["Polaritép"]=="NE"]
    list_NE1= list_NE1[list_NE1["Présentation"]==1]
    print ('list_NE1 : ',list_NE1)
    list_NE2= apc[apc["Polaritéc"]=="NE"]
    list_NE2= list_NE2[list_NE2["Contenu"]==1]
    print ('list_NE2 : ',list_NE2)
    list_NE3= aps[aps["Polarités"]=="NE"]
    list_NE3= list_NE3[list_NE3["Structure"]==1]
    print ('list_NE3 : ',list_NE3)
    list_NE4= apco[apco["Polaritéco"]=="NE"]
    list_NE4= list_NE4[list_NE4["Communication"]==1]
    print ('list_NE4 : ',list_NE4)
    list_NE5= apd[apd["Polaritéd"]=="NE"]
    list_NE5= list_NE5[list_NE5["Design"]==1]
    print ('list_NE5 : ',list_NE5)
    list_NE6= apg[apg["Polaritég"]=="NE"]
    list_NE6= list_NE6[list_NE6["Général"]==1]
    print ('list_NE6 : ',list_NE6)

    listvaluespa_NE = [] 
    for x in list_NE2["Commentaire"]:
      listvaluespa_NE.append(x)
    for x in list_NE4["Commentaire"]:
      listvaluespa_NE.append(x)
    for x in list_NE5["Commentaire"]:
      listvaluespa_NE.append(x)
    for x in list_NE6["Commentaire"]:
      listvaluespa_NE.append(x)
    for x in list_NE1["Commentaire"]:
      listvaluespa_NE.append(x)
    for x in list_NE3["Commentaire"]:
      listvaluespa_NE.append(x)
    print ('listvaluespa_NE : ', listvaluespa_NE)

    list_P= g[g["Polarité"]=="P"]
    list_N= g[g["Polarité"]=="N"]
    list_NE= g[g["Polarité"]=="NE"]

    

    listvaluesg_P=list_P["Commentaire"].tolist()
    listvaluesg_N=list_N["Commentaire"].tolist()
    listvaluesg_NE=list_NE["Commentaire"].tolist()

    

    aspect_s= [] 
    for x in gb_sent["Aspect"]:
        aspect_s.append(x)

    listkeys_As=dict(Counter(aspect_s)).keys()
    

    listkeys_Ags=[]
    for x in listkeys_As:
        listkeys_Ags.append(x)

   

    lists_J= gb_sent[gb_sent["Sentiment"]=="J"]
    lists_M= gb_sent[gb_sent["Sentiment"]=="M"]
    lists_NE= gb_sent[gb_sent["Sentiment"]=="NE"]
    lists_E= gb_sent[gb_sent["Sentiment"]=="E"]
    lists_A= gb_sent[gb_sent["Sentiment"]=="A"]
    lists_C= gb_sent[gb_sent["Sentiment"]=="C"]
    lists_F= gb_sent[gb_sent["Sentiment"]=="F"]

    listvaluess_J=lists_J["Commentaire"].tolist()
    listvaluess_M=lists_M["Commentaire"].tolist()
    listvaluess_NE=lists_NE["Commentaire"].tolist()
    listvaluess_E=lists_E["Commentaire"].tolist()
    listvaluess_A=lists_A["Commentaire"].tolist()
    listvaluess_C=lists_C["Commentaire"].tolist()
    listvaluess_F=lists_F["Commentaire"].tolist()


    aspect_a= [] 
    for x in gb_atti["Aspect"]:
        aspect_a.append(x)

    listkeys_At=dict(Counter(aspect_a)).keys()
    

    listkeys_Agt=[]
    for x in listkeys_At:
        listkeys_Agt.append(x)

    

    liste_A= gb_atti[gb_atti["Attitude"]=="A"]
    liste_H= gb_atti[gb_atti["Attitude"]=="H"]
    liste_NE= gb_atti[gb_atti["Attitude"]=="NE"]

    listvaluest_A=liste_A["Commentaire"].tolist()
    listvaluest_H=liste_H["Commentaire"].tolist()
    listvaluest_NE=liste_NE["Commentaire"].tolist()
    


    polarité_keys = polarité_stat.keys()
    polarité_values = polarité_stat.values()

    valeur_keys = valeur_stat.keys()
    valeur_values = valeur_stat.values()
 


    aspect_keys = aspect_stat.keys()
    aspect_values = aspect_stat.values()
    sentiment_keys = sentiment_stat.keys()
    sentiment_values = sentiment_stat.values()
    attitude_keys = attitude_stat.keys()
    attitude_values = attitude_stat.values()
    module_keys=module_stat.keys()
    module_values= module_stat.values()

    attente_keys = attente_stat.keys()
    attente_values = attente_stat.values()
    



    asp_dom=max(aspect_values)
    aspN_dom=max(aspect_stat,key=aspect_stat.get)
    polN_dom=max(polarité_values)
    pol_dom=max(polarité_stat, key=polarité_stat.get)
   

    VN_dom=max(valeur_values)
    V_dom=max(valeur_stat, key=valeur_stat.get)
  


    Ex_dom=max(attente_stat, key=attente_stat.get)
    ExN_dom=max(attente_values)
    
    emoN_dom=max(sentiment_values)
    emo_dom=max(sentiment_stat, key=sentiment_stat.get)
    attiN_dom=max(attitude_values)
    atti_dom=max(attitude_stat,key=attitude_stat.get)


 


    listkeys_P = []
    listvalues_P= []

    listkeys_V = []
    listvalues_V= []

    listkeys_A = []
    listvalues_A = []
    listkeys_S = []
    listvalues_S = []
    listkeys_T = []
    listvalues_T = []
    listkeys_M=[]
    listevalues_M=[]

    listkeys_Ex = []
    listvalues_Ex= []

    for x in valeur_keys:
        listkeys_V.append(x)

    for y in valeur_values:
        listvalues_V.append(y)


    for x in polarité_keys:
        listkeys_P.append(x)

    for y in polarité_values:
        listvalues_P.append(y)
    
    for x in aspect_keys:
        listkeys_A.append(x)

    for y in aspect_values:
        listvalues_A.append(y)
    
    for x in sentiment_keys:
        listkeys_S.append(x)

    for y in sentiment_values:
        listvalues_S.append(y)
    
    for x in attitude_keys:
        listkeys_T.append(x)

    for y in attitude_values:
        listvalues_T.append(y)

    for x in module_keys:
        listkeys_M.append(x)

    for y in module_values:
        listevalues_M.append(y)

    for x in attente_keys:
        listkeys_Ex.append(x)

    for y in attente_values:
        listvalues_Ex.append(y)     
     
    listkeysA=[] 
    for i in listkeys_A:
      if i=='Pr':
        j='Présentation'
        listkeysA.append(j)
      elif i=='G':
        j='Général'
        listkeysA.append(j)
      elif i=='C':
       j='Contenu'
       listkeysA.append(j)
      elif i=='S':
        j='Structure'
        listkeysA.append(j)
      elif i=='D':
        j='Design'
        listkeysA.append(j)
      else: 
        j='Communication'
        listkeysA.append(j)


    listkeysV=[] 
    for i in listkeys_V:
      if i=='U':
        j='Utilité'
        listkeysV.append(j)
      elif i=='I':
        j='Intrinsèque'
        listkeysV.append(j)
      elif i=='A':
       j='Accomplissement'
       listkeysV.append(j)
      else: 
        j='Cout'
        listkeysV.append(j)


    listkeysP=[] 
    for i in listkeys_P:
      if i=='P':
        j='Positive'
        listkeysP.append(j)
      elif i=='N':
        j='Négative'
        listkeysP.append(j)
      else:
       j='Neutre'
       listkeysP.append(j)

    listkeysEx=[] 
    for i in listkeys_Ex:
      if i=='P':
        j='Positive'
        listkeysEx.append(j)
      elif i=='N':
        j='Négative'
        listkeysEx.append(j)
      else:
       j='Neutre'
       listkeysEx.append(j)
     
    listkeysS=[]
    for i in listkeys_S:
      if i=='J':
        j='Joie'
        listkeysS.append(j)
      elif i=='M':
        j='Mécontentement'
        listkeysS.append(j)
      elif i=='A':
       j='Anxiété'
       listkeysS.append(j)
      elif i=='E':
        j='Ennui'
        listkeysS.append(j)
      elif i=='C':
        j='Colère'
        listkeysS.append(j)
      elif i=='F':
        j='Confusion'
        listkeysS.append(j)
      else: 
        j='Neutre'
        listkeysS.append(j)
    
    listkeysAt=[]
    for i in listkeys_T:
      if i=='NE':
        j='Neutre'
        listkeysAt.append(j)
      elif i=='A':
        j='Amicale'
        listkeysAt.append(j)
      else:
       j='Hostile'
       listkeysAt.append(j)
    

    listkeysAg=[]
    for i in listkeys_Ag:
      if i=='Pr':
        j='Présentation'
        listkeysAg.append(j)
      elif i=='G':
        j='Général'
        listkeysAg.append(j)
      elif i=='C':
       j='Contenu'
       listkeysAg.append(j)
      elif i=='S':
        j='Structure'
        listkeysAg.append(j)
      elif i=='D':
        j='Design'
        listkeysAg.append(j)
      else: 
        j='Communication'
        listkeysAg.append(j)
     
    
    listkeysAgs=[]
    for i in listkeys_Ags:
      if i=='Pr':
        j='Présentation'
        listkeysAgs.append(j)
      elif i=='G':
        j='Général'
        listkeysAgs.append(j)
      elif i=='C':
       j='Contenu'
       listkeysAgs.append(j)
      elif i=='S':
        j='Structure'
        listkeysAgs.append(j)
      elif i=='D':
        j='Design'
        listkeysAgs.append(j)
      else: 
        j='Communication'
        listkeysAgs.append(j)
    

    listkeysAgt=[]
    for i in listkeys_Agt:
      if i=='Pr':
        j='Présentation'
        listkeysAgt.append(j)
      elif i=='G':
        j='Général'
        listkeysAgt.append(j)
      elif i=='C':
       j='Contenu'
       listkeysAgt.append(j)
      elif i=='S':
        j='Structure'
        listkeysAgt.append(j)
      elif i=='D':
        j='Design'
        listkeysAgt.append(j)
      else: 
        j='Communication'
        listkeysAgt.append(j)

      



    context = {
        'data':out,
        'message':message,
        'headers':headers,
        'listkeys_M':listkeys_M,
        'listvalues_M':listevalues_M,
        'listkeys_A': listkeysA,
        'listvalues_A': listvalues_A,
        'listkeys_P': listkeysP,
        'listvalues_P': listvalues_P,

        'listkeys_V': listkeysV,
        'listvalues_V': listvalues_V,

        'listkeys_Ex': listkeysEx,
        'listvalues_Ex': listvalues_Ex,
        'listkeys_S': listkeysS,
        'listvalues_S': listvalues_S,
        'listkeys_T': listkeysAt,
        'listvalues_T': listvalues_T,
        'totalCom':totalCom,
        'aspDom':asp_dom,
        'aspNdom':aspN_dom,
        'polDom':pol_dom, 
        'polNdom':polN_dom,

        'VDom':V_dom, 
        'VNdom':VN_dom,

        'ExDom':Ex_dom, 
        'ExNdom':ExN_dom,
        'emoDom':emo_dom,
        'emoNdom':emoN_dom,
        'attiDom':atti_dom,
        'attiNdom':attiN_dom,

        'keys_Ag':listkeysAg,

        'listkeys_Ags':listkeysAgs,
        'listkeys_Agt':listkeysAgt,

        'listvaluesg_P':listvaluesg_P,
        'listvaluesg_N':listvaluesg_N,
        'listvaluesg_NE':listvaluesg_NE,
  
        'listvaluespa_P':listvaluespa_P,
        'listvaluespa_N':listvaluespa_N,
        'listvaluespa_NE':listvaluespa_NE,

        'listvaluess_J':listvaluess_J,
        'listvaluess_M':listvaluess_M,
        'listvaluess_NE':listvaluess_NE,
        'listvaluess_E':listvaluess_E,
        'listvaluess_A':listvaluess_A,
        'listvaluess_C':listvaluess_C,
        'listvaluess_F':listvaluess_F,
        'listvaluest_A':listvaluest_A,
        'listvaluest_H':listvaluest_H,
        'listvaluest_NE':listvaluest_NE,
    

    }

    return render(request, 'emotion\Dash.html', context)

def association(request):
    context={}
    if request.method == 'POST':
      MinSup = 'MinSup' in request.POST and request.POST['MinSup']
      MinConf = 'MinConf' in request.POST and request.POST['MinConf']
    
      df = data.drop(columns=['Commentaire','Séance','Module'])
      df=pd.get_dummies(df)
      frequent_itemsets_ap = apriori(df, min_support=float(MinSup), use_colnames=True)
      print(frequent_itemsets_ap)
      rules_ap = association_rules(frequent_itemsets_ap, metric="confidence", min_threshold=float(MinConf))
    
      rules_ap=rules_ap.drop(columns=['antecedent support','consequent support','leverage','conviction'])
      rules_ap["antecedents"] = rules_ap["antecedents"].apply(lambda x: ', '.join(list(x))).astype("unicode")
      rules_ap["consequents"] = rules_ap["consequents"].apply(lambda x: ', '.join(list(x))).astype("unicode")
      rules_ap["lift"]=sorted(rules_ap["lift"],reverse=True)
      headers=rules_ap.columns
      out=rules_ap.values.tolist()
    
    
      context ={
        'data':out,
        'headers':headers,
        'rules_ap':rules_ap,
      }
    else:
        MinSup=0.3
        MinConf=0.7
        df = data.drop(columns=['Commentaire','Séance','Module'])
        df=pd.get_dummies(df)
        frequent_itemsets_ap = apriori(df, min_support=MinSup, use_colnames=True)
      
        rules_ap = association_rules(frequent_itemsets_ap, metric="confidence", min_threshold=MinConf)
    
        rules_ap=rules_ap.drop(columns=['antecedent support','consequent support','leverage','conviction'])
        headers=rules_ap.columns
        rules_ap["antecedents"] = rules_ap["antecedents"].apply(lambda x: ', '.join(list(x))).astype("unicode")
        rules_ap["consequents"] = rules_ap["consequents"].apply(lambda x: ', '.join(list(x))).astype("unicode")
        rules_ap["lift"]=sorted(rules_ap["lift"],reverse=True)
        out=rules_ap.values.tolist()
        context ={
        'data':out,
        'headers':headers,
        'rules_ap':rules_ap,
      }
    return  render(request, 'emotion/Association.html', context)

############################################Clustering##################################################

def clustering(request): #Kmeans

    context={}
    df=data.drop(columns=['Commentaire','Séance','Module','Aspect'])
    mca = prince.MCA()
    mca.fit(df)
    k=mca.transform(df)
    x=k.values

    wcss=[]
    for i in range(1,11):
      kmeans = KMeans(n_clusters= i, init='k-means++', random_state=0)
      kmeans.fit(x)
      wcss.append(kmeans.inertia_)
   
    
    silhouette=[]
    for n_clusters in range(2,11):
     clusterer = KMeans(n_clusters=n_clusters, random_state=10)
     cluster_labels = clusterer.fit_predict(x)
     silhouette_avg = silhouette_score(x, cluster_labels)
     silhouette.append(silhouette_avg)


    listvaluesg_P=[]
    listvaluesg_N=[]
    listvaluesg_NE=[]
    listvaluess_J=[]
    listvaluess_M=[]
    listvaluess_NE=[]
    listvaluess_E=[]
    listvaluess_A=[]
    listvaluess_C=[]
    listvaluess_F=[]
    listvaluest_A=[]
    listvaluest_H=[]
    listvaluest_NE=[]
    listkeys_Cg=[]
    listkeys_cluster = []
    listvalues_cluster= []








    if request.method == 'POST':
      k = 'k' in request.POST and request.POST['k']
      kmeansmodel = KMeans(n_clusters=int(k), random_state=10)
      y_kmeans= kmeansmodel.fit_predict(x)
      combinedDf=data.assign(cluster_predicted=y_kmeans)
      
      g=combinedDf.groupby(["cluster_predicted","Polarité"],as_index=False)["Commentaire"].count()
      gb_sent=combinedDf.groupby(["cluster_predicted","Sentiment"],as_index=False)['Commentaire'].count()
      gb_atti=combinedDf.groupby(["cluster_predicted","Attitude"],as_index=False)['Commentaire'].count()

      clusters = [] 
      for x in combinedDf["cluster_predicted"]:
        clusters.append(x)

      cluster_stat=dict(Counter(clusters))
      cluster_keys = cluster_stat.keys()
      cluster_values = cluster_stat.values()

      


      for x in cluster_keys:
        listkeys_cluster.append(x)

      for y in cluster_values:
        listvalues_cluster.append(y)



      cluster_g = [] 
      for x in g["cluster_predicted"]:
        cluster_g.append(x)
     
      listkeys_CgStat=dict(Counter(cluster_g)).keys()
    

     
      for x in listkeys_CgStat:
        listkeys_Cg.append(x)


      list_P= g[g["Polarité"]=="P"]
      list_N= g[g["Polarité"]=="N"]
      list_NE= g[g["Polarité"]=="NE"]

      list_P= pd.DataFrame(data=list_P)
      list_P.reset_index(inplace=True)
      list_N= pd.DataFrame(data=list_N)
      list_N.reset_index(inplace=True)
      list_NE= pd.DataFrame(data=list_NE)
      list_NE.reset_index(inplace=True)
    

      
######################### P in cluster##############################################################
      for i in listkeys_Cg:
        arret=False
        for j in list_P.index:
          print(j)
          if (list_P.loc[list_P.index[j], 'cluster_predicted']==listkeys_Cg[i] and arret==False):
            listvaluesg_P.append(list_P["Commentaire"][j])
            arret=True
            break
           
        if arret==False:
           listvaluesg_P.append(0)
######################### N in cluster##############################################################
      for i in listkeys_Cg:
        arret=False
        for j in list_N.index:
          print(j)
          if (list_N.loc[list_N.index[j], 'cluster_predicted']==listkeys_Cg[i] and arret==False):
            listvaluesg_N.append(list_N["Commentaire"][j])
            arret=True
            break
           
        if arret==False:
           listvaluesg_N.append(0)
######################### NE in cluster##############################################################
      for i in listkeys_Cg:
        arret=False
        for j in list_NE.index:
          print(j)
          if (list_NE.loc[list_NE.index[j], 'cluster_predicted']==listkeys_Cg[i] and arret==False):
            listvaluesg_NE.append(list_NE["Commentaire"][j])
            arret=True
            break
           
        if arret==False:
           listvaluesg_NE.append(0)
########################################################################################################
#########################################Sentiment###################################################### 
      lists_J= gb_sent[gb_sent["Sentiment"]=="J"]
      lists_M= gb_sent[gb_sent["Sentiment"]=="M"]
      lists_NE= gb_sent[gb_sent["Sentiment"]=="NE"]
      lists_E= gb_sent[gb_sent["Sentiment"]=="E"]
      lists_A= gb_sent[gb_sent["Sentiment"]=="A"]
      lists_C= gb_sent[gb_sent["Sentiment"]=="C"]
      lists_F= gb_sent[gb_sent["Sentiment"]=="F"]
    
      lists_J= pd.DataFrame(data=lists_J)
      lists_J.reset_index(inplace=True)
      lists_M= pd.DataFrame(data=lists_M)
      lists_M.reset_index(inplace=True)
      lists_NE= pd.DataFrame(data=lists_NE)
      lists_NE.reset_index(inplace=True)
      lists_C= pd.DataFrame(data=lists_C)
      lists_C.reset_index(inplace=True)
      lists_F= pd.DataFrame(data=lists_F)
      lists_F.reset_index(inplace=True)
      lists_A= pd.DataFrame(data=lists_A)
      lists_A.reset_index(inplace=True)
      lists_E= pd.DataFrame(data=lists_E)
      lists_E.reset_index(inplace=True)
    
  
################################################Joie################################################################
      for i in listkeys_Cg:
        arret=False
        for j in lists_J.index:
          if (lists_J.loc[lists_J.index[j], 'cluster_predicted']==listkeys_Cg[i] and arret==False):
            listvaluess_J.append(lists_J["Commentaire"][j])
            arret=True
            break
           
        if arret==False:
           listvaluess_J.append(0)
########################################################Mecontentement############################################
      for i in listkeys_Cg:
        arret=False
        for j in lists_M.index:
          print(j)
          if (lists_M.loc[lists_M.index[j], 'cluster_predicted']==listkeys_Cg[i] and arret==False):
            listvaluess_M.append(lists_M["Commentaire"][j])
            arret=True
            break
           
        if arret==False:
           listvaluess_M.append(0)
################################################################Neutre############################################
      for i in listkeys_Cg:
        arret=False
        for j in lists_NE.index:
          print(j)
          if (lists_NE.loc[lists_NE.index[j], 'cluster_predicted']==listkeys_Cg[i] and arret==False):
            listvaluess_NE.append(lists_NE["Commentaire"][j])
            arret=True
            break
           
        if arret==False:
           listvaluess_NE.append(0)
##############################################Ennui################################################################
      for i in listkeys_Cg:
        arret=False
        for j in lists_E.index:
          print(j)
          if (lists_E.loc[lists_E.index[j], 'cluster_predicted']==listkeys_Cg[i] and arret==False):
            listvaluess_E.append(lists_E["Commentaire"][j])
            arret=True
            break
           
        if arret==False:
           listvaluess_E.append(0)
##################################################Colère#########################################################
      for i in listkeys_Cg:
        arret=False
        for j in lists_C.index:
          print(j)
          if (lists_C.loc[lists_C.index[j], 'cluster_predicted']==listkeys_Cg[i] and arret==False):
            listvaluess_C.append(lists_C["Commentaire"][j])
            arret=True
            break
           
        if arret==False:
           listvaluess_C.append(0)
######################################################Confusion################################################
      for i in listkeys_Cg:
        arret=False
        for j in lists_F.index:
          print(j)
          if (lists_F.loc[lists_F.index[j], 'cluster_predicted']==listkeys_Cg[i] and arret==False):
            listvaluess_F.append(lists_F["Commentaire"][j])
            arret=True
            break
           
        if arret==False:
           listvaluess_F.append(0)
#####################################################Anxiété###############################################
      for i in listkeys_Cg:
        arret=False
        for j in lists_A.index:
          print(j)
          if (lists_A.loc[lists_A.index[j], 'cluster_predicted']==listkeys_Cg[i] and arret==False):
            listvaluess_A.append(lists_A["Commentaire"][j])
            arret=True
            break
           
        if arret==False:
           listvaluess_A.append(0)
###########################################################################################################
############################################Attitude######################################################

      liste_A= gb_atti[gb_atti["Attitude"]=="A"]
      liste_H= gb_atti[gb_atti["Attitude"]=="H"]
      liste_NE= gb_atti[gb_atti["Attitude"]=="NE"]

      liste_A= pd.DataFrame(data=liste_A)
      liste_A.reset_index(inplace=True)
      liste_H= pd.DataFrame(data=liste_H)
      liste_H.reset_index(inplace=True)
      liste_NE= pd.DataFrame(data=liste_NE)
      liste_NE.reset_index(inplace=True)

      
################################################Amicale####################################################
      for i in listkeys_Cg: 
        arret=False
        for j in liste_A.index:
          print(j)
          if (liste_A.loc[liste_A.index[j], 'cluster_predicted']==listkeys_Cg[i] and arret==False):
            listvaluest_A.append(liste_A["Commentaire"][j])
            arret=True
            break
           
        if arret==False:
          listvaluest_A.append(0)
##############################################Hostile#################################################
      for i in listkeys_Cg:
        arret=False
        for j in liste_H.index:
          print(j)
          if (liste_H.loc[liste_H.index[j], 'cluster_predicted']==listkeys_Cg[i] and arret==False):
            listvaluest_H.append(liste_H["Commentaire"][j])
            arret=True
            break
           
        if arret==False:
           listvaluest_H.append(0)

##########################################Neutre####################################################
      for i in listkeys_Cg:
        arret=False
        for j in liste_NE.index:
          print(j)
          if (liste_NE.loc[liste_NE.index[j], 'cluster_predicted']==listkeys_Cg[i] and arret==False):
            listvaluest_NE.append(liste_NE["Commentaire"][j])
            arret=True
            break
           
        if arret==False:
           listvaluest_NE.append(0)
####################################################################################################






    context ={
       'Kmeans_elbow':wcss,
        'Silhouette_kmeans':silhouette,
        'listKeys_cluster':listkeys_cluster,
        'listvalues_cluster':listvalues_cluster,
        'listkeys_Cg':listkeys_Cg,
        'listvaluesg_P':listvaluesg_P,
        'listvaluesg_N':listvaluesg_N,
        'listvaluesg_NE':listvaluesg_NE,
        'listvaluess_J':listvaluess_J,
        'listvaluess_M':listvaluess_M,
        'listvaluess_NE':listvaluess_NE,
        'listvaluess_E':listvaluess_E,
        'listvaluess_A':listvaluess_A,
        'listvaluess_C':listvaluess_C,
        'listvaluess_F':listvaluess_F,
        'listvaluest_A':listvaluest_A,
        'listvaluest_H':listvaluest_H,
        'listvaluest_NE':listvaluest_NE,
    }






    return  render(request, 'emotion/Clusteringkmeans.html', context)

    #################################Kmodes#################################
    
def clustering_kmodes(request):
    context={}
    df=data.drop(columns=['Commentaire','Séance','Module','Aspect'])
    mca = prince.MCA()
    mca.fit(df)

    le = preprocessing.LabelEncoder() 
    data_kmode = df.apply(le.fit_transform)
    
    cost=[]
    for i in range(1,11):
      kmodes = KModes(n_clusters=i, init = "Cao", n_init = 1, verbose=1)
      kmodes.fit(data_kmode)
      cost.append(kmodes.cost_)

    
    
    silhouette=[]
    for i in range(2,11):
      clusterer = KModes(n_clusters=i, init = "Cao", n_init = 1, verbose=1)
      cluster_labels = clusterer.fit_predict(data_kmode)
      silhouette_avg = silhouette_score(data_kmode, cluster_labels, metric='jaccard')
      silhouette.append(silhouette_avg)
     

    listvaluesg_P=[]
    listvaluesg_N=[]
    listvaluesg_NE=[]
    listvaluess_J=[]
    listvaluess_M=[]
    listvaluess_NE=[]
    listvaluess_E=[]
    listvaluess_A=[]
    listvaluess_C=[]
    listvaluess_F=[]
    listvaluest_A=[]
    listvaluest_H=[]
    listvaluest_NE=[]
    listkeys_Cg=[]
    listkeys_cluster = []
    listvalues_cluster= []








    if request.method == 'POST':
      k = 'k' in request.POST and request.POST['k']
      kmodemodel = KModes(n_clusters=int(k), init = "Cao", n_init = 1, verbose=1)
      y_kmode= kmodemodel.fit_predict(data_kmode)

      combinedDf=data.assign(cluster_predicted=y_kmode)
 
      g=combinedDf.groupby(["cluster_predicted","Polarité"],as_index=False)["Commentaire"].count()
  
      gb_sent=combinedDf.groupby(["cluster_predicted","Sentiment"],as_index=False)['Commentaire'].count()
      
      gb_atti=combinedDf.groupby(["cluster_predicted","Attitude"],as_index=False)['Commentaire'].count()
      

      clusters = [] 
      for x in combinedDf["cluster_predicted"]:
        clusters.append(x)

      cluster_stat=dict(Counter(clusters))
      cluster_keys = cluster_stat.keys()
      cluster_values = cluster_stat.values()

      


      for x in cluster_keys:
        listkeys_cluster.append(x)

      for y in cluster_values:
        listvalues_cluster.append(y)



      cluster_g = [] 
      for x in g["cluster_predicted"]:
        cluster_g.append(x)
     
      listkeys_CgStat=dict(Counter(cluster_g)).keys()
    

     
      for x in listkeys_CgStat:
        listkeys_Cg.append(x)


      list_P= g[g["Polarité"]=="P"]
      list_N= g[g["Polarité"]=="N"]
      list_NE= g[g["Polarité"]=="NE"]

      list_P= pd.DataFrame(data=list_P)
      list_P.reset_index(inplace=True)
      list_N= pd.DataFrame(data=list_N)
      list_N.reset_index(inplace=True)
      list_NE= pd.DataFrame(data=list_NE)
      list_NE.reset_index(inplace=True)
    

      
######################### P in cluster##############################################################
      for i in listkeys_Cg:
        arret=False
        for j in list_P.index:
          print(j)
          if (list_P.loc[list_P.index[j], 'cluster_predicted']==listkeys_Cg[i] and arret==False):
            listvaluesg_P.append(list_P["Commentaire"][j])
            arret=True
            break
           
        if arret==False:
           listvaluesg_P.append(0)
######################### N in cluster##############################################################
      for i in listkeys_Cg:
        arret=False
        for j in list_N.index:
          print(j)
          if (list_N.loc[list_N.index[j], 'cluster_predicted']==listkeys_Cg[i] and arret==False):
            listvaluesg_N.append(list_N["Commentaire"][j])
            arret=True
            break
           
        if arret==False:
           listvaluesg_N.append(0)
######################### NE in cluster##############################################################
      for i in listkeys_Cg:
        arret=False
        for j in list_NE.index:
          print(j)
          if (list_NE.loc[list_NE.index[j], 'cluster_predicted']==listkeys_Cg[i] and arret==False):
            listvaluesg_NE.append(list_NE["Commentaire"][j])
            arret=True
            break
           
        if arret==False:
           listvaluesg_NE.append(0)
########################################################################################################
#########################################Sentiment###################################################### 
      lists_J= gb_sent[gb_sent["Sentiment"]=="J"]
      lists_M= gb_sent[gb_sent["Sentiment"]=="M"]
      lists_NE= gb_sent[gb_sent["Sentiment"]=="NE"]
      lists_E= gb_sent[gb_sent["Sentiment"]=="E"]
      lists_A= gb_sent[gb_sent["Sentiment"]=="A"]
      lists_C= gb_sent[gb_sent["Sentiment"]=="C"]
      lists_F= gb_sent[gb_sent["Sentiment"]=="F"]
    
      lists_J= pd.DataFrame(data=lists_J)
      lists_J.reset_index(inplace=True)
      lists_M= pd.DataFrame(data=lists_M)
      lists_M.reset_index(inplace=True)
      lists_NE= pd.DataFrame(data=lists_NE)
      lists_NE.reset_index(inplace=True)
      lists_C= pd.DataFrame(data=lists_C)
      lists_C.reset_index(inplace=True)
      lists_F= pd.DataFrame(data=lists_F)
      lists_F.reset_index(inplace=True)
      lists_A= pd.DataFrame(data=lists_A)
      lists_A.reset_index(inplace=True)
      lists_E= pd.DataFrame(data=lists_E)
      lists_E.reset_index(inplace=True)
    
  
################################################Joie################################################################
      for i in listkeys_Cg:
        arret=False
        for j in lists_J.index:
          print(j)
          if (lists_J.loc[lists_J.index[j], 'cluster_predicted']==listkeys_Cg[i] and arret==False):
            listvaluess_J.append(lists_J["Commentaire"][j])
            arret=True
            break
           
        if arret==False:
           listvaluess_J.append(0)
########################################################Mecontentement############################################
      for i in listkeys_Cg:
        arret=False
        for j in lists_M.index:
          print(j)
          if (lists_M.loc[lists_M.index[j], 'cluster_predicted']==listkeys_Cg[i] and arret==False):
            listvaluess_M.append(lists_M["Commentaire"][j])
            arret=True
            break
           
        if arret==False:
           listvaluess_M.append(0)
################################################################Neutre############################################
      for i in listkeys_Cg:
        arret=False
        for j in lists_NE.index:
          print(j)
          if (lists_NE.loc[lists_NE.index[j], 'cluster_predicted']==listkeys_Cg[i] and arret==False):
            listvaluess_NE.append(lists_NE["Commentaire"][j])
            arret=True
            break
           
        if arret==False:
           listvaluess_NE.append(0)
##############################################Ennui################################################################
      for i in listkeys_Cg:
        arret=False
        for j in lists_E.index:
          print(j)
          if (lists_E.loc[lists_E.index[j], 'cluster_predicted']==listkeys_Cg[i] and arret==False):
            listvaluess_E.append(lists_E["Commentaire"][j])
            arret=True
            break
           
        if arret==False:
           listvaluess_E.append(0)
##################################################Colère#########################################################
      for i in listkeys_Cg:
        arret=False
        for j in lists_C.index:
          print(j)
          if (lists_C.loc[lists_C.index[j], 'cluster_predicted']==listkeys_Cg[i] and arret==False):
            listvaluess_C.append(lists_C["Commentaire"][j])
            arret=True
            break
           
        if arret==False:
           listvaluess_C.append(0)
######################################################Confusion################################################
      for i in listkeys_Cg:
        arret=False
        for j in lists_F.index:
          print(j)
          if (lists_F.loc[lists_F.index[j], 'cluster_predicted']==listkeys_Cg[i] and arret==False):
            listvaluess_F.append(lists_F["Commentaire"][j])
            arret=True
            break
           
        if arret==False:
           listvaluess_F.append(0)
#####################################################Anxiété###############################################
      for i in listkeys_Cg:
        arret=False
        for j in lists_A.index:
          print(j)
          if (lists_A.loc[lists_A.index[j], 'cluster_predicted']==listkeys_Cg[i] and arret==False):
            listvaluess_A.append(lists_A["Commentaire"][j])
            arret=True
            break
           
        if arret==False:
           listvaluess_A.append(0)
###########################################################################################################
############################################Attitude######################################################

      liste_A= gb_atti[gb_atti["Attitude"]=="A"]
      liste_H= gb_atti[gb_atti["Attitude"]=="H"]
      liste_NE= gb_atti[gb_atti["Attitude"]=="NE"]

      liste_A= pd.DataFrame(data=liste_A)
      liste_A.reset_index(inplace=True)
      liste_H= pd.DataFrame(data=liste_H)
      liste_H.reset_index(inplace=True)
      liste_NE= pd.DataFrame(data=liste_NE)
      liste_NE.reset_index(inplace=True)

      
################################################Amicale####################################################
      for i in listkeys_Cg: 
        arret=False
        for j in liste_A.index:
          print(j)
          if (liste_A.loc[liste_A.index[j], 'cluster_predicted']==listkeys_Cg[i] and arret==False):
            listvaluest_A.append(liste_A["Commentaire"][j])
            arret=True
            break
           
        if arret==False:
          listvaluest_A.append(0)
##############################################Hostile#################################################
      for i in listkeys_Cg:
        arret=False
        for j in liste_H.index:
          print(j)
          if (liste_H.loc[liste_H.index[j], 'cluster_predicted']==listkeys_Cg[i] and arret==False):
            listvaluest_H.append(liste_H["Commentaire"][j])
            arret=True
            break
           
        if arret==False:
           listvaluest_H.append(0)

##########################################Neutre####################################################
      for i in listkeys_Cg:
        arret=False
        for j in liste_NE.index:
          print(j)
          if (liste_NE.loc[liste_NE.index[j], 'cluster_predicted']==listkeys_Cg[i] and arret==False):
            listvaluest_NE.append(liste_NE["Commentaire"][j])
            arret=True
            break
           
        if arret==False:
           listvaluest_NE.append(0)
####################################################################################################






    context ={
       'Kmode_elbow':cost,
        'Silhouette_kmode':silhouette,
        'listKeys_cluster':listkeys_cluster,
        'listvalues_cluster':listvalues_cluster,
        'listkeys_Cg':listkeys_Cg,
        'listvaluesg_P':listvaluesg_P,
        'listvaluesg_N':listvaluesg_N,
        'listvaluesg_NE':listvaluesg_NE,
        'listvaluess_J':listvaluess_J,
        'listvaluess_M':listvaluess_M,
        'listvaluess_NE':listvaluess_NE,
        'listvaluess_E':listvaluess_E,
        'listvaluess_A':listvaluess_A,
        'listvaluess_C':listvaluess_C,
        'listvaluess_F':listvaluess_F,
        'listvaluest_A':listvaluest_A,
        'listvaluest_H':listvaluest_H,
        'listvaluest_NE':listvaluest_NE,
    }


    return  render(request, 'emotion/ClusteringKmodes.html', context)  
