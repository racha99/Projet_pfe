from enum import auto
from django.http.response import JsonResponse
from django.shortcuts import render, redirect
from django.http import HttpResponse, HttpResponseRedirect
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, hamming_loss
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


def hamming_score(y_true, y_pred, normalize=True, sample_weight=None):
    '''
    Compute the Hamming score (a.k.a. label-based accuracy) for the multi-label case
    http://stackoverflow.com/q/32239577/395857
    '''
    acc_list = []
    for i in range(y_true.shape[0]):
        set_true = set( np.where(y_true[i])[0] )
        set_pred = set( np.where(y_pred[i])[0] )
        tmp_a = None
        if len(set_true) == 0 and len(set_pred) == 0:
            tmp_a = 1
        else:
            tmp_a = len(set_true.intersection(set_pred))/\
                    float( len(set_true.union(set_pred)) )
       
        acc_list.append(tmp_a)
    return np.mean(acc_list)


    

def readfile(filename):

    #we have to create those in order to be able to access it around
    # use panda to read the file because i can use DATAFRAME to read the file
    #column;culumn2;column
    global rows,columns,data,my_file,missing_values,df_html
     #read the missing data - checking if there is a null
    missingvalue = ['?', '--']

    my_file = pd.read_csv(filename, sep=None,na_values=missingvalue, engine='python',encoding='utf-8', error_bad_lines=False)
    data = pd.DataFrame(data=my_file, index=None)
    
    
    df_html = data.to_html(index=False)
   

    rows = len(data.axes[0])
    columns = len(data.axes[1])


    null_data = data[data.isnull().any(axis=1)] # find where is the missing data #na null =['x1','x13']
    missing_values = len(null_data)

def Construction(request): 
    


    context = {'d': 0}
    return render(request, 'Construction.html', context) 
     

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


def Niveau(x):
    if (x == 'p'):
        x="Polarité"
    else:
        
        if (x=='at'):
          x = "Attitude"   
        
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

def train_values(clas,rep): 
    x=0
    
    if(clas=="mlknn"):
        x = MLkNN(k=27)
    if(clas=="brrf"):  
        x = BinaryRelevance(classifier = RandomForestClassifier(), require_dense = [False, True])
    if(clas=="brsvm"):  
        x = BinaryRelevance(classifier = svm.SVC(C=1.0, kernel='linear', degree=3, gamma='auto'), require_dense = [False, True])
    if(clas=="brknn"):  
        x = BinaryRelevance(classifier = KNeighborsClassifier(n_neighbors = 27,metric='euclidean'), require_dense = [False, True])
    if(clas=="brmlp"):  
        x = BinaryRelevance(classifier = MLPClassifier(), require_dense = [False, True])

    if(clas=="lprf"):  
        x = LabelPowerset(classifier = RandomForestClassifier(), require_dense = [False, True])
    if(clas=="lpsvm"):  
        x = LabelPowerset(classifier = svm.SVC(C=1.0, kernel='linear', degree=3, gamma='auto'), require_dense = [False, True])
    if(clas=="lpknn"):  
        x = LabelPowerset(classifier = KNeighborsClassifier(n_neighbors = 29,metric='euclidean'), require_dense = [False, True])
    if(clas=="lpmlp"):  
        x = LabelPowerset(classifier = MLPClassifier(), require_dense = [False, True])

       
    return (x) 

def train_sentiments(clas,rep): 
    x=0
    
    if(clas=="mlknn"):
        x = MLkNN(k=27)
        

    if(clas=="brrf"):  
        x = BinaryRelevance(classifier = RandomForestClassifier(), require_dense = [False, True])
    if(clas=="brsvm"):  
        x = BinaryRelevance(classifier = svm.SVC(C=1.0, kernel='linear', degree=3, gamma='auto'), require_dense = [False, True])
    if(clas=="brknn"):  
        x = BinaryRelevance(classifier = KNeighborsClassifier(n_neighbors = 23,metric='euclidean'), require_dense = [False, True])
    if(clas=="brmlp"):  
        x = BinaryRelevance(classifier = MLPClassifier(), require_dense = [False, True])

    if(clas=="lprf"):  
        x = LabelPowerset(classifier = RandomForestClassifier(), require_dense = [False, True])
    if(clas=="lpsvm"):  
        x = LabelPowerset(classifier = svm.SVC(C=1.0, kernel='linear', degree=3, gamma='auto'), require_dense = [False, True])
    if(clas=="lpknn"):  
        x = LabelPowerset(classifier = KNeighborsClassifier(n_neighbors = 19,metric='euclidean'), require_dense = [False, True])
    if(clas=="lpmlp"):  
        x = LabelPowerset(classifier = MLPClassifier(), require_dense = [False, True])

    return (x)

def train_aspects(clas,rep): 
    x=0
     
    if(clas=="mlknn"):
        x = MLkNN(k=27)
    
    if(clas=="brrf"):  
        x = BinaryRelevance(classifier = RandomForestClassifier(), require_dense = [False, True])
    if(clas=="brsvm"):  
        x = BinaryRelevance(classifier = svm.SVC(C=1.0, kernel='linear', degree=3, gamma='auto'), require_dense = [False, True])
    if(clas=="brknn"):  
        x = BinaryRelevance(classifier = KNeighborsClassifier(n_neighbors = 11,metric='euclidean'), require_dense = [False, True])
    if(clas=="brmlp"):  
        x = BinaryRelevance(classifier = MLPClassifier(), require_dense = [False, True])

    if(clas=="lprf"):  
        x = LabelPowerset(classifier = RandomForestClassifier(), require_dense = [False, True])
    if(clas=="lpsvm"):  
        x = LabelPowerset(classifier = svm.SVC(C=1.0, kernel='linear', degree=3, gamma='auto'), require_dense = [False, True])
    if(clas=="lpknn"):  
        x = LabelPowerset(classifier = KNeighborsClassifier(n_neighbors = 27,metric='euclidean'), require_dense = [False, True])
    if(clas=="lpmlp"):  
        x = LabelPowerset(classifier = MLPClassifier(), require_dense = [False, True])

       
    return (x) 


def classification_model(nv,classi,rep):
    x=0
    if(nv=="p"):
        x=train_polarite(classi,rep)
    if(nv=="at"):
        x= train_attitude(classi,rep)
    if(nv=="e"):
        x=train_attente(classi,rep)
    if(nv=="t"):
        x=train_values(classi,rep) 
    if(nv=="ts"):
        x=train_sentiments(classi,rep)
    if(nv=="ta"):
        x=train_aspects(classi,rep) 
    return x                 

def corpus_pretraitement(request):
 
 if is_ajax(request):
        filt = request.POST.get('filtrage', None) # getting data from first_name input 
        lem = request.POST.get('lemmatisation', None)
        vide = request.POST.get('mots', None)
        
        my_file = pd.read_csv("media/corpus.csv", sep=None, engine='python',encoding='utf-8', error_bad_lines=False)
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
           
          if some_variable['name'] == 'classif': 
            classi=some_variable['value'] 
          if some_variable['name'] == 'nvclass': 
             niveau=some_variable['value']      
        
       
        repr="13"
        teste="30"
            
        my_file = pd.read_csv("media/corpus.csv", sep=None, engine='python',encoding='utf-8', error_bad_lines=False)
        data = pd.DataFrame(data=my_file, index=None)
        Commentaire = [] 
        for x in data['Commentaire']:
            Commentaire.append(x)
        
        global nlp 
        nlp = fr_core_news_sm.load()
        table= []
        for i in data['Commentaire']:
            a =Pretraitement(i,True,True,True)
            table.append(a)
        
        data=data.assign(CommentairePrep=table)
        
        d,vectors=ngram_tfidf(data,1,3)
        rows1 = len(d.axes[0])
        columns1 = len(d.axes[1])
       
        
        if (niveau == "t" or niveau == "ts" or niveau == "ta"): 
          if (niveau == "t"): 
             y= data[['Utilité', 'Intrinsèque', 'Accomplissement', 'Cout']]
             label_names=["Utilité","Intrinsèque","Accomplissement","Cout"]
          elif (niveau == "ts"):
             y= data[['Anxiété', 'Colère', 'Ennui', 'Joie', 'Mécontentement', 'Confusion',"Satisfaction","Gratitude"]]
             label_names=["Anxiété","Colère","Ennui","Joie","Mécontentement","Confusion","Satisfaction","Gratitude"]
          else :
             y= data[['Présentation', 'Contenu', 'Design', 'Communication', 'Structure','Général', ]]
             label_names=["Présentation","Contenu","Design","Communication","Structure","Général",]

          X_train, X_test, y_train, y_test = train_test_split(d, y, test_size = 0.33, random_state = 42)
          classification = classification_model(niveau,classi,repr)

          X_train = lil_matrix(X_train).toarray()
          y_train = lil_matrix(y_train).toarray()
          X_test = lil_matrix(X_test).toarray()

          classification.fit(X_train, y_train)
          prediction=(classification.predict(X_test))
         
          prediction = prediction.toarray()
          
        else :
          y=data[Niveau(niveau)]
          X_train, X_test, y_train, y_test = train_test_split(d, y, test_size = 0.33, random_state = 42)
          classification = classification_model(niveau,classi,repr)
          classification.fit(X_train,y_train)
          prediction=(classification.predict(X_test))

       
        
        treatment ="FLV"
       

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
    
       
 
    
    for i in range(len(array)):
      s.append(array[i])

    result = classification_report(test,classificatio.predict(x_test))
    
    
    context['test'] = s[3]
    context['represent'] = s[1]
    context['classifieur'] = s[2]
    performances = result.split()
    hamming_exist=False
    if ((s[4]=="t") or (s[4]=="ta") or (s[4]=="ts")):
        hamming_exist=True
        p = classificatio.predict(x_test)
        val_hamming_loss =round(hamming_loss(test, p),2)
       
        context['hamming_loss'] = val_hamming_loss
      
         
       
        context['hamming_score'] = round(hamming_score(np.array(test), p.toarray()),2)

    if (s[4]=="p"):
        context['niveau'] = s[4]
        negative = performances[5],performances[6],performances[7]
        neutre = performances[10],performances[11],performances[12]
        positive = performances[15],performances[16],performances[17]
        context['positive'] = positive
        context['negative'] = negative
        context['neutre'] = neutre

    if(s[4]=="ta"):
        context['niveau'] = s[4]
        presentation = performances[5],performances[6],performances[7]
        contenu  = performances[10],performances[11],performances[12]
        design = performances[15],performances[16],performances[17]
        communication = performances[20],performances[21],performances[22]
        structure = performances[25],performances[26],performances[27]
        general = performances[30],performances[31],performances[32]
        context['contenu'] = contenu
        context['communication'] = communication
        context['design'] = design
        context['structure'] = structure
        context['presentation'] = presentation
        context['general'] = general

    if(s[4]=="ts"):
        context['niveau'] = s[4]
        anxiete = performances[5],performances[6],performances[7]
        colere = performances[10],performances[11],performances[12]
        ennui = performances[15],performances[16],performances[17]
        joie= performances[20],performances[21],performances[22]
        mecontentement= performances[25],performances[26],performances[27]
        confusion= performances[30],performances[31],performances[32]
        satisfaction = performances[35],performances[36],performances[37]
        gratitude=performances[40],performances[41],performances[42]
        context['anxiete'] = anxiete
        context['colere'] = colere
        context['ennui'] = ennui
        context['confusion'] = confusion
        context['joie'] = joie
        context['mecontentement'] = mecontentement
        context['satisfaction'] = satisfaction
        context['gratitude']=gratitude

    if (s[4]=="at"):
        context['niveau'] = s[4]
        amical = performances[5],performances[6],performances[7]
        hostile = performances[10],performances[11],performances[12]
        neutre = performances[15],performances[16],performances[17]
        context['amical'] = amical
        context['hostile'] = hostile
        context['neutre'] = neutre
    
    if (s[4]=="t"):
        context['niveau'] = s[4]
        utilite = performances[5],performances[6],performances[7]
        intrinseque = performances[10],performances[11],performances[12] 
        
        accomplissement = performances[15],performances[16],performances[17]
        cout = performances[20],performances[21],performances[22]
       
    
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
       
    context['hamming_exist']=hamming_exist
    

    return render(request, 'Classification.html', context)


def change_modele(request):     
    nom = request.POST.get('Nom', None)   
    
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
    
 
    
    for i in range(len(array)):
      s.append(array[i])

    result = classification_report(test,classificatio.predict(x_test))
    context={}
    
    context['test'] = s[3]
    context['represent'] = s[1]
    context['classifieur'] = s[2]
    performances = result.split()
    hamming_exist=False

    if ((s[4]=="t") or (s[4]=="ta") or (s[4]=="ts")):
      p = classificatio.predict(x_test)
      val_hamming_loss = round(hamming_loss(test, p),2)

      context['hamming_loss'] = val_hamming_loss
      hamming_exist=True
        
     
      context['hamming_score'] = round(hamming_score(np.array(test), p.toarray()),2)

    if (s[4]=="p"):
        context['niveau'] = s[4]
        negative = performances[5],performances[6],performances[7]
        neutre = performances[10],performances[11],performances[12]
        positive = performances[15],performances[16],performances[17]
        context['positive'] = positive
        context['negative'] = negative
        context['neutre'] = neutre

    if(s[4]=="ta"):
        context['niveau'] = s[4]
        presentation = performances[5],performances[6],performances[7]
        contenu  = performances[10],performances[11],performances[12]
        design = performances[15],performances[16],performances[17]
        communication = performances[20],performances[21],performances[22]
        structure = performances[25],performances[26],performances[27]
        general = performances[30],performances[31],performances[32]
        context['contenu'] = contenu
        context['communication'] = communication
        context['design'] = design
        context['structure'] = structure
        context['presentation'] = presentation
        context['general'] = general

    if(s[4]=="ts"):
        context['niveau'] = s[4]
        anxiete = performances[5],performances[6],performances[7]
        colere = performances[10],performances[11],performances[12]
        ennui = performances[15],performances[16],performances[17]
        joie= performances[20],performances[21],performances[22]
        mecontentement= performances[25],performances[26],performances[27]
        confusion= performances[30],performances[31],performances[32]
        satisfaction = performances[35],performances[36],performances[37]
        gratitude=performances[40],performances[41],performances[42]
        context['anxiete'] = anxiete
        context['colere'] = colere
        context['ennui'] = ennui
        context['confusion'] = confusion
        context['joie'] = joie
        context['mecontentement'] = mecontentement
        context['satisfaction'] = satisfaction
        context['gratitude']=gratitude
    if (s[4]=="at"):
        context['niveau'] = s[4]
        amical = performances[5],performances[6],performances[7]
        hostile = performances[10],performances[11],performances[12]
        neutre = performances[15],performances[16],performances[17]
        context['amical'] = amical
        context['hostile'] = hostile
        context['neutre'] = neutre

    if (s[4]=="t"):
        context['niveau'] = s[4]
        utilite = performances[5],performances[6],performances[7]
        intrinseque = performances[10],performances[11],performances[12] 
        
        accomplissement = performances[15],performances[16],performances[17]
        cout = performances[20],performances[21],performances[22]
        
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
    
    
    context['hamming_exist']=hamming_exist

    
    
    return JsonResponse(context)


def polarite_prediction(a):
    x=""
    if (a=="P"):
        x= "Votre commentaire est POSITIF"
    if (a=="N"):
        x= "Votre commentaire est NEGATIF"
    if (a=="NE"):
        x= "Votre commentaire est NEUTRE"
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
            com= Pretraitement(commentaire,True,True,True)
            vector = vectors.transform([com]).todense()
            result = classificatio.predict((vector[0]))

            if (s[4]=="t" or s[4]=="ts" or s[4]=="ta" ):
                 result =result.toarray()
                
            if(s[4]=="p"):
              response = {'msg':polarite_prediction(result)}
           
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
                                c = ['Anxiété', 'Colère', 'Ennui', 'Joie', 'Mécontentement', 'Confusion',"Satisfaction","Gratitude"]
                                msg = ""
                                for i in range(8):
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
              
           
            return JsonResponse(response) # return response as JSON


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


def results(request):
    # prepare the visualization
                                #12                          
    message = 'Nous avons trouvé ' + str(rows) + ' lignes et ' + str(columns) + ' colonnes. Données manquantes: ' + str(missing_values)
    messages.warning(request, message)
    headers=[col for col in data.columns]
    out = data.values.tolist()
   
    context = {
        'data':out,
        'message':message,
        'headers':headers,
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


