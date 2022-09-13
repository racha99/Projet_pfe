from django.shortcuts import render
from numpy import true_divide
import pandas as pd
from collections import Counter
from django.shortcuts import redirect, reverse
from regex import F

from wordcloud import WordCloud
import matplotlib.pyplot as plt
from .forms import AjouterComment
# Create your views here.
import os
from pyrebase import pyrebase
import datetime
from .models import User, Student, Teacher, Comment, Module, comment_traité
import json
import pickle

from django.conf import settings # import the settings file


def TestTemplate(request):
  return  render(request, 'test_new/index.html')

global selected_cours 
selected_cours = None

global selected_module 
selected_module = None
global selected_niv 
selected_niv = None

global selected_annee
selected_annee=None

def select_niv(request):       
  global selected_niv
  if request.method == "POST":
      selected_niv = request.POST.get("niveau")
      return redirect(administration)

def select_annee(request):       
    global selected_annee
    if request.method == "POST":
        selected_annee = request.POST.get("annee")
        if request.user.is_student: 
            return redirect(Input)
        elif request.user.is_teacher: 
            return redirect(Dashboard)
        else: 
            return redirect(administration)



config = {
    "apiKey": "AIzaSyCUuMFEEcYy20VP78UBmUQn2V0zpHjw5jo",
    "authDomain": "pfe-test-ab7fc.firebaseapp.com",
    "projectId": "pfe-test-ab7fc",
    "storageBucket": "pfe-test-ab7fc.appspot.com",
    "messagingSenderId": "679303919609",
    "appId": "1:679303919609:web:ec43f4ba85fbf39931af28",
    #"measurementId": "G-measurement_id",
    "databaseURL": "https://pfe-test-ab7fc-default-rtdb.firebaseio.com"
}

firebase = pyrebase.initialize_app(config)
database=firebase.database()

def data_display(request):
  data = database.child('data').get().val()
  df2 = pd.read_json(data, orient ='index')
  
  print(data)

  return  render(request, 'test_json.html',{'data': data})

# custom redirection
def login_success(request):
    """
    Redirects users based on whether they are in the admins group
    """
    global selected_cours 
    selected_cours = None

    global selected_module 
    selected_module = None
    global selected_niv 
    selected_niv = None

    global selected_annee
    selected_annee=None
    
    if request.user.is_student:
        # user is an admin
        return redirect("inputStudent")
    elif request.user.is_teacher:
        return redirect("dashboardTeacher")
    elif request.user.is_admin:
        return redirect('administration')
    elif request.user.is_app_admin:
        return redirect('Home')

# def login(request):
#     global selected_cours
#     selected_cours="all"
#     return  render(request, 'login.html')


def Dashboard(request,k=None):
  global selected_cours
  
  if request.user.is_authenticated: 
    if (request.user.is_teacher or request.user.is_admin): 
     #bdlt here ...... from here to 'polarité générale'
      today = datetime.datetime.now()
      year = today.year
  
      if selected_annee : 
        year = int(selected_annee)

      modules=[]
      if request.user.is_admin :
        module=k 
      
      else : 
        teacher = request.user.teacher
        modules = Module.objects.filter(teachers=teacher.id)
        module = modules[0].code
        if selected_cours :
          print ('iff')
          module = selected_cours 

      if (year<= 2021):
        my_file = pd.read_csv("media/Neww.csv", sep=None, engine='python',encoding='utf-8', error_bad_lines=False)
        data = pd.DataFrame(data=my_file, index=None)

        data = data[data['Module']==module]

        # data['Séance'] = (pd.to_datetime(data['Séance'])).dt.year
        # data = data[data['Séance']==year]
        
        data['Séance'] = pd.to_datetime(data['Séance']).dt.strftime('%Y-%m-%d')
        data=data[(data['Séance'] > '2020-12-31') & (data['Séance'] < '2022-01-01')]
        print("this data",data)
        

      else :
        cmnts_all = Comment.objects.filter(Module=module)
        cmnts=[]
        for c in cmnts_all :
          if (c.get_year()==year):
            cmnts.append(c.id)
        data = pd.DataFrame(list(comment_traité.objects.filter(comment_id__in=cmnts).values()))
        
      print ("data ta3 3chiya",data)

   
    #wordcloud
     #bdlt here
    if not data.empty : 
      empty = 0
      exclure_mots = ['dans','avec','elles','elle','il','ils','je',"j'",'d', 'du', 'de', 'la', 'des', 'le', 'et', 'est', 'elle', 'une', 'en', 'que', 'aux', 'qui', 'ces', 'les', 'dans', 'sur', 'l', 'un', 'pour', 'par', 'il', 'ou', 'à', 'ce', 'a', 'sont', 'cas', 'plus', 'leur', 'se', 's', 'vous', 'au', 'c', 'aussi', 'toutes', 'autre', 'comme', 'mais', 'pas', 'ou']
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
      final_wordcloud.to_file("static/images/wordcloud.png")

  # ------------------------------------------Polarité générale---------------------------------------------- 
      polarité = [] 
      for x in data["Polarité"]:
        polarité.append(x)

      polarité_stat = dict(Counter(polarité))
      

    
      if "P" in polarité_stat:
        p = polarité_stat["P"]
      else:
        p=0
    
      if "N" in polarité_stat:
        n = polarité_stat["N"]
      else:
        n=0
      
      if "NE" in polarité_stat:
        ne = polarité_stat["NE"]
      else:
        ne=0
    
    

      # ---------------------------------------Polarité par aspect----------------------------------------------
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
      
      listkeys_AgStat=dict(Counter(aspect_pol)).keys()

      listkeys_Ag=[]
      for x in listkeys_AgStat:
          listkeys_Ag.append(x)

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
  

      list_P1= app[app["Polaritép"]=="P"]
      list_P1= list_P1[list_P1["Présentation"]==1]
      print("list_p1",list_P1)

      list_P2= apc[apc["Polaritéc"]=="P"]
      list_P2= list_P2[list_P2["Contenu"]==1]
  
      list_P3= aps[aps["Polarités"]=="P"]
      list_P3= list_P3[list_P3["Structure"]==1]

      list_P4= apco[apco["Polaritéco"]=="P"]
      list_P4= list_P4[list_P4["Communication"]==1]
    
      list_P5= apd[apd["Polaritéd"]=="P"]
      list_P5= list_P5[list_P5["Design"]==1]

      list_P6= apg[apg["Polaritég"]=="P"]
      list_P6= list_P6[list_P6["Général"]==1]
  

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
      
      
      list_N1= app[app["Polaritép"]=="N"]
      list_N1= list_N1[list_N1["Présentation"]==1]

      list_N2= apc[apc["Polaritéc"]=="N"]
      list_N2= list_N2[list_N2["Contenu"]==1]
    
      list_N3= aps[aps["Polarités"]=="N"]
      list_N3= list_N3[list_N3["Structure"]==1]
    
      list_N4= apco[apco["Polaritéco"]=="N"]
      list_N4= list_N4[list_N4["Communication"]==1]

      list_N5= apd[apd["Polaritéd"]=="N"]
      list_N5= list_N5[list_N5["Design"]==1]
    
      list_N6= apg[apg["Polaritég"]=="N"]
      list_N6= list_N6[list_N6["Général"]==1]
    

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
  

      list_NE1= app[app["Polaritép"]=="NE"]
      list_NE1= list_NE1[list_NE1["Présentation"]==1]
  
      list_NE2= apc[apc["Polaritéc"]=="NE"]
      list_NE2= list_NE2[list_NE2["Contenu"]==1]
  
      list_NE3= aps[aps["Polarités"]=="NE"]
      list_NE3= list_NE3[list_NE3["Structure"]==1]

      list_NE4= apco[apco["Polaritéco"]=="NE"]
      list_NE4= list_NE4[list_NE4["Communication"]==1]

      list_NE5= apd[apd["Polaritéd"]=="NE"]
      list_NE5= list_NE5[list_NE5["Design"]==1]
  
      list_NE6= apg[apg["Polaritég"]=="NE"]
      list_NE6= list_NE6[list_NE6["Général"]==1]
    

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

      # --------------------------------------------------Valeurs--------------------------------------------------------  
      valeur = [] 
      for x in data["Utilité"]:
        if x==1 : valeur.append("U")
      for x in data["Intrinsèque"]:
        if x==1 : valeur.append("I") 
      for x in data["Accomplissement"]:
        if x==1 : valeur.append("A")
      for x in data["Cout"]:
        if x==1 : valeur.append("C")

      valeur_stat = dict(Counter(valeur))
      valeur_keys = valeur_stat.keys()
      valeur_values = valeur_stat.values()

      listkeys_V = []
      listvalues_V= []

      for x in valeur_keys:
        listkeys_V.append(x)

      for y in valeur_values:
        listvalues_V.append(y)

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
      

      # --------------------------------------------------Attente générale---------------------------------------------------
      attente = [] 
      for x in data["Attente"]:
          attente.append(x)

      attente_stat = dict(Counter(attente)) 
      attente_keys = attente_stat.keys()
      attente_values = attente_stat.values()

      listkeys_Ex = []
      listvalues_Ex= []
      for x in attente_keys:
        listkeys_Ex.append(x)

      for y in attente_values:
        listvalues_Ex.append(y)  
      
      if "P" in polarité_stat:
        p = polarité_stat["P"]
      else:
        p=0



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


      # ------------------------------------------------Attitude générale-------------------------------------------
      print("attitude general")
      attitude = [] 
      for x in data["Attitude"]:
        attitude.append(x)
      print(data["Attitude"])
      
      attitude_stat = dict(Counter(attitude)) 
      attitude_keys = attitude_stat.keys()
      attitude_values = attitude_stat.values()

      listkeys_T = []
      listvalues_T = []
      for x in attitude_keys:
        listkeys_T.append(x)

      for y in attitude_values:
        listvalues_T.append(y)
      
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
      print("attitude",listkeysAt[0])

      attitude_ne=0
      attitude_a=0
      attitude_h=0
      if "NE" in listkeys_T:
        attitude_ne=listvalues_T[0]
      if "A" in listkeys_T:
        attitude_a=listvalues_T[1]
      if "H" in listkeys_T:
        attitude_h=listvalues_T[2]

      


      

      S=attitude_ne+attitude_a+attitude_h

      attitude_ne=round((attitude_ne/S)*100)
      attitude_a=round((attitude_a/S)*100)
      attitude_h=round((attitude_h/S)*100)
      print("comme +",S,attitude_ne,attitude_a,attitude_h)




      # -----------------------------------------------Evolutions des émotions-------------------------------------------
      data["Séance"]=pd.to_datetime(data["Séance"]).dt.strftime('%Y-%m-%d')
      # data["Séance"]=pd.to_datetime(data["Séance"], format='%d/%m/%Y')
      
      data_sorted=data.sort_values(by="Séance")

      print("seances",data_sorted)
      # for x in data["Séance"]:
      #     séance.append(x)
      
      #print("these are seance",séance)

      séance_stat=dict(Counter(data_sorted))
      séance_keys=séance_stat.keys()

      # print("these are seance stat and keys",séance_stat,séance_keys)

      listkeys_Se=[]
      
      for x in séance_keys:
        listkeys_Se.append(x)

      #print("these are listkeys_Se",listkeys_Se)

      s1= data_sorted.groupby(["Joie","Séance"],as_index=False)['Commentaire'].count()
      s2= data_sorted.groupby(["Mécontentement","Séance"],as_index=False)['Commentaire'].count()
      s3= data_sorted.groupby(["Anxiété","Séance"],as_index=False)['Commentaire'].count()
      s4= data_sorted.groupby(["Colère","Séance"],as_index=False)['Commentaire'].count()
      s5= data_sorted.groupby(["Ennui","Séance"],as_index=False)['Commentaire'].count()
      s6= data_sorted.groupby(["Confusion","Séance"],as_index=False)['Commentaire'].count()
      s7= data_sorted.groupby(["Gratitude","Séance"],as_index=False)['Commentaire'].count()
      s8= data_sorted.groupby(["Satisfaction","Séance"],as_index=False)['Commentaire'].count()
      

      list_S1= s1[s1["Joie"]==1]
      list_S2= s2[s2["Mécontentement"]==1]
      list_S3= s3[s3["Anxiété"]==1]
      list_S4= s4[s4["Colère"]==1]
      list_S5= s5[s5["Ennui"]==1]
      list_S6= s6[s6["Confusion"]==1]
      list_S7= s7[s7["Gratitude"]==1]
      list_S8= s8[s8["Satisfaction"]==1]
      
      print("these are S1",s8)
      print("these are S1",list_S8)
      #print (sorted(list_S1,key=lambda x:datetime.datetime.strptime(x[2],"%d/%m/%Y")))
      listvalues_S1 = [] 
      listvalues_S1Y = [] 
      listvalues_S1X = [] 

      for x in list_S1["Séance"]:
        listvalues_S1X.append(x)
      for x in list_S1["Commentaire"]:
        listvalues_S1Y.append(x)

      listvalues_S2 = [] 
      listvalues_S2Y = [] 
      listvalues_S2X = [] 
      for x in list_S2["Séance"]:
        listvalues_S2X.append(x)
      for x in list_S2["Commentaire"]:
        listvalues_S2Y.append(x)

      listvalues_S3=[]
      listvalues_S3Y = [] 
      listvalues_S3X = [] 
      for x in list_S3["Séance"]:
        listvalues_S3X.append(x)
      for x in list_S3["Commentaire"]:
        listvalues_S3Y.append(x)

      listvalues_S4=[]
      listvalues_S4Y = [] 
      listvalues_S4X = [] 
      for x in list_S4["Séance"]:
        listvalues_S4X.append(x)
      for x in list_S4["Commentaire"]:
        listvalues_S4Y.append(x)
      

      listvalues_S5=[]
      listvalues_S5Y = [] 
      listvalues_S5X = [] 
      for x in list_S5["Séance"]:
        listvalues_S5X.append(x)
      for x in list_S5["Commentaire"]:
        listvalues_S5Y.append(x)
      

      listvalues_S6=[]
      listvalues_S6Y = [] 
      listvalues_S6X = [] 
      for x in list_S6["Séance"]:
        listvalues_S6X.append(x)
      for x in list_S6["Commentaire"]:
        listvalues_S6Y.append(x)
      

      listvalues_S7=[]
      listvalues_S7Y = [] 
      listvalues_S7X = [] 
      for x in list_S7["Séance"]:
        listvalues_S7X.append(x)
      for x in list_S7["Commentaire"]:
        listvalues_S7Y.append(x)
      

      listvalues_S8=[]
      listvalues_S8Y = [] 
      listvalues_S8X = [] 
      for x in list_S8["Séance"]:
        listvalues_S8X.append(x)
      for x in list_S8["Commentaire"]:
        listvalues_S8Y.append(x)
      
      data = []
      jsonList=[]

      for i in range(0,len(listvalues_S1X)):
        listvalues_S1.append({"x" : listvalues_S1X[i], "y" : listvalues_S1Y[i]})

      for i in range(0,len(listvalues_S2X)):
        listvalues_S2.append({"x" : listvalues_S2X[i], "y" : listvalues_S2Y[i]})
      
      for i in range(0,len(listvalues_S3X)):
        listvalues_S3.append({"x" : listvalues_S3X[i], "y" : listvalues_S3Y[i]})

      for i in range(0,len(listvalues_S4X)):
        listvalues_S4.append({"x" : listvalues_S4X[i], "y" : listvalues_S4Y[i]})

      for i in range(0,len(listvalues_S5X)):
        listvalues_S5.append({"x" : listvalues_S5X[i], "y" : listvalues_S5Y[i]})
      
      for i in range(0,len(listvalues_S6X)):
        listvalues_S6.append({"x" : listvalues_S6X[i], "y" : listvalues_S6Y[i]})
      
      for i in range(0,len(listvalues_S7X)):
        listvalues_S7.append({"x" : listvalues_S7X[i], "y" : listvalues_S7Y[i]})
      
      for i in range(0,len(listvalues_S8X)):
        listvalues_S8.append({"x" : listvalues_S8X[i], "y" : listvalues_S8Y[i]})
      
      
      # jsonList


      print("date S8",listvalues_S8)
        
      # listvalues_S2 = [] 
      # for x in list_S2["Commentaire"]:
      #   listvalues_S2.append(x)
      # listvalues_S3 = [] 
      # for x in list_S3["Commentaire"]:
      #   listvalues_S3.append(x)

      
      print ( "listevaluesS1",listvalues_S1X,listvalues_S1Y)
      


      context = {
              #bdlt here
              
              'empty' : empty,
              'p' : p,
              'n' : n,
              'ne' : ne,
              # 'jsonized':jsonized,
              "jsonList": jsonList,
              'attitude_ne':attitude_ne,
              'attitude_a':attitude_a,
              'attitude_h':attitude_h,
              'listkeys_V': listkeysV,
              'listvalues_V': listvalues_V,

              'listkeys_Ex': listkeysEx,
              'listvalues_Ex': listvalues_Ex,

              'listkeys_Se': listkeys_Se,

              'listvalues_S1':listvalues_S1,
              'listvalues_S2':listvalues_S2,
              'listvalues_S3':listvalues_S3,

              'listvalues_S4':listvalues_S4,
              'listvalues_S5':listvalues_S5,
              'listvalues_S6':listvalues_S6,

              'listvalues_S7':listvalues_S7,
              'listvalues_S8':listvalues_S8,
              

              'listkeys_T': listkeysAt,
              'listvalues_T': listvalues_T,

              'keys_Ag':listkeysAg,
      
              'listvaluespa_P':listvaluespa_P,
              'listvaluespa_N':listvaluespa_N,
              'listvaluespa_NE':listvaluespa_NE,
              'list_cours':modules,

      

      

      }

      
    
      return  render(request, 'dashboards/dashTeacher.html',context)
    else : 
        msg = "Pas encore de données pour les valeurs sélectionées"
        context = {
        'msg' : msg,
        'year' : year,
        'modules' : modules, 
        }
        return render(request, 'dashboards/dashTeacher.html', context)



def select_cours_dash(request):       
    global selected_cours
    if request.method == "POST":
       
        selected_cours = request.POST.get("cours")

    return redirect(Dashboard)

def select_cours_comment(request):       
    global selected_cours
    if request.method == "POST":
       
        selected_cours = request.POST.get("cours")

    return redirect(DashboardViewComments)

def select_module_etudiant(request):       
    global selected_module
    if request.method == "POST":
       
        selected_module = request.POST.get("module")

    return redirect(Input)


def DashboardViewComments(request):
    global selected_cours
    print("this cours that reached",selected_cours)
    #load csv file
    my_file = pd.read_csv("media/Neww.csv", sep=None, engine='python',encoding='utf-8', error_bad_lines=False)
    data = pd.DataFrame(data=my_file, index=None)
    
    cours=data.loc[:,"Module"]
    list_cours=cours.values.tolist()
    if(selected_cours!=None):
      data=data.query("Module=="+"'"+selected_cours+"'")
    
    # if selected_cours!="all":
    #   odd = filter(lambda p : p%2 != 0, nums)
   
    list_cours = list(dict.fromkeys(list_cours))
    out = data.values.tolist()
    
   # filtered=filter(lambda Module : Module=="Esi",data)
    print("is this list of all filtered",data.query("Module== 'reuissir-le-changement'"))
    totalCom=len(data.index)
    context = {
        'data':out,
        'totalCom':totalCom,
        'list_cours':list_cours,
        'selected_cours':selected_cours,
    
    }
  
    return  render(request, 'dashboards/dashTeacherCommentsTab.html',context)

# ______________________________________traitement commentaire_______________________________

def traitement_comment(comment): 
 c = comment_traité.objects.create(comment_id=comment.id) 
 c.Commentaire=comment.text
 c.Module=comment.Module
 c.Séance=comment.Séance
 modeles = ["FLV_12_mlknn_30_t", "FLV_12_mlknn_30_ta", "FLV_12_mlknn_30_ts", "FLV_12_svm_30_p", "FLV_12_svm_30_at", "FLV_12_svm_30_e"]
 for i in modeles : 
   modele = i
   print ('modele : ', modele)
   array = modele.split("_")
   s=[]
   for i in range(len(array)):
      s.append(array[i])
   print ('s : ', s)

   pathfile = os.path.join(settings.MODELS,modele)
   with open(pathfile,'rb') as p:
             loaded_model = pickle.load(p)
   classificatio=loaded_model['classificat']
   vectors=loaded_model['vectors']

   if comment : #cheking if comment have value
       vector = vectors.transform([comment.text]).todense()
       result = classificatio.predict((vector[0]))
       if (s[4]=="t"):
         result =result.toarray() 
         print ('result : ', result)
         c.Utilité = result[0,0]
         c.Intrinsèque = result[0,1]
         c.Accomplissement = result[0,2]
         c.Cout = result [0,3]
       elif (s[4]=="ta"):
         result =result.toarray() 
         print ('result : ', result)
         c.Présentation = result[0,0]
         c.Contenu = result[0,1]
         c.Design = result[0,2]
         c.Général = result [0,3]
         c.Communication = result[0,4]
         c.Structure = result [0,5]
       elif (s[4]=="ts"):
         result =result.toarray() 
         print ('result : ', result)
         c.Anxieté = result[0,0]
         c.Colère = result[0,1]
         c.Ennui = result[0,2]
         c.Joie = result[0,3]
         c.Mécontentement = result [0,4]
         c.Confusion = result[0,5]
         c.Satisfaction = result [0,6]
         c.Gratitude = result [0,7]
       elif (s[4]=="p"):
         print ('result : ', result)
         c.Polarité = result[0]
       elif (s[4]=="at"):
         print ('result : ', result)
         c.Attitude = result[0]
       elif (s[4]=="e"):
         print ('result : ', result)
         c.Attente = result[0]
 c.save()
 return (c)


#_________________________administration__________________________
# def administration(request):
#   return  render(request,'dashboards/dashAdministration.html')



def administration(request):
  global selected_niv
  global selected_annee 

  if request.user.is_authenticated: 
    if request.user.is_admin:
      mod_1cp=Module.objects.filter(niveau='1CP') 
      mod_2cp=Module.objects.filter(niveau='2CP')
      mod_1cs=Module.objects.filter(niveau='1CS')
      mod_2cs_sit=Module.objects.filter(niveau='2CS', specialite = 'SIT')
      mod_2cs_sil=Module.objects.filter(niveau='2CS', specialite = 'SIL')
      mod_2cs_siq=Module.objects.filter(niveau='3CS', specialite = 'SIQ')
      mod_3cs_sit=Module.objects.filter(niveau='3CS', specialite = 'SIT')
      mod_3cs_sil=Module.objects.filter(niveau='3CS', specialite = 'SIL')
      mod_3cs_siq=Module.objects.filter(niveau='3CS', specialite = 'SIQ')

      today = datetime.datetime.now()
      year = today.year

      
  
      if selected_annee : 
        year = int(selected_annee)

      if (year<= 2021):
        my_file = pd.read_csv("media/Neww.csv", sep=None, engine='python',encoding='utf-8', error_bad_lines=False)
        df = pd.DataFrame(data=my_file, index=None)
        # date_after = datetime.date(2020, 12,31)
        # date_before= datetime.date(2022, 1, )
        df['Séance'] = (pd.to_datetime(df['Séance'], format="%Y-%m-%d"))
        df=df[(df['Séance'] > '2020-12-31') & (df['Séance'] < '2022-01-01')]
        # df = df[df['Séance']==year]
        print ("this is", df)
        if selected_niv :
          df = df[df['Niveau']==selected_niv]
          print (df)
        

      else :
        modules_set = Module.objects.all()
        if selected_niv :
          modules_set = Module.objects.filter(niveau=selected_niv)
        test1=[]
        for t in modules_set : 
            test1.append(t.code) 
        print ('codes modules', test1)
        cmnts_all = Comment.objects.filter(Module__in=test1)
        print (cmnts_all)  
        cmnts=[]
        for c in cmnts_all :
          if (c.get_year()==year):
            cmnts.append(c.id)
        print (cmnts)
        cmnts_trait= comment_traité.objects.filter(comment_id__in=cmnts)
        print (cmnts_trait) 
        df = pd.DataFrame(list(comment_traité.objects.filter(comment_id__in=cmnts).values()))
        print (df)
      
      # ------------------------------------------Polarité générale--------------------------------------------
     #bdlt here
      if not df.empty : 
        empty = 0
     
        polarité = [] 
        for x in df["Polarité"]:
          polarité.append(x)

        polarité_stat = dict(Counter(polarité))
          
        
        if "P" in polarité_stat : 
          p = polarité_stat["P"]
        else : 
          p = 0 
        if "N" in polarité_stat : 
          n = polarité_stat["N"]
        else : 
          n = 0 
        if "NE" in polarité_stat : 
          ne = polarité_stat["NE"]
        else : 
          ne = 0 

        # ---------------------------------------Polarité par module----------------------------------------------
        pol_mod= df.groupby(["Module","Polarité"],as_index=False)['Commentaire'].count()

        mod= [] 
        for x in pol_mod["Module"]:
            mod.append(x)

        listkeys_pm=dict(Counter(mod)).keys()
        
        listkeys_pol_mod=[]
        listvalues_P=[]
        listvalues_N=[]
        listvalues_NE=[]
        for x in listkeys_pm:
            listkeys_pol_mod.append(x)
        print("listkeys_pol_mod",listkeys_pol_mod)
        print("pol_mod",pol_mod)
        exi=False
    
        for m in listkeys_pol_mod:
          row=pol_mod[pol_mod["Module"]==m]
          for r in row.values.tolist():
            if r[1]=="P":
                listvalues_P.append(r[2])
                exi=True
          if not(exi):
            listvalues_P.append(0)
          exi=False
          
        for m in listkeys_pol_mod:
          row=pol_mod[pol_mod["Module"]==m]
          for r in row.values.tolist():
            if r[1]=="N":
                listvalues_N.append(r[2])
                exi=True
          if not(exi):
            listvalues_N.append(0)
          exi=False
        
        for m in listkeys_pol_mod:
          row=pol_mod[pol_mod["Module"]==m]
          for r in row.values.tolist():
            if r[1]=="NE":
                listvalues_NE.append(r[2])
                exi=True
          if not(exi):
            listvalues_NE.append(0)
          exi=False

        
        
      
        

      





        # liste_P= pol_mod[pol_mod["Polarité"]=="P"]
        # liste_N= pol_mod[pol_mod["Polarité"]=="N"]
        # liste_NE= pol_mod[pol_mod["Polarité"]=="NE"]
        # print("liste_P",liste_P)

        # listvalues_P=liste_P["Commentaire"].tolist()
        # listvalues_N=liste_N["Commentaire"].tolist()
        # listvalues_NE=liste_NE["Commentaire"].tolist()
        print(" listvalues_P", listvalues_P)

        # -----------------------------------------------Nuage des mots-------------------------------------------
        exclure_mots = ['avec','car','cela','sans','ça' ,'d', 'du', 'de', 'la', 'des', 'le', 'et', 'est','elle', 'une', 'en', 'que', 'aux', 'qui', 'ces', 'les', 'dans', 'sur', 'l', 'un', 'pour', 'par', 'il', 'ou', 'à', 'ce', 'a', 'sont', 'cas', 'plus', 'leur', 'se', 's', 'vous','je','tu', 'au', 'c', 'aussi', 'toutes','tout','tous' , 'autre', 'comme', 'mais', 'pas', 'ou']

        comment_words = " "
        for i in df['Commentaire']: 
              i = str(i) 
              separate = i.split() 
              for j in range(len(separate)): 
                separate[j] = separate[j].lower() 
              comment_words += " ".join(separate)+" " 
            
                          # Creating the Word Cloud
        final_wordcloud = WordCloud(width = 800, height = 800, background_color ='white', stopwords = exclure_mots, min_font_size = 10).generate(comment_words)

                          # Displaying the WordCloud                    
        plt.figure(figsize = (10, 10), facecolor = None) 
        plt.imshow(final_wordcloud) 
        plt.axis("off") 
        plt.tight_layout(pad = 0) 
        
                          # Save the image in the img folder:
        final_wordcloud.to_file("static/images/wordcloudall.png")

        context = {
        #bdlt here
        'empty' : empty, 
          

        'year' : year, 
        'p' : p,
        'n' : n,
        'ne' : ne,
        'listvalues_P' : listvalues_P, 
        'listvalues_N' : listvalues_N, 
        'listvalues_NE' : listvalues_NE, 
        'listkeys_pol_mod' : listkeys_pol_mod, 
        'mod_1cp' : mod_1cp,
        'mod_2cp' : mod_2cp,
        'mod_1cs' : mod_1cs,
        'mod_2cs_sit' : mod_2cs_sit, 
        'mod_2cs_sil' : mod_2cs_sil, 
        'mod_2cs_siq' : mod_2cs_siq, 
        'mod_3cs_sit' : mod_3cs_sit, 
        'mod_3cs_sil' : mod_3cs_sil, 
        'mod_3cs_siq' : mod_3cs_siq, 
        }
        return render(request,'dashboards/dashAdministration.html', context)
      else : 
        msg = "Pas encore de données pour les valeurs sélectionées"
        context = {
        'msg' : msg,
        'year' : year, 
        'mod_1cp' : mod_1cp,
        'mod_2cp' : mod_2cp,
        'mod_1cs' : mod_1cs,
        'mod_2cs_sit' : mod_2cs_sit, 
        'mod_2cs_sil' : mod_2cs_sil, 
        'mod_2cs_siq' : mod_2cs_siq, 
        'mod_3cs_sit' : mod_3cs_sit, 
        'mod_3cs_sil' : mod_3cs_sil, 
        'mod_3cs_siq' : mod_3cs_siq, 
        }
        return render(request, 'dashboards/dashAdministration.html', context)

# def Input(request):
#   return render(request, 'dashboards/inputStudent.html')

def Input(request):
  global selected_module
  if request.user.is_authenticated: 
    if request.user.is_student: 
      student = request.user.student
      n = student.niveau
      s = student.specialite
      modules = Module.objects.filter(niveau=n,specialite=s) 
      comments = Comment.objects.filter(student_id=student.id)
      form = AjouterComment()
      if selected_module : 
        
        comments = Comment.objects.filter(student_id=student.id, Module=selected_module) 
        form = AjouterComment()      
        if request.method == 'POST':
            form = AjouterComment(request.POST)
            if form.is_valid():
                comment = form.save(commit=False)
                comment.Module= selected_module
                comment.student= student
                comment.save()
                comment_traité=traitement_comment(comment)

                print(comments)    
    return render(request, 'dashboards/inputStudent.html', {'form': form,'comments': comments,'modules':modules})
  
      # return render(request,'student.html',{'form': form, 'comments': comments, 'modules':modules})

    # 
    
    

