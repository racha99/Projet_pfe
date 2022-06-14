from django.shortcuts import render
import pandas as pd
from collections import Counter
from django.shortcuts import redirect

from wordcloud import WordCloud
import matplotlib.pyplot as plt
# Create your views here.


def login(request):
    global selected_cours
    selected_cours="all"
    return  render(request, 'login.html')


def Dashboard(request):
    global selected_cours
    
    #load the data
    my_file = pd.read_csv("media/Neww.csv", sep=None, engine='python',encoding='utf-8', error_bad_lines=False)
    data = pd.DataFrame(data=my_file, index=None)
    cours=data.loc[:,"Module"]
    list_cours=cours.values.tolist()
    list_cours = list(dict.fromkeys(list_cours))
    if(selected_cours!="all"):
      data=data.query("Module=="+"'"+selected_cours+"'")
   
    #wordcloud
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
  
    p = polarité_stat["P"]
    n = polarité_stat["N"]
    ne = polarité_stat["NE"]


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
    attitude = [] 
    for x in data["Attitude"]:
      attitude.append(x)
    
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


    # -----------------------------------------------Evolutions des émotions-------------------------------------------
    séance=[]
    for x in data["Séance"]:
        séance.append(x)

    séance_stat=dict(Counter(séance))
    séance_keys=séance_stat.keys()

    listkeys_Se=[]
    
    for x in séance_keys:
      listkeys_Se.append(x)


    s1= data.groupby(["Joie","Séance"],as_index=False)['Commentaire'].count()
    s2= data.groupby(["Mécontentement","Séance"],as_index=False)['Commentaire'].count()
    s3= data.groupby(["Anxiété","Séance"],as_index=False)['Commentaire'].count()

    list_S1= s1[s1["Joie"]==1]
    list_S2= s2[s2["Mécontentement"]==1]
    list_S3= s3[s3["Anxiété"]==1]

    listvalues_S1 = [] 
    for x in list_S1["Commentaire"]:
      listvalues_S1.append(x)
    listvalues_S2 = [] 
    for x in list_S2["Commentaire"]:
      listvalues_S2.append(x)
    listvalues_S3 = [] 
    for x in list_S3["Commentaire"]:
      listvalues_S3.append(x)
    


    context = {
            'p' : p,
            'n' : n,
            'ne' : ne,
            'listkeys_V': listkeysV,
            'listvalues_V': listvalues_V,

            'listkeys_Ex': listkeysEx,
            'listvalues_Ex': listvalues_Ex,

            'listkeys_Se': listkeys_Se,

            'listvalues_S1':listvalues_S1,
            'listvalues_S2':listvalues_S2,
            'listvalues_S3':listvalues_S3,

            'listkeys_T': listkeysAt,
            'listvalues_T': listvalues_T,

            'keys_Ag':listkeysAg,
    
            'listvaluespa_P':listvaluespa_P,
            'listvaluespa_N':listvaluespa_N,
            'listvaluespa_NE':listvaluespa_NE,
            'list_cours':list_cours

    

    

    }

    print("attitude",listvalues_T )

    return  render(request, 'dashboards/dashTeacher.html',context)



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




def DashboardViewComments(request):
    global selected_cours
    print("this cours that reached",selected_cours)
    #load csv file
    my_file = pd.read_csv("media/Neww.csv", sep=None, engine='python',encoding='utf-8', error_bad_lines=False)
    data = pd.DataFrame(data=my_file, index=None)
    
    cours=data.loc[:,"Module"]
    list_cours=cours.values.tolist()
    if(selected_cours!="all"):
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

def Input(request):
    return  render(request, 'dashboards/inputStudent.html')

