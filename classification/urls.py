from django.urls import path, re_path

from . import views

urlpatterns = [
   # path('', views.Home, name='Home'),
    path('classification', views.Classification, name='Classification'),
   
    path('construction', views.Construction, name='Construction'),
    
    path('ajax-posting/', views.ajax_posting, name='ajax_posting'),# ajax-posting / name = that we will use in ajax url
    path('modele_creation/', views.modele_creation, name='modele_creation'),
    path('corpus_pretraitement/', views.corpus_pretraitement, name='corpus_pretraitement'),
    path('change_modele/', views.change_modele, name='change_modele'),

    path('', views.index, name='index'),
    path('results', views.results, name='results'),
    path('association', views.association, name='association'),
    
    
    ]