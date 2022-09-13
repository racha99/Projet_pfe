from django.urls import path, re_path

from . import views
from django.urls import re_path
from django.urls import include


urlpatterns = [
 # path('', views.Home, name='Home'),
 # path('', views.login, name='login'),
   path('dashboard', views.Dashboard, name='dashboardTeacher'),
   path('test-json', views.data_display, name='test-json'),
   path('test-template', views.TestTemplate, name='test-template'),
   path('input', views.Input,name='inputStudent'),
   path('administration', views.administration, name='administration'),
   #bdlt here
   path('administration/<str:k>/dashboard', views.Dashboard, name='dashboard2'), 
   path('dashboard/comments/',views.DashboardViewComments, name='dashboardComments'),
   path('dashboard/comments',views.DashboardViewComments, name='dashboardComments'),
   #re_path(r'^dashboard/comments/(?P<cours>)',views.DashboardViewComments, name='dashboardComments'),  # good
   path('dashboard/commentsCours',views.select_cours_comment, name ='select_coursComments'),
   path('dashboard/DashCours',views.select_cours_dash, name ='select_coursDash'),
   path('input/commentsEtudiant',views.select_module_etudiant, name ='select_module_etudiant'),
   path('select_annee', views.select_annee, name='select_annee'),
   path('select_niv', views.select_niv, name='select_niv'),
   # path('',views.select_cours, name ='select_cours')
   re_path(r'$', views.login_success, name='login_success')
   
    
    ]