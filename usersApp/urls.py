from django.urls import path, re_path

from . import views
from django.urls import re_path
from django.urls import include


urlpatterns = [
 
   path('dashboard', views.Dashboard, name='dashboardTeacher'),
   path('input', views.Input,name='inputStudent'),
   path('administration', views.administration, name='administration'),
   path('administration/<str:k>/dashboard', views.Dashboard, name='dashboard2'), 
   path('dashboard/comments/',views.DashboardViewComments, name='dashboardComments'),
   path('dashboard/comments',views.DashboardViewComments, name='dashboardComments'),
  
   path('dashboard/commentsCours',views.select_cours_comment, name ='select_coursComments'),
   path('dashboard/DashCours',views.select_cours_dash, name ='select_coursDash'),
   path('input/commentsEtudiant',views.select_module_etudiant, name ='select_module_etudiant'),
   path('select_annee', views.select_annee, name='select_annee'),
   path('select_niv', views.select_niv, name='select_niv'),
  
   re_path(r'$', views.login_success, name='login_success')
   
    
    ]