from django.urls import path, re_path

from . import views
from django.urls import re_path

urlpatterns = [
   # path('', views.Home, name='Home'),
   path('', views.login, name='login'),
   path('dashboard', views.Dashboard, name='dashboardTeacher'),
   path('input', views.Input),
   path('dashboard/comments/',views.DashboardViewComments, name='dashboardComments'),
   path('dashboard/comments',views.DashboardViewComments, name='dashboardComments'),
   #re_path(r'^dashboard/comments/(?P<cours>)',views.DashboardViewComments, name='dashboardComments'),  # good
   path('dashboard/commentsCours',views.select_cours_comment, name ='select_coursComments'),
   path('dashboard/DashCours',views.select_cours_dash, name ='select_coursDash')
   # path('',views.select_cours, name ='select_cours')
   
    
    ]