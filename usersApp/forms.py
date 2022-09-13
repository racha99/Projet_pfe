from django import forms
from .models import User, Student, Teacher, Comment, Module
import datetime
from django.forms.widgets import SelectDateWidget
from django.utils import timezone



class LoginForm(forms.Form):
    username = forms.CharField(
        widget= forms.TextInput(
            attrs={
                "class": "form-control"
            }
        )
    )
    password = forms.CharField(
        widget=forms.PasswordInput(
            attrs={
                "class": "form-control"
            }
        )
    )

class AjouterComment(forms.ModelForm):
    class Meta:
        model = Comment
        
        fields = ('Séance', 'text')    
        labels = {
        "text": "Commentaire",
        "Séance": "Séance",
        }  

        widgets={
            'Séance': forms.TextInput(attrs={'class':'form-control'}),
            'text': forms.Textarea(attrs={'class':'form-control'})

        }




# class AjouterComment(forms.ModelForm):
#     class Meta:
#         model = Comment
        
#         fields = ('Séance', 'text')    
#         labels = {
#         "text": "Commentaire",
#         "Séance": "Séance",
#         }  




# class AjouterCommentTest(forms.ModelForm):
#     class Meta:
#         model = Comment
        
#         fields = ('seance','text')    
#         labels = {
#         "text": "Commentaire",
#         "seance": "Séance",
#         }  

#         widgets = {
#             'seance': forms.SelectDateWidget(),
#             'text': forms.Textarea()  

#         }

    