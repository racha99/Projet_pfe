from django.db import models
from django.contrib.auth.models import AbstractUser
from datetime import datetime
from django.db.models.signals import post_save
from django.dispatch import receiver
from django.urls import reverse


SPECIALITE=(
    ('SIT','SIT'),
    ('SIL','SIL'),
    ('SIQ','SIQ'),
    )

NIVEAU=(
    ('1CP','1CP'),
    ('2CP','2CP'),
    ('1CS','1CS'),
    ('2CS','2CS'),
    ('3CS','3CS'),
    )

VALEUR=(
    ('P','P'),
    ('N','N'),
    ('NE','NE'),
    )

VALEUR_ATITUDE=(
    ('A','A'),
    ('H','H'),
    ('NE','NE'),
    )

BOOL=(
    (1,'1'),
    (0,'0'),
    )


class User(AbstractUser):
    is_teacher= models.BooleanField(default=False)
    is_student = models.BooleanField(default=False)
    is_admin = models.BooleanField(default=False)
    is_app_admin = models.BooleanField(default=False)

class Teacher(models.Model):
    user = models.OneToOneField(User, on_delete=models.CASCADE, related_name='teacher')    

class Student(models.Model):
    user = models.OneToOneField(User, on_delete=models.CASCADE, related_name='student')
    specialite=models.CharField(max_length=3, choices=SPECIALITE, blank=True)
    niveau=models.CharField(max_length=3, choices=NIVEAU)

def create_user_profile(sender, instance, created, **kwargs):
	if instance.is_student:
	   Student.objects.get_or_create(user = instance)
	else:
	   Teacher.objects.get_or_create(user = instance)
	


class Comment(models.Model):
    student = models.ForeignKey(Student, on_delete=models.CASCADE)
    text = models.TextField(max_length=500)
    Séance = models.DateField(default=datetime.now)
    date = models.DateTimeField(auto_now_add=True)
    Module = models.TextField(max_length=10)

    def get_year(self): 
        return self.Séance.year

class comment_traité(models.Model):
    comment = models.ForeignKey(Comment, on_delete=models.CASCADE)
    Commentaire = models.TextField(max_length=500, default="")
    Module = models.TextField(max_length=10, default="")
    Séance = models.DateField(default=datetime.now)
    Polarité = models.CharField(max_length=2, choices=VALEUR, blank=True)
    Attitude = models.CharField(max_length=2, choices=VALEUR_ATITUDE, blank=True)
    Attente = models.CharField(max_length=2, choices=VALEUR, blank=True)
    Utilité = models.IntegerField(choices=BOOL, blank=True, default = 0)
    Intrinsèque = models.IntegerField(choices=BOOL, blank=True, default = 0)
    Accomplissement = models.IntegerField(choices=BOOL, blank=True, default = 0)
    Cout = models.IntegerField(choices=BOOL, blank=True, default = 0)
    Présentation = models.IntegerField(choices=BOOL, blank=True, default = 0)
    Contenu = models.IntegerField(choices=BOOL, blank=True, default = 0)
    Design = models.IntegerField(choices=BOOL, blank=True, default = 0)
    Général = models.IntegerField(choices=BOOL, blank=True, default = 0)
    Communication = models.IntegerField(choices=BOOL, blank=True, default = 0)
    Structure = models.IntegerField(choices=BOOL, blank=True, default = 0)
    Anxiété = models.IntegerField(choices=BOOL, blank=True, default = 0)
    Colère = models.IntegerField(choices=BOOL, blank=True, default = 0)
    Ennui = models.IntegerField(choices=BOOL, blank=True, default = 0)
    Joie = models.IntegerField(choices=BOOL, blank=True, default = 0)
    Mécontentement = models.IntegerField(choices=BOOL, blank=True, default = 0)
    Confusion = models.IntegerField(choices=BOOL, blank=True, default = 0)
    Satisfaction = models.IntegerField(choices=BOOL, blank=True, default = 0)
    Gratitude = models.IntegerField(choices=BOOL, blank=True, default = 0)

    def get_year(self): 
        return self.Séance.year

class Module(models.Model):
    teachers = models.ManyToManyField(Teacher)
    titre = models.TextField(max_length=100)
    code = models.TextField(max_length=10)
    specialite=models.CharField(max_length=3, choices=SPECIALITE, blank=True)
    niveau=models.CharField(max_length=3, choices=NIVEAU)

    #bdlt here
    def get_absolute_url(self):
        """Returns the url to access a particular author instance."""
        return reverse('dashboard2', args=[str(self.code)])
 

