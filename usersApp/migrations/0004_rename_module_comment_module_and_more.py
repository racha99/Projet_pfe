# Generated by Django 4.0.4 on 2022-06-22 21:59

from django.db import migrations


class Migration(migrations.Migration):

    dependencies = [
        ('usersApp', '0003_rename_accompl_comment_traité_accomplissement_and_more'),
    ]

    operations = [
        migrations.RenameField(
            model_name='comment',
            old_name='module',
            new_name='Module',
        ),
        migrations.RenameField(
            model_name='comment',
            old_name='seance',
            new_name='Séance',
        ),
    ]
