from django.db import models

# Create your models here.
from django import forms

class UploadImageForm(forms.Form):
    image = forms.ImageField()

