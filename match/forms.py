
# match/forms.py

from django import forms

class UploadImageForm(forms.Form):
    image = forms.ImageField()
