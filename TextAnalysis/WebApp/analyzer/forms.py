from django import forms

class TextAnalysisForm(forms.Form):
    text = forms.CharField(widget=forms.Textarea, required=True)
    analysis_type = forms.ChoiceField(choices=[
        ('emotions', 'Emotions')
    ])

class URLCheckForm(forms.Form):
    url = forms.URLField(required=True)

class PlagiarismForm(forms.Form):
    text1 = forms.CharField(widget=forms.Textarea, required=True)
    text2 = forms.CharField(widget=forms.Textarea, required=True)
