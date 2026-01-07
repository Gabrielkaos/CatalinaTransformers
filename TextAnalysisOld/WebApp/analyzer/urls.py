from django.urls import path
from . import views

urlpatterns = [
    path('', views.index, name='index'),
    path('analyze/', views.analyze_text, name='analyze_text'),
    path('phishing/', views.check_phishing, name='check_phishing'),
    path('plagiarism/', views.check_plagiarism, name='check_plagiarism'),
    path('aihuman/', views.check_ai, name='check_ai'),
]
