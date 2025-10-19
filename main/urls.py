from django.urls import path
from . import views

urlpatterns = [
    path('', views.analyzer_page, name='analyzer'),
    path('analyze/', views.analyze, name='analyze'),
]
