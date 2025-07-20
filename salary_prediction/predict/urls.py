from django.contrib import admin
from django.urls import path ,include
from . import views
urlpatterns = [
    path('',views.home),
    path('form',views.form),
    path('lr', views.predict_salar_with_linear_regression),
    
]
