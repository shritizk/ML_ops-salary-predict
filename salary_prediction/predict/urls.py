from django.contrib import admin
from django.urls import path ,include
from views import *
urlpatterns = [
    path('lr/', predict_salar_with_linear_regression),
    
]
