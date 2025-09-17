from django.urls import path
from . import views

urlpatterns = [
    path('vistaLogin/', views.vistaLogin, name='vistaLogin'),
    path('login/', views.iniciarSesion, name='login'),
    path('registro/', views.vistaRegistro, name='registro'),
    path('dev/', views.userDev),
]