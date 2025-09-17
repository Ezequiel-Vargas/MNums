from . import views
from django.urls import path

urlpatterns = [
    path('historial/', views.vistaHistorial, name='historial'),
    path('p/', views.imprimir, name='p'),
    
]