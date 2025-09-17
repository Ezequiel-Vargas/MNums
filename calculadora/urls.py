from . import views
from django.urls import path

urlpatterns = [
    #URLS DE INTERFACES
    path('', views.vistaIndex, name='index'),
    path('acerca-de/', views.vistaAcercaDe, name='acerca-de'),
    
    #URLS DE FUNCIONALIDADES/CALCULOS
    path('derivacion/', views.derivacion, name='derivacion'),
    path('euler/', views.euler, name='euler'),
    path('euler/mejorado/', views.eulerMejorado, name='euler-mejorado'),
    path('interpolacion/linear/', views.interpolacionLinear, name='interpolacion-linear'),
    path('interpolacion/lagrange/', views.interpolacionMejorado, name='interpolacion-mejorado'),
    path('newton-raphson/', views.newtonRaphson, name='newton-raphson'),
    path('runge-kutta/', views.rungeKutta, name='runge-kutta'),
    path('aggCat/', views.insertarCategorias, name='aggCat'),
]
