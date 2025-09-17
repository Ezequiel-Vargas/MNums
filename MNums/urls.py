from django.urls import path, include
from django.conf import settings 
from django.conf.urls.static import static

urlpatterns = [
    path('', include('historial.urls')),
    path('', include('usuarios.urls')),
    path('', include('calculadora.urls')),
]

if settings.DEBUG:
    urlpatterns += static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)