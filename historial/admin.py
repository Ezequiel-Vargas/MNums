from django.contrib import admin
from .models import (
    calculo, categoria_metodo, metodo_numerico
)
from usuarios.models import usuario

@admin.register(usuario)
class UsuarioAdmin(admin.ModelAdmin):
    list_display = ['nombre_usuario', 'ultima_sesion']
    list_filter = ['ultima_sesion']
    search_fields = ['nombre_usuario']
    readonly_fields = ['ultima_sesion', 'id']

@admin.register(categoria_metodo)
class CategoriaMetodoAdmin(admin.ModelAdmin):
    list_display = ['nombre', 'descripcion']
    search_fields = ['nombre']
@admin.register(calculo)
class CalculoAdmin(admin.ModelAdmin):
    list_display = ['parametros_entrada', 'id_usuario', 'id_metodo', 'resultado', 'procedimiento', 'mensaje_error', 'fecha_calculo']
    list_filter = ['id', 'id_metodo__id_categoria', 'fecha_calculo']
    search_fields = ['usuario__nombre_usuario', 'fecha_calculo']
    readonly_fields = ['fecha_calculo', 'mensaje_error', 'id', 'parametros_entrada', 'procedimiento', 'resultado']
