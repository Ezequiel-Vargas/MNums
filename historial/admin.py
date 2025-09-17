from django.contrib import admin
from .models import (
    calculo, categoria_metodo, metodo_numerico, parametro_metodo
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

class ParametroMetodoInline(admin.TabularInline):
    model = parametro_metodo
    extra = 1

@admin.register(metodo_numerico)
class MetodoNumericoAdmin(admin.ModelAdmin):
    list_display = ['nombre', 'id_categoria', 'documentacion']
    list_filter = ['id_categoria']
    search_fields = ['nombre', 'id']
    readonly_fields = ['documentacion', 'id']
    inlines = [ParametroMetodoInline]

@admin.register(calculo)
class CalculoAdmin(admin.ModelAdmin):
    list_display = ['parametros_entrada', 'id_usuario', 'id_metodo', 'resultado', 'procedimiento', 'mensaje_error', 'fecha_calculo']
    list_filter = ['id', 'id_metodo__id_categoria', 'fecha_calculo']
    search_fields = ['usuario__nombre_usuario', 'fecha_calculo']
    readonly_fields = ['fecha_calculo', 'mensaje_error', 'id', 'parametros_entrada', 'procedimiento', 'resultado']
