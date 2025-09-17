import os
from django.core import serializers
from django.shortcuts import render, redirect
from json import JSONDecodeError
from usuarios.models import usuario
from django.contrib import messages
from .models import calculo, metodo_numerico, categoria_metodo
import json
import ast
from django.forms.models import model_to_dict

# Función auxiliar para decodificar JSON y limpiar diccionarios
def cargarJSON(datos):
    if not datos:
        return {}
    if isinstance(datos, dict):
        return datos
    try:
        # Si el formato es tipo dict con comillas simples
        return ast.literal_eval(datos)
    except (ValueError, SyntaxError):
        return {}
    
# Función que renderiza la interfaz y carga el historial de cálculos del usuario
def vistaHistorial(request):
    # verificar la variable de sesión
    usuario_id = request.session.get("usuario_id")
    if not usuario_id:
        messages.error(request, "Es necesario iniciar sesión para ver el historial")
        print("Es necesario iniciar sesión para ver el historial")
        return redirect("login")
    
    #instanciar el usuario
    try:
        objUsuario = usuario.objects.get(pk = usuario_id)
    except usuario.DoesNotExist:
        messages.error(request, "Usuario no encontrado")
        print("Usuario no encontrado")
        return redirect("login")

    # instanciar el historial de usuario
    objCalculos = calculo.objects.filter(id_usuario = objUsuario)
    print(objCalculos.__len__)

    # Verificar que exista historial
    if objCalculos.exists():
        # Queryset del historial de cálculos del usuario
        
        print("Hay historial")
        """# Se dividen los qurysets según los métodos"""
        # donde solo se necesita la ecuación
        objCalculosEc = objCalculos.exclude(id_metodo__in=[3, 4]) 
        datosEc = []
        # Donde se necesitan todos los parámetros
        objCalculosP = objCalculos.filter(id_metodo__in=[3, 4])
        datosP = []
        """# Formatear y almacenar los datos de los cálculos """
        # Métodos con ecuación
        for item in objCalculosEc:
            parametros = cargarJSON(item.parametros_entrada)
            resultado = cargarJSON(item.resultado)
            """En el método de derivación, no se utiliza una ecuación sino una función, aunque para fines prácticos, se le llamará ecuación (en el backend) en este caso para rescatar el dato y mostrarlo en el template"""
            if parametros.get("ecuacion"):
                ecuacion = parametros.get("ecuacion")
            else: 
                ecuacion = parametros.get("funcion") 

            datosEc.append({
                "id": item.id,
                "metodo": item.id_metodo.nombre,
                "ecuacion": ecuacion,
                "resultado": resultado,
                "mensaje_error": item.mensaje_error
            })

        # Métodos con todos los parámetros
        for item in objCalculosP:
            parametros = cargarJSON(item.parametros_entrada)
            resultado = cargarJSON(item.resultado)

            datosP.append({
                "id": item.id,
                "metodo": item.id_metodo.nombre,
                "parametros": parametros,
                "resultado": resultado,
                "mensaje_error": item.mensaje_error
            })
        template_vars = {
            "calculo_ec": datosEc,
            "calculo_p": datosP
        }
        
        return render(request, 'historial/historialC.html', template_vars)      
    
    else:
        messages.info(request, "No hay historial para mostrar")
        print("No hay historial")

def imprimir(request):
    """c = calculo.objects.filter(id__range=(64, 67))
    for obj in c:
        for field in obj._meta.fields:
            nombre = field.name
            valor = getattr(obj, nombre)
            print(f"{nombre}: {valor}")"""
    
    c = calculo.objects.get(id = 181)
    print("historial:", model_to_dict(c))
    print("usuario:", model_to_dict(c.id_usuario))
    print("metodo_numerico:", model_to_dict(c.id_metodo))
    print("categoria_metodo:", model_to_dict(c.id_metodo.id_categoria))

    return redirect("/")

def create_fixtures(request):
    # Obtener algunos registros a formatear 
    datos = usuario.objects.all()
    
    # Serializar a JSON
    serialized_data = serializers.serialize('json', datos, indent=2)
    
    # Crear directorio fixtures si no existe
    os.makedirs('fixtures', exist_ok=True)
    
    # Guardar con encoding UTF-8
    with open('fixtures/sample_data.json', 'w', encoding='utf-8') as f:
        f.write(serialized_data)
    
    print("✅ Fixtures creados exitosamente!")

    return redirect("/")