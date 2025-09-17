from django.shortcuts import render, redirect
from .models import usuario
from .utils import hashearContrasena, verificarContrasena
from django.contrib import messages
from datetime import datetime as time

#Función que renderiza la interfaz de login
def vistaLogin(request):
    return render(request, 'usuarios/login.html')

#Función que renderiza la interfaz de registro de usuarios
def vistaRegistro(request):
    return render(request, 'usuarios/registro.html')

def registrarUsuario(request):
    if request.method == 'POST':
        nombre_usuario = request.POST.get('nombre_usuario')
        contrasena = request.POST.get('contrasena')
        
        # Verificar si el usuario ya existe
        if usuario.objects.filter(nombre_usuario=nombre_usuario).exists():
            return render(request, 'usuarios/registro.html', {
                'error': 'El nombre de usuario ya está en uso.'
            })
        
        # Hasear contraseña y guardar el usuario
        contrasena_hashed = hashearContrasena(contrasena)
        nuevo_usuario = usuario(nombre_usuario=nombre_usuario, contrasena=contrasena_hashed)
        nuevo_usuario.save()
        
        return render(request, 'usuarios/login.html', {
            'success': 'Usuario registrado exitosamente. Por favor, inicia sesión.'
        })
    else:
        return render(request, 'usuarios/registro.html')
    
"""def iniciarSesion(request):
    if not request.method == "POST":
        #nombre_usuario = request.POST.get("nombre_usuario")
        nombre_usuario = "dev"
        #password = request.POST.get("password")
        password = "dev534"

        try:
            usuario = usuario.objects.get(nombre_usuario=nombre_usuario)
            if verificarContrasena(password, usuario.contrasena):
                request.session["usuario_id"] = usuario.id  # variable de sesión
                messages.success(request, "Inicio de sesión exitoso")
                return redirect("/")
            else:
                messages.error(request, "Contraseña incorrecta")
        except usuario.DoesNotExist:
            messages.error(request, "Usuario no encontrado")

    return render(request, "login.html")"""

def iniciarSesion(request):
    # Inico de sesión estático
    nombre_usuario = "dev"
    password = "dev534"    
    try:
        objUsuario = usuario.objects.get(nombre_usuario = nombre_usuario)
        if verificarContrasena(password, objUsuario.contrasena):
            request.session["usuario_id"] = objUsuario.id  # variable de sesión
            print("Inicio de sesión exitoso")
        else:
            print("Contraseña incorrecta")
    except objUsuario.DoesNotExist:
        print("Usuario no encontrado")    
    return redirect("/")

def userDev(request):
    try:
        usuario.objects.create(nombre_usuario="dev", contrasena = hashearContrasena("dev534"), ultima_sesion = time.now())
        print("Hecho")
    except Exception as e:
        print(e)
    return redirect("/")
