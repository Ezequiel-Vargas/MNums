# app/utils.py
from django.contrib.auth.hashers import make_password, check_password

def hashearContrasena(contrasena):
    return make_password(contrasena)

def verificarContrasena(contrasena, contrasenaHasheada):
    return check_password(contrasena, contrasenaHasheada)
