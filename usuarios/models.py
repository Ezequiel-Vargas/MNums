from django.db import models

# Modelo de usuarios
class usuario(models.Model):
    nombre_usuario = models.CharField(max_length=150, unique=True)
    contrasena = models.CharField(max_length=255)
    ultima_sesion = models.DateTimeField(auto_now=True)

    class meta:
        db_table = 'usuario'
        verbose_name = 'usuario'
        verbose_name_plural = 'usuarios'

    def __str__(self):
        return self.nombre_usuario
