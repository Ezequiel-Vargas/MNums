from django.db import models
from usuarios.models import usuario
import json

# Modelo de categorias para distinguir los métodos numéricos
class categoria_metodo(models.Model):
    nombre = models.CharField(max_length=50, unique=True)
    descripcion = models.TextField(blank=True)
    
    class meta:
        db_table = 'categoria_metodo'
        verbose_name = 'categoría de método'
        verbose_name_plural = 'categorias de métodos'
        ordering = ['nombre']

# Modelo de los métodos disponibles
class metodo_numerico(models.Model):
    nombre = models.CharField(max_length=50)
    id_categoria = models.ForeignKey(
        categoria_metodo,
        on_delete=models.CASCADE,
        related_name='metodos'
    )
    documentacion = models.TextField(blank=True)
    
    class meta:
        db_table = 'metodo_numerico'
        verbose_name = 'método numérico'
        verbose_name_plural = 'métodos numéricos'
        ordering = ['categoria', 'nombre']
        unique_together = ['nombre', 'categoria']

# Modelo para almacenar los cálculos realizados
class calculo(models.Model):
    id_usuario = models.ForeignKey(
        usuario,
        on_delete=models.CASCADE,
        related_name='calculos'
    )
    id_metodo = models.ForeignKey(
        metodo_numerico,
        on_delete=models.CASCADE,
        related_name='calculos'
    )
    parametros_entrada = models.JSONField()
    resultado = models.JSONField(null=True, blank=True)
    procedimiento = models.JSONField(null=True, blank=True)
    mensaje_error = models.TextField(blank=True)
    fecha_calculo = models.DateTimeField(auto_now_add=True)
    
    class meta:
        db_table = 'calculo'
        verbose_name = 'cálculo'
        verbose_name_plural = 'cálculos'
        ordering = ['-fecha_calculo']
    
    # Formatea los parámetros de entrada y los retorna
    def formatearInputs(self):
        if isinstance(self.parametros_entrada, str):
            try:
                return json.loads(self.parametros_entrada)
            except json.JSONDecodeError:
                return {}
        return self.parametros_entrada or {}
    
    # Formatea el resultado y lo retorna
    def formatearResultado(self):
        if isinstance(self.resultado, str):
            try:
                return json.loads(self.resultado)
            except json.JSONDecodeError:
                return {}
        return self.resultado or {}
