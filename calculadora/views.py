import base64
from io import BytesIO
from django.contrib import messages
from django.shortcuts import redirect, render
import matplotlib
import numpy as np
import sympy as sp
import matplotlib.pyplot as plt
matplotlib.use('Agg')
import math
from historial.models import metodo_numerico, calculo, categoria_metodo
from usuarios.models import usuario
from decimal import Decimal, InvalidOperation
import re
from collections import Counter

# Función que renderiza la interfaz Index
def vistaIndex(request):
    return render (request, 'index.html')  

# Función que renderiza la interfaz informativa Acerca de
def vistaAcercaDe(request):
    return render (request, 'acercaDe.html')

# Función que preprocesa la función de entrada para que sea compatible con SymPy, maneja funciones trigonométricas, exponenciales, logarítmicas, etc.
def preprocesarFuncion(funcion_str):
    print(funcion_str)
    # Eliminar espacios
    funcion = funcion_str.replace(' ', '')
    print(funcion)
    # Diccionario de reemplazos para hacer la función compatible con SymPy
    reemplazos = {
        # Funciones trigonométricas
        'seno': 'sin',
        'sen': 'sin',
        'coseno': 'cos',
        'cos': 'cos',
        'tangente': 'tan',
        'tan': 'tan',
        'tg': 'tan',
        'cotangente': 'cot',
        'cotg': 'cot',
        'cot': 'cot',
        'secante': 'sec',
        'sec': 'sec',
        'cosecante': 'csc',
        'cosec': 'csc',
        'csc': 'csc',
        
        # Funciones trigonométricas inversas
        'arcsin': 'asin',
        'arccos': 'acos',
        'arctan': 'atan',
        'arccot': 'acot',
        'arcsec': 'asec',
        'arccsc': 'acsc',
        
        # Funciones hiperbólicas
        'senh': 'sinh',
        'cosh': 'cosh',
        'tanh': 'tanh',
        'coth': 'coth',
        'sech': 'sech',
        'csch': 'csch',
        
        # Exponenciales y logaritmos
        'exp': 'exp',
        'ln': 'log',
        'log10': 'log(x, 10)',
        'log': 'log',
        'lg': 'log(x, 10)',
        
        # Constantes matemáticas
        'pi': 'pi',
        'π': 'pi',
        'e': 'E',
        
        # Raíz cuadrada
        'sqrt': 'sqrt',
        'raiz': 'sqrt',
        '√': 'sqrt',
    }
    
    # Aplicar reemplazos básicos
    for original, reemplazo in reemplazos.items():
        funcion = funcion.replace(original, reemplazo)

    print(f"Después de reemplazos básicos: '{funcion}'")
    
    # Manejar casos especiales de logaritmos
    funcion = re.sub(r'log10\(([^)]+)\)', r'log(\1, 10)', funcion)# log10(x) pasa a ser -> log(x, 10)
    funcion = re.sub(r'lg\(([^)]+)\)', r'log(\1, 10)', funcion)
    
    # Manejar potencias con ^
    funcion = funcion.replace('^', '**')

    print(f"Después de potencias: '{funcion}'")

    # Lista de funciones matemáticas conocidas para evitar transformaciones incorrectas
    funciones_matematicas = [
        'sinh', 'cosh', 'tanh', 'coth', 'sech', 'csch',
        'asin', 'acos', 'atan', 'acot', 'asec', 'acsc',
        'sqrt', 'floor', 'ceil',
        'sin', 'cos', 'tan', 'cot', 'sec', 'csc',
        'exp', 'log', 'abs' 
    ]

   # Método alternativo: procesar multiplicación implícita paso a paso
    
    # 1. Primero, manejar números seguidos de funciones específicas: 2sin -> 2*sin
    for func in funciones_matematicas:
        # Patrón: número seguido directamente de función
        funcion = re.sub(rf'(\d)({func})\b', r'\1*\2', funcion)
    
    print(f"Después de multiplicación número*función: '{funcion}'")
    
    # 2. Número seguido de variable (pero no de función): 2x -> 2*x
    # Crear patrón que excluye funciones conocidas
    patron_no_funciones = '(?!' + '|'.join(funciones_matematicas) + r'\b)'
    funcion = re.sub(rf'(\d)({patron_no_funciones}[a-zA-Z])', r'\1*\2', funcion)
    
    print(f"Después de multiplicación número*variable: '{funcion}'")
    
    # 3. Variable seguida de número: x2 -> x*2
    funcion = re.sub(r'([a-zA-Z])(\d)', r'\1*\2', funcion)
    
    # 4. Variable seguida de paréntesis (excluyendo funciones): x( -> x*(
    # Verificar que lo que precede al paréntesis no sea una función conocida
    temp_funcion = funcion
    for func in funciones_matematicas:
        # Reemplazar temporalmente las funciones para evitar conflictos
        temp_funcion = temp_funcion.replace(f'{func}(', f'__FUNC__{func}__(')
    
    # Aplicar la regla de multiplicación implícita para variables
    temp_funcion = re.sub(r'([a-zA-Z])\(', r'\1*(', temp_funcion)
    
    # Restaurar las funciones
    for func in funciones_matematicas:
        temp_funcion = temp_funcion.replace(f'__FUNC__{func}__(', f'{func}(')
    
    funcion = temp_funcion
    print(f"Después de variable*paréntesis: '{funcion}'")
    
    # 5. Paréntesis seguido de variable: )x -> )*x
    funcion = re.sub(r'\)([a-zA-Z])', r')*\1', funcion)
    
    # 6. Paréntesis seguido de número: )2 -> )*2
    funcion = re.sub(r'\)(\d)', r')*\1', funcion)
    
    # 7. Número seguido de paréntesis: 2( -> 2*(
    funcion = re.sub(r'(\d)\(', r'\1*(', funcion)
    
    # 8. Paréntesis seguido de función: )sin -> )*sin
    for func in funciones_matematicas:
        funcion = re.sub(rf'\)({func})\b', r')*\1', funcion)
    
    # Manejar constantes comunes
    funcion = re.sub(r'(\d+)pi', r'\1*pi', funcion)         # 2pi -> 2*pi
    funcion = re.sub(r'pi(\w)', r'pi*\1', funcion)          # pix -> pi*x
    funcion = re.sub(r'(\d+)E\b', r'\1*E', funcion)         # 2E -> 2*E
    funcion = re.sub(r'E(\w)', r'E*\1', funcion)            # Ex -> E*x
    
    print(f"Función procesada: '{funcion}'")
    return funcion

#Funcion que evalúa una función f(x) con SymPy en un punto específico
def evaluarFuncion(f, x_symbol, valor):
    try:
        # Evaluación utilizando el método evalf
        resultado = f.subs(x_symbol, valor).evalf()
        
        # Verificar si el resultado es complejo
        if resultado.is_real is False or (hasattr(resultado, 'im') and abs(resultado.im) > 1e-10):
            raise ValueError(f"La función produce valores complejos en x = {valor}")
        
        # Convertir a float usando N() para mayor precisión
        valor_float = float(resultado.n())
        
        # Verificar que el resultado es finito
        if not np.isfinite(valor_float):
            raise ValueError(f"La función no está definida o es infinita en x = {valor}")
            
        return valor_float
        
    except (TypeError, ValueError, AttributeError) as e:
        # Método alternativo: usar método lambdify
        try:
            func_numerica = sp.lambdify(x_symbol, f, modules=['numpy', 'math'])
            resultado = func_numerica(valor)
            
            if not np.isfinite(resultado):
                raise ValueError(f"La función no está definida o es infinita en x = {valor}")
                
            return float(resultado)
            
        except Exception as e2:
            raise ValueError(f"Error al evaluar la función en x = {valor}: {str(e)}. {str(e2)}")

# Funcion que evalúa una función multivariable con SymPy 
def evaluarFuncionMultiple(f, variables_valores):
    try:
        # Convertir a diccionario si es necesario
        if isinstance(variables_valores, list):
            variables_valores = dict(variables_valores)
        
        # Sustituir valores
        resultado = f.subs(variables_valores).evalf()
        
        # Verificar si el resultado es complejo
        if resultado.is_real is False or (hasattr(resultado, 'im') and abs(resultado.im) > 1e-10):
            vars_str = ", ".join([f"{k} = {v}" for k, v in variables_valores.items()])
            raise ValueError(f"La función produce valores complejos en {vars_str}")
        
        # Convertir a float
        valor_float = float(resultado.n())
        
        # Verificar que el resultado es finito
        if not np.isfinite(valor_float):
            vars_str = ", ".join([f"{k} = {v}" for k, v in variables_valores.items()])
            raise ValueError(f"La función no está definida o es infinita en {vars_str}")
            
        return valor_float
        
    except (TypeError, ValueError, AttributeError) as e:
        # Método alternativo: usar lambdify
        try:
            simbolos = list(variables_valores.keys())
            valores = list(variables_valores.values())
            
            func_numerica = sp.lambdify(simbolos, f, modules=['numpy', 'math'])
            resultado = func_numerica(*valores)
            
            if not np.isfinite(resultado):
                vars_str = ", ".join([f"{k} = {v}" for k, v in variables_valores.items()])
                raise ValueError(f"La función no está definida o es infinita en {vars_str}")
                
            return float(resultado)
            
        except Exception as e2:
            vars_str = ", ".join([f"{k} = {v}" for k, v in variables_valores.items()])
            raise ValueError(f"Error al evaluar la función en {vars_str}: {str(e)}. {str(e2)}")

# Función que evalúa una función SymPy de manera inteligente, decidiendo automáticamente si usar evaluarFuncion (para una variable) o evaluarFuncionMultiple (para múltiples variables)
def evaluarFuncionInteligente(f, **kwargs):
    try:
        # Obtener las variables libres en la función
        variables_en_funcion = f.free_symbols
        
        # Crear símbolos para las variables proporcionadas
        simbolos_kwargs = {sp.symbols(nombre): valor for nombre, valor in kwargs.items()}
        
        # Filtrar solo las variables que realmente están en la función
        variables_necesarias = {
            sym: val for sym, val in simbolos_kwargs.items()
            if sym in variables_en_funcion
        }
        
        # Determinar qué función usar basándose en el número de variables
        num_variables = len(variables_necesarias)
        
        if num_variables == 0:# Función constante
            resultado = float(f.evalf())
            return resultado
        elif num_variables == 1:# Una sola variable, se usa evaluación de f(x)
            simbolo = list(variables_necesarias.keys())[0]
            valor = list(variables_necesarias.values())[0]
            return evaluarFuncion(f, simbolo, valor)
        else:# Múltiples variables se usa evaluación multivariable
            return evaluarFuncionMultiple(f, variables_necesarias)
            
    except Exception as e:
        vars_str = ", ".join([f"{k} = {v}" for k, v in kwargs.items()])
        raise ValueError(f"Error al evaluar la función con {vars_str}: {str(e)}")

# Funcion que Limpia y valida formato de entrada
def limpiarValores(valorInput, nombreCampo):
    # Remover espacios extra
    valorLimpio = re.sub(r'\s+', '', valorInput)
    
    # Remover comas finales si existen
    valorLimpio = valorLimpio.rstrip(',')
    
    # Verificar que no esté vacío después de limpiar
    if not valorLimpio:
        raise ValueError(f"{nombreCampo} no puede estar vacío")

    return valorLimpio

# Función para detercar las variables que necesita una ecuación
def obtenerVariablesRequeridas(ecuacion):
    try:
        # Se preprocesa la ecuación
        ecuacion_procesada = preprocesarFuncion(ecuacion)

        # Convertir a expresión de SymPy
        f = sp.sympify(ecuacion_procesada)

        # Obtener las variables de la ecuación
        variables = f.free_symbols

        # Retornar las variables encontradas en la ecuación
        return {str(var) for var in variables}
    except Exception as e:
        raise ValueError(f"Error al analizar la ecuación '{ecuacion}': {str(e)}")

# Función para evaluar expreciones discerniendo entre el tipo de ecuación
def evaluarDerivada(x_val, y_val, ecuacion):
    try:
        # Se preprocesa la ecuación
        ecuacion_procesada = preprocesarFuncion(ecuacion)
        
        # Convertir a expresión de SymPy
        f = sp.sympify(ecuacion_procesada)
        
        # Usar el evaluador inteligente, según las condiciones de la ecuación decide qué método usar
        resultado = evaluarFuncionInteligente(f, x=x_val, y=y_val)
        
        return resultado
    
    except Exception as e:
        raise ValueError(f"Error al evaluar la derivada: {str(e)}")
    
# Función auxiliar para calcular derivada numérica de funciones de una variable
def calcular_derivada_numerica_univariable(f, var_simbolo, x0, h, metodo, funcion_original, derivada_analitica):
    procedimiento = []
    
    if metodo == 'hacia_adelante':
        f_x0 = evaluarFuncion(f, var_simbolo, x0)
        f_x0_h = evaluarFuncion(f, var_simbolo, x0 + h)
        derivada_numerica = (f_x0_h - f_x0) / h

        procedimiento = [
            f"Función: f({var_simbolo}) = {funcion_original}",
            f"Punto {var_simbolo}0: {x0}",
            f"Tamaño de paso h: {h}",
            f"Método: Diferencia hacia adelante",
            f"f({x0}) = {f_x0}",
            f"f({x0 + h}) = {f_x0_h}",
            f"Derivada numérica: f'({x0}) = (f({x0 + h}) - f({x0})) / {h} = {float(derivada_numerica)}",
            f"Derivada analítica: f'({x0}) = {derivada_analitica}"
        ]

    elif metodo == 'hacia_atras':
        f_x0 = evaluarFuncion(f, var_simbolo, x0)
        f_x0_menos_h = evaluarFuncion(f, var_simbolo, x0 - h)
        derivada_numerica = (f_x0 - f_x0_menos_h) / h

        procedimiento = [
            f"Función: f({var_simbolo}) = {funcion_original}",
            f"Punto {var_simbolo}0: {x0}",
            f"Tamaño de paso h: {h}",
            f"Método: Diferencia hacia atrás",
            f"f({x0}) = {f_x0}",
            f"f({x0 - h}) = {f_x0_menos_h}",
            f"Derivada numérica: f'({x0}) = (f({x0}) - f({x0 - h})) / {h} = {float(derivada_numerica)}",
            f"Derivada analítica: f'({x0}) = {derivada_analitica}"
        ]

    elif metodo == 'central':
        f_x0_mas_h = evaluarFuncion(f, var_simbolo, x0 + h)
        f_x0_menos_h = evaluarFuncion(f, var_simbolo, x0 - h)
        derivada_numerica = (f_x0_mas_h - f_x0_menos_h) / (2 * h)

        procedimiento = [
            f"Función: f({var_simbolo}) = {funcion_original}",
            f"Punto {var_simbolo}0: {x0}",
            f"Tamaño de paso h: {h}",
            f"Método: Diferencia central",
            f"f({x0 + h}) = {f_x0_mas_h}",
            f"f({x0 - h}) = {f_x0_menos_h}",
            f"Derivada numérica: f'({x0}) = (f({x0 + h}) - f({x0 - h})) / (2*{h}) = {float(derivada_numerica)}",
            f"Derivada analítica: f'({x0}) = {derivada_analitica}"
        ]

    return float(derivada_numerica), procedimiento


# Función para calcular derivada analítica numérica de funciones multivariables
def calcular_derivada_numerica_multivariable(f, var_simbolo, variable_derivar, valores_variables, h, metodo, funcion_original, derivada_analitica):
    procedimiento = []
    
    # Preparar valores base
    valores_base = valores_variables.copy()
    valor_variable = valores_variables[variable_derivar]
    
    # Crear descripción de variables
    vars_descripcion = ", ".join([f"{k} = {v}" for k, v in valores_variables.items()])
    
    if metodo == 'hacia_adelante':
        # Evaluar en el punto base
        f_base = evaluarFuncionInteligente(f, **valores_base)
        
        # Evaluar en el punto base + h
        valores_adelante = valores_base.copy()
        valores_adelante[variable_derivar] = valor_variable + h
        f_adelante = evaluarFuncionInteligente(f, **valores_adelante)
        
        derivada_numerica = (f_adelante - f_base) / h

        procedimiento = [
            f"Función: f = {funcion_original}",
            f"Derivando respecto a: {variable_derivar}",
            f"Tamaño de paso h: {h}",
            f"Método: Diferencia hacia adelante",
            f"f({vars_descripcion}) = {f_base}",
            f"f(..., {variable_derivar} = {valor_variable + h}, ...) = {f_adelante}",
            f"∂f/∂{variable_derivar} ≈ ({f_adelante} - {f_base}) / {h} = {float(derivada_numerica)}",
            f"Derivada analítica: ∂f/∂{variable_derivar} = {derivada_analitica}"
        ]

    elif metodo == 'hacia_atras':
        # Evaluar en el punto base
        f_base = evaluarFuncionInteligente(f, **valores_base)
        
        # Evaluar en el punto base - h
        valores_atras = valores_base.copy()
        valores_atras[variable_derivar] = valor_variable - h
        f_atras = evaluarFuncionInteligente(f, **valores_atras)
        
        derivada_numerica = (f_base - f_atras) / h

        procedimiento = [
            f"Función: f = {funcion_original}",
            f"Derivando respecto a: {variable_derivar}",
            f"Tamaño de paso h: {h}",
            f"Método: Diferencia hacia atrás",
            f"f({vars_descripcion}) = {f_base}",
            f"f(..., {variable_derivar} = {valor_variable - h}, ...) = {f_atras}",
            f"∂f/∂{variable_derivar} ≈ ({f_base} - {f_atras}) / {h} = {float(derivada_numerica)}",
            f"Derivada analítica: ∂f/∂{variable_derivar} = {derivada_analitica}"
        ]

    elif metodo == 'central':
        # Evaluar en punto + h
        valores_adelante = valores_base.copy()
        valores_adelante[variable_derivar] = valor_variable + h
        f_adelante = evaluarFuncionInteligente(f, **valores_adelante)
        
        # Evaluar en punto - h
        valores_atras = valores_base.copy()
        valores_atras[variable_derivar] = valor_variable - h
        f_atras = evaluarFuncionInteligente(f, **valores_atras)
        
        derivada_numerica = (f_adelante - f_atras) / (2 * h)

        procedimiento = [
            f"Función: f = {funcion_original}",
            f"Derivando respecto a: {variable_derivar}",
            f"Tamaño de paso h: {h}",
            f"Método: Diferencia central",
            f"f(..., {variable_derivar} = {valor_variable + h}, ...) = {f_adelante}",
            f"f(..., {variable_derivar} = {valor_variable - h}, ...) = {f_atras}",
            f"∂f/∂{variable_derivar} ≈ ({f_adelante} - {f_atras}) / (2*{h}) = {float(derivada_numerica)}",
            f"Derivada analítica: ∂f/∂{variable_derivar} = {derivada_analitica}"
        ]

    return float(derivada_numerica), procedimiento

# Función que renderiza la interfaz y realiza el cálculo del método de derivación
def derivacion(request):
    # DOCUMENTACIÓN del método
    documentacion = {
        'nombre': 'Método de Derivación',
        'definicion': '​La derivación numérica es una técnica utilizada para estimar la derivada de una función en un punto específico, especialmente cuando la función es compleja o no se dispone de su expresión analítica. Se basa en aproximaciones que emplean valores discretos de la función para calcular tasas de cambio.',
        'formulas': '''
            <p><strong> Fórmula general por definición: </strong> <br>
                f'(x) = lim h->0 [(f(x + h) - f(x)) / h] 
            <br> 
                Donde la variable 'h' representa el tamaño  del intervalo entre los puntos donde se evalúa la función.
            </p>
            
            <p>
                <strong>Diferencia hacia adelante.</strong> 
                Utiliza el valor de la función en el punto de interés y en un punto ligeramente adelantado: 
                <br>
                f'(x) ≈ [f(x+h) - f(x)] / h
            </p>
            
            <p><strong>Diferenciahacia atrás.</strong>  
                Emplea el valor de la función en el punto de interés y en un punto ligeramente retrasado: 
                <br>
                f'(x) ≈ [f(x) - f(x-h)] / h
            </p>
            

            <p><strong>Diferencia central.</strong> 
                Promedia las diferencias hacia adelante y hacia atrás, proporcionando una estimación más precisa:​ 
                <br>
                f'(x) ≈ [f(x+h) - f(x-h)] / (2h)
            </p>
            
            <p>Esta última fórmula es preferida por su mayor exactitud al considerar información de ambos lados del punto x.</p>
        ''',
        'usos': '''
            <p>
                La derivación numérica se aplica en diversas áreas, tales como:​
            </p>
            <ul>
                <li>Análisis de datos: Para determinar tasas de cambio en   conjuntos de datos empíricos.​</li>
                <li>Simulaciones físicas: En modelos que requieren la   estimación de derivadas de funciones representativas de   fenómenos físicos.​</li>
                <li>Optimización: En algoritmos que necesitan el cálculo de     gradientes para mejorar soluciones</li>
            </ul>
        '''
    }

    # CÁLCULO del método
    # Declararación de variables
    resultado = None
    mensajeError = None
    procedimiento = []
    valores_variables = {}

    if request.method == 'POST':
        try:
            # Obtener datos del formulario
            funcion = request.POST.get('funcion', '').strip()
            variable_derivar = request.POST.get('variable_derivar', '').strip()
            h = float(request.POST.get('h'))
            metodo = request.POST.get('metodo')

            # Asumir valores iniciales como 0 si no se envian
            try:
                x0 = float(request.POST.get('x0'))
            except:
                x0 = None
            try:
                y0 = float(request.POST.get('y0'))
            except:
                y0 = None
            try:
                z0 = float(request.POST.get('z0'))
            except:
                z0 = None
        
        except ValueError as e:
            raise ValueError("Los valores numéricos deben ser números validos")
        
        # Lista de métodos validos
        metodos_validos = ['hacia_adelante', 'hacia_atras', 'central']

        # Lista de métodos validos
        variables_validas = ['x', 'y', 'z']
        
        # Crear una Instancia del usuario
        objUsuario = usuario.objects.get(id = request.session["usuario_id"])

        parametros_entrada = {
            'funcion': funcion,
            'variable_derivar': variable_derivar,
            'x0': x0,
            'z0': z0,
            'y0': y0,
            'h': h,
            'metodo': metodo
        }
        
        try:
            # Validaciones numéricas
            if h <= 0:
                raise ValueError("El paso h debe ser mayor que cero")

            if h > 1:
                raise ValueError("El paso h es demasiado grande. Se recomienda h < 1 para mejor precisión")

            if h < 1e-10:
                raise ValueError("El paso h es demasiado pequeño. Puede causar errores de redondeo numérico")

            if metodo not in metodos_validos:
                raise ValueError(f"Método no válido. Debe ser uno de: {', '.join(metodos_validos)}")
            
            if variable_derivar not in variables_validas:
                raise ValueError(f"Variable no válida. Debe ser una de: {', '.join(variables_validas)}")

            # Preprocesar la función para SymPy
            funcion_procesada = preprocesarFuncion(funcion)

            # Debug: mostrar el procesamiento
            print(f"Función original: '{funcion}'")
            print(f"Función procesada: '{funcion_procesada}'")

            # Crear símbolo simbólico para la función
            x = sp.Symbol('x')
            y = sp.Symbol('y')
            z = sp.Symbol('z')

            # Validar que la función pueda ser interpretada por Sympy
            try:
                f = sp.sympify(funcion_procesada)
                print(f"Función SymPy: {f}")
            except Exception as e:
                raise ValueError(f"Error al interpretar la función '{funcion}'. Verifica la sintaxis. Error: {str(e)}")
            
            # Determinar las variables presentes en la función
            variables_en_funcion = obtenerVariablesRequeridas(funcion)
            print(f"Variables detectadas en la función: {variables_en_funcion}")

            # Validar que la variable a derivar esté en la función
            if variable_derivar not in variables_en_funcion and len(variables_en_funcion) > 0:
                raise ValueError(f"La variable '{variable_derivar}' no está presente en la función. Variables disponibles: {', '.join(variables_en_funcion)}")
            
            # Agregar las variables que están presentes en la función
            if 'x' in variables_en_funcion:
                if x0 is None:
                    raise ValueError("La función contiene la variable 'x', pero no se proporcionó el valor x0")
                valores_variables['x'] = float(x0)
            if 'y' in variables_en_funcion:
                if y0 is None:
                    raise ValueError("La función contiene la variable 'y', pero no se proporcionó el valor y0")
                valores_variables['y'] = float(y0)
            if 'z' in variables_en_funcion:
                if z0 is None:
                    raise ValueError("La función contiene la variable 'z', pero no se proporcionó el valor z0")
                valores_variables['z'] = float(z0)

            # Validar que la variable a derivar esté en la función
            if variable_derivar not in variables_en_funcion and len(variables_en_funcion) > 0:
                raise ValueError(f"La variable '{variable_derivar}' no está presente en la función. Variables disponibles: {', '.join(variables_en_funcion)}")

            # Determinar el tipo de función 
            if len(variables_en_funcion) == 0:# Función constante
                resultado = 0.0  # La derivada de una constante es 0
                derivadaAnalitica = 0.0
                
                procedimiento = [
                    f"Función: f = {funcion}",
                    "Esta es una función constante",
                    "La derivada de una constante es siempre 0",
                    f"Derivada numérica: 0",
                    f"Derivada analítica: 0",
                    "Error porcentual: 0%"
                ]
            # Función de una sola variable
            elif len(variables_en_funcion) == 1:
                var_simbolo = list(f.free_symbols)[0]
                
                # Validar puntos necesarios para el método seleccionado
                puntos_a_validar = [valores_variables[str(var_simbolo)]]
                if metodo == 'hacia_adelante':
                    puntos_a_validar.append(valores_variables[str(var_simbolo)] + h)
                elif metodo == 'hacia_atras':
                    puntos_a_validar.append(valores_variables[str(var_simbolo)] - h)
                elif metodo == 'central':
                    puntos_a_validar.extend([valores_variables[str(var_simbolo)] + h, valores_variables[str(var_simbolo)] - h])

                # Validar todos los puntos
                for punto in puntos_a_validar:
                    evaluarFuncion(f, var_simbolo, punto)

                # Calcular derivada analítica
                try:
                    derivada_simbolica = sp.diff(f, var_simbolo)
                    derivadaAnalitica = evaluarFuncion(derivada_simbolica, var_simbolo, valores_variables[str(var_simbolo)])
                except Exception as e:
                    raise ValueError(f"Error al calcular la derivada analítica: {str(e)}")

                # Aplicar método de derivación numérica
                resultado, procedimiento = calcular_derivada_numerica_univariable(
                    f, var_simbolo, valores_variables[str(var_simbolo)], h, metodo, funcion, derivadaAnalitica
                )
            # Función multivariable
            else:
                # Determinar la variable respecto a la cual derivar
                if variable_derivar == 'x':
                    var_simbolo = x
                    valor_var = x0
                elif variable_derivar == 'y':
                    var_simbolo = y
                    valor_var = y0
                elif variable_derivar == 'z':
                    var_simbolo = z
                    valor_var = z0
                else:
                    raise ValueError(f"Variable de derivación no soportada: {variable_derivar}")

                # Calcular derivada analítica (parcial)
                try:
                    derivada_simbolica = sp.diff(f, var_simbolo)
                    derivadaAnalitica = evaluarFuncionInteligente(derivada_simbolica, **valores_variables)
                except Exception as e:
                    raise ValueError(f"Error al calcular la derivada parcial analítica: {str(e)}")

                # Aplicar método de derivación numérica para funciones multivariables
                resultado, procedimiento = calcular_derivada_numerica_multivariable(
                    f, var_simbolo, variable_derivar, valores_variables, h, metodo, funcion, derivadaAnalitica
                )

            # Calcular error si la derivada analítica no es cero
            if hasattr(locals(), 'derivadaAnalitica') and abs(derivadaAnalitica) > 1e-10:
                error = abs((resultado - derivadaAnalitica) / derivadaAnalitica) * 100
            else:
                error = abs(resultado - derivadaAnalitica) * 100

            if len(procedimiento) > 0 and not any("Error porcentual" in p for p in procedimiento):
                procedimiento.append(f"Error porcentual: {error:.6f}%")

            # GUARDAR EN BASE DE DATOS - CÁLCULO EXITOSO
            try:
                print("Iniciando guardado en BD...")  # Debug
                
                # Buscar o crear categoria
                categoria, _ = categoria_metodo.objects.get_or_create(
                    nombre = "Derivación Numérica",
                    descripcion = "Método numérico para estimar derivadas de funciones"
                )

                # Buscar o crear el método de Derivación
                metodo, creado = metodo_numerico.objects.get_or_create(
                    nombre='Derivación Numérica',
                    defaults={
                        'id_categoria_id': 7,  
                        'documentacion': documentacion['definicion']
                    }
                )
                
                print(f"Método obtenido/creado: {metodo.nombre}, Creado: {creado}")  # Debug

                # Crear el registro del cálculo
                objCalculo = calculo.objects.create(
                    id_usuario = objUsuario,
                    id_metodo = metodo,
                    parametros_entrada = parametros_entrada,
                    resultado = {'derivada numerica': resultado},
                    procedimiento = procedimiento
                )

                print(f'Cálculo guardado exitosamente con ID: {objCalculo.id}')
                messages.success(request, 'Cálculo realizado y guardado correctamente')

            except Exception as db_error:
                messages.warning(request, f'Cálculo realizado correctamente, pero no se pudo guardar en la base de datos: {str(db_error)}')
                print(f"Error de BD en cálculo exitoso: {db_error}")

        except Exception as e:
            # Error en el cálculo principal
            mensajeError = f"Error en el cálculo: {str(e)}"
            print(f"Error en cálculo: {e}")
            
            # Intentar guardar el error en la BD
            try: 
                print("Guardando error en BD...")  # Debug
                
                metodo, _ = metodo_numerico.objects.get_or_create(
                    nombre = 'Derivación Numérica',
                    defaults ={
                        'id_categoria_id': 7,
                        'documentacion': documentacion['definicion']
                    }
                )

                objCalculo = calculo.objects.create(
                    id_usuario = objUsuario,
                    id_metodo = metodo,
                    parametros_entrada = parametros_entrada,
                    resultado=None,
                    procedimiento=None,
                    mensaje_error=mensajeError
                )
                print(f'Error guardado en BD con ID: {objCalculo.id}')
                
            except Exception as db_error_2:
                print(f'Error al guardar error en BD: {db_error_2}')
            

        except ValueError as e:
            mensajeError = f"Error de validación: {str(e)}"
            print(f"ValueError: {e}")
        except ZeroDivisionError as e:
            mensajeError = "Error: División por cero en el cálculo de la    derivada"
            print(f"ZeroDivisionError: {e}")
        except OverflowError as e:
            mensajeError = "Error: Los valores se volvieron demasiado   grandes durante el cálculo"
            print(f"OverflowError: {e}")
        except Exception as e:
            mensajeError = f"Error inesperado: {str(e)}"
            print(f"Unexpected error: {e}")

        template_vars = {
            'resultado': resultado,
            'procedimiento': procedimiento,
            'error': mensajeError,
            'funcion': funcion,
            'variable_derivar': variable_derivar,
            'x0': x0,
            'z0': z0,
            'y0': y0,
            'h': h,
            'metodo': metodo,
            'documentacion': documentacion
        }

        return render(request, 'calculadora/derivacion.html', template_vars)
    else:
        return render(request, 'calculadora/derivacion.html',{
            'documentacion': documentacion
        })
    
# Función que renderiza la interfaz y realiza el cálculo para el método de euler
def euler(request):
    # DOCUMENTACIÓN del método
    documentacion = {
        'nombre': 'Método de Euler clásico',
        'definicion': 'El método de Euler clásico ees un procedimiento numérico de primer orden para resolver ecuaciones diferenciales ordinarias (EDO) con un valor inicial dado. Es el método explícito más básico para la integración numérica de ecuaciones diferenciales ordinarias y es el método de Runge-Kutta más simple. Es especialmente útil cuando no es posible obtener soluciones analíticas exactas.',
        'formulas': '''
            <p><strong> Considerando una ecuación diferencial ordinaria de la forma: </strong> <br>
                y' = f(x,y),    y(x0) = y0
            </p> 
            <p>
                <strong>El método de Euler aproxima la solución construyendo una sucesión de valores (xn, yn) </strong> mediante la fórmula:
                <br>
                xn+1 = xn + h*yn+1 = yn + h * f(xn,yn)
            </p>   
            <p> 
                Aquí, h es el tamaño del paso, que determina la distancia entre los puntos donde se calcula la solución aproximada. El valor yn+1 se obtiene sumando al valor anterior yn el producto del tamaño del paso h por la pendiente de la función en el punto (xn,yn).
            <p>
        ''',
        'usos': '''
            <p>
                El método clásico de Euler se aplica en contextos como:
            </p>
            <ul>
                <li>Modelado de sistemas dinámicos: Permite simular el comportamiento de sistemas físicos y biológicos descritos por EDOs.</li>
                <li>Procesos de crecimiento y decaimiento exponencial: Es útil para aproximar soluciones en modelos de poblaciones o desintegración radiactiva</li>
                <li>Circuitos eléctricos: Ayuda en la resolución de ecuaciones que describen la evolución temporal de corrientes y voltajes.</li>
            </ul>
        '''
    }

    # CÁLCULO del método
    # Declararación de variables
    resultado = None
    mensajeError = None
    procedimiento = []

    if request.method == 'POST':
        try:
            # Obtener datos del formulario
            ecuacion = request.POST.get('ecuacion', '').strip()
            x0 = float(request.POST.get('x0'))
            y0 = float(request.POST.get('y0'))
            print(f"y: {y0}")
            x_final = float(request.POST.get('x_final'))
            h = float(request.POST.get('h'))

        except ValueError as e:
            raise ValueError("Todos los valores numéricos deben ser números válidos")
            
        # Crear una Instancia del usuario
        objUsuario = usuario.objects.get(id=request.session["usuario_id"])

        parametros_entrada = {
            'ecuacion': ecuacion,
            'x0': x0,
            'y0': y0,
            'x final': x_final,
            'h': h
        }
        
        try:
            # Validar campos vacios
            if not all([ecuacion, x0, y0, x_final, h]):
                raise ValueError("Todos los campos son obligatorios")
            
            # Validaciones numéricas
            if h <= 0:
                raise ValueError("El paso h debe ser mayor que cero")

            if x_final <= x0:
                raise ValueError("x final debe ser mayor que x0")

            if h > (x_final - x0):
                raise ValueError(f"El paso h es demasiado grande. Debe ser   menor que {(x_final - x0)}")
            
            # Validar que la ecuación sea evaluable
            if evaluarDerivada(str(x0), str(y0), ecuacion):
                pass # si no hay errores en la derivación, continua el flujo

            # Calcular número de pasos
            n = int(round((x_final - x0) / h))
            # Ajustar h para llegar exactamente a x_final
            h_ajustado = (x_final - x0) / n

            if abs(h_ajustado - h) > 0.0001:  # Si la diferencia es     significativa
                procedimiento.append(f"Nota: h ajustado de {h} a {h_ajustado:.6f} para llegar exactamente a {x_final}")
                h = h_ajustado

            # Validar número máximo de pasos para evitar cálculos excesivos
            if n > 100:
                raise ValueError("Número de pasos demasiado grande (máximo: 100). Use un h mayor.")
            
            # Inicializar arreglos para x, y
            x_valores = np.zeros(n + 1)
            y_valores = np.zeros(n + 1)
            
            # Condiciones iniciales
            x_valores[0] = x0
            y_valores[0] = y0
            
            # Implementación del método de Euler
            for i in range(1, n + 1):
                try:
                    # Guardar el ultimo valor de x,y
                    x_prev = x_valores[i-1]
                    y_prev = y_valores[i-1]

                    # Calcular la pendiente
                    pendiente = evaluarDerivada(x_prev,  y_prev, ecuacion)

                    # Actualizar x, y
                    x_valores[i] = x_prev + h_ajustado
                    y_valores[i] = y_prev + h_ajustado * pendiente

                    # Registrar pasos
                    procedimiento.append(
                        f"Paso {i}: x = {x_valores[i]:.4f}, y = {y_valores[i] :.4f}, "
                        f"pendiente = {pendiente:.6f}"
                    )
                except Exception as e:
                    raise ValueError(f"Error en el paso {i}: {str(e)}")

            # Resultado final
            resultado = {
                'x_final': f"Valor final de x: {x_valores[-1]}",
                'y_final': f"Valor final de y: {y_valores[-1]}"
            }

            # GUARDAR EN BASE DE DATOS - CÁLCULO EXITOSO
            try:
                print("Iniciando guardado en BD...")  # Debug
                
                # Buscar o crear categoria
                categoria, _ = categoria_metodo.objects.get_or_create(
                    nombre = "Euler Clásico",
                    descripcion = "Método numérico básico para resolver EDOs"
                )

                # Buscar o crear el método Runge-Kutta
                metodo, creado = metodo_numerico.objects.get_or_create(
                    nombre='Euler Clásico',
                    defaults={
                        'id_categoria_id': 6,  
                        'documentacion': documentacion['definicion']
                    }
                )
                
                print(f"Método obtenido/creado: {metodo.nombre}, Creado: {creado}")  # Debug

                # Crear el registro del cálculo
                objCalculo = calculo.objects.create(
                    id_usuario = objUsuario,
                    id_metodo = metodo,
                    parametros_entrada = parametros_entrada,
                    resultado ={
                        'x final': x_valores[-1],
                        'y final': y_valores[-1]
                    },
                    procedimiento = procedimiento
                )

                print(f'Cálculo guardado exitosamente con ID: {objCalculo.id}')
                messages.success(request, 'Cálculo realizado y guardado correctamente')

            except Exception as db_error:
                messages.warning(request, f'Cálculo realizado correctamente, pero no se pudo guardar en la base de datos: {str(db_error)}')
                print(f"Error de BD en cálculo exitoso: {db_error}")

        except Exception as e:
            # Error en el cálculo principal
            mensajeError = f"Error en el cálculo: {str(e)}"
            print(f"Error en cálculo: {e}")
            
            # Intentar guardar el error en la BD
            try:
                print("Guardando error en BD...")  # Debug
                
                metodo, _ = metodo_numerico.objects.get_or_create(
                    nombre = 'Euler Clásico',
                    defaults ={
                        'id_categoria_id': 6,
                        'documentacion': documentacion['definicion']
                    }
                )

                objCalculo = calculo.objects.create(
                    id_usuario = objUsuario,
                    id_metodo = metodo,
                    parametros_entrada = parametros_entrada,
                    resultado=None,
                    procedimiento=None,
                    mensaje_error=mensajeError
                )
                print(f'Error guardado en BD con ID: {objCalculo.id}')
                
            except Exception as db_error_2:
                print(f'Error al guardar error en BD: {db_error_2}')
        
        except ValueError as e:
            mensajeError = f"Error de validación: {str(e)}"
            print(f"ValueError: {e}")
        except ZeroDivisionError as e:
            mensajeError = "Error: División por cero en el cálculo"
            print(f"ZeroDivisionError: {e}")
        except OverflowError as e:
            mensajeError = "Error: Los valores se volvieron demasiado   grandes durante el cálculo"
            print(f"OverflowError: {e}")
        except Exception as e:
            mensajeError = f"Error inesperado: {str(e)}"
            print(f"Unexpected error: {e}")

        template_vars = {
            'resultado': resultado,
            'procedimiento': procedimiento,
            'error': mensajeError,
            'ecuacion': ecuacion,
            'x0': x0,
            'x_final': x_final,
            'y0': y0,
            'h': h,
            'documentacion': documentacion
        }

        return render(request, 'calculadora/euler.html', template_vars)
    else:
        return render(request, 'calculadora/euler.html',{
            'documentacion': documentacion
        })

# Función que renderiza la interfaz y realiza el cálculo para el método de euler mejorado
def eulerMejorado(request):
    # DOCUMENTACIÓN del método
    documentacion = {
        'nombre': 'Método de Euler mejorado',
        'definicion': 'El método de Euler mejorado, también conocido como método del punto medio o método de Heun, es una técnica numérica utilizada para aproximar soluciones de ecuaciones diferenciales ordinarias (EDO). Este método mejora la precisión del método de Euler clásico al considerar la pendiente en dos puntos dentro de cada intervalo, promediándolas para obtener una mejor estimación del valor siguiente de la solución.',
        'formulas': '''
            <p>El procedimiento de este método se puede describir en los siguientes pasos:
            <br>
            <strong> Cálculo de la pendiente inicial (k1): </strong> <br>
                k1 = f(xi,yi)
            <br>
            Esta es la pendiente de la función en el punto actuaal (xi,yi)
            </p> 
            <p>
                <strong>Estimación preliminar del siguiente valor de y (yi+1):</strong>
                <br>
                yi+1 = yi + h * ki
                <br>
                Aquí, h es el tamaño del paaso y yi+1 es una estimación inicial del valor de y en xi+1 = xi + h utilizando la pendiente inicial.
            </p>   
            <p> 
                <strong>Cálculo de la pendiente en el punto estimado (k2):</strong>
                <br>
                k2 = f(xi+1,yi+1)
                <br>
                Esta es la pendiente de la función en el punto estimado (xi+1,yi+1).
            <p>
            <p> 
                <strong>Cálculo de la pendiente en el punto estimado (k2):</strong>
                <br>
                k2 = f(xi+1,yi+1)
                <br>
                Esta es la pendiente de la función en el punto estimado (xi+1,yi+1).
            <p>
            <p> 
                <strong>Cálculo del valor corregido de y en xi+1:</strong>
                <br>
                yi+1 = yi + (h/2)(ki + k2)
                <br>
                Este es el valor final corregido de y, obtenido promediando las pendientes k1 y k2.
            <p>
        ''',
        'usos': '''
            <p>
                El método clásico de Euler se aplica en campos matemáticos y de ingeniería para resolver probllemas que involucran EDOs, especialmente cuando no es posible obtener soluciones analíticas exactas, por ejemplo:
            </p>
            <ul>
                <li>EDOs rígidas: En ecuaciones donde hay una gran disparidad en las escalas de tiempo de las soluciones, el método puede requerir pasos extremadamente pequeños para mantener la estabilidad y precisión, lo que lo hace ineficiente.</li>
                <li>Sistemas altamente no lineales: En ecuaciones con comportamientos altamente no lineales o caóticos, las aproximaciones lineales del método pueden no capturar adecuadamente la dinámica del sistema.</li>
            </ul>
            <p>
                En problemas donde pequeñas variaciones en las condiciones iniciales provocan grandes cambios en la solución, el método puede acumular errores significativos. En tales casos, se recomienda considerar métodos numéricos más avanzados, como los métodos de Runge-Kutta de órdenes superiores, que ofrecen mayor estabilidad y precisión. 
            </p>
        '''
    }

    # CÁLCULO del método
    # Declararación de variables
    resultado = None
    mensajeError = None
    procedimiento = []

    if request.method == 'POST':
        try:
            # Obtener datos del formulario
            ecuacion = request.POST.get('ecuacion', '').strip()
            x0 = float(request.POST.get('x0'))
            y0 = float(request.POST.get('y0'))
            x_final = float(request.POST.get('x_final'))
            h = float(request.POST.get('h'))

        except ValueError as e:
            raise ValueError("Todos los valores numéricos deben ser números válidos")
        
        # Crear una Instancia del usuario
        objUsuario = usuario.objects.get(id=request.session["usuario_id"])

        # Preparar parametros de entrada para guardar
        parametros_entrada = {
            'ecuacion': ecuacion,
            'x0': x0,
            'y0': y0,
            'x final': x_final,
            'h': h
        }
        
        try:
            # Validar campos vacíos
            if not all([ecuacion, x0, y0, x_final, h]):
                raise ValueError("Todos los campos son obligatorios")
            
            # Validaciones numéricas
            if h <= 0:
                raise ValueError("El paso h debe ser mayor que cero")

            if x_final <= x0:
                raise ValueError("x final debe ser mayor que x0")

            if h > (x_final - x0):
                raise ValueError(f"El paso h es demasiado grande. Debe ser   menor que {(x_final - x0)}")

            # Validar que la ecuación sea evaluable
            if evaluarDerivada(str(x0), str(y0), ecuacion):
                pass # si no hay errores en la derivación, continua el flujo

            # Calcular número de pasos
            n = int(round((x_final - x0) / h))
            # Ajustar h para llegar exactamente a x_final
            h_ajustado = (x_final - x0) / n

            if abs(h_ajustado - h) > 0.0001:  # Si la diferencia es     significativa
                procedimiento.append(f"Nota: h ajustado de {h} a {h_ajustado:.6f} para llegar exactamente a {x_final}")
                h = h_ajustado

            # Validar número máximo de pasos para evitar cálculos excesivos
            if n > 100:
                raise ValueError("Número de pasos demasiado grande (máximo: 100). Use un h mayor.")
            
            # Inicializar arreglos para resultados
            x_valores = np.zeros(n + 1)
            y_valores = np.zeros(n + 1)
            
            # Condiciones iniciales
            x_valores[0] = x0
            y_valores[0] = y0
            
            # Implementar Método de Heun
            for i in range(1, n + 1):
                try:
                    # Guardar el ultimo valor de x,y
                    x_prev = x_valores[i-1]
                    y_prev = y_valores[i-1]

                    # Predictor (primer paso de Euler)
                    k1 = evaluarDerivada(x_prev, y_prev, ecuacion)
                    y_pred = y_prev + h * k1

                    # Corrector (promedio de pendientes)
                    k2 = evaluarDerivada(x_prev + h, y_pred, ecuacion)
                    y_next = y_prev + (h/2) * (k1 + k2)
                    x_next = x_prev + h

                    # Almacenar valores
                    x_valores[i] = x_next
                    y_valores[i] = y_next

                    # Registrar paso
                    procedimiento.append(
                        f"Paso {i}: " +
                        f"x = {x_next:.4f}, " +
                        f"k1 = {k1}, " +
                        f"y predicho = {y_pred:.4f}, " +
                        f"k2 = {k2}, " +
                        f"y corregido = {y_next:.4f}"
                    )
            
                except Exception as e:
                    mensajeError = f"Error en el paso {i}: {str(e)}"
                    print(e)
            
            # resultado final
            resultado = {
                'x_final': f"Valor final de x: {x_valores[-1]}",
                'y_final': f"Valor final de y: {y_valores[-1]}"
            }

            # GUARDAR EN BASE DE DATOS - CÁLCULO EXITOSO
            try:
                print("Iniciando guardado en BD...")  # Debug
                
                # Buscar o crear categoria
                categoria, _ = categoria_metodo.objects.get_or_create(
                    nombre = "Euler Mejorado (Heun)",
                    descripcion = "Método numérico para resolver EDOs con mayor precisión que el método de Euler clásico"
                )

                # Buscar o crear el método Runge-Kutta
                metodo, creado = metodo_numerico.objects.get_or_create(
                    nombre='Euler Mejorado (Heun)',
                    defaults={
                        'id_categoria_id': 5,  
                        'documentacion': documentacion['definicion']
                    }
                )
                
                print(f"Método obtenido/creado: {metodo.nombre}, Creado: {creado}")  # Debug

                # Crear el registro del cálculo
                objCalculo = calculo.objects.create(
                    id_usuario = objUsuario,
                    id_metodo = metodo,
                    parametros_entrada = parametros_entrada,
                    resultado ={
                        'x final': x_valores[-1],
                        'y final': y_valores[-1]
                    },
                    procedimiento = procedimiento
                )

                print(f'Cálculo guardado exitosamente con ID: {objCalculo.id}')
                messages.success(request, 'Cálculo realizado y guardado correctamente')

            except Exception as db_error:
                messages.warning(request, f'Cálculo realizado correctamente, pero no se pudo guardar en la base de datos: {str(db_error)}')
                print(f"Error de BD en cálculo exitoso: {db_error}")

        except Exception as e:
            # Error en el cálculo principal
            mensajeError = f"Error en el cálculo: {str(e)}"
            print(f"Error en cálculo: {e}")
            
            # Intentar guardar el error en la BD
            try:
                print("Guardando error en BD...")  # Debug
                
                metodo, _ = metodo_numerico.objects.get_or_create(
                    nombre = 'Euler Mejorado (Heun)',
                    defaults ={
                        'id_categoria_id': 5,
                        'documentacion': documentacion['definicion']
                    }
                )

                objCalculo = calculo.objects.create(
                    id_usuario = objUsuario,
                    id_metodo = metodo,
                    parametros_entrada = parametros_entrada,
                    resultado=None,
                    procedimiento=None,
                    mensaje_error=mensajeError
                )
                print(f'Error guardado en BD con ID: {objCalculo.id}')
                
            except Exception as db_error_2:
                print(f'Error al guardar error en BD: {db_error_2}')
            
        except ValueError as e:
            mensajeError = f"Error de validación: {str(e)}"
            print(f"ValueError: {e}")
        except ZeroDivisionError as e:
            mensajeError = "Error: División por cero en el cálculo"
            print(f"ZeroDivisionError: {e}")
        except OverflowError as e:
            mensajeError = "Error: Los valores se volvieron demasiado   grandes durante el cálculo"
            print(f"OverflowError: {e}")
        except Exception as e:
            mensajeError = f"Error inesperado: {str(e)}"
            print(f"Unexpected error: {e}")

        template_vars ={
            'resultado': resultado,
            'procedimiento': procedimiento,
            'error': mensajeError,
            'ecuacion': ecuacion,
            'x0': x0,
            'x_final': x_final,
            'y0': y0,
            'h': h,
            'documentacion': documentacion
        }

        return render(request, 'calculadora/eulerMejorado.html', template_vars)
    else:
        return render(request, 'calculadora/eulerMejorado.html',{
            'documentacion': documentacion
        })
    
# Función que renderiza la interfaz y realiza el cálculo para el método de interpolación linear
def interpolacionLinear(request):
    # DOCUMENTACIÓN del método
    documentacion = {
        'nombre': 'Método de Interpolación',
        'definicion': 'La interpolación es un método numérico que permite estimar valores intermedios de una función a partir de un conjunto discreto de datos conocidos. Consiste en construir una función que pase exactamente por estos puntos dados, facilitando la aproximación de valores donde no se tienen mediciones directas.',
        'formulas': '''
            <p>La variante <strong>Interpolación Linear</strong> es la forma más sencilla de este método, donde se conectan dos puntos consecutivos con una línea recta. <strong> La expresión de la interpolación lineal se obtiene del polinomio interpolador de Newton de grado uno: </strong> <br>
                f(x) = f(xi) + (x-xi)/(xi+i-xi) * (f(xi+i) - f(xi)) 
            <br>
            Donde x es el valor en el cual se desea interpolar, xi y xi+1 son los puntos conocidos más cercanos y F(xi), f(xi+i) son los valores de la función en esos puntos.
            </p> 
            <p>
                La variante <strong>Interpolación de Langrage</strong> es un método que construye un polinomio que pasa por todos los puntos dados sin necesidad de calcular diferencias divididas. <strong>La fórmula del polinomio de Lagrange de grado n es:</strong>
                <br>
                P(x) = \sum _{n=0}^{\infty }\: f(xj) * Lj(x)
                <br>
                Donde Lj(x) son los polinomios básicos de Lagrange definidos como:
                <br>
                Lj(x) = \prod _{i=0}^n\left, i\nej (x-xi)/(xj-xi)
                <br>
                Aquí, xj son los puntos conocidos y f(xj) sus correspondientes valores.
            </p>   
            <p> 
                La variante <strong>Interpolación de Newton</strong> utiliza diferencias divididas para construir el polinomio interpolante. <strong>La fórmula general es:</strong>
                <br>
                P(x) = f[x0] + (x-x0)f[x0,x1] + (x-x0)(x-x1)f[x0,x1,x2] + ...
                <br>
                Donde f[x0,x1,...,xk] son las diferencias divididas calculadasa a partir de los valores conocidos.
            <p>
        ''',
        'usos': '''
            <p>
                El método de interpolación tiene diversas aplicaciones en el campo matemático, tales como:
            </p>
            <ul>
                <li>Estimación de valores intermedios: Cuando se tienen datos discretos, la interpolación permite estimar valores en puntos no medidos. Por ejemplo, determinar la temperatura en un momento específico basándose en registros horarios.</li>
                <li>Aproximación de funciones complejas: En situaciones donde una función es difícil de evaluar directamente, la interpolación ofrece una aproximación basada en un conjunto de puntos conocidos.</li>
                <li>Integración y diferenciación numérica: Métodos como la integración y diferenciación numérica se derivan de fórmulas de interpolación polinómica.</li>
            </ul>
            <p><strong style="color: red">Nota:</strong> En esta sección solo se calcula la interpolación linear.</p>
        '''
    }

    # CÁLCULO del método de interpolación lineal corregido
    # Declaración de variables
    resultado = None
    mensajeError = None
    procedimiento = []

    if request.method == 'POST':
        try:
            # Obtener datos del formulario
            encontrar = request.POST.get('encontrar')
            x_valores_str = request.POST.get('x_valores')
            y_valores_str = request.POST.get('y_valores')
            # Asumir x, y en 0 si no se envian
            try:
                x_buscado = float(request.POST.get('x_buscado', ''))
            except:
                x_buscado = 0
            try:
                y_buscado = float(request.POST.get('y_buscado', ''))
            except:
                y_buscado = 0

            # Crear una Instancia del usuario
            objUsuario = usuario.objects.get(id=request.session["usuario_id"])

            # Preparar parametros de entrada para guardar
            parametros_entrada = {
                'encontrar': encontrar,
                'valores de x': x_valores_str,
                'valores de y': y_valores_str,
                'x buscado': x_buscado,
                'y buscado': y_buscado,
            }

            # Validar que los campos no estén vacíos
            if not x_valores_str or not y_valores_str:
                raise ValueError("Los valores de X e Y no pueden estar  vacíos")

            # Convertir strings a listas de floats
            x_valores = list(map(float, x_valores_str.split(',')))
            y_valores = list(map(float, y_valores_str.split(',')))

            # Validaciones básicas
            if len(x_valores) != len(y_valores):
                raise ValueError("Los arrays de X e Y deben tener la misma  longitud")

            if len(x_valores) < 2:
                raise ValueError("Se necesitan al menos 2 puntos para   realizar interpolación")
            
            if len(x_valores) > 20 or len(y_valores) > 20:
                raise ValueError("Máximo 20 puntos permitidos para evitar inestabilidad numérica")

            # Verificar que no haya valores duplicados en X (para   interpolación de Y)
            # y que no haya valores duplicados en Y (para interpolación de  X)
            if encontrar == 'y' and len(set(x_valores)) != len(x_valores):
                raise ValueError("Los valores de X no pueden tener  duplicados para interpolar Y")
            elif encontrar == 'x' and len(set(y_valores)) != len(y_valores):
                raise ValueError("Los valores de Y no pueden tener  duplicados para interpolar X")

            # Convertir a arrays de NumPy
            x = np.array(x_valores)
            y = np.array(y_valores)

            if encontrar == 'y':
                # Ordenar los datos por X para interpolación correcta
                indices_ordenados = np.argsort(x)
                x_ordenado = x[indices_ordenados]
                y_ordenado = y[indices_ordenados]

                # Verificar si el punto está dentro del rango
                if x_buscado < x_ordenado[0]:
                    # Extrapolación hacia la izquierda
                    x1, x2 = x_ordenado[0], x_ordenado[1]
                    y1, y2 = y_ordenado[0], y_ordenado[1]
                    y_interpolado = y1 + (x_buscado - x1) * (y2 - y1) / (x2- x1)

                    procedimiento = [
                        f"Datos ordenados: x = {x_ordenado.tolist()}, y =   {y_ordenado.tolist()}",
                        f"Punto buscado: x = {x_buscado}",
                        f"EXTRAPOLACIÓN: x está fuera del rango de datos    (menor que {x_ordenado[0]})",
                        f"Puntos usados: (x1, y1) = ({x1}, {y1}), (x2, y2)  = ({x2}, {y2})",
                        f"Fórmula: y = y1 + (x - x1) * (y2 - y1) / (x2 - x1)    ",
                        f"Cálculo: y = {y1} + ({x_buscado} - {x1}) * ({y2}  - {y1}) / ({x2} - {x1})",
                        f"Resultado de extrapolación: y = {y_interpolado:.  6f}"
                    ]

                elif x_buscado > x_ordenado[-1]:
                    # Extrapolación hacia la derecha
                    x1, x2 = x_ordenado[-2], x_ordenado[-1]
                    y1, y2 = y_ordenado[-2], y_ordenado[-1]
                    y_interpolado = y1 + (x_buscado - x1) * (y2 - y1) / (x2     - x1)

                    procedimiento = [
                        f"Datos ordenados: x = {x_ordenado.tolist()}, y =   {y_ordenado.tolist()}",
                        f"Punto buscado: x = {x_buscado}",
                        f"EXTRAPOLACIÓN: x está fuera del rango de datos    (mayor que {x_ordenado[-1]})",
                        f"Puntos usados: (x1, y1) = ({x1}, {y1}), (x2, y2)  = ({x2}, {y2})",
                        f"Fórmula: y = y1 + (x - x1) * (y2 - y1) / (x2 - x1)    ",
                        f"Cálculo: y = {y1} + ({x_buscado} - {x1}) * ({y2}  - {y1}) / ({x2} - {x1})",
                        f"Resultado de extrapolación: y = {y_interpolado:.  6f}"
                    ]

                else:
                    # Interpolación dentro del rango
                    # Buscar el intervalo correcto
                    indice_derecho = np.searchsorted(x_ordenado, x_buscado)

                    # Si x_buscado es exactamente igual a un punto conocido
                    if indice_derecho < len(x_ordenado) and x_ordenado  [indice_derecho] == x_buscado:
                        y_interpolado = y_ordenado[indice_derecho]
                        procedimiento = [
                            f"Datos: x = {x_valores}, y = {y_valores}",
                            f"Punto buscado: x = {x_buscado}",
                            f"El punto x = {x_buscado} existe exactamente   en los datos",
                            f"Resultado: y = {y_interpolado}"
                        ]
                    else:
                        # Interpolación lineal entre dos puntos
                        x1, x2 = x_ordenado[indice_derecho-1], x_ordenado   [indice_derecho]
                        y1, y2 = y_ordenado[indice_derecho-1], y_ordenado   [indice_derecho]

                        y_interpolado = y1 + (x_buscado - x1) * (y2 - y1) /     (x2 - x1)

                        procedimiento = [
                            f"Puntos cercanos: (x1, y1) = ({x1}, {y1}),     (x2, y2) = ({x2}, {y2})",
                            f"Fórmula: y = y1 + (x - x1) * (y2 - y1) / (x2  - x1)",
                            f"Cálculo: y = {y1} + ({x_buscado} - {x1}) *    ({y2} - {y1}) / ({x2} - {x1})",
                            f"Resultado de interpolación: y =   {y_interpolado:.6f}"
                        ]

                resultado = y_interpolado

            else:  # encontrar x entre los valores de y sin asumir que están ordenados

                # Encontrar el rango de Y
                y_min, y_max = np.min(y), np.max(y)

                if y_buscado < y_min:
                    # Extrapolación: encontrar los dos puntos con Y mínimos
                    indices_min = np.argsort(y)[:2]
                    x1, x2 = x[indices_min[0]], x[indices_min[1]]
                    y1, y2 = y[indices_min[0]], y[indices_min[1]]

                    x_interpolado = x1 + (y_buscado - y1) * (x2 - x1) / (y2     - y1)

                    procedimiento = [
                        f"Datos: x = {x_valores}, y = {y_valores}",
                        f"Punto buscado: y = {y_buscado}",
                        f"EXTRAPOLACIÓN: y está fuera del rango de datos    (menor que {y_min})",
                        f"Puntos usados: (x1, y1) = ({x1}, {y1}), (x2, y2)  = ({x2}, {y2})",
                        f"Fórmula: x = x1 + (y - y1) * (x2 - x1) / (y2 - y1)    ",
                        f"Cálculo: x = {x1} + ({y_buscado} - {y1}) * ({x2}  - {x1}) / ({y2} - {y1})",
                        f"Resultado de extrapolación: x = {x_interpolado:.  6f}"
                    ]

                elif y_buscado > y_max:
                    # Extrapolación: encontrar los dos puntos con Y máximos
                    indices_max = np.argsort(y)[-2:]
                    x1, x2 = x[indices_max[0]], x[indices_max[1]]
                    y1, y2 = y[indices_max[0]], y[indices_max[1]]

                    x_interpolado = x1 + (y_buscado - y1) * (x2 - x1) / (y2     - y1)

                    procedimiento = [
                        f"Datos: x = {x_valores}, y = {y_valores}",
                        f"Punto buscado: y = {y_buscado}",
                        f"EXTRAPOLACIÓN: y está fuera del rango de datos    (mayor que {y_max})",
                        f"Puntos usados: (x1, y1) = ({x1}, {y1}), (x2, y2)  = ({x2}, {y2})",
                        f"Fórmula: x = x1 + (y - y1) * (x2 - x1) / (y2 - y1)    ",
                        f"Cálculo: x = {x1} + ({y_buscado} - {y1}) * ({x2}  - {x1}) / ({y2} - {y1})",
                        f"Resultado de extrapolación: x = {x_interpolado:.  6f}"
                    ]

                else:
                    # Interpolación: encontrar dos puntos consecutivos que  contengan y_buscado
                    punto_encontrado = False

                    # Verificar si y_buscado es exactamente igual a algún   punto
                    for i, yi in enumerate(y):
                        if abs(yi - y_buscado) < 1e-10:  # Tolerancia para  comparación de flotantes
                            x_interpolado = x[i]
                            procedimiento = [
                                f"Datos: x = {x_valores}, y = {y_valores}",
                                f"Punto buscado: y = {y_buscado}",
                                f"El punto y = {y_buscado} existe   exactamente en los datos",
                                f"Resultado: x = {x_interpolado}"
                            ]
                            punto_encontrado = True
                            break
                        
                    if not punto_encontrado:
                        # Buscar el par de puntos adecuado para     interpolación
                        mejor_par = None
                        menor_distancia = float('inf')

                        for i in range(len(y)):
                            for j in range(i+1, len(y)):
                                y1, y2 = y[i], y[j]
                                # Verificar si y_buscado está entre y1 e y2
                                if (y1 <= y_buscado <= y2) or (y2 <=    y_buscado <= y1):
                                    distancia = abs(y1 - y2)
                                    if distancia < menor_distancia and  distancia > 1e-10:
                                        menor_distancia = distancia
                                        mejor_par = (i, j)

                        if mejor_par is None:
                            raise ValueError(f"No se puede interpolar: el   valor y = {y_buscado} no está en el rango de  los datos")

                        i, j = mejor_par
                        x1, x2 = x[i], x[j]
                        y1, y2 = y[i], y[j]

                        # Evitar división por cero
                        if abs(y2 - y1) < 1e-10:
                            raise ValueError("División por cero: los    valores de Y son iguales")

                        x_interpolado = x1 + (y_buscado - y1) * (x2 - x1) /     (y2 - y1)

                        procedimiento = [
                            f"Puntos cercanos: (x1, y1) = ({x1}, {y1}),     (x2, y2) = ({x2}, {y2})",
                            f"Fórmula: x = x1 + (y - y1) * (x2 - x1) / (y2  - y1)",
                            f"Cálculo: x = {x1} + ({y_buscado} - {y1}) *    ({x2} - {x1}) / ({y2} - {y1})",
                            f"Resultado de interpolación: x =   {x_interpolado:.6f}"
                        ]

                resultado = x_interpolado
            
            # GUARDAR EN BASE DE DATOS - CÁLCULO EXITOSO
            try:
                print("Iniciando guardado en BD...")  # Debug
                
                # Buscar o crear categoria
                categoria, _ = categoria_metodo.objects.get_or_create(
                    nombre = "Interpolación linear",
                    descripcion = "Método numérico para resolver problemas de interpolación linear"
                )

                # Buscar o crear el método Runge-Kutta
                metodo, creado = metodo_numerico.objects.get_or_create(
                    nombre='Interpolación linear',
                    defaults={
                        'id_categoria_id': 4,  
                        'documentacion': documentacion['definicion']
                    }
                )
                
                print(f"Método obtenido/creado: {metodo.nombre}, Creado: {creado}")  # Debug

                # Crear el registro del cálculo
                if encontrar == 'x':
                    objCalculo = calculo.objects.create(
                        id_usuario = objUsuario,
                        id_metodo = metodo,
                        parametros_entrada = parametros_entrada,
                        resultado = {'x interpolado': resultado},
                        procedimiento = procedimiento
                    )
                else:
                    objCalculo = calculo.objects.create(
                        id_usuario = objUsuario,
                        id_metodo = metodo,
                        parametros_entrada = parametros_entrada,
                        resultado = {'y interpolado': resultado},
                        procedimiento = procedimiento
                    )

                print(f'Cálculo guardado exitosamente con ID: {objCalculo.id}')
                messages.success(request, 'Cálculo realizado y guardado correctamente')

            except Exception as db_error:
                messages.warning(request, f'Cálculo realizado correctamente, pero no se pudo guardar en la base de datos: {str(db_error)}')
                print(f"Error de BD en cálculo exitoso: {db_error}")

        except Exception as e:
            # Error en el cálculo principal
            mensajeError = f"Error en el cálculo: {str(e)}"
            print(f"Error en cálculo: {e}")
            
            # Intentar guardar el error en la BD
            try:
                print("Guardando error en BD...")  # Debug
                
                metodo, _ = metodo_numerico.objects.get_or_create(
                    nombre = 'Interpolación linear',
                    defaults ={
                        'id_categoria_id': 4,
                        'documentacion': documentacion['definicion']
                    }
                )

                objCalculo = calculo.objects.create(
                    id_usuario = objUsuario,
                    id_metodo = metodo,
                    parametros_entrada = parametros_entrada,
                    resultado=None,
                    procedimiento=None,
                    mensaje_error=mensajeError
                )
                print(f'Error guardado en BD con ID: {objCalculo.id}')
                
            except Exception as db_error_2:
                print(f'Error al guardar error en BD: {db_error_2}')

        except ValueError as e:
            mensajeError = f"Error de validación: {str(e)}"
            print(f"ValueError: {e}")
        except ZeroDivisionError as e:
            mensajeError = "Error: División por cero en el cálculo de   interpolación"
            print(f"ZeroDivisionError: {e}")
        except Exception as e:
            mensajeError = f"Error inesperado: {str(e)}"
            print(f"Unexpected error: {e}")

        template_vars = {
            'resultado': resultado,
            'procedimiento': procedimiento,
            'error': mensajeError,
            'x_valores': x_valores_str,
            'y_valores': y_valores_str,
            'x_buscado': x_buscado,
            'y_buscado': y_buscado,
            'documentacion': documentacion
        }

        return render(request, 'calculadora/interpolacionLinear.html', template_vars)
    else:
        return render(request, 'calculadora/interpolacionLinear.html',{
            'documentacion': documentacion
        })
 
# Función que renderiza la interfaz y realiza el cálculo para el método de interpolación de lagrange
def interpolacionMejorado(request):
    # DOCUMENTACIÓN del método
    documentacion = {
        'nombre': 'Método de Interpolación mejorado',
        'definicion': 'La interpolación mejorada es una técnica fundamental en matemáticas y ciencias aplicadas que permite estimar valores intermedios dentro del rango de un conjunto de datos discretos. Entre los métodos más reconocidos se encuentran la interpolación de Lagrange y la interpolación de Newton.',
        'formulas': '''
            <p>
                La <strong>Interpolación de Langrage</strong> es un método que construye un polinomio que pasa por todos los puntos dados sin necesidad de calcular diferencias divididas. <strong>La fórmula del polinomio de Lagrange de grado n es:</strong>
                <br>
                P(x) = \sum _{n=0}^{\infty }\: f(xj) * Lj(x)
                <br>
                Donde Lj(x) son los polinomios básicos de Lagrange definidos como:
                <br>
                Lj(x) = \prod _{i=0}^n\left, i\nej (x-xi)/(xj-xi)
                <br>
                Aquí, xj son los puntos conocidos y f(xj) sus correspondientes valores.
            </p>   
            <p> 
                La <strong>Interpolación de Newton</strong> utiliza diferencias divididas para construir el polinomio interpolante. <strong>La fórmula general es:</strong>
                <br>
                P(x) = f[x0] + (x-x0)f[x0,x1] + (x-x0)(x-x1)f[x0,x1,x2] + ...
                <br>
                Donde f[x0,x1,...,xk] son las diferencias divididas calculadasa a partir de los valores conocidos.
            <p>
        ''',
        'usos': '''
            <p>
                El método de interpolación tiene diversas aplicaciones en el campo matemático, tales como:
            </p>
            <ul>
                <li><strong>La interpolación de Lagrange</strong> es útil en situaciones donde se requiere estimar valores intermedios de una función desconocida basándose en un conjunto discreto de datos. Se aplica en áreas como la ingeniería, la física y la economía para aproximar funciones complejas o cuando los datos experimentales son limitados</li>
                <li><strong>La interpolación de Newton es adecuada</strong> cuando los datos están tabulados con intervalos iguales o desiguales y se requiere eficiencia en el cálculo al incorporar nuevos puntos. Es ampliamente utilizado en análisis numérico y aplicaciones científicas.</li>
            </ul>
            <p><strong style="color: red">Nota:</strong> En esta sección solo se calcula la interpolación de Lagrange.</p>
        '''
    }

    # CÁLCULO del método
    # Declararación de variables
    resultado = None
    mensajeError = None
    procedimiento = ["Cálculo del polinomio de Lagrange:"]
    grafica_url = None

    if request.method == 'POST':
        try:
            # Obtener datos del formulario
            encontrar = request.POST.get('encontrar')
            x_valores_str = request.POST.get('x_valores', '').strip()
            y_valores_str = request.POST.get('y_valores', '').strip()
            # Asumir x, y en 0 si no se envian
            try:
                x_buscado = float(request.POST.get('x_buscado', ''))
            except:
                x_buscado = float(0)
            try:
                y_buscado = float(request.POST.get('y_buscado', ''))
            except:
                y_buscado = float(0)

            # Crear una Instancia del usuario
            objUsuario = usuario.objects.get(id=request.session["usuario_id"])

            # Preparar parametros de entrada para guardar
            parametros_entrada = {
                "valores de x": x_valores_str,
                "valores de y": y_valores_str,
                "x buscado": x_buscado,
                "y buscado": y_buscado,
                "encontrar": encontrar,
            }

            # Validar inputs no vacios
            if not all([x_valores_str, y_valores_str, encontrar]):
                raise ValueError("Todos los campos son obligatorios")
            
            # Convertir strings a listas de floats
            x_valores = [float(x) for x in x_valores_str.split(',')]
            y_valores = [float(y) for y in y_valores_str.split(',')]

            # Validaciones matemáticas
            if len(x_valores) < 2 or len(y_valores) < 2:
                raise ValueError("Se necesitan al menos 2 puntos para   realizar la interpolación")

            if len(x_valores) > 20 or len(y_valores) > 20:
                raise ValueError("Máximo 20 puntos permitidos para evitar inestabilidad numérica")

            if len(x_valores) != len(y_valores):
                raise ValueError("La cantidad de valores x e y debe ser la misma")
            
            if encontrar == 'y' and len(set(x_valores)) != len(x_valores):
                raise ValueError("Los valores de X no pueden tener  duplicados para interpolar Y")
            elif encontrar == 'x' and len(set(y_valores)) != len(y_valores):
                raise ValueError("Los valores de Y no pueden tener  duplicados para interpolar X")

            # Verificar que todos los valores sean finitos
            for i, x_val in enumerate(x_valores):
                if not np.isfinite(x_val):
                    raise ValueError(f"El valor x[{i+1}] = {x_val} no es un número finito")

            for i, y_val in enumerate(y_valores):
                if not np.isfinite(y_val):
                    raise ValueError(f"El valor y[{i+1}] = {y_val} no es un número finito")

            if not np.isfinite(x_buscado):
                raise ValueError("El valor a interpolar debe ser un número finito")
            
            if encontrar == 'y':
                # Implementación del método de Lagrange (en función de x)
                n = len(x_valores)
                y_buscado = 0

                # Cálculo del valor interpolado usando polinomios de Lagrange
                for i in range(n):
                    # Calcular el término L_i(x)
                    L_i = 1
                    formula_parts = []

                    for j in range(n):
                        if j != i:
                            L_i *= (x_buscado - x_valores[j]) / (x_valores[i] -  x_valores[j])
                            formula_parts.append(f"(x - {x_valores[j]}) /   ({x_valores[i]} - {x_valores[j]})")

                    term_value = L_i * y_valores[i]
                    y_buscado += term_value

                    # Documentar el proceso
                    procedimiento.append(f"L{i}({x_buscado}) = {' * '.join   (formula_parts)} = {L_i}")
                    procedimiento.append(f"Término {i+1}: {L_i} * {y_valores[i]} = {term_value}")

                procedimiento.append(f"Resultado final: y({x_buscado}) =  {y_buscado}")
                resultado = y_buscado
            else:
                # Implementación del método de Lagrange (en función de y)
                n = len(y_valores)
                x_buscado = 0
                
                # Cálculo del valor interpolado usando polinomios de Lagrange
                for i in range(n):
                    # Calcular el término L_i(y)
                    L_i = 1
                    formula_parts = []
                    
                    for j in range(n):
                        if j != i:
                            L_i *= (y_buscado - y_valores[j]) / (y_valores[i] - y_valores[j])
                    
                    term_value = L_i * x_valores[i]
                    x_buscado += term_value
                    
                    # Documentar el proceso
                    procedimiento.append(f"L{i}({y_buscado}) = {' * '.join(formula_parts)} = {L_i}")

                procedimiento.append(f"Resultado final: x({y_buscado}) = {x_buscado}")
                resultado = x_buscado

            # Crear gráfica
            x_range = np.linspace(min(x_valores) - 1, max(x_valores) + 1, 100)
            y_range = []
            
            for x in x_range:
                y_val = 0
                for i in range(n):
                    L_i = 1
                    for j in range(n):
                        if j != i:
                            L_i *= (x - x_valores[j]) / (x_valores[i] - x_valores[j])
                    y_val += L_i * y_valores[i]
                y_range.append(y_val)
            
            plt.figure(figsize=(10, 6))
            plt.plot(x_range, y_range, 'b-', label='Polinomio interpolador')
            plt.plot(x_valores, y_valores, 'ro', label='Puntos originales')
            plt.plot(x_buscado, y_buscado, 'go', label='Punto interpolado')
            plt.axhline(y=0, color='k', linestyle='-', alpha=0.3)
            plt.axvline(x=0, color='k', linestyle='-', alpha=0.3)
            plt.grid(True, alpha=0.3)
            plt.legend()
            plt.title('Interpolación Polinómica de Lagrange')
            plt.xlabel('x')
            plt.ylabel('y')
            
            # Guardar gráfica como imagen para mostrar en template
            buffer = BytesIO()
            plt.savefig(buffer, format='png')
            buffer.seek(0)
            imagen_png = buffer.getvalue()
            buffer.close()
            grafica_url = base64.b64encode(imagen_png).decode('utf-8')

            try:
                print("Iniciando guardado en BD...")  # Debug

                # Buscar o crear categoria
                categoria, _ = categoria_metodo.objects.get_or_create(
                    nombre = "Interpolación de Lagrange",
                    descripcion = "Método numérico que utiliza polinomios de Lagrange para interpolar valores entre un conjunto de datos conociddos"
                )

                # Buscar o crear el método de Lagrange
                metodo, creado = metodo_numerico.objects.get_or_create(
                    nombre='Interpolación de Lagrange',
                    defaults={
                        'id_categoria_id': 3,  
                        'documentacion': documentacion['definicion']
                    }
                )

                print(f"Método obtenido/creado: {metodo.nombre}, Creado: {creado}")  # Debug

                # Crear el registro del cálculo
                objCalculo = calculo.objects.create(
                    id_usuario = objUsuario,
                    id_metodo = metodo,
                    parametros_entrada = parametros_entrada,
                    resultado = {'y interpolado': resultado},
                    procedimiento = procedimiento
                )

                print(f'Cálculo guardado exitosamente con ID: {objCalculo.id}')
                messages.success(request, 'Cálculo realizado y guardado correctamente')

            except Exception as db_error:
                messages.warning(request, f'Cálculo realizado correctamente, pero no se pudo guardar en la base de datos: {str(db_error)}')
                print(f"Error de BD en cálculo exitoso: {db_error}")
            
        except Exception as e:
            # Error en el cálculo principal
            mensajeError = f"Error en el cálculo: {str(e)}"
            print(f"Error en cálculo: {e}")

            # Intentar guardar el error en la BD
            try:
                print("Guardando error en BD...")  # Debug
                
                metodo, _ = metodo_numerico.objects.get_or_create(
                    nombre = 'Interpolación de Lagrange',
                    defaults ={
                        'id_categoria_id': 3,
                        'documentacion': documentacion['definicion']
                    }
                )

                objCalculo = calculo.objects.create(
                    id_usuario = objUsuario,
                    id_metodo = metodo,
                    parametros_entrada = parametros_entrada,
                    resultado = None,
                    procedimiento = None,
                    mensaje_error = mensajeError
                )
                print(f'Error guardado en BD con ID: {objCalculo.id}')
                
            except Exception as db_error_2:
                print(f'Error al guardar error en BD: {db_error_2}')

        template_values = {
            'resultado': resultado,
            'procedimiento': procedimiento,
            'error': mensajeError,
            'x_valores': x_valores_str,
            'y_valores': y_valores_str,
            'x_buscado': x_buscado,
            'y_buscado': y_buscado,
            'documentacion': documentacion,
            'grafica_url': grafica_url
        }

        return render(request, 'calculadora/interpolacionLagrange.html', template_values)
    else:
        return render(request, 'calculadora/interpolacionLagrange.html',{
            'documentacion': documentacion
        })
 
# Función que renderiza la interfaz y realiza el cálculo para el método de newton raphson
def newtonRaphson(request):
    # DOCUMENTACIÓN del método
    documentacion = {
        'nombre': 'Método de Newton Raphson',
        'definicion': 'El método de Newton-Raphson es un algoritmo iterativo ampliamente utilizado en el análisis numérico para la localización de raíces de funciones reales o complejas. Este método se basa en la aproximación lineal de una función a través de la tangente en un punto de partida, obteniendo iterativamente una mejor aproximación de la raíz.',
        'formulas': '''
            <p>
                El método se fundamenta en el desarrollo en serie de Taylor de una función  f(x) alrededor de un punto :
                <br>
                f(x) ≈ f(xn) + f'(xn)(x-xn)
            </p> 
            <p> 
                La aproximación lineal o la recta tangente en el punto (xn, f(xn)) se utiliza para encontrar una nueva aproximación xn+1 mediante la intersección de dicha tangente con el eje x. Esto conduce a la <strong>fórmula iterativa</strong>: 
                <br>
                xn+1 = xn - f(xn)/f'(xn)
            </p>
            <p>
                Geométricamente, en cada iteración se traza la recta tangente a la curva y = f(x) en el punto (xn, f(xn)) y se toma la abscisa del punto de corte de dicha tangente con el eje x como la siguiente aproximación xn+1.  
            </p>
            
        ''',
        'usos': '''
            <p>
                Este métdo se aplica en campos como:
            </p>
            <ul>
                <li>Resolución de ecuaciones no lineales: Es útil para encontrar raíces de ecuaciones donde los métodos analíticos son complicados o inviables.</li>
                <li>Optimización: Al localizar ceros de la derivada de una función, ayuda a identificar máximos y mínimos locales.</li>
                <li>Cálculo de valores propios: Se emplea en álgebra lineal para determinar valores propios de matrices.</li>
            </ul>
        '''
    }

    # CÁLCULO del método
    # Declararación de variables
    resultado = None
    mensajeError = None
    procedimiento = []
    iteraciones = []
    grafica = None

    if request.method == 'POST':
        try:
            # Obtener datos del formulario
            funcion_texto = request.POST.get('funcion', '').strip()
            x_inicial = float(request.POST.get('x0', ''))
            tolerancia = float(request.POST.get('tolerancia', ''))
            max_iteraciones = int(request.POST.get('max_iteraciones', ''))

            # Validaciones de parámetros
            if tolerancia <= 0:
                raise ValueError("La tolerancia debe ser un número positivo")
        
            if tolerancia > 1:
                raise ValueError("La tolerancia debe ser menor a 1")

            if tolerancia < 1e-15:
                raise ValueError("La tolerancia es demasiado pequeña (menor     a 1e-15)")

            if max_iteraciones <= 0:
                raise ValueError("El número máximo de iteraciones debe ser  positivo")

            if max_iteraciones > 10000:
                raise ValueError("Número máximo de iteraciones excesivo     (máximo: 10000)")

            if not math.isfinite(x_inicial):
                raise ValueError("El valor inicial debe ser un número finito")
            
            # Límite seguridad para el valor inicial
            if abs(x_inicial) > 1e10:
                raise ValueError("El valor inicial es demasiado grande")

            # Crear una Instancia del usuario
            objUsuario = usuario.objects.get(id = request.session["usuario_id"])

            # Preparar parámetros de entreada para guardar
            parametros_entrada = {
                'ecuacion': funcion_texto,
                'x0': x_inicial,
                'tolerancia': tolerancia,
                'iteraciones máximas': max_iteraciones
            }
            
            # Validar y procesar la función
            try:
                x = sp.Symbol('x')

                # Verificar que la función sea válida sintácticamente
                funcion = sp.sympify(funcion_texto)

                # Verificar que la expresión contenga solo la variable x
                variables_expr = funcion.free_symbols
                variables_permitidas = {x}
                variables_no_permitidas = variables_expr -  variables_permitidas

                if variables_no_permitidas:
                    raise ValueError(f"Variables no permitidas en la    función: {variables_no_permitidas}")

                # Calcular la derivada simbólicamente
                derivada = sp.diff(funcion, x)

                # Verificar que la derivada no sea identicamente cero
                if derivada == 0:
                    raise ValueError("La derivada de la función es  identicamente cero. No se puede aplicar Newton-Raphson")

                # Convertir función y derivada a funciones numéricas
                f = sp.lambdify(x, funcion, 'numpy')
                df = sp.lambdify(x, derivada, 'numpy')

                # Prueba inicial de las funciones
                try:
                    f_test = f(x_inicial)
                    df_test = df(x_inicial)

                    if not np.isfinite(f_test):
                        raise ValueError("La función no está definida en el     punto inicial")

                    if not np.isfinite(df_test):
                        raise ValueError("La derivada no está definida en   el punto inicial")

                except (ZeroDivisionError, OverflowError, RuntimeWarning):
                    raise ValueError("La función o su derivada no están     definidas en el punto inicial")

            except sp.SympifyError:
                raise ValueError("Función matemática inválida.  Verifique la     sintaxis")
            except Exception as func_error:
                raise ValueError(f"Error en la función: {str(func_error)}")
            
            # Implementación del método de Newton-Raphson
            xi = x_inicial
            iteracion = 0
            error = float('inf')
            
            while error > tolerancia and iteracion < max_iteraciones:
                # Calcular f(xi) y f'(xi)
                f_xi = f(xi)
                df_xi = df(xi)
                
                # Verificar si la derivada es cercana a cero
                if abs(df_xi) < 1e-10:
                    mensajeError = "La derivada es cercana a cero. El método no puede continuar."
                    break
                
                # Actualizar x usando la fórmula de Newton-Raphson
                x_siguiente = xi - f_xi / df_xi
                
                # Calcular el error
                error = abs(x_siguiente - xi)
                
                # Guardar información de la iteración
                iteraciones.append({
                    'iteracion': iteracion + 1,
                    'xi': xi,
                    'f(xi)': f_xi,
                    'df(xi)': df_xi,
                    'x siguiente': x_siguiente,
                    'error': error
                })
                
                # Actualizar xi para la próxima iteración
                xi = x_siguiente
                iteracion += 1
            
            # Verificar resultado
            if iteracion >= max_iteraciones and error > tolerancia:
                procedimiento.append(f"El método alcanzó el máximo de iteraciones ({max_iteraciones}) sin converger.")
                resultado = xi
            else:
                procedimiento.append(f"El método convergió en {iteracion} iteraciones.")
                resultado = xi
                
                # Verificar que realmente sea una raíz
                valor_funcion = f(resultado)
                if abs(valor_funcion) > tolerancia:
                    procedimiento.append(f"Advertencia: f({resultado}) = {valor_funcion}, que no es exactamente cero.")
                else:
                    procedimiento.append(f"Verificación: f({resultado}) = {valor_funcion} ≈ 0")

            try:
                print("Guardando en bd")
                # Buscar o crear categoria
                categoria, _ = categoria_metodo.objects.get_or_create(
                    nombre = "Newton Raphson",
                    descripcion = "Método numérico para encontrar el valor de raíces en ecuaciones reales no lineales"
                )

                # Buscar o crear el método
                metodo, creado = metodo_numerico.objects.get_or_create(
                    nombre = "Newton Raphson",
                    defaults={
                        'id_categoria_id': 2,  
                        'documentacion': documentacion['definicion']
                    }
                )

                print(f"Método obtenido/creado: {metodo.nombre}, Creado: {creado}")  # Debug

                # Crear registro del cálculo
                objCalculo = calculo.objects.create(
                    id_usuario = objUsuario,
                    id_metodo = metodo,
                    parametros_entrada = parametros_entrada,
                    resultado ={
                        'raiz aproximada': float(resultado),
                    },
                    procedimiento = {
                        'procedimiento': procedimiento,
                        'iteraciones': iteraciones
                    }
                )

                print(f'Cálculo guardado exitosamente con ID: {objCalculo.id}')
                messages.success(request, 'Cálculo realizado y guardado correctamente')
            
            except Exception as db_error:
                messages.warning(request, f'Cálculo realizado correctamente, pero no se pudo guardar en la base de datos: {str(db_error)}')
                print(f"Error de BD en cálculo exitoso: {db_error}")

        except Exception as e:
            # Error en el cálculo principal
            mensajeError = f"Error en el cálculo: {str(e)}"
            print(f"Error en cálculo: {e}")

            # Intentar guardar el error en la BD
            try:
                print("Guardando error en BD...")  # Debug
                
                metodo, _ = metodo_numerico.objects.get_or_create(
                    nombre = 'Newton Raphson',
                    defaults ={
                        'id_categoria_id': 2,
                        'documentacion': documentacion['definicion']
                    }
                )

                objCalculo = calculo.objects.create(
                    id_usuario = objUsuario,
                    id_metodo = metodo,
                    parametros_entrada = parametros_entrada,
                    resultado = None,
                    procedimiento = None,
                    mensaje_error = mensajeError
                )
                print(f'Error guardado en BD con ID: {objCalculo.id}')
                
            except Exception as db_error_2:
                print(f'Error al guardar error en BD: {db_error_2}')

        template_vars = {
            'resultado': resultado,
            'procedimiento': procedimiento,
            'iteraciones': iteraciones,
            'error': mensajeError,
            'funcion': funcion_texto,
            'x0': x_inicial,
            'tolerancia': tolerancia,
            'max_iteraciones': max_iteraciones,
            'documentacion': documentacion,
            'grafica_url': grafica
        }

        return render(request, 'calculadora/newtonRaphson.html', template_vars)
    else:
        return render(request, 'calculadora/newtonRaphson.html',{
            'documentacion': documentacion
        })
 
# Función que renderiza la interfaz y realiza el cálculo para el método de runge kutta (2do orden)
def rungeKutta(request):
    # DOCUMENTACIÓN del método
    documentacion = {
        'nombre': 'Métodos de Runge-Kutta',
        'definicion': '​El método de Runge-Kutta es una familia de técnicas numéricas utilizadas para resolver ecuaciones diferenciales ordinarias (EDOs). Estas técnicas permiten aproximar la solución de una EDO sin necesidad de conocer su solución analítica, lo que es especialmente útil en problemas donde dicha solución es difícil o imposible de obtener.',
        'formulas': '''
            <p>
                Los métodos de Runge-Kutta se basan en la evaluación de la función derivada en varios puntos dentro de un intervalo para calcular una estimación más precisa de la solución. El método clásico de cuarto orden (RK4) es uno de los más utilizados debido a su equilibrio entre precisión y eficiencia computacional. Para una EDO de la forma y' = f(t,y) con condición inicial y(t0) = y0. Existen cuatro ordenes de este método:
                <br>
                <ul>
                    <li>k1 = f(tn,yn)</li>
                    <li>k2 = f(tn + h/2, yn + h/2 * k1)</li>
                    <li>k3 = f(tn + h/2, yn + h/2 * k2)</li>
                    <li>k4 = f(tn + h, yn + hk3)</li>
                </ul>
            </p> 
            <p> 
                Donde: 
                <br>
                h es el tamaño del paso.
                <br>
                k1, k2, k3, k4 son las pendientes internasa que se calculan evaluando la función f en los puntos específicos dentro del intervalo.
            </p>
            
        ''',
        'usos': '''
            <p>
                El método de Runge-Kutta se aplica ampliamente en diversas áreas de las matemáticas y la ingeniería para resolver EDOs que modelan fenómenos complejos. Algunos ejemplos incluyen::
            </p>
            <ul>
                <li>Dinámica de poblaciones: Modelos que describen el crecimiento de una población, como una colonia de bacterias, pueden representarse mediante EDOs. El método de Runge-Kutta permite predecir el tamaño de la población en momentos futuros basándose en tasas de crecimiento específicas.</li>
                <li>Ingeniería eléctrica: Se emplea en el análisis dinámico y de estabilidad de sistemas de potencia, permitiendo simular el comportamiento de redes eléctricas durante transitorios y evaluar su respuesta ante diferentes condiciones operativas. ​</li>
                <li>Ingeniería mecánica: Es útil en la resolución de problemas dinámicos no lineales, como el estudio de sistemas de vibraciones, amortiguadores y otros componentes mecánicos que pueden modelarse mediante EDOs complejas.</li>
            </ul>
        '''
    }

    # CÁLCULO del método RK4
    # Declararación de variables
    resultado = None
    mensajeError = None
    procedimiento = []
    iteraciones = []
    grafica = None

    if request.method == 'POST':
        try:
            # Obtener datos del formulario con validaciones iniciales
            ecuacion = request.POST.get('ecuacion', '').strip()
            
            # Validar que todos los campos estén presentes
            if not all([ecuacion, request.POST.get('x0'), request.POST.get('y0'), 
                       request.POST.get('h'), request.POST.get('x_final')]):
                raise ValueError("Todos los campos son obligatorios")
            
            # Conversiones con manejo de errores
            try:
                x0 = float(request.POST.get('x0', ''))
                y0 = float(request.POST.get('y0', ''))
                h = float(request.POST.get('h', ''))
                xf = float(request.POST.get('x_final', ''))
            except (ValueError, TypeError):
                raise ValueError("Los valores numéricos deben ser números válidos")
            
            # Validaciones matemáticas
            if h == 0:
                raise ValueError("El paso h no puede ser cero")
            
            if h < 0 and xf > x0:
                raise ValueError("Para xf > x0, el paso h debe ser positivo")
            
            if h > 0 and xf < x0:
                raise ValueError("Para xf < x0, el paso h debe ser negativo")
            
            if abs(h) < 1e-10:
                raise ValueError("El paso h es demasiado pequeño (menor a 1e-10)")
            
            if abs(h) > abs(xf - x0):
                raise ValueError("El paso h es mayor que el rango de integración")
            
            # Calcular número de pasos y validar
            n_pasos = int(abs((xf - x0) / h))
            
            # Límite de seguridad para evitar cálculos infinitos
            MAX_PASOS = 10000
            if n_pasos > MAX_PASOS:
                raise ValueError(f"Número de pasos excesivo ({n_pasos}). Máximo permitido: {MAX_PASOS}")
            
            if n_pasos <= 0:
                raise ValueError("El número de pasos debe ser mayor a cero")
            
            titulo_calculo = "Runge Kutta 4to orden"
            
            # Crear una Instancia del usuario
            objUsuario = usuario.objects.get(id=request.session["usuario_id"])
            
            # Preparar parámetros de entrada para guardar
            parametros_entrada = {
                'ecuacion': ecuacion,
                'x0': x0,
                'y0': y0,
                'h': h,
                'x final': xf,
                'metodo': 'RK4'
            }
            
            # Validar y procesar la ecuación diferencial
            try:
                x, y = sp.symbols('x y')
                
                # Verificar que la ecuación sea válida sintácticamente
                f_expr = sp.sympify(ecuacion)
                
                # Verificar que la expresión contenga las variables correctas
                variables_expr = f_expr.free_symbols
                variables_permitidas = {x, y}
                variables_no_permitidas = variables_expr - variables_permitidas
                
                if variables_no_permitidas:
                    raise ValueError(f"Variables no permitidas en la ecuación: {variables_no_permitidas}")
                
                # Convertir la expresión simbólica a una función numérica
                f = sp.lambdify((x, y), f_expr, 'numpy')
                
                # Prueba inicial de la función
                try:
                    test_result = f(x0, y0)
                    if not np.isfinite(test_result):
                        raise ValueError("La función no está definida en el punto inicial")
                except (ZeroDivisionError, OverflowError, RuntimeWarning):
                    raise ValueError("La función no está definida en el punto inicial (división por cero o overflow)")
                
            except sp.SympifyError:
                raise ValueError("Ecuación matemática inválida. Verifique la sintaxis")
            except Exception as eq_error:
                raise ValueError(f"Error en la ecuación: {str(eq_error)}")
            
            # Ajustar el paso según la dirección
            h_step = h if xf > x0 else -abs(h)
            
            # Vectores para almacenar resultados
            x_valores = [x0]
            y_valores = [y0]
            
            # Cálculo del método RK4
            x_actual = x0
            y_actual = y0
            
            # Control de overflow y valores inválidos
            MAX_VALUE = 1e10
            MIN_VALUE = -1e10
            
            for i in range(n_pasos):
                try:
                    # Calcular k1, k2, k3 y k4 con validaciones
                    k1 = f(x_actual, y_actual)
                    
                    # Verificar que k1 sea válido
                    if not np.isfinite(k1):
                        raise ValueError(f"k1 no es finito en iteración {i+1}")
                    
                    k2 = f(x_actual + h_step/2, y_actual + h_step/2 * k1)
                    if not np.isfinite(k2):
                        raise ValueError(f"k2 no es finito en iteración {i+1}")
                    
                    k3 = f(x_actual + h_step/2, y_actual + h_step/2 * k2)
                    if not np.isfinite(k3):
                        raise ValueError(f"k3 no es finito en iteración {i+1}")
                    
                    k4 = f(x_actual + h_step, y_actual + h_step * k3)
                    if not np.isfinite(k4):
                        raise ValueError(f"k4 no es finito en iteración {i+1}")
                    
                    # Calcular siguiente valor de y con fórmula RK4
                    y_siguiente = y_actual + (h_step/6) * (k1 + 2*k2 + 2*k3 + k4)
                    
                    # Verificar overflow
                    if not np.isfinite(y_siguiente):
                        raise ValueError(f"Overflow en iteración {i+1}: y = {y_siguiente}")
                    
                    if abs(y_siguiente) > MAX_VALUE:
                        raise ValueError(f"Valor excesivo en iteración {i+1}: |y| > {MAX_VALUE}")
                    
                    # Guardar esta iteración con sus valores k
                    iteraciones.append({
                        'i': i,
                        'x': round(x_actual, 10),
                        'y': round(y_actual, 10),
                        'f': round(f(x_actual, y_actual), 10),
                        'detalles': {
                            'k1': round(k1, 10),
                            'k2': round(k2, 10),
                            'k3': round(k3, 10),
                            'k4': round(k4, 10),
                            'y calculada': f"y_{i+1} = y_{i} + h/6 * (k1 + 2*k2 + 2*k3 + k4) = {y_actual:.6f} + {h_step}/6 * ({k1:.6f} + 2*{k2:.6f} + 2*{k3:.6f} + {k4:.6f})"
                        }
                    })
                    
                    # Actualizar para la siguiente iteración
                    x_actual += h_step
                    x_valores.append(round(x_actual, 10))
                    y_valores.append(round(y_siguiente, 10))
                    y_actual = y_siguiente
                    
                except (ZeroDivisionError, OverflowError, FloatingPointError) as math_error:
                    raise ValueError(f"Error matemático en iteración {i+1}: {str(math_error)}")
                except Exception as iter_error:
                    raise ValueError(f"Error en iteración {i+1}: {str(iter_error)}")
            
            # Resultado final
            resultado = {
                'x_final': f"x = {xf}",
                'y_final': f"y = {y_valores[-1]:.10f}"
            }

            # GUARDAR EN BASE DE DATOS - CÁLCULO EXITOSO
            try:
                print("Iniciando guardado en BD...")  # Debug
                
                # Buscar o crear categoria
                categoria, _ = categoria_metodo.objects.get_or_create(
                    nombre = "Runge-Kutta 4to Orden",
                    descripcion = "Método numérico para resolver ecuaciones diferenciales ordinarias de primer orden mediante la aproximación iterativa de 4to orden"
                )

                # Buscar o crear el método Runge-Kutta
                metodo, creado = metodo_numerico.objects.get_or_create(
                    nombre='Runge-Kutta 4to Orden',
                    defaults={
                        'id_categoria_id': 1,  
                        'documentacion': documentacion['definicion']
                    }
                )
                
                print(f"Método obtenido/creado: {metodo.nombre}, Creado: {creado}")  # Debug

                # Crear el registro del cálculo
                objCalculo = calculo.objects.create(
                    id_usuario = objUsuario,
                    id_metodo = metodo,
                    parametros_entrada = parametros_entrada,
                    resultado ={
                        'x': float(xf),
                        'y': float(y_valores[-1]),
                        'valores de x': [float(x) for x in x_valores],
                        'valores de y': [float(y) for y in y_valores],
                        'número de paso': n_pasos,
                        'paso en h': h,
                        'precisión estimada': abs(h**4)  # Error teórico de RK4
                    },
                    procedimiento = iteraciones
                )

                print(f'Cálculo guardado exitosamente con ID: {objCalculo.id}')
                messages.success(request, 'Cálculo realizado y guardado correctamente')

            except Exception as db_error:
                messages.warning(request, f'Cálculo realizado correctamente, pero no se pudo guardar en la base de datos: {str(db_error)}')
                print(f"Error de BD en cálculo exitoso: {db_error}")

        except Exception as e:
            # Error en el cálculo principal
            mensajeError = f"Error en el cálculo: {str(e)}"
            print(f"Error en cálculo: {e}")
            
            # Intentar guardar el error en la BD
            try:
                print("Guardando error en BD...")  # Debug
                
                metodo, _ = metodo_numerico.objects.get_or_create(
                    nombre = 'Runge-Kutta 4to Orden',
                    defaults ={
                        'id_categoria_id': 1,
                        'documentacion': documentacion['definicion']
                    }
                )

                objCalculo = calculo.objects.create(
                    id_usuario = objUsuario,
                    id_metodo = metodo,
                    parametros_entrada = parametros_entrada,
                    resultado=None,
                    procedimiento=None,
                    mensaje_error=mensajeError
                )
                print(f'Error guardado en BD con ID: {objCalculo.id}')
                
            except Exception as db_error_2:
                print(f'Error al guardar error en BD: {db_error_2}')

        template_vars = {
            'resultado': resultado,
            'procedimiento': procedimiento,
            'error': mensajeError,
            'iteraciones': iteraciones,
            'ecuacion': ecuacion,
            'x0': x0,
            'y0': y0,
            'h': h,
            'x_final': xf,
            'documentacion': documentacion
        }
        return render(request, 'calculadora/rungeKutta.html', template_vars)
    else:
        return render(request, 'calculadora/rungeKutta.html', {
            'documentacion': documentacion
        })

def insertarCategorias(request):
    try:
        nuevaCategoria = categoria_metodo.objects.create(nombre="Interpolación linear", descripcion="Método numérico para resolver problemas de interpolación linear")
        print(f'Categoria agregada con el id: {nuevaCategoria.pk}' )
    except Exception as e:
        print(f'Error {e}')
    return redirect('/')
