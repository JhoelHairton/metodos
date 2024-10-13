import numpy as np
from scipy.interpolate import lagrange, CubicSpline
from sklearn.metrics import mean_squared_error, r2_score
from numpy.polynomial import Polynomial

# Método de Lagrange
def lagrange_interpolation(x, y):
    if len(x) < 2 or len(y) < 2:
        raise ValueError("Se necesitan al menos 2 puntos de datos para la interpolación de Lagrange")
    
    poly = lagrange(x, y)
    return poly(x)

# Método de Diferencias Divididas de Newton
def newton_interpolation(x, y):
    """
    Implementación optimizada del método de interpolación de Newton.
    Maneja eficientemente conjuntos de datos grandes y evita problemas de división por cero.
    """
    n = len(x)
    if n < 2:
        raise ValueError("Se necesitan al menos 2 puntos de datos para la interpolación de Newton")
    
    coef = np.copy(y)
    
    for j in range(1, n):
        for i in range(n - 1, j - 1, -1):
            # Evitar divisiones por cero o diferencias muy pequeñas
            denominator = (x[i] - x[i - j])
            if abs(denominator) < 1e-10:
                raise ZeroDivisionError(f"División por cero detectada en diferencias divididas de Newton entre x[{i}] y x[{i-j}]")
            coef[i] = (coef[i] - coef[i - 1]) / denominator

    def newton_poly(val):
        result = coef[-1]
        for i in range(n - 2, -1, -1):
            result = result * (val - x[i]) + coef[i]
        return result

    return [newton_poly(val) for val in x]

# Mínimos Cuadrados (Regresión Lineal y Polinómica)
def least_squares(x, y, degree=1):
    if len(x) < 2 or len(y) < 2:
        raise ValueError("Se necesitan al menos 2 puntos de datos para la regresión")
    
    coef = np.polyfit(x, y, degree)
    return np.polyval(coef, x)  # Usamos polyval para evaluar el polinomio

# Spline Cúbico
def cubic_spline(x, y):
    if len(x) < 2:
        raise ValueError("Se necesitan al menos 2 puntos de datos para construir un spline cúbico")
    
    spline = CubicSpline(x, y)
    return spline(x)

# Cálculo del Error Cuadrático Medio (MSE)
def calculate_mse(y_true, y_pred):
    if len(y_true) != len(y_pred):
        raise ValueError("Las longitudes de y_true y y_pred deben coincidir")
    
    return mean_squared_error(y_true, y_pred)

# Cálculo del coeficiente de determinación R²
def calculate_r2(y_true, y_pred):
    if len(y_true) != len(y_pred):
        raise ValueError("Las longitudes de y_true y y_pred deben coincidir")
    
    return r2_score(y_true, y_pred)
