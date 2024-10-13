from flask import Flask, request, jsonify
import pandas as pd
import numpy as np
from models.interpolation_methods import (
    lagrange_interpolation, newton_interpolation,
    least_squares, cubic_spline, calculate_mse, calculate_r2
)
from flask_cors import CORS  # Importar CORS

app = Flask(__name__)
CORS(app)  # Habilitar CORS para todas las rutas

# Variable para almacenar los datos cargados
uploaded_data = None

@app.route('/upload', methods=['POST'])
def upload_file():
    global uploaded_data
    if 'file' not in request.files:
        return jsonify({"error": "No file uploaded"}), 400
    
    file = request.files['file']

    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400

    try:
        # Leer el archivo CSV usando el separador correcto
        uploaded_data = pd.read_csv(file, sep=';')  # Ajustar el separador si es necesario

        # Limpiar los nombres de las columnas (eliminar espacios en blanco)
        uploaded_data.columns = uploaded_data.columns.str.strip()

        # Verificar si las columnas necesarias están presentes
        if 'MES_FACTURACION' not in uploaded_data.columns or 'PROMEDIO_CONSUMO' not in uploaded_data.columns:
            return jsonify({"error": "Archivo CSV no tiene las columnas necesarias"}), 400

        return jsonify({"message": "Datos cargados correctamente"}), 200

    except Exception as e:
        return jsonify({"error": f"Error al procesar el archivo: {str(e)}"}), 500

@app.route('/compare', methods=['POST'])
def compare_methods():
    if uploaded_data is None:
        return jsonify({"error": "No se han cargado datos aún"}), 400

    methods = request.json.get('methods')
    num_predictions = request.json.get('num_predictions', 0)

    x = uploaded_data['MES_FACTURACION'].values
    y = uploaded_data['PROMEDIO_CONSUMO'].values

    # Convertir a tipo numérico si es necesario
    x = pd.to_numeric(x, errors='coerce')
    y = pd.to_numeric(y, errors='coerce')

    # Eliminar valores nulos
    x = x[~np.isnan(x)]
    y = y[~np.isnan(y)]

    # Verificar si hay suficientes datos
    if len(x) < 1000 or len(y) < 1000:
        return jsonify({"error": "Se necesitan al menos 1000 puntos de datos"}), 400

    # Tomar solo los primeros 1000 datos para probar
    x_sample = x[:1000]
    y_sample = y[:1000]

    # Generar valores futuros si se solicitan
    future_x = pd.Series(range(x_sample.max() + 1, x_sample.max() + num_predictions + 1)) if num_predictions > 0 else None

    results = []
    for method in methods:
        if method == 'lagrange':
            historical_result = lagrange_interpolation(x_sample, y_sample)
            future_result = lagrange_interpolation(future_x, y_sample) if future_x is not None else []
        elif method == 'newton':
            historical_result = newton_interpolation(x_sample, y_sample)
            future_result = newton_interpolation(future_x, y_sample) if future_x is not None else []
        elif method == 'least_squares_linear':
            historical_result = least_squares(x_sample, y_sample, degree=1)
            future_result = least_squares(future_x, y_sample, degree=1) if future_x is not None else []
        elif method == 'least_squares_poly':
            historical_result = least_squares(x_sample, y_sample, degree=2)
            future_result = least_squares(future_x, y_sample, degree=2) if future_x is not None else []
        elif method == 'cubic_spline':
            historical_result = cubic_spline(x_sample, y_sample)
            future_result = cubic_spline(future_x, y_sample) if future_x is not None else []

        results.append({
            "historical": historical_result.tolist(),
            "future": future_result.tolist(),
        })

    return jsonify({
        "historical_x": x_sample.tolist(),
        "future_x": future_x.tolist() if future_x is not None else [],
        "results": results
    }), 200

@app.route('/validate', methods=['POST'])
def validate_methods():
    if uploaded_data is None:
        return jsonify({"error": "No se han cargado datos aún"}), 400

    methods = request.json.get('methods')

    x = uploaded_data['MES_FACTURACION'].values
    y_true = uploaded_data['PROMEDIO_CONSUMO'].values

    # Convertir a tipo numérico si es necesario
    x = pd.to_numeric(x, errors='coerce')
    y_true = pd.to_numeric(y_true, errors='coerce')

    # Eliminar valores nulos
    x = x[~np.isnan(x)]
    y_true = y_true[~np.isnan(y_true)]

    # Tomar solo los primeros 1000 datos para probar
    x_sample = x[:1000]
    y_sample = y_true[:1000]

    results = []
    for method in methods:
        if method == 'lagrange':
            y_pred = lagrange_interpolation(x_sample, y_sample)
        elif method == 'newton':
            y_pred = newton_interpolation(x_sample, y_sample)
        elif method == 'least_squares_linear':
            y_pred = least_squares(x_sample, y_sample, degree=1)
        elif method == 'least_squares_poly':
            y_pred = least_squares(x_sample, y_sample, degree=2)
        elif method == 'cubic_spline':
            y_pred = cubic_spline(x_sample, y_sample)

        # Cálculo de las métricas de precisión
        mse = calculate_mse(y_sample, y_pred)
        r2 = calculate_r2(y_sample, y_pred)

        results.append({
            "method": method,
            "mse": mse,
            "r2": r2
        })

    return jsonify({"results": results}), 200

if __name__ == '__main__':
    app.run(debug=True)
