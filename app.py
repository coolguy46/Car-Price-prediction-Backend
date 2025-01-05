from flask import Flask, request, jsonify
from flask_cors import CORS # type: ignore
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from car_price_model import CarPriceModel  # Import your model class

app = Flask(__name__)
CORS(app)

# Load the saved model and preprocessing objects
try:
    checkpoint = torch.load('car_price_model.pth')
    model = CarPriceModel(input_dim=3)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    le = checkpoint['label_encoder']
    scaler_X = checkpoint['scaler_X']
    scaler_y = checkpoint['scaler_y']

    # Get and clean car models list
    KNOWN_CARS = sorted(le.classes_.tolist())
    print(f"Loaded {len(KNOWN_CARS)} car models")

except Exception as e:
    print(f"Error loading model: {e}")
    raise

@app.route('/cars', methods=['GET'])
def get_cars():
    search_term = request.args.get('search', '').lower()
    
    if search_term:
        # Filter cars based on search term
        filtered_cars = [car for car in KNOWN_CARS if search_term in car.lower()]
        # Limit to first 100 matches to prevent overwhelming the frontend
        return jsonify({
            'cars': filtered_cars[:100],
            'total': len(filtered_cars)
        })
    else:
        # Return first 100 cars if no search term
        return jsonify({
            'cars': KNOWN_CARS[:100],
            'total': len(KNOWN_CARS)
        })

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.json
        
        # Extract features
        name = data['name']
        
        # Check if car model is known
        if name not in KNOWN_CARS:
            return jsonify({
                'error': f'Unknown car model. Please select from available models.',
                'success': False
            }), 400
            
        year = float(data['year'])
        miles = float(data['miles'])
        
        # Basic validation
        current_year = 2025  # You might want to use current year from system
        if year < 1900 or year > current_year:
            return jsonify({
                'error': f'Year must be between 1900 and {current_year}',
                'success': False
            }), 400
            
        if miles < 0:
            return jsonify({
                'error': 'Mileage cannot be negative',
                'success': False
            }), 400
        
        # Encode and scale input
        name_encoded = le.transform([name])[0]
        input_features = np.array([[name_encoded, year, miles]])
        input_scaled = scaler_X.transform(input_features)
        
        # Make prediction
        with torch.no_grad():
            input_tensor = torch.FloatTensor(input_scaled)
            prediction_scaled = model(input_tensor)
            prediction = scaler_y.inverse_transform(
                prediction_scaled.numpy().reshape(-1, 1)
            )[0][0]
        
        return jsonify({
            'predicted_price': round(float(prediction), 2),
            'success': True
        })
        
    except KeyError as e:
        return jsonify({
            'error': f'Missing required field: {str(e)}',
            'success': False
        }), 400
    except Exception as e:
        return jsonify({
            'error': str(e),
            'success': False
        }), 400

@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({
        'status': 'healthy',
        'model_loaded': model is not None,
        'num_car_models': len(KNOWN_CARS)
    })



# Add these imports to your app.py
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from datetime import datetime

# Add these new endpoints to your Flask app
@app.route('/model-stats', methods=['GET'])
def get_model_stats():
    try:
        # Load your training data
        df = pd.read_csv('C:/Users/ankey/Downloads/archive/carvana.csv')
        
        # Prepare features and target
        X = df[['Name', 'Year', 'Miles']].copy()
        y = df['Price'].values
        
        # Encode car names
        X['Name_encoded'] = le.transform(X['Name'])
        X_prepared = X[['Name_encoded', 'Year', 'Miles']].values
        
        # Scale features and target
        X_scaled = scaler_X.transform(X_prepared)
        y_scaled = scaler_y.transform(y.reshape(-1, 1)).flatten()
        
        # Make predictions
        with torch.no_grad():
            input_tensor = torch.FloatTensor(X_scaled)
            predictions_scaled = model(input_tensor)
            predictions = scaler_y.inverse_transform(
                predictions_scaled.numpy().reshape(-1, 1)
            ).flatten()
        
        # Calculate metrics
        mse = mean_squared_error(y, predictions)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y, predictions)
        r2 = r2_score(y, predictions)
        
        # Calculate price range distribution
        price_ranges = [0, 10000, 20000, 30000, 50000, 100000, float('inf')]
        price_labels = ['0-10k', '10k-20k', '20k-30k', '30k-50k', '50k-100k', '100k+']
        price_dist = pd.cut(y, bins=price_ranges, labels=price_labels).value_counts().sort_index()
        
        # Calculate error distribution
        errors = predictions - y
        error_stats = {
            'mean_error': float(np.mean(errors)),
            'std_error': float(np.std(errors)),
            'error_percentiles': {
                '25th': float(np.percentile(errors, 25)),
                '50th': float(np.percentile(errors, 50)),
                '75th': float(np.percentile(errors, 75))
            }
        }
        
        # Calculate prediction accuracy by price range
        accuracy_by_range = {}
        for i in range(len(price_ranges)-1):
            mask = (y >= price_ranges[i]) & (y < price_ranges[i+1])
            if mask.any():
                range_errors = np.abs(errors[mask])
                accuracy = np.mean(range_errors <= 0.1 * y[mask])  # Within 10% of actual price
                accuracy_by_range[price_labels[i]] = float(accuracy)
        
        # Calculate year-wise performance
        year_performance = {}
        for year in df['Year'].unique():
            mask = df['Year'] == year
            if mask.any():
                year_rmse = np.sqrt(mean_squared_error(y[mask], predictions[mask]))
                year_performance[int(year)] = float(year_rmse)
        
        return jsonify({
            'metrics': {
                'mse': float(mse),
                'rmse': float(rmse),
                'mae': float(mae),
                'r2': float(r2)
            },
            'price_distribution': {
                'labels': price_labels,
                'values': price_dist.tolist()
            },
            'error_distribution': error_stats,
            'accuracy_by_range': accuracy_by_range,
            'year_performance': year_performance,
            'sample_size': len(y),
            'timestamp': datetime.now().isoformat()
        })
        
    except Exception as e:
        return jsonify({
            'error': str(e),
            'success': False
        }), 400

@app.route('/prediction-scatter', methods=['GET'])
def get_scatter_data():
    try:
        # Load your training data
        df = pd.read_csv('C:/Users/ankey/Downloads/archive/carvana.csv')
        
        # Prepare features and target
        X = df[['Name', 'Year', 'Miles']].copy()
        y = df['Price'].values
        
        # Encode car names
        X['Name_encoded'] = le.transform(X['Name'])
        X_prepared = X[['Name_encoded', 'Year', 'Miles']].values
        
        # Scale features and target
        X_scaled = scaler_X.transform(X_prepared)
        
        # Make predictions
        with torch.no_grad():
            input_tensor = torch.FloatTensor(X_scaled)
            predictions_scaled = model(input_tensor)
            predictions = scaler_y.inverse_transform(
                predictions_scaled.numpy().reshape(-1, 1)
            ).flatten()
        
        # Sample 1000 points for scatter plot to avoid overwhelming the frontend
        indices = np.random.choice(len(y), min(1000, len(y)), replace=False)
        
        scatter_data = {
            'actual': y[indices].tolist(),
            'predicted': predictions[indices].tolist(),
            'years': df['Year'].iloc[indices].tolist(),
            'miles': df['Miles'].iloc[indices].tolist()
        }
        
        return jsonify(scatter_data)
        
    except Exception as e:
        return jsonify({
            'error': str(e),
            'success': False
        }), 400
    
if __name__ == '__main__':
    app.run(debug=True, port=5000)