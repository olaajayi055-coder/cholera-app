import numpy as np
import pandas as pd
from flask import Flask, render_template, request, jsonify
from sklearn.ensemble import VotingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
import joblib
import os
import datetime

app = Flask(__name__)

# --- CONFIGURATION ---
# Ensure the models directory exists (Critical for cloud deployment)
MODEL_DIR = 'models'
os.makedirs(MODEL_DIR, exist_ok=True)

MODEL_PATH = os.path.join(MODEL_DIR, 'cholera_ensemble.pkl')
SCALER_PATH = os.path.join(MODEL_DIR, 'scaler.pkl')
DATA_PATH = 'cholera_dataset.csv'

# Global variables
scaler = None
ensemble_model = None

# --- 1. DATA LOADING FUNCTIONS ---

def generate_simulated_data():
    """Generates realistic simulated data if CSV is missing."""
    print("⚠️ Generating simulated dataset...")
    np.random.seed(42)
    n_samples = 600
    
    age = np.random.randint(1, 80, n_samples)
    fever = np.random.randint(1, 11, n_samples)
    dehydration = np.random.randint(1, 11, n_samples)
    water_quality = np.random.randint(1, 6, n_samples)
    sanitation = np.random.randint(1, 6, n_samples)
    
    # Realistic logic: High fever + dehydration + poor water/sanitation = High Risk
    risk_score = (fever * 0.35) + (dehydration * 0.45) + (water_quality * 0.25) - (sanitation * 0.15)
    target = (risk_score > 5.5).astype(int)
    
    # Add noise (5% flip)
    noise_indices = np.random.choice(n_samples, size=int(n_samples * 0.05), replace=False)
    target[noise_indices] = 1 - target[noise_indices]
    
    X = np.column_stack((age, fever, dehydration, water_quality, sanitation)) 
    y = target
    
    # Save generated data to CSV for transparency
    df = pd.DataFrame({
        'age': age, 'fever': fever, 'dehydration': dehydration,
        'water_quality': water_quality, 'sanitation': sanitation,
        'cholera_positive': target
    })
    df.to_csv(DATA_PATH, index=False)
    print(f"✅ Simulated data saved to {DATA_PATH}")
    
    return X, y

def load_real_data():
    """Loads data from CSV or generates if missing."""
    if os.path.exists(DATA_PATH):
        try:
            print(f"📂 Loading data from {DATA_PATH}...")
            df = pd.read_csv(DATA_PATH)
            
            required_cols = ['age', 'fever', 'dehydration', 'water_quality', 'sanitation', 'cholera_positive']
            if not all(col in df.columns for col in required_cols):
                raise ValueError("Missing columns in CSV")
            
            X = df[['age', 'fever', 'dehydration', 'water_quality', 'sanitation']].values
            y = df['cholera_positive'].values
            print(f"✅ Data loaded successfully. Shape: {X.shape}")
            return X, y
        except Exception as e:
            print(f"❌ Error reading CSV ({e}). Regenerating data...")
            return generate_simulated_data()
    else:
        print("📄 CSV not found. Generating new dataset...")
        return generate_simulated_data()

# --- 2. MODEL TRAINING & LOADING ---

def train_and_save_model():
    """Trains the model and saves it to disk."""
    global scaler, ensemble_model
    
    print("\n--- 🚀 Starting Model Training ---")
    X, y = load_real_data()
    
    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Define Ensemble
    rf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)    lr = LogisticRegression(max_iter=1000, random_state=42)
    svc = SVC(kernel='rbf', probability=True, random_state=42)
    
    ensemble_model = VotingClassifier(
        estimators=[('rf', rf), ('lr', lr), ('svc', svc)],
        voting='soft'
    )
    
    print("🧠 Training ensemble (Random Forest + Logistic Regression + SVM)...")
    ensemble_model.fit(X_scaled, y)
    
    # Evaluate
    preds = ensemble_model.predict(X_scaled)
    acc = accuracy_score(y, preds)
    print(f"✅ Training complete. Accuracy: {acc:.2f}")
    
    # Save models
    print(f"💾 Saving models to {MODEL_DIR}...")
    joblib.dump(ensemble_model, MODEL_PATH)
    joblib.dump(scaler, SCALER_PATH)
    
    # Save a timestamp file to track when it was trained
    with open(os.path.join(MODEL_DIR, 'last_trained.txt'), 'w') as f:
        f.write(str(datetime.datetime.now()))
        
    print("✅ Models saved successfully!\n")

def load_saved_model():
    """Loads existing models from disk if available."""
    global scaler, ensemble_model
    
    if os.path.exists(MODEL_PATH) and os.path.exists(SCALER_PATH):
        print(f"📂 Loading pre-trained models from {MODEL_DIR}...")
        try:
            ensemble_model = joblib.load(MODEL_PATH)
            scaler = joblib.load(SCALER_PATH)
            
            # Check when it was trained
            ts_file = os.path.join(MODEL_DIR, 'last_trained.txt')
            if os.path.exists(ts_file):
                with open(ts_file, 'r') as f:
                    print(f"🕒 Model last trained: {f.read().strip()}")
            
            print("✅ Pre-trained models loaded successfully!\n")
            return True
        except Exception as e:
            print(f"❌ Error loading models ({e}). Retraining...")
            return False
    else:
        print("⚠️ No saved models found. Will train new ones.")        return False

# --- INITIALIZATION ---
# Try to load, otherwise train
if not load_saved_model():
    train_and_save_model()

# --- 3. WEB ROUTES ---

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if ensemble_model is None or scaler is None:
        return jsonify({'error': 'Model not ready. Please wait for training to complete.'}), 503
        
    try:
        data = request.form
        
        # Parse inputs
        input_values = [
            float(data['age']),
            float(data['fever']),
            float(data['dehydration']),
            float(data['water_quality']),
            float(data['sanitation'])
        ]
        
        input_data = np.array([input_values])
        input_scaled = scaler.transform(input_data)
        
        # Predict
        prediction = ensemble_model.predict(input_scaled)[0]
        probabilities = ensemble_model.predict_proba(input_scaled)[0]
        probability = probabilities[1] # Probability of class 1 (Positive)
        
        result = "High Risk of Cholera" if prediction == 1 else "Low Risk of Cholera"
        
        if probability > 0.7:
            risk_level = "Critical"
        elif probability > 0.4:
            risk_level = "Moderate"
        else:
            risk_level = "Low"
        
        return jsonify({
            'result': result,
            'probability': f"{probability * 100:.2f}%",            'risk_level': risk_level
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/status')
def status():
    """Endpoint to check system health"""
    is_trained = os.path.exists(MODEL_PATH)
    ts = "Unknown"
    if os.path.exists(os.path.join(MODEL_DIR, 'last_trained.txt')):
        with open(os.path.join(MODEL_DIR, 'last_trained.txt'), 'r') as f:
            ts = f.read().strip()
            
    return jsonify({
        "status": "online",
        "model_trained": is_trained,
        "last_trained": ts,
        "message": "System operational"
    })

if __name__ == '__main__':
    # Debug mode is False for production, but we keep it True for local dev
    # When deployed on Render, Gunicorn ignores this and uses start.sh
    app.run(host='0.0.0.0', port=5000, debug=True)