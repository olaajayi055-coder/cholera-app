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

# Import custom modules
from analytics import track_prediction
from database import save_prediction, get_statistics, init_db
from auth import login_manager, init_auth, get_user_by_username, create_user, login_required, current_user, login_user, logout_user

app = Flask(__name__)
app.secret_key = 'super-secret-key-change-in-production' # Change this for real deployment

# Initialize Login Manager
login_manager.init_app(app)
login_manager.login_view = 'login'

# --- CONFIGURATION ---
MODEL_DIR = 'models'
os.makedirs(MODEL_DIR, exist_ok=True)

MODEL_PATH = os.path.join(MODEL_DIR, 'cholera_ensemble.pkl')
SCALER_PATH = os.path.join(MODEL_DIR, 'scaler.pkl')
DATA_PATH = 'cholera_dataset.csv'

scaler = None
ensemble_model = None

# --- 1. DATA LOADING FUNCTIONS ---

def generate_simulated_data():
    print("⚠️ Generating simulated dataset...")
    np.random.seed(42)
    n_samples = 600
    
    age = np.random.randint(1, 80, n_samples)
    fever = np.random.randint(1, 11, n_samples)
    dehydration = np.random.randint(1, 11, n_samples)
    water_quality = np.random.randint(1, 6, n_samples)
    sanitation = np.random.randint(1, 6, n_samples)
    
    risk_score = (fever * 0.35) + (dehydration * 0.45) + (water_quality * 0.25) - (sanitation * 0.15)
    target = (risk_score > 5.5).astype(int)    
    noise_indices = np.random.choice(n_samples, size=int(n_samples * 0.05), replace=False)
    target[noise_indices] = 1 - target[noise_indices]
    
    X = np.column_stack((age, fever, dehydration, water_quality, sanitation))
    y = target
    
    df = pd.DataFrame({
        'age': age, 'fever': fever, 'dehydration': dehydration,
        'water_quality': water_quality, 'sanitation': sanitation,
        'cholera_positive': target
    })
    df.to_csv(DATA_PATH, index=False)
    print(f"✅ Simulated data saved to {DATA_PATH}")
    return X, y

def load_real_data():
    if os.path.exists(DATA_PATH):
        try:
            print(f"📂 Loading data from {DATA_PATH}...")
            df = pd.read_csv(DATA_PATH)
            required_cols = ['age', 'fever', 'dehydration', 'water_quality', 'sanitation', 'cholera_positive']
            if not all(col in df.columns for col in required_cols):
                raise ValueError("Missing columns")
            X = df[['age', 'fever', 'dehydration', 'water_quality', 'sanitation']].values
            y = df['cholera_positive'].values
            print(f"✅ Data loaded. Shape: {X.shape}")
            return X, y
        except Exception as e:
            print(f"❌ Error reading CSV: {e}. Regenerating...")
            return generate_simulated_data()
    else:
        print("📄 CSV not found. Generating...")
        return generate_simulated_data()

# --- 2. MODEL TRAINING ---

def train_and_save_model():
    global scaler, ensemble_model
    print("\n--- 🚀 Starting Model Training ---")
    X, y = load_real_data()
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    rf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    lr = LogisticRegression(max_iter=1000, random_state=42)
    svc = SVC(kernel='rbf', probability=True, random_state=42)
    
    ensemble_model = VotingClassifier(        estimators=[('rf', rf), ('lr', lr), ('svc', svc)],
        voting='soft'
    )
    
    print("🧠 Training ensemble...")
    ensemble_model.fit(X_scaled, y)
    
    preds = ensemble_model.predict(X_scaled)
    acc = accuracy_score(y, preds)
    print(f"✅ Training complete. Accuracy: {acc:.2f}")
    
    print(f"💾 Saving models...")
    joblib.dump(ensemble_model, MODEL_PATH)
    joblib.dump(scaler, SCALER_PATH)
    
    with open(os.path.join(MODEL_DIR, 'last_trained.txt'), 'w') as f:
        f.write(str(datetime.datetime.now()))
    print("✅ Models saved!\n")

def load_saved_model():
    global scaler, ensemble_model
    if os.path.exists(MODEL_PATH) and os.path.exists(SCALER_PATH):
        print(f"📂 Loading pre-trained models...")
        try:
            ensemble_model = joblib.load(MODEL_PATH)
            scaler = joblib.load(SCALER_PATH)
            ts_file = os.path.join(MODEL_DIR, 'last_trained.txt')
            if os.path.exists(ts_file):
                with open(ts_file, 'r') as f:
                    print(f"🕒 Last trained: {f.read().strip()}")
            print("✅ Models loaded!\n")
            return True
        except Exception as e:
            print(f"❌ Error loading: {e}. Retraining...")
            return False
    else:
        print("⚠️ No saved models. Training new ones...")
        return False

# --- INITIALIZATION ---
init_db()
init_auth()
if not load_saved_model():
    train_and_save_model()

# --- 3. WEB ROUTES ---

@app.route('/')
def home():
    return render_template('index.html')
@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form.get('username')
        password = request.form.get('password')
        user = get_user_by_username(username)
        if user and user.check_password(password):
            login_user(user)
            return jsonify({'success': True, 'message': 'Login successful'})
        return jsonify({'success': False, 'message': 'Invalid credentials'}), 401
    return render_template('login.html')

@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        username = request.form.get('username')
        password = request.form.get('password')
        if len(username) < 3 or len(password) < 6:
            return jsonify({'success': False, 'message': 'Username min 3 chars, password min 6'}), 400
        if create_user(username, password):
            return jsonify({'success': True, 'message': 'Registration successful'})
        return jsonify({'success': False, 'message': 'Username exists'}), 400
    return render_template('register.html')

@app.route('/logout')
@login_required
def logout():
    logout_user()
    return jsonify({'success': True, 'message': 'Logged out'})

@app.route('/predict', methods=['POST'])
def predict():
    if ensemble_model is None or scaler is None:
        return jsonify({'error': 'Model not ready'}), 503
        
    try:
        data = request.form
        age = int(float(data['age']))
        fever = int(float(data['fever']))
        dehydration = int(float(data['dehydration']))
        water_quality = int(float(data['water_quality']))
        sanitation = int(float(data['sanitation']))
        
        input_data = np.array([[age, fever, dehydration, water_quality, sanitation]])
        input_scaled = scaler.transform(input_data)
        
        prediction = ensemble_model.predict(input_scaled)[0]
        probabilities = ensemble_model.predict_proba(input_scaled)[0]
        probability = probabilities[1]        
        result = "High Risk of Cholera" if prediction == 1 else "Low Risk of Cholera"
        risk_level = "Critical" if probability > 0.7 else "Moderate" if probability > 0.4 else "Low"
        prob_str = f"{probability * 100:.2f}%"
        
        ip_address = request.remote_addr
        save_prediction(age, fever, dehydration, water_quality, sanitation, result, prob_str, risk_level, ip_address)
        track_prediction(result, prob_str, risk_level)
        
        return jsonify({
            'result': result,
            'probability': prob_str,
            'risk_level': risk_level
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/stats')
def api_stats():
    stats = get_statistics()
    return jsonify(stats)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)