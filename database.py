import sqlite3
import os

DATABASE = 'predictions.db'

def init_db():
    """Initialize the database with required tables"""
    conn = sqlite3.connect(DATABASE)
    cursor = conn.cursor()
    
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS predictions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
            age INTEGER,
            fever INTEGER,
            dehydration INTEGER,
            water_quality INTEGER,
            sanitation INTEGER,
            result TEXT,
            probability REAL,
            risk_level TEXT,
            ip_address TEXT
        )
    ''')
    
    conn.commit()
    conn.close()
    print("✅ Database initialized")

def save_prediction(age, fever, dehydration, water_quality, sanitation, result, probability, risk_level, ip_address='unknown'):
    """Save a prediction to the database"""
    try:
        conn = sqlite3.connect(DATABASE)
        cursor = conn.cursor()
        
        # Remove '%' sign for storage
        prob_val = float(probability.replace('%', ''))
        
        cursor.execute('''
            INSERT INTO predictions (age, fever, dehydration, water_quality, sanitation, result, probability, risk_level, ip_address)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (age, fever, dehydration, water_quality, sanitation, result, prob_val, risk_level, ip_address))
        
        conn.commit()
        conn.close()
        
    except Exception as e:
        print(f"Database save error: {e}")

def get_statistics():
    """Get prediction statistics"""
    try:
        conn = sqlite3.connect(DATABASE)
        cursor = conn.cursor()
        
        cursor.execute('SELECT COUNT(*) FROM predictions')
        total = cursor.fetchone()[0] or 0
        
        cursor.execute("SELECT COUNT(*) FROM predictions WHERE result LIKE '%High Risk%'")
        high_risk = cursor.fetchone()[0] or 0
        
        cursor.execute('SELECT AVG(probability) FROM predictions')
        avg_prob = cursor.fetchone()[0] or 0.0
        
        conn.close()
        
        return {
            'total': total,
            'high_risk': high_risk,
            'avg_probability': round(avg_prob, 2)
        }
        
    except Exception as e:
        print(f"Statistics error: {e}")
        return {'total': 0, 'high_risk': 0, 'avg_probability': 0.0}