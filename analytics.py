import os
from datetime import datetime

def track_prediction(result, probability, risk_level):
    """Track prediction statistics locally"""
    try:
        print(f"📊 PREDICTION TRACKED: {result} ({probability}) - {risk_level}")
        
        # Save to local file for simple logging
        log_file = 'prediction_logs.txt'
        with open(log_file, 'a') as f:
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            f.write(f"{timestamp} | {result} | {probability} | {risk_level}\n")
            
    except Exception as e:
        print(f"Analytics error: {e}")