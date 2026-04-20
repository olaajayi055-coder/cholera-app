import pandas as pd
import numpy as np

def create_realistic_cholera_data():
    print("Generating realistic cholera dataset...")
    np.random.seed(42)
    
    n_samples = 600  # Creating 600 realistic records
    
    # Generate features
    age = np.random.randint(1, 80, n_samples)
    fever = np.random.randint(1, 11, n_samples)       # 1-10 scale
    dehydration = np.random.randint(1, 11, n_samples) # 1-10 scale
    water_quality = np.random.randint(1, 6, n_samples) # 1 (Good) to 5 (Poor)
    sanitation = np.random.randint(1, 6, n_samples)    # 1 (Poor) to 5 (Good)
    
    # LOGIC: Simulate real-world risk factors
    # High fever + High dehydration + Poor Water + Poor Sanitation = High Risk
    # Note: Sanitation score is inverted in logic (5 is good, so it reduces risk)
    
    risk_score = (
        (fever * 0.35) + 
        (dehydration * 0.45) + 
        (water_quality * 0.25) - 
        (sanitation * 0.15) +
        (age * 0.02) # Slight increase in risk for very young/old could be added, keeping simple here
    )
    
    # Determine target (1 = Cholera Positive, 0 = Negative)
    # Threshold adjusted to create a realistic mix of cases (~30-40% positive rate)
    threshold = 5.5 
    target = (risk_score > threshold).astype(int)
    
    # Add some "noise" to make it look like real medical data (not perfectly linear)
    # Flip about 5% of the results randomly to simulate misdiagnosis or asymptomatic cases
    noise_indices = np.random.choice(n_samples, size=int(n_samples * 0.05), replace=False)
    target[noise_indices] = 1 - target[noise_indices]
    
    # Create DataFrame
    df = pd.DataFrame({
        'age': age,
        'fever': fever,
        'dehydration': dehydration,
        'water_quality': water_quality,
        'sanitation': sanitation,
        'cholera_positive': target
    })
    
    # Save to CSV
    filename = 'cholera_dataset.csv'
    df.to_csv(filename, index=False)
    
    print(f"✅ Success! Created '{filename}' with {len(df)} records.")
    print(f"   - Positive Cases (1): {df['cholera_positive'].sum()}")
    print(f"   - Negative Cases (0): {(df['cholera_positive'] == 0).sum()}")
    print(f"   - File saved to: {os.path.abspath(filename)}")

if __name__ == "__main__":
    import os
    create_realistic_cholera_data()