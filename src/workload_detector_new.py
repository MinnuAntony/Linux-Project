import pandas as pd
import numpy as np
import xgboost as xgb
import joblib
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import random
import os
import pickle

data = [
    [0.8, 0.1, 0.05, 100, 200, 50],
    [0.85, 0.10, 0.05, 110, 220, 55],
    [0.78, 0.12, 0.07, 130, 260, 60],
    [0.88, 0.09, 0.06, 115, 210, 52],
    [0.81, 0.11, 0.08, 120, 230, 58],
    [0.90, 0.08, 0.04, 140, 280, 65],
    [0.75, 0.14, 0.05, 100, 200, 50],
    [0.80, 0.10, 0.07, 125, 240, 57],
    [0.83, 0.09, 0.06, 135, 270, 63],
    [0.77, 0.15, 0.05, 105, 190, 45],
    [0.82, 0.13, 0.07, 128, 250, 59],
    
    [0.40, 0.85, 0.05, 180, 250, 120],
    [0.35, 0.80, 0.06, 170, 240, 110],
    [0.38, 0.85, 0.05, 180, 250, 120],
    [0.40, 0.88, 0.07, 190, 260, 130],
    [0.42, 0.92, 0.08, 200, 270, 140],
    [0.37, 0.78, 0.06, 175, 245, 115],
    [0.39, 0.84, 0.07, 185, 255, 125],
    [0.41, 0.90, 0.05, 195, 265, 135],
    [0.36, 0.82, 0.08, 165, 230, 100],
    [0.34, 0.76, 0.06, 160, 225, 95],
    [0.43, 0.95, 0.07, 210, 280, 145],
    
    [0.25, 0.15, 0.95, 1200, 1400, 600],
    [0.20, 0.18, 0.90, 1100, 1300, 500],
    [0.25, 0.15, 0.95, 1200, 1400, 600],
    [0.22, 0.14, 0.93, 1150, 1350, 550],
    [0.18, 0.12, 0.88, 1050, 1250, 450],
    [0.21, 0.16, 0.92, 1120, 1320, 520],
    [0.24, 0.17, 0.94, 1180, 1380, 580],
    [0.19, 0.13, 0.89, 1080, 1280, 470],
    [0.23, 0.19, 0.91, 1160, 1360, 540],
    [0.26, 0.11, 0.96, 1220, 1420, 620],
    [0.27, 0.10, 0.97, 1250, 1450, 650]
]

#data = [[0.068, 0.519, 0.0008192,100,200,50]]
"""data = [
    [0.8, 0.1, 0.05, 100, 200, 50],
    [0.40, 0.85, 0.05, 180, 250, 120],
    [0.25, 0.15, 0.95, 1200, 1400, 600]
]"""

# Always select a new sample each time the script runs
sample_counters = random.choice(data)
with open("input.txt", "wb") as file:
    pickle.dump(sample_counters, file)
print("\n📊 Input System Counters :\n", sample_counters[0:3])
print("\n")

CLASSIFICATION_MODEL_FILE = "../data/workload_classification_model.pkl"
THROUGHPUT_MODEL_FILE = "../data/workload_throughput_model.pkl"

# Train workload classification and throughput models
def train_workload_models(data_file):
    df = pd.read_csv(data_file)
    X = df.drop(columns=['workload_type', 'throughput'])
    y_class = df['workload_type']
    y_throughput = df['throughput']
    
    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X)
    
    X_train_c, X_test_c, y_train_c, _ = train_test_split(X_scaled, y_class, test_size=0.2)
    classification_model = xgb.XGBClassifier(eval_metric='mlogloss')
    classification_model.fit(X_train_c, y_train_c)
    joblib.dump(classification_model, CLASSIFICATION_MODEL_FILE)
    print(f"Workload classification model saved to {CLASSIFICATION_MODEL_FILE}")
    
    X_train_t, _, y_train_t, _ = train_test_split(X_scaled, y_throughput, test_size=0.2)
    throughput_model = xgb.XGBRegressor()
    throughput_model.fit(X_train_t, y_train_t)
    joblib.dump(throughput_model, THROUGHPUT_MODEL_FILE)
    print(f"Workload throughput model saved to {THROUGHPUT_MODEL_FILE}")

# Detect workload type and calculate similarity with known profiles
def detect_workload(performance_counters):
    classification_model = joblib.load(CLASSIFICATION_MODEL_FILE)
    throughput_model = joblib.load(THROUGHPUT_MODEL_FILE)

    workload_type = classification_model.predict(np.array(performance_counters).reshape(1, -1))[0]
    predicted_throughput = throughput_model.predict(np.array(performance_counters).reshape(1, -1))[0]

    known_profiles = {"CPU-intensive": [1, 0, 0], "Memory-intensive": [0, 1, 0], "Disk-intensive": [0, 0, 1]}
    unknown_features = np.array(performance_counters[:3]).reshape(1, -1)

    similarities = {}
    for workload, profile in known_profiles.items():
        similarity = cosine_similarity(unknown_features, np.array(profile).reshape(1, -1))[0][0]
        similarities[workload] = similarity

    total_similarity = sum(similarities.values()) or 1
    weight_vector = {k: v / total_similarity for k, v in similarities.items()}
    
    WORKLOAD_MAPPING = {0: "CPU-intensive", 1: "Memory-intensive", 2: "Disk-intensive"}
    workload_name = WORKLOAD_MAPPING.get(workload_type, "Unknown")
    print(f"Detected workload type: {workload_type} {workload_name}")
    #print(f"Workload weight vector: {weight_vector}")
    
    
    

    
    return workload_type, predicted_throughput, weight_vector

if __name__ == "__main__":
    print("Training workload models using XGBoost...")
    train_workload_models("../data/performance_counters.csv")
    detect_workload(sample_counters)

