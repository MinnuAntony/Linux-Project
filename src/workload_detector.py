import pandas as pd
import numpy as np
import xgboost as xgb
import joblib
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import random
import os

data = [
    [0.8, 0.1, 0.05, 100, 200, 50],
    [0.40, 0.85, 0.05, 180, 250, 120],
    [0.25, 0.15, 0.95, 1200, 1400, 600]
]

# Select a new workload
sample_counters = random.choice(data)

# Save it safely
np.save("sample_counters.npy", sample_counters)

# Ensure the file is written immediately
with open("sample_counters.npy", "rb") as f:
    os.fsync(f.fileno())



"""data = [ [0.8, 0.1, 0.05, 100, 200, 50],  [0.40, 0.85, 0.05, 180, 250, 120],[0.25, 0.15, 0.95, 1200, 1400, 600] ]
sample_counter = random.choice(data)
sample_counters = sample_counter
print(sample_counters)
#sample_counters = [0.40, 0.85, 0.05, 180, 250, 120]
#sample_counters = [0.25, 0.15, 0.95, 1200, 1400, 600]
#sample_counters = [0.8, 0.1, 0.05, 100, 200, 50]"""


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
    classification_model = xgb.XGBClassifier(use_label_encoder=False, eval_metric='mlogloss')
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
    
    print(f"Detected workload type: {workload_type}")
   # print(f"Predicted throughput: {predicted_throughput}")
    print(f"Workload weight vector: {weight_vector}")
    
    return workload_type, predicted_throughput, weight_vector

if __name__ == "__main__":
    print("Training workload models using XGBoost...")
    train_workload_models("../data/performance_counters.csv")
    #sample_counters = [0.8, 0.1, 0.05, 100, 200, 50]
    #sample_counters = [0.40, 0.85, 0.05, 180, 250, 120]
    #sample_counters =  [0.25, 0.15, 0.95, 1200, 1400, 600]
    """data = [ [0.8, 0.1, 0.05, 100, 200, 50],  [0.40, 0.85, 0.05, 180, 250, 120],[0.25, 0.15, 0.95, 1200, 1400, 600] ]
    sample_counters = random.choice(data)"""
    print(sample_counters)
    detect_workload(sample_counters)
 #randomized choosing the workloads but since its not saved ... diffenrent files showing diff outcome for workload
