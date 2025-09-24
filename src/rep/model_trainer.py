"""import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.ensemble import IsolationForest, RandomForestRegressor
from sklearn.cluster import KMeans
import xgboost as xgb
import lightgbm as lgb
import catboost as cb
from sklearn.metrics import mean_absolute_error
import joblib
import subprocess
import os
import random

# Function to compute similarity using cosine similarity
def cosine_similarity(vec1, vec2):
    dot_product = np.dot(vec1, vec2)
    norm1 = np.linalg.norm(vec1)
    norm2 = np.linalg.norm(vec2)
    return dot_product / (norm1 * norm2 + 1e-6)

# Function to classify workload using EPTMW approach
def classify_workload(row, known_workloads):
    feature_vector = np.array([row['cpu_load'], row['memory_load'], row['disk_load']])
    similarities = {}
    
    for workload_type, workload_samples in known_workloads.items():
        max_similarity = max([cosine_similarity(feature_vector, sample) for sample in workload_samples])
        similarities[workload_type] = max_similarity
    
    return max(similarities, key=similarities.get)

# Function to train classification and regression models
def train_models(data_file, model_prefix):
    df = pd.read_csv(data_file)
    print(df.columns)

    # Remove outliers
    iso = IsolationForest(contamination=0.05)
    df = df[iso.fit_predict(df) == 1]
    
    # Normalize features
    scaler = MinMaxScaler()
    X = scaler.fit_transform(df[['cpu_load', 'memory_load', 'disk_load']])
    
    # Apply PCA for dimensionality reduction
    pca = PCA(n_components=3)
    X_pca = pca.fit_transform(X)
    
    # Cluster workloads dynamically using K-Means
    kmeans = KMeans(n_clusters=3, random_state=42)
    cluster_labels = kmeans.fit_predict(X_pca)
    df['workload_type'] = cluster_labels
    
    # Map clusters to workload types based on cluster centroids
    centroids = kmeans.cluster_centers_
    workload_types = ['cpu', 'memory', 'disk']
    sorted_indices = np.argsort(np.argmax(centroids, axis=1))
    cluster_mapping = {sorted_indices[i]: workload_types[i] for i in range(3)}
    df['workload_type'] = df['workload_type'].map(cluster_mapping)
    
    # Prepare training data
    y_class = df['workload_type'].astype('category').cat.codes
    
    X_train, X_test, y_class_train, y_class_test = train_test_split(X_pca, y_class, test_size=0.3, random_state=42)
    
    # Train XGBoost classifier
    classification_model = xgb.XGBClassifier(eval_metric='mlogloss')
    classification_model.fit(X_train, y_class_train)
    joblib.dump(classification_model, f'{model_prefix}_classification.pkl')
    print("Workload classification model trained and saved.")

    # Collect performance counters using 'perf'
    performance_data = []
    for _, row in df.iterrows():
        try:
            cycles_output = subprocess.getoutput("perf stat -e cycles -x ',' sleep 0.1 2>&1 | grep cycles | awk -F',' '{print $1}'").strip()
            llc_output = subprocess.getoutput("perf stat -e cache-misses -x ',' sleep 0.1 2>&1 | grep cache-misses | awk -F',' '{print $1}'").strip()
        
            cycles = int(cycles_output.replace(',', '')) if cycles_output.replace(',', '').isdigit() else 0
            llc_miss = int(llc_output.replace(',', '')) if llc_output.replace(',', '').isdigit() else 0
        
            if cycles == 0 or llc_miss == 0:
                print("Warning: Failed to collect perf metrics.")
                continue
        
            throughput = (cycles - llc_miss) / max(cycles, 1)
        
            performance_data.append({
                'cpu_usage': row['cpu_load'] * 100,
                'memory_usage': row['memory_load'] * 100,
                'disk_usage': row['disk_load'] * 100,
                'cycles': cycles,
                'llc_miss': llc_miss,
                'workload_type': row['workload_type'],
                'throughput': throughput
            })
        except Exception as e:
            print(f"Error collecting performance counters: {e}")
    
    performance_df = pd.DataFrame(performance_data)
    performance_df.to_csv('../data/performance_counters.csv', index=False)
    print("performance_counters.csv generated successfully!")

# Function to create energy efficiency model
def create_energy_model(metrics_file, model_path):
    df = pd.read_csv(metrics_file)
    X = df[['cpu_load', 'memory_load', 'disk_load']]
    y = df['power']
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model.fit(X_train, y_train)
    print(f"Energy Model MAE: {mean_absolute_error(y_test, model.predict(X_test)):.4f}")
    joblib.dump(model, model_path)
    print(f"✅ Energy model saved to {model_path}")

# Function to create performance model
def create_performance_model(performance_file, model_path):
    df = pd.read_csv(performance_file)
    X = df[['cpu_usage', 'memory_usage', 'disk_usage', 'cycles', 'llc_miss']]
    y = df['throughput']
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model.fit(X_train, y_train)
    print(f"Performance Model MAE: {mean_absolute_error(y_test, model.predict(X_test)):.4f}")
    joblib.dump(model, model_path)
    print(f"✅ Performance model saved to {model_path}")

if __name__ == "__main__":
    train_models('../data/workloads.csv', '../data/eptmw_model')
    create_energy_model('../data/metrics.csv', '../data/energy_efficiency_model.pkl')
    create_performance_model('../data/performance_counters.csv', '../data/performance_model.pkl')"""


import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import IsolationForest
import xgboost as xgb
import lightgbm as lgb
import catboost as cb
from sklearn.metrics import mean_absolute_error
import numpy as np
import joblib
import subprocess
import random

# Function to assign workload type based on resource usage
"""def assign_workload_type(row):
    cpu_factor = row['cpu_load'] / (row['power'] + 1e-6)
    memory_factor = row['memory_load'] / (row['power'] + 1e-6)
    disk_factor = row['disk_load'] / (row['power'] + 1e-6)
    max_factor = max(cpu_factor, memory_factor, disk_factor)
    return 0 if max_factor == cpu_factor else 1 if max_factor == memory_factor else 2"""
def assign_workload_type(row):
    max_usage = max(row['cpu_load'], row['memory_load'], row['disk_load'])
    return 0 if max_usage == row['cpu_load'] else 1 if max_usage == row['memory_load'] else 2


# Function to train classification and regression models
def train_models(data_file, model_prefix):
    df = pd.read_csv(data_file)
    print(df.columns)

    # Assign workload types and remove outliers
    df['workload_type'] = df.apply(assign_workload_type, axis=1)
    iso = IsolationForest(contamination=0.05)
    df = df[iso.fit_predict(df.drop(columns=['power', 'workload_type'])) == 1]

    # Normalize features
    scaler = MinMaxScaler()
    X = scaler.fit_transform(df[['cpu_load', 'memory_load', 'disk_load']])
    y_class = df['workload_type']
    y_reg = df['power']

    # Split data for training and testing
    X_train, X_test, y_class_train, y_class_test = train_test_split(X, y_class, test_size=0.3, random_state=42)
    _, _, y_reg_train, y_reg_test = train_test_split(X, y_reg, test_size=0.3, random_state=42)

    # Train XGBoost classifier
    classification_model = xgb.XGBClassifier(eval_metric='mlogloss')
    classification_model.fit(X_train, y_class_train)
    joblib.dump(classification_model, f'{model_prefix}_classification.pkl')
    print("Workload classification model trained and saved.")

    # Train ensemble power prediction models
    models = {'xgb': xgb.XGBRegressor(), 'lgb': lgb.LGBMRegressor(), 'cat': cb.CatBoostRegressor(verbose=0)}
    predictions = [model.fit(X_train, y_reg_train).predict(X_test) for model in models.values()]
    mae = mean_absolute_error(y_reg_test, np.mean(predictions, axis=0))
    print(f"Power Model Ensemble MAE: {mae}")
    joblib.dump(models, f'{model_prefix}_throughput_ensemble.pkl')
    print("Power prediction model ensemble trained and saved.")

    # Collect performance counters using 'perf'
    performance_data = []
    for _, row in df.iterrows():
        # Capture cycles and LLC misses with error handling
        cycles_output = subprocess.getoutput("perf stat -e cycles -x ',' sleep 0.1 2>&1 | grep cycles | awk -F',' '{print $1}'").strip()
        llc_output = subprocess.getoutput("perf stat -e cache-misses -x ',' sleep 0.1 2>&1 | grep cache-misses | awk -F',' '{print $1}'").strip()

        cycles = int(cycles_output.replace(',', '')) if cycles_output.replace(',', '').isdigit() else 0
        llc_miss = int(llc_output.replace(',', '')) if llc_output.replace(',', '').isdigit() else 0

        if cycles == 0 or llc_miss == 0:
            print(f"Warning: Failed to collect perf metrics (cycles: {cycles}, llc_miss: {llc_miss}).")

        ipc = cycles / (llc_miss + 1e-6) if llc_miss > 0 else 0
        throughput = ipc * cycles

        # Record performance metrics
        performance_data.append({
            'cpu_usage': row['cpu_load'] * 100,
            'memory_usage': row['memory_load'] * 100,
            'disk_usage': row['disk_load'] * 100,
            'interrupts': random.randint(1000, 5000),
            'cycles': cycles,
            'llc_miss': llc_miss,
            'workload_type': row['workload_type'],
            'throughput': throughput
        })

    # Save performance counters to CSV
    performance_df = pd.DataFrame(performance_data)
    performance_df.to_csv('../data/performance_counters.csv', index=False)
    print("performance_counters.csv generated successfully!")

if __name__ == "__main__":
    train_models('../data/metrics.csv', '../data/eptmw_model')


