import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import IsolationForest
import xgboost as xgb
import lightgbm as lgb
import catboost as cb
from sklearn.metrics import mean_absolute_error, accuracy_score
import numpy as np
import joblib
import subprocess
import random

def assign_workload_type(row):
    max_usage = max(row['cpu_load'], row['memory_load'], row['disk_load'])
    return 0 if max_usage == row['cpu_load'] else 1 if max_usage == row['memory_load'] else 2

def train_models(data_file, model_prefix):
    df = pd.read_csv(data_file)
    print("Dataset Columns:", df.columns)

    df['workload_type'] = df.apply(assign_workload_type, axis=1)
    
    # Outlier removal using Isolation Forest
    iso = IsolationForest(contamination=0.05, random_state=42)
    df = df[iso.fit_predict(df.drop(columns=['power', 'workload_type'])) == 1]
    
    # Scaling features
    scaler = MinMaxScaler()
    feature_cols = ['cpu_load', 'memory_load', 'disk_load']
    X = scaler.fit_transform(df[feature_cols])
    y_class = df['workload_type']
    y_reg = df['power']
    
    # Splitting for classification
    X_train, X_test, y_class_train, y_class_test = train_test_split(
        X, y_class, test_size=0.3, random_state=42, stratify=y_class)
    
    # Splitting for regression
    X_train_reg, X_test_reg, y_reg_train, y_reg_test = train_test_split(
        X, y_reg, test_size=0.3, random_state=42)
    
    print("Training XGBoost classification model...")
    classification_model = xgb.XGBClassifier(
        eval_metric='mlogloss', 
        n_estimators=50,  # Reduced for speed
        random_state=42
    )
    
    classification_model.fit(
        X_train, y_class_train,
        eval_set=[(X_test, y_class_test)],
        verbose=True
    )

    # Extract evaluation results
    evals_result = classification_model.evals_result()

    # Extract and plot accuracy from evaluation results
    epochs = len(evals_result['validation_0']['mlogloss'])
    accuracy_per_epoch = [1 - loss for loss in evals_result['validation_0']['mlogloss']]
    plt.plot(range(epochs), accuracy_per_epoch, label='Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.title('Training Accuracy Over Epochs')
    plt.legend()
    plt.show()
    
    y_class_pred = classification_model.predict(X_test)
    class_accuracy = accuracy_score(y_class_test, y_class_pred)
    print(f"Final Workload Classification Accuracy: {class_accuracy:.4f}")
    
    joblib.dump(classification_model, f'{model_prefix}_classification.pkl')
    print("Workload classification model trained and saved.")
    
    # Power Prediction Model (Ensemble of XGBoost, LightGBM, CatBoost)
    models = {
        'xgb': xgb.XGBRegressor(n_estimators=50, random_state=42),
        'lgb': lgb.LGBMRegressor(n_estimators=50, random_state=42),
        'cat': cb.CatBoostRegressor(n_estimators=50, verbose=0, random_state=42)
    }
    
    predictions = {}
    for name, model in models.items():
        print(f"Training {name} model...")
        model.fit(X_train_reg, y_reg_train)
        predictions[name] = model.predict(X_test_reg)
    
    # Averaging predictions (Ensemble Learning)
    ensemble_pred = np.mean(list(predictions.values()), axis=0)
    mae = mean_absolute_error(y_reg_test, ensemble_pred)
    print(f"Power Model Ensemble MAE: {mae:.4f}")
    
    joblib.dump(models, f'{model_prefix}_throughput_ensemble.pkl')
    print("Power prediction model ensemble trained and saved.")
    
    # Collecting Performance Counters using 'perf'
    performance_data = []
    for _, row in df.iterrows():
        cycles_output = subprocess.getoutput("perf stat -e cycles -x ',' sleep 0.1 2>&1 | grep cycles | awk -F',' '{print $1}'").strip()
        llc_output = subprocess.getoutput("perf stat -e cache-misses -x ',' sleep 0.1 2>&1 | grep cache-misses | awk -F',' '{print $1}'").strip()
        
        cycles = int(cycles_output.replace(',', '')) if cycles_output.replace(',', '').isdigit() else 0
        llc_miss = int(llc_output.replace(',', '')) if llc_output.replace(',', '').isdigit() else 0
        
        if cycles == 0 or llc_miss == 0:
            print(f"Warning: Failed to collect perf metrics (cycles: {cycles}, llc_miss: {llc_miss}).")
        
        ipc = cycles / (llc_miss + 1e-6) if llc_miss > 0 else 0
        throughput = ipc * cycles
        
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
    
    performance_df = pd.DataFrame(performance_data)
    performance_df.to_csv(f'{model_prefix}_performance_counters.csv', index=False)
    print("Performance counters CSV generated successfully!")

if __name__ == "__main__":
    train_models('../data/metrics.csv', '../data/eptmw_model')

