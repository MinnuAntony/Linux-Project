
#got this from https://chatgpt.com/canvas/shared/67b38db9bd7081919d55f92b9d957e5d
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
import joblib
import os

def create_energy_model(metrics_file, model_path):
    """Generate energy efficiency model from metrics.csv."""
    df = pd.read_csv(metrics_file)

    # Validate columns
    required_cols = {'cpu_load', 'memory_load', 'disk_load', 'power'}
    if not required_cols.issubset(df.columns):
        raise ValueError(f"Missing columns in {metrics_file}: {required_cols - set(df.columns)}")

    # Train model
    X = df[['cpu_load', 'memory_load', 'disk_load']]
    y = df['power']
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model.fit(X_train, y_train)

    # Evaluate and save
    y_pred = model.predict(X_test)
    print(f"Energy Model MAE: {mean_absolute_error(y_test, y_pred):.4f}")
    joblib.dump(model, model_path)
    print(f"✅ Energy model saved to {model_path}")

def create_performance_model(performance_file, model_path):
    """Generate performance model from performance_counters.csv."""
    df = pd.read_csv(performance_file)

    # Validate columns
    required_cols = {'cpu_usage', 'memory_usage', 'disk_usage', 'interrupts', 'cycles', 'llc_miss', 'throughput'}
    if not required_cols.issubset(df.columns):
        raise ValueError(f"Missing columns in {performance_file}: {required_cols - set(df.columns)}")

    # Train model
    X = df[['cpu_usage', 'memory_usage', 'disk_usage', 'interrupts', 'cycles', 'llc_miss']]
    y = df['throughput']
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model.fit(X_train, y_train)

    # Evaluate and save
    y_pred = model.predict(X_test)
    print(f"Performance Model MAE: {mean_absolute_error(y_test, y_pred):.4f}")
    joblib.dump(model, model_path)
    print(f"✅ Performance model saved to {model_path}")

def main():
    metrics_file = '../data/metrics.csv'
    #metrics_file = '../data/metrics_cpu_80.csv'
    performance_file = '../data/performance_counters.csv'
    energy_model = '../data/energy_efficiency_model.pkl'
    performance_model = '../data/performance_model.pkl'

    if not os.path.exists(metrics_file):
        raise FileNotFoundError(f"File not found: {metrics_file}")
    if not os.path.exists(performance_file):
        raise FileNotFoundError(f"File not found: {performance_file}")

    print("🚀 Generating Energy Efficiency Model...")
    create_energy_model(metrics_file, energy_model)

    print("\n🚀 Generating Performance Model...")
    create_performance_model(performance_file, performance_model)

    print("\n✅ Both models have been generated successfully!")

if __name__ == "__main__":
    main()

