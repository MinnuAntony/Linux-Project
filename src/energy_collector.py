# energy_collector.py content

"""import pandas as pd
import random
import time

def collect_metrics(workload_file, output_file):
    #Simulate power metrics collection based on workloads.
    workloads = pd.read_csv(workload_file)
    results = []
    for _, row in workloads.iterrows():
        time.sleep(0.5)  # Simulate delay
        power = (row['cpu_load'] * 100) + (row['memory_load'] * 50) + (row['disk_load'] * 30) + random.uniform(0, 20)
        results.append({
            'cpu_load': row['cpu_load'],
            'memory_load': row['memory_load'],
            'disk_load': row['disk_load'],
            'power': power
        })
    pd.DataFrame(results).to_csv(output_file, index=False)
    print("Simulated power metrics collected.")

if __name__ == "__main__":
    collect_metrics('../data/workloads.csv', '../data/metrics.csv')"""
    
import pandas as pd
import random
import time

def collect_metrics(workload_file, output_file):
    """Simulate power metrics collection based on workloads with laptop-specific scaling."""
    
    BASE_POWER = 8  # More realistic idle power for your laptop
    CPU_FACTOR = 0.9  # Adjusted for efficiency of mobile CPUs
    MEMORY_FACTOR = 0.5  # Less impact from memory
    DISK_FACTOR = 0.4  # Even lower impact for SSD-based systems
    
    workloads = pd.read_csv(workload_file)
    results = []
    
    for _, row in workloads.iterrows():
        time.sleep(0.5)  # Simulate delay
        
        # Improved power model with realistic laptop behavior
        cpu_power = CPU_FACTOR * (row['cpu_load'] ** 1.3)  # Slightly non-linear CPU scaling
        memory_power = MEMORY_FACTOR * row['memory_load']
        disk_power = DISK_FACTOR * row['disk_load']
        
        # Total power consumption
        power = BASE_POWER + cpu_power + memory_power + disk_power + random.uniform(-2, 2)  # Less noise variation
        
        results.append({
            'cpu_load': row['cpu_load'],
            'memory_load': row['memory_load'],
            'disk_load': row['disk_load'],
            'power': round(power, 2)
        })
    
    pd.DataFrame(results).to_csv(output_file, index=False)
    print("Simulated power metrics collected with laptop-specific scaling.")

if __name__ == "__main__":
    collect_metrics('../data/workloads.csv', '../data/metrics.csv')


