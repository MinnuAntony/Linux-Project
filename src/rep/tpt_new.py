import numpy as np
import time
import os
import pickle
import matplotlib.pyplot as plt
from workload_detector import detect_workload

with open("input.txt", "rb") as file:
    sample_counters = pickle.load(file)
print("\n📊 Input System Counters :\n", sample_counters[0:3])
print("\n")

EFFICIENCY_MULTIPLIERS = {
    "CPU-intensive": {"vm.swappiness": {10: 1.05}},
    "Memory-intensive": {
        "vm.swappiness": {80: 0.95},
        "vm.dirty_background_ratio": {5: 1.02},
        "vm.dirty_ratio": {30: 1.03},
        "vm.min_free_kbytes": {131072: 1.01},
    },
    "Disk-intensive": {
        "vm.dirty_expire_centisecs": {100: 1.04},
        "vm.dirty_writeback_centisecs": {50: 1.03},
    }
}

def calculate_base_throughput(sc, workload_type):
    cpu_component, mem_component, disk_component = sc[:3]
    
    weights = {
        "CPU-intensive": {"cpu": 0.7, "mem": 0.2, "disk": 0.1},
        "Memory-intensive": {"cpu": 0.2, "mem": 0.7, "disk": 0.1},
        "Disk-intensive": {"cpu": 0.15, "mem": 0.15, "disk": 0.7},
    }.get(workload_type, {"cpu": 0.33, "mem": 0.33, "disk": 0.34})

    return (weights["cpu"] * cpu_component +
            weights["mem"] * mem_component +
            weights["disk"] * disk_component) * 100

def estimate_throughput(sc, efficiency_multipliers, workload_type):
    base_throughput = calculate_base_throughput(sc, workload_type)
    efficiency_factor = 1.0

    if efficiency_multipliers:
        for values in efficiency_multipliers.values():
            for multiplier in values.values():
                efficiency_factor *= multiplier

    return base_throughput * efficiency_factor

def visualize_throughput(before, after, workload_type):
    labels = ["Before Tuning", "After Tuning"]
    values = [before, after]

    plt.figure(figsize=(6, 4))
    bars = plt.bar(labels, values, color=['lightgreen', 'green'])

    # Add text annotations on top of the bars
    for bar, value in zip(bars, values):
        plt.text(bar.get_x() + bar.get_width() / 2, 
                 bar.get_height() + 1,  # Slightly above the bar
                 f"{value:.2f}",  # Format with 2 decimal places
                 ha='center', va='bottom', fontsize=12, weight="bold")

    plt.xlabel("Tuning Stage")
    plt.ylabel("Estimated Throughput")
    plt.title(f"Throughput Estimation for {workload_type}")
    plt.ylim(0, max(values) * 1.2)  # Add some space above bars

    plt.savefig("throughput_comparison.png")  # Save as an image
    plt.show()


def main():
    workload_prediction = detect_workload(sample_counters)
    workload_type_probs = workload_prediction[2]
    workload_type = max(workload_type_probs, key=workload_type_probs.get)
    
    sc = sample_counters[:3]
    before_tuning = calculate_base_throughput(sc, workload_type)
    multipliers = EFFICIENCY_MULTIPLIERS.get(workload_type, {})
    after_tuning = estimate_throughput(sc, multipliers, workload_type)
    
    print(f"Estimated Base Throughput Before Tuning: {before_tuning}")
    print(f"Estimated Throughput After Tuning: {after_tuning}")
    
    with open("throughput_estimation_results.csv", "a") as log_file:
        log_file.write(f"{workload_type},{before_tuning},{after_tuning}\n")
    
    print("📊 Estimated throughput impact logged.")
    visualize_throughput(before_tuning, after_tuning, workload_type)

if __name__ == "__main__":
    main()
