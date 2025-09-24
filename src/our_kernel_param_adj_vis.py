import subprocess
import numpy as np
import time
import os
import pickle
import matplotlib.pyplot as plt
from workload_detector import detect_workload

# Kernel parameter profiles based on workload types
WORKLOAD_PROFILES = {
    "CPU-intensive": {"vm.swappiness": 10},
    "Memory-intensive": {
        "vm.swappiness": 80,
        "vm.dirty_background_ratio": 5,
        "vm.dirty_ratio": 30,
        "vm.min_free_kbytes": 131072,
    },
    "Disk-intensive": {
        "vm.dirty_expire_centisecs": 100,
        "vm.dirty_writeback_centisecs": 50,
    }
}

# Mapping workload type numbers to profile names
WORKLOAD_MAPPING = {0: "CPU-intensive", 1: "Memory-intensive", 2: "Disk-intensive"}

def set_sysctl(param, value):
    """Apply sysctl parameter."""
    try:
        subprocess.run(["sudo", "sysctl", f"{param}={value}"], check=True)
        print(f"Set {param} to {value}")
    except Exception as e:
        print(f"Failed to set {param}: {e}")

def display_results(workload_name, applied_params):
    """Display the detected workload and kernel parameters using Matplotlib."""
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.set_xticks([])
    ax.set_yticks([])
    ax.axis("off")

    text = f"DETECTED WORKLOAD TYPE: {workload_name}\n\n"
    text += "ADJUSTED KERNEL PARAMETERS:\n\n"
    
    if applied_params:
        for param, value in applied_params.items():
            text += f"  - {param}: {value}\n"
    else:
        text += "  No parameters applied.\n"

    ax.text(0.05, 0.95, text, fontsize=12, va="top", ha="left", family="monospace")

    # Save the image
    plt.savefig("kernel_param_adjustment.png", dpi=200, bbox_inches="tight")

    # Show the image as a popup
    plt.show()

def apply_kernel_params_for_workload(workload_features):
    """Adjust kernel parameters based on detected workload."""
    workload_type, _, _ = detect_workload(workload_features)
    workload_name = WORKLOAD_MAPPING.get(workload_type, "Unknown")
    
    applied_params = {}

    if workload_name in WORKLOAD_PROFILES:
        for param, value in WORKLOAD_PROFILES[workload_name].items():
            set_sysctl(param, value)
            applied_params[param] = value
        print(f"✅ Applied parameters for {workload_name} workload.")
    else:
        print(f"⚠️ No profile found for workload type: {workload_name}")

    # Display results with visualization
    display_results(workload_name, applied_params)

if __name__ == "__main__":
    with open("input.txt", "rb") as file:
        sample_counters = pickle.load(file)
    print("\n📊 Input System Counters:\n", sample_counters[0:3])

    apply_kernel_params_for_workload(sample_counters)

