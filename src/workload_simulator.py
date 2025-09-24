# workload_simulator.py content
import numpy as np
import pandas as pd
from pyDOE import lhs

def generate_workloads(output_file, num_samples=2000):
    #Generate mixed workloads using LHS (CPU, memory, disk)
    samples = lhs(3, samples=num_samples)
    workloads = pd.DataFrame(samples, columns=['cpu_load', 'memory_load', 'disk_load'])
    workloads.to_csv(output_file, index=False)
    print("Mixed workloads generated using LHS.")

if __name__ == "__main__":
    generate_workloads('../data/workloads.csv') 
   



