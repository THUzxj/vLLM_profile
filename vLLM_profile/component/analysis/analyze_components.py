import pandas as pd

data = pd.read_csv('ncu_profile_result_v4_4B/flash_attn_kernel_metrics.csv')

print(data.head())

data["actual_memory_bandwidth_gbs"] = data['device_memory_mb'] / data['duration']

print(data)