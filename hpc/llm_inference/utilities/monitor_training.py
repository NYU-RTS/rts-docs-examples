#!/usr/bin/env python3
# monitor_training.py

import psutil
import torch
import time
import json
from datetime import datetime

def monitor_system():
    """Monitor system resources during training"""
    
    stats = {
        'timestamp': datetime.now().isoformat(),
        'cpu_percent': psutil.cpu_percent(),
        'memory': {
            'total': psutil.virtual_memory().total,
            'available': psutil.virtual_memory().available,
            'percent': psutil.virtual_memory().percent
        }
    }
    
    # GPU stats if available
    if torch.cuda.is_available():
        gpu_stats = []
        for i in range(torch.cuda.device_count()):
            gpu_memory = torch.cuda.get_device_properties(i).total_memory
            gpu_allocated = torch.cuda.memory_allocated(i)
            gpu_cached = torch.cuda.memory_reserved(i)
            
            gpu_stats.append({
                'device': i,
                'name': torch.cuda.get_device_name(i),
                'total_memory': gpu_memory,
                'allocated_memory': gpu_allocated,
                'cached_memory': gpu_cached,
                'memory_percent': (gpu_allocated / gpu_memory) * 100
            })
        
        stats['gpus'] = gpu_stats
    
    return stats

def log_stats(output_file="training_monitor.jsonl"):
    """Log system stats to file"""
    stats = monitor_system()
    
    with open(output_file, 'a') as f:
        f.write(json.dumps(stats) + '\n')
    
    return stats

if __name__ == "__main__":
    import sys
    interval = int(sys.argv[1]) if len(sys.argv) > 1 else 60
    
    print(f"Monitoring system every {interval} seconds...")
    
    try:
        while True:
            stats = log_stats()
            print(f"[{stats['timestamp']}] CPU: {stats['cpu_percent']:.1f}% | "
                  f"Memory: {stats['memory']['percent']:.1f}%")
            
            if 'gpus' in stats:
                for gpu in stats['gpus']:
                    print(f"  GPU {gpu['device']}: {gpu['memory_percent']:.1f}% memory")
            
            time.sleep(interval)
    except KeyboardInterrupt:
        print("\nMonitoring stopped.")