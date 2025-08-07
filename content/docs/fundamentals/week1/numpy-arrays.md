---
title: "NumPy for Infrastructure Metrics"
description: "Efficient numerical operations for DevOps metrics and performance analysis"
weight: 2
---

## Overview
NumPy is essential for efficient numerical operations in DevOps/SRE contexts, particularly when dealing with large-scale metrics, performance data, and system monitoring. It provides the foundation for analyzing infrastructure patterns and anomalies.

## Core Concepts with DevOps Applications

### 1. Arrays for Metrics Collection
**DevOps Context**: Storing and processing time-series metrics

```python
import numpy as np
from datetime import datetime, timedelta

# CPU metrics collected every minute for the last hour
cpu_metrics = np.array([
    45.2, 48.3, 52.1, 67.8, 72.3, 68.9, 71.2, 75.6, 82.3, 88.1,
    92.5, 94.2, 91.8, 87.3, 83.2, 79.5, 76.8, 73.2, 69.8, 66.4,
    62.1, 58.9, 55.3, 52.8, 49.2, 46.8, 44.3, 42.1, 40.8, 39.2,
    38.5, 37.9, 38.2, 39.8, 42.3, 45.6, 48.9, 52.3, 56.7, 61.2,
    65.8, 70.3, 74.8, 78.2, 81.5, 84.2, 86.3, 87.8, 88.9, 89.5,
    90.1, 89.8, 88.3, 86.2, 83.8, 81.2, 78.5, 75.8, 73.2, 70.8
])

# Quick statistics
print(f"Average CPU: {np.mean(cpu_metrics):.2f}%")
print(f"Peak CPU: {np.max(cpu_metrics):.2f}%")
print(f"Min CPU: {np.min(cpu_metrics):.2f}%")
print(f"Std Dev: {np.std(cpu_metrics):.2f}")

# Find periods of high load (> 80%)
high_load_indices = np.where(cpu_metrics > 80)[0]
print(f"High load periods (minutes): {high_load_indices}")
```

### 2. Multi-dimensional Arrays for Server Farms
**DevOps Context**: Managing metrics across multiple servers

```python
# Metrics for 10 servers over 24 hours (hourly samples)
# Dimensions: [servers, hours, metrics_type]
# metrics_type: 0=CPU, 1=Memory, 2=Disk I/O

server_metrics = np.random.rand(10, 24, 3) * 100

# Server names for reference
server_names = [f"web-{i:02d}" for i in range(10)]

def analyze_server_health(metrics, server_names):
    """Analyze health metrics across server farm"""
    
    # Average metrics per server
    avg_per_server = np.mean(metrics, axis=1)
    
    # Find servers with high average CPU (> 75%)
    high_cpu_servers = np.where(avg_per_server[:, 0] > 75)[0]
    
    # Find servers with high memory usage (> 85%)
    high_mem_servers = np.where(avg_per_server[:, 1] > 85)[0]
    
    # Calculate server health score (lower is better)
    health_scores = np.sum(avg_per_server, axis=1) / 3
    
    # Rank servers by health
    ranked_indices = np.argsort(health_scores)[::-1]
    
    print("Server Health Analysis:")
    print("-" * 40)
    for idx in ranked_indices[:5]:  # Top 5 problematic servers
        print(f"{server_names[idx]}: Score={health_scores[idx]:.1f} "
              f"CPU={avg_per_server[idx, 0]:.1f}% "
              f"Mem={avg_per_server[idx, 1]:.1f}% "
              f"IO={avg_per_server[idx, 2]:.1f}%")
    
    return {
        "high_cpu": [server_names[i] for i in high_cpu_servers],
        "high_memory": [server_names[i] for i in high_mem_servers],
        "health_scores": dict(zip(server_names, health_scores))
    }

# Analyze the server farm
results = analyze_server_health(server_metrics, server_names)
```

### 3. Statistical Operations for Anomaly Detection
**DevOps Context**: Detecting unusual patterns in system behavior

```python
def detect_anomalies(metrics, window_size=10, threshold=3):
    """
    Detect anomalies using moving average and standard deviation
    Used for identifying unusual spikes in metrics
    """
    # Calculate moving average
    moving_avg = np.convolve(metrics, np.ones(window_size)/window_size, mode='valid')
    
    # Calculate moving standard deviation
    moving_std = np.array([
        np.std(metrics[i:i+window_size]) 
        for i in range(len(metrics) - window_size + 1)
    ])
    
    # Detect anomalies (values beyond threshold * std from mean)
    anomalies = []
    for i in range(len(moving_avg)):
        actual_value = metrics[i + window_size - 1]
        if abs(actual_value - moving_avg[i]) > threshold * moving_std[i]:
            anomalies.append({
                'index': i + window_size - 1,
                'value': actual_value,
                'expected': moving_avg[i],
                'deviation': abs(actual_value - moving_avg[i]) / moving_std[i]
            })
    
    return anomalies

# Example: Detect anomalies in response times
response_times = np.random.normal(100, 15, 1000)  # Normal: 100ms ± 15ms
# Inject some anomalies
response_times[150] = 400  # Spike
response_times[500] = 350  # Another spike
response_times[750] = 10   # Drop

anomalies = detect_anomalies(response_times)
print(f"Found {len(anomalies)} anomalies in response times")
for anomaly in anomalies[:5]:
    print(f"  Time index {anomaly['index']}: {anomaly['value']:.1f}ms "
          f"(expected: {anomaly['expected']:.1f}ms, "
          f"{anomaly['deviation']:.1f} std devs)")
```

### 4. Array Operations for Capacity Planning
**DevOps Context**: Predicting resource needs based on historical data

```python
def capacity_planning(historical_usage, growth_rate=0.1, forecast_days=30):
    """
    Predict future capacity needs based on historical usage patterns
    """
    # Calculate trend using linear regression
    days = np.arange(len(historical_usage))
    coefficients = np.polyfit(days, historical_usage, 1)
    trend_line = np.poly1d(coefficients)
    
    # Project future usage
    future_days = np.arange(len(historical_usage), 
                           len(historical_usage) + forecast_days)
    projected_usage = trend_line(future_days)
    
    # Add growth factor
    projected_usage *= (1 + growth_rate)
    
    # Calculate required capacity (with 20% buffer)
    required_capacity = np.max(projected_usage) * 1.2
    
    # Find when we'll exceed current capacity
    current_capacity = 1000  # GB
    days_until_capacity = np.where(projected_usage > current_capacity)[0]
    
    return {
        'current_avg': np.mean(historical_usage),
        'projected_avg': np.mean(projected_usage),
        'peak_projected': np.max(projected_usage),
        'required_capacity': required_capacity,
        'days_until_capacity_exceeded': days_until_capacity[0] if len(days_until_capacity) > 0 else None
    }

# Historical storage usage (GB) over 90 days
storage_usage = np.linspace(600, 850, 90) + np.random.normal(0, 20, 90)

planning = capacity_planning(storage_usage)
print("Capacity Planning Report:")
print(f"  Current Average: {planning['current_avg']:.1f} GB")
print(f"  30-day Projected Average: {planning['projected_avg']:.1f} GB")
print(f"  Required Capacity: {planning['required_capacity']:.1f} GB")
if planning['days_until_capacity_exceeded']:
    print(f"  WARNING: Will exceed capacity in {planning['days_until_capacity_exceeded']} days!")
```

### 5. Performance Optimization with NumPy
**DevOps Context**: Efficient processing of large-scale metrics

```python
import time

def process_metrics_python(metrics):
    """Process metrics using pure Python (slow)"""
    result = []
    for value in metrics:
        if value > 50:
            result.append(value * 1.1)
        else:
            result.append(value * 0.9)
    return result

def process_metrics_numpy(metrics):
    """Process metrics using NumPy (fast)"""
    result = np.where(metrics > 50, metrics * 1.1, metrics * 0.9)
    return result

# Compare performance
large_metrics = np.random.rand(1000000) * 100

# Python approach
start = time.time()
python_result = process_metrics_python(large_metrics.tolist())
python_time = time.time() - start

# NumPy approach
start = time.time()
numpy_result = process_metrics_numpy(large_metrics)
numpy_time = time.time() - start

print(f"Python processing time: {python_time:.4f} seconds")
print(f"NumPy processing time: {numpy_time:.4f} seconds")
print(f"Speed improvement: {python_time/numpy_time:.1f}x faster")
```

## Real-World DevOps Examples

### Example 1: Network Traffic Analysis

```python
import numpy as np

class NetworkTrafficAnalyzer:
    """Analyze network traffic patterns for capacity planning"""
    
    def __init__(self, sampling_rate=60):  # seconds
        self.sampling_rate = sampling_rate
        self.traffic_data = []
    
    def add_sample(self, bytes_in, bytes_out, packets_in, packets_out):
        """Add a traffic sample"""
        self.traffic_data.append([bytes_in, bytes_out, packets_in, packets_out])
    
    def analyze_patterns(self):
        """Analyze traffic patterns"""
        if len(self.traffic_data) < 2:
            return None
        
        data = np.array(self.traffic_data)
        
        # Calculate bandwidth utilization (Mbps)
        bytes_total = data[:, 0] + data[:, 1]
        bandwidth_mbps = (bytes_total * 8) / (self.sampling_rate * 1_000_000)
        
        # Packet analysis
        packets_total = data[:, 2] + data[:, 3]
        avg_packet_size = bytes_total / np.maximum(packets_total, 1)
        
        # Detect traffic spikes (>2 std dev from mean)
        mean_traffic = np.mean(bandwidth_mbps)
        std_traffic = np.std(bandwidth_mbps)
        spikes = np.where(bandwidth_mbps > mean_traffic + 2 * std_traffic)[0]
        
        # Calculate percentiles for SLA monitoring
        percentiles = np.percentile(bandwidth_mbps, [50, 95, 99])
        
        return {
            'avg_bandwidth_mbps': mean_traffic,
            'peak_bandwidth_mbps': np.max(bandwidth_mbps),
            'p50_bandwidth': percentiles[0],
            'p95_bandwidth': percentiles[1],
            'p99_bandwidth': percentiles[2],
            'avg_packet_size': np.mean(avg_packet_size),
            'traffic_spikes': len(spikes),
            'spike_indices': spikes.tolist()
        }
    
    def predict_peak_hours(self, hourly_data):
        """Identify peak traffic hours"""
        hourly_avg = np.mean(hourly_data.reshape(-1, 24), axis=0)
        peak_hours = np.argsort(hourly_avg)[-3:]  # Top 3 hours
        return peak_hours, hourly_avg[peak_hours]

# Example usage
analyzer = NetworkTrafficAnalyzer()

# Simulate network traffic data
for _ in range(1440):  # 24 hours of minute-by-minute data
    bytes_in = np.random.exponential(1000000)  # Exponential distribution for traffic
    bytes_out = np.random.exponential(500000)
    packets_in = int(bytes_in / 1500)  # Assuming ~1500 byte packets
    packets_out = int(bytes_out / 1500)
    analyzer.add_sample(bytes_in, bytes_out, packets_in, packets_out)

results = analyzer.analyze_patterns()
print("Network Traffic Analysis:")
for key, value in results.items():
    if isinstance(value, float):
        print(f"  {key}: {value:.2f}")
    else:
        print(f"  {key}: {value}")
```

### Example 2: Load Balancer Distribution Analysis

```python
def analyze_load_distribution(request_counts, server_names):
    """
    Analyze how well load is distributed across servers
    """
    total_requests = np.sum(request_counts)
    expected_per_server = total_requests / len(request_counts)
    
    # Calculate distribution metrics
    distribution = request_counts / total_requests * 100
    std_dev = np.std(request_counts)
    cv = (std_dev / np.mean(request_counts)) * 100  # Coefficient of variation
    
    # Chi-square test for uniformity
    chi_square = np.sum((request_counts - expected_per_server) ** 2 / expected_per_server)
    
    # Identify over/under utilized servers
    deviation_pct = ((request_counts - expected_per_server) / expected_per_server) * 100
    overloaded = np.where(deviation_pct > 20)[0]
    underutilized = np.where(deviation_pct < -20)[0]
    
    print("Load Balancer Analysis:")
    print(f"  Total Requests: {total_requests:,.0f}")
    print(f"  Expected per server: {expected_per_server:,.0f}")
    print(f"  Standard Deviation: {std_dev:,.0f}")
    print(f"  Coefficient of Variation: {cv:.1f}%")
    print(f"  Chi-square statistic: {chi_square:.2f}")
    
    if len(overloaded) > 0:
        print(f"\n  Overloaded servers (>20% above expected):")
        for idx in overloaded:
            print(f"    {server_names[idx]}: {request_counts[idx]:,.0f} "
                  f"({deviation_pct[idx]:+.1f}%)")
    
    if len(underutilized) > 0:
        print(f"\n  Underutilized servers (>20% below expected):")
        for idx in underutilized:
            print(f"    {server_names[idx]}: {request_counts[idx]:,.0f} "
                  f"({deviation_pct[idx]:+.1f}%)")
    
    return {
        'distribution': distribution,
        'cv': cv,
        'chi_square': chi_square,
        'balanced': cv < 10  # Consider balanced if CV < 10%
    }

# Example: Analyze load distribution across 8 servers
server_names = [f"app-{i:02d}" for i in range(8)]
request_counts = np.array([98500, 102300, 99800, 121000, 95600, 97200, 103400, 98200])

analysis = analyze_load_distribution(request_counts, server_names)
```

### Example 3: SLA Compliance Monitoring

```python
def calculate_sla_metrics(response_times, sla_target=200):
    """
    Calculate SLA compliance metrics for response times
    """
    # Remove outliers using IQR method
    q1, q3 = np.percentile(response_times, [25, 75])
    iqr = q3 - q1
    lower_bound = q1 - 1.5 * iqr
    upper_bound = q3 + 1.5 * iqr
    filtered_times = response_times[(response_times >= lower_bound) & 
                                   (response_times <= upper_bound)]
    
    # Calculate percentiles
    percentiles = np.percentile(filtered_times, [50, 90, 95, 99, 99.9])
    
    # SLA compliance
    compliance_rate = (np.sum(response_times <= sla_target) / 
                      len(response_times)) * 100
    
    # Apdex score (Application Performance Index)
    satisfied = np.sum(response_times <= sla_target)
    tolerating = np.sum((response_times > sla_target) & 
                       (response_times <= sla_target * 4))
    apdex = (satisfied + tolerating * 0.5) / len(response_times)
    
    print("SLA Compliance Report:")
    print(f"  Target SLA: {sla_target}ms")
    print(f"  Compliance Rate: {compliance_rate:.2f}%")
    print(f"  Apdex Score: {apdex:.3f}")
    print(f"  Median (P50): {percentiles[0]:.1f}ms")
    print(f"  P90: {percentiles[1]:.1f}ms")
    print(f"  P95: {percentiles[2]:.1f}ms")
    print(f"  P99: {percentiles[3]:.1f}ms")
    print(f"  P99.9: {percentiles[4]:.1f}ms")
    
    return {
        'compliance_rate': compliance_rate,
        'apdex': apdex,
        'percentiles': dict(zip(['p50', 'p90', 'p95', 'p99', 'p99.9'], percentiles))
    }

# Generate sample response times (mix of normal and some slow requests)
normal_responses = np.random.normal(150, 30, 9500)
slow_responses = np.random.normal(400, 100, 500)
response_times = np.concatenate([normal_responses, slow_responses])
np.random.shuffle(response_times)

sla_metrics = calculate_sla_metrics(response_times)
```

## Practice Exercises

1. **Metric Aggregation**: Write a function that aggregates metrics from multiple data centers and calculates weighted averages based on traffic volume.

2. **Trend Detection**: Implement a function that detects upward or downward trends in system metrics using linear regression.

3. **Resource Correlation**: Create a script that finds correlations between different metrics (CPU, memory, network) to identify resource bottlenecks.

4. **Percentile Monitoring**: Build a monitoring system that tracks percentile-based SLIs (Service Level Indicators) over time.

## References and Learning Resources

- [NumPy Official Documentation](https://numpy.org/doc/stable/)
- [NumPy for Data Science](https://www.datacamp.com/community/tutorials/python-numpy-tutorial)
- [Scientific Computing with NumPy](https://scipy-lectures.org/intro/numpy/index.html)
- [NumPy Illustrated: The Visual Guide](https://betterprogramming.pub/numpy-illustrated-the-visual-guide-to-numpy-3b1d4976de1d)
- [Performance Python: NumPy](https://www.oreilly.com/library/view/high-performance-python/9781492055013/)
- [Time Series Analysis with NumPy](https://machinelearningmastery.com/time-series-data-stationary-python/)
- [NumPy Financial Functions](https://numpy.org/numpy-financial/latest/)
- [NumPy Random Sampling for Simulations](https://numpy.org/doc/stable/reference/random/index.html)