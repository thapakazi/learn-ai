---
title: "Pandas for Log Analysis"
description: "Log analysis and infrastructure data management with Pandas DataFrames"
weight: 3
---

## Overview
Pandas is the go-to library for analyzing structured data in DevOps/SRE contexts. It excels at processing logs, analyzing incidents, tracking deployments, and generating reports from various infrastructure data sources.

## Core Concepts with DevOps Applications

### 1. DataFrames for Log Analysis
**DevOps Context**: Parsing and analyzing application logs

```python
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import re

# Sample log data
log_data = """
2024-01-15 10:23:45 INFO [api-gateway] Request from 192.168.1.100 - GET /api/users - 200 - 145ms
2024-01-15 10:23:46 ERROR [auth-service] Authentication failed for user john.doe - 401 - 23ms
2024-01-15 10:23:47 INFO [api-gateway] Request from 192.168.1.101 - POST /api/orders - 201 - 523ms
2024-01-15 10:23:48 WARN [database] Query timeout on orders table - 1245ms
2024-01-15 10:23:49 INFO [api-gateway] Request from 192.168.1.100 - GET /api/products - 200 - 89ms
2024-01-15 10:23:50 ERROR [payment-service] Payment processing failed - Transaction ID: TX123456 - 500 - 2341ms
2024-01-15 10:23:51 INFO [api-gateway] Request from 192.168.1.102 - GET /api/users/123 - 404 - 12ms
2024-01-15 10:23:52 WARN [cache-service] Cache miss rate high: 45% - Performance degradation expected
2024-01-15 10:23:53 INFO [api-gateway] Request from 192.168.1.100 - DELETE /api/sessions - 204 - 34ms
2024-01-15 10:23:54 ERROR [api-gateway] Request from 192.168.1.103 - GET /api/reports - 503 - Service Unavailable
"""

def parse_logs_to_dataframe(log_text):
    """Parse log text into a structured DataFrame"""
    
    lines = log_text.strip().split('\n')
    parsed_logs = []
    
    for line in lines:
        # Parse log line using regex
        pattern = r'(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}) (\w+) \[([^\]]+)\] (.+)'
        match = re.match(pattern, line)
        
        if match:
            timestamp_str, level, service, message = match.groups()
            
            # Extract response time if present
            time_match = re.search(r'(\d+)ms', message)
            response_time = int(time_match.group(1)) if time_match else None
            
            # Extract status code if present
            status_match = re.search(r'\b(\d{3})\b', message)
            status_code = int(status_match.group(1)) if status_match else None
            
            parsed_logs.append({
                'timestamp': pd.to_datetime(timestamp_str),
                'level': level,
                'service': service,
                'message': message,
                'response_time_ms': response_time,
                'status_code': status_code
            })
    
    return pd.DataFrame(parsed_logs)

# Parse logs into DataFrame
df_logs = parse_logs_to_dataframe(log_data)

# Analyze log data
print("Log Analysis Summary:")
print("-" * 50)
print(f"Total log entries: {len(df_logs)}")
print(f"\nLog levels distribution:")
print(df_logs['level'].value_counts())
print(f"\nServices with errors:")
error_services = df_logs[df_logs['level'] == 'ERROR']['service'].value_counts()
print(error_services)
print(f"\nAverage response time by service:")
response_times = df_logs.groupby('service')['response_time_ms'].mean().dropna()
print(response_times.round(2))
```

### 2. Time Series Analysis for Metrics
**DevOps Context**: Analyzing infrastructure metrics over time

```python
# Generate sample infrastructure metrics
def generate_infrastructure_metrics():
    """Generate realistic infrastructure metrics data"""
    
    # Create time index
    date_range = pd.date_range(
        start='2024-01-01 00:00:00',
        end='2024-01-07 23:59:59',
        freq='1H'
    )
    
    # Generate metrics with daily and weekly patterns
    n_points = len(date_range)
    
    # CPU usage with daily pattern (higher during business hours)
    base_cpu = 30
    daily_pattern = np.sin(np.arange(n_points) * 2 * np.pi / 24) * 20
    weekly_pattern = np.where(
        pd.Series(date_range).dt.dayofweek.isin([5, 6]),  # Weekend
        -10, 0
    )
    cpu_usage = base_cpu + daily_pattern + weekly_pattern + np.random.normal(0, 5, n_points)
    cpu_usage = np.clip(cpu_usage, 5, 95)
    
    # Memory usage (gradual increase with occasional drops)
    memory_usage = 40 + np.cumsum(np.random.normal(0.1, 2, n_points)) / 50
    memory_usage = np.clip(memory_usage, 20, 90)
    
    # Request count (business hours pattern)
    hour_of_day = pd.Series(date_range).dt.hour
    is_business_hours = (hour_of_day >= 8) & (hour_of_day <= 18)
    is_weekday = ~pd.Series(date_range).dt.dayofweek.isin([5, 6])
    request_count = np.where(
        is_business_hours & is_weekday,
        np.random.poisson(1000, n_points),
        np.random.poisson(200, n_points)
    )
    
    # Error rate (increases with load)
    error_rate = (request_count / 1000) * np.random.uniform(0.5, 2, n_points)
    
    df = pd.DataFrame({
        'timestamp': date_range,
        'cpu_usage': cpu_usage,
        'memory_usage': memory_usage,
        'request_count': request_count,
        'error_rate': error_rate,
        'response_time_p99': 100 + (cpu_usage * 2) + np.random.normal(0, 20, n_points)
    })
    
    return df

# Create metrics DataFrame
df_metrics = generate_infrastructure_metrics()

# Set timestamp as index for time series operations
df_metrics.set_index('timestamp', inplace=True)

# Resample to different time windows
hourly_avg = df_metrics.resample('1H').mean()
daily_avg = df_metrics.resample('1D').mean()
daily_max = df_metrics.resample('1D').max()

print("Infrastructure Metrics Analysis:")
print("-" * 50)
print("\nDaily Average Metrics:")
print(daily_avg[['cpu_usage', 'memory_usage', 'error_rate']].round(2))

# Identify peak usage times
peak_hours = df_metrics.groupby(df_metrics.index.hour)['request_count'].mean().sort_values(ascending=False)
print(f"\nTop 5 Peak Hours (by average request count):")
print(peak_hours.head().round(0))

# Correlation analysis
print("\nMetric Correlations:")
correlations = df_metrics[['cpu_usage', 'memory_usage', 'request_count', 'error_rate', 'response_time_p99']].corr()
print(correlations.round(3))
```

### 3. Incident Analysis and Reporting
**DevOps Context**: Tracking and analyzing production incidents

```python
# Create incident tracking DataFrame
incidents_data = {
    'incident_id': ['INC001', 'INC002', 'INC003', 'INC004', 'INC005', 'INC006', 'INC007', 'INC008'],
    'timestamp': pd.to_datetime([
        '2024-01-01 03:45:00', '2024-01-02 14:30:00', '2024-01-03 09:15:00',
        '2024-01-04 22:00:00', '2024-01-05 11:20:00', '2024-01-06 16:45:00',
        '2024-01-07 08:30:00', '2024-01-08 20:15:00'
    ]),
    'service': ['auth-service', 'database', 'api-gateway', 'payment-service', 
                'cache-service', 'database', 'auth-service', 'api-gateway'],
    'severity': ['P1', 'P2', 'P3', 'P1', 'P2', 'P1', 'P3', 'P2'],
    'duration_minutes': [45, 120, 15, 180, 60, 240, 30, 90],
    'root_cause': ['Memory leak', 'Slow queries', 'Network timeout', 'Third-party API down',
                   'Redis connection pool', 'Disk space', 'Certificate expired', 'DDoS attack'],
    'affected_users': [5000, 2000, 500, 10000, 3000, 15000, 1000, 8000],
    'resolved_by': ['John', 'Sarah', 'Mike', 'John', 'Sarah', 'Mike', 'John', 'Sarah']
}

df_incidents = pd.DataFrame(incidents_data)

# Add calculated fields
df_incidents['mttr_hours'] = df_incidents['duration_minutes'] / 60
df_incidents['impact_score'] = (
    df_incidents['affected_users'] * 
    df_incidents['severity'].map({'P1': 3, 'P2': 2, 'P3': 1}) *
    df_incidents['duration_minutes'] / 60
)

# Incident analysis
print("Incident Analysis Report:")
print("-" * 50)

# MTTR by severity
mttr_by_severity = df_incidents.groupby('severity')['mttr_hours'].agg(['mean', 'median', 'max'])
print("\nMTTR by Severity (hours):")
print(mttr_by_severity.round(2))

# Most problematic services
service_incidents = df_incidents.groupby('service').agg({
    'incident_id': 'count',
    'duration_minutes': 'sum',
    'affected_users': 'sum',
    'impact_score': 'sum'
}).rename(columns={'incident_id': 'incident_count', 'duration_minutes': 'total_downtime_min'})

service_incidents = service_incidents.sort_values('impact_score', ascending=False)
print("\nService Reliability Report:")
print(service_incidents)

# Root cause analysis
root_cause_freq = df_incidents['root_cause'].value_counts()
print("\nTop Root Causes:")
print(root_cause_freq)

# On-call performance
oncall_stats = df_incidents.groupby('resolved_by').agg({
    'incident_id': 'count',
    'mttr_hours': 'mean',
    'severity': lambda x: (x == 'P1').sum()
}).rename(columns={'incident_id': 'incidents_resolved', 'severity': 'p1_incidents'})

print("\nOn-Call Engineer Performance:")
print(oncall_stats.round(2))
```

### 4. Deployment Tracking and Analysis
**DevOps Context**: Monitoring deployment success rates and rollback patterns

```python
# Create deployment tracking data
deployments = {
    'deployment_id': [f'DEP{i:04d}' for i in range(1, 51)],
    'timestamp': pd.date_range(start='2024-01-01', periods=50, freq='4H'),
    'service': np.random.choice(['api', 'auth', 'database', 'frontend', 'backend'], 50),
    'environment': np.random.choice(['dev', 'staging', 'production'], 50, p=[0.4, 0.3, 0.3]),
    'version': [f'v1.{i//10}.{i%10}' for i in range(50)],
    'deployment_method': np.random.choice(['blue-green', 'canary', 'rolling'], 50),
    'duration_seconds': np.random.normal(300, 100, 50).clip(60, 600),
    'status': np.random.choice(['success', 'failed', 'rolled_back'], 50, p=[0.8, 0.1, 0.1]),
    'deployed_by': np.random.choice(['CI/CD', 'Manual'], 50, p=[0.7, 0.3])
}

df_deployments = pd.DataFrame(deployments)

# Add calculated fields
df_deployments['date'] = df_deployments['timestamp'].dt.date
df_deployments['hour'] = df_deployments['timestamp'].dt.hour
df_deployments['weekday'] = df_deployments['timestamp'].dt.day_name()
df_deployments['is_business_hours'] = df_deployments['hour'].between(9, 17)

# Deployment success analysis
print("Deployment Analysis Report:")
print("-" * 50)

# Success rate by environment
success_by_env = pd.crosstab(
    df_deployments['environment'], 
    df_deployments['status'],
    normalize='index'
) * 100

print("\nSuccess Rate by Environment (%):")
print(success_by_env.round(1))

# Deployment frequency and patterns
deploy_by_day = df_deployments.groupby('weekday')['deployment_id'].count()
deploy_by_day = deploy_by_day.reindex(['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday'])
print("\nDeployments by Day of Week:")
print(deploy_by_day)

# Best practices compliance
df_deployments['follows_best_practice'] = (
    (df_deployments['environment'] != 'production') | 
    (df_deployments['is_business_hours'] & (df_deployments['weekday'].isin(['Monday', 'Tuesday', 'Wednesday', 'Thursday'])))
)

compliance_rate = df_deployments['follows_best_practice'].mean() * 100
print(f"\nBest Practice Compliance Rate: {compliance_rate:.1f}%")

# Service deployment reliability
service_reliability = df_deployments.groupby('service').agg({
    'status': lambda x: (x == 'success').mean() * 100,
    'duration_seconds': 'mean',
    'deployment_id': 'count'
}).rename(columns={
    'status': 'success_rate',
    'duration_seconds': 'avg_duration',
    'deployment_id': 'total_deployments'
})

print("\nService Deployment Reliability:")
print(service_reliability.round(1))
```

### 5. Cost Analysis and Optimization
**DevOps Context**: Analyzing cloud infrastructure costs

```python
# Generate cloud cost data
def generate_cloud_costs():
    dates = pd.date_range(start='2024-01-01', end='2024-01-31', freq='D')
    
    services = []
    for date in dates:
        # Compute costs (higher on weekdays)
        is_weekday = date.weekday() < 5
        compute_base = 500 if is_weekday else 300
        
        services.append({
            'date': date,
            'service': 'EC2',
            'cost': compute_base + np.random.normal(0, 50),
            'usage_hours': (24 * 20) if is_weekday else (24 * 10),
            'region': 'us-east-1'
        })
        
        # Storage costs (gradually increasing)
        day_num = (date - dates[0]).days
        services.append({
            'date': date,
            'service': 'S3',
            'cost': 100 + day_num * 2 + np.random.normal(0, 10),
            'usage_gb': 5000 + day_num * 100,
            'region': 'us-east-1'
        })
        
        # Database costs
        services.append({
            'date': date,
            'service': 'RDS',
            'cost': 200 + np.random.normal(0, 20),
            'usage_hours': 24,
            'region': 'us-east-1'
        })
        
        # Network costs
        services.append({
            'date': date,
            'service': 'CloudFront',
            'cost': 50 + np.random.exponential(30),
            'usage_gb': np.random.exponential(1000),
            'region': 'global'
        })
    
    return pd.DataFrame(services)

df_costs = generate_cloud_costs()

# Cost analysis
print("Cloud Cost Analysis:")
print("-" * 50)

# Total costs by service
total_by_service = df_costs.groupby('service')['cost'].sum().sort_values(ascending=False)
print("\nTotal Costs by Service (January):")
print(total_by_service.round(2))

# Daily cost trends
daily_costs = df_costs.groupby('date')['cost'].sum()
print(f"\nDaily Cost Statistics:")
print(f"  Average: ${daily_costs.mean():.2f}")
print(f"  Minimum: ${daily_costs.min():.2f}")
print(f"  Maximum: ${daily_costs.max():.2f}")
print(f"  Total: ${daily_costs.sum():.2f}")

# Cost efficiency metrics
df_costs['cost_per_unit'] = df_costs.apply(
    lambda row: row['cost'] / row.get('usage_hours', row.get('usage_gb', 1)),
    axis=1
)

efficiency = df_costs.groupby('service')['cost_per_unit'].mean()
print("\nCost Efficiency (per unit):")
print(efficiency.round(4))

# Identify cost anomalies
service_daily_avg = df_costs.groupby(['service', 'date'])['cost'].sum().reset_index()
for service in df_costs['service'].unique():
    service_data = service_daily_avg[service_daily_avg['service'] == service]['cost']
    mean_cost = service_data.mean()
    std_cost = service_data.std()
    anomalies = service_data[service_data > mean_cost + 2 * std_cost]
    if len(anomalies) > 0:
        print(f"\nCost anomalies detected for {service}:")
        print(f"  Normal range: ${mean_cost - 2*std_cost:.2f} - ${mean_cost + 2*std_cost:.2f}")
        print(f"  Anomalies found: {len(anomalies)} days")
```

## Real-World DevOps Examples

### Example 1: API Performance Dashboard Data Preparation

```python
class APIPerformanceAnalyzer:
    """Analyze API performance metrics from logs"""
    
    def __init__(self):
        self.df = None
    
    def load_api_logs(self, log_file=None):
        """Load and parse API access logs"""
        # Simulate API log data
        n_records = 10000
        
        endpoints = ['/api/users', '/api/orders', '/api/products', '/api/auth/login', '/api/health']
        methods = ['GET', 'POST', 'PUT', 'DELETE']
        
        data = {
            'timestamp': pd.date_range(start='2024-01-01', periods=n_records, freq='1min'),
            'endpoint': np.random.choice(endpoints, n_records, p=[0.3, 0.25, 0.25, 0.15, 0.05]),
            'method': np.random.choice(methods, n_records, p=[0.5, 0.3, 0.15, 0.05]),
            'status_code': np.random.choice([200, 201, 400, 401, 404, 500, 503], n_records, 
                                          p=[0.7, 0.1, 0.05, 0.03, 0.05, 0.05, 0.02]),
            'response_time_ms': np.random.lognormal(4, 1, n_records),
            'user_id': np.random.choice(range(1, 1001), n_records),
            'ip_address': [f"192.168.1.{np.random.randint(1, 255)}" for _ in range(n_records)]
        }
        
        self.df = pd.DataFrame(data)
        self.df['success'] = self.df['status_code'] < 400
        self.df['hour'] = self.df['timestamp'].dt.hour
        
    def calculate_sli_metrics(self, time_window='1H'):
        """Calculate Service Level Indicators"""
        
        # Resample data by time window
        resampled = self.df.set_index('timestamp').resample(time_window)
        
        sli_metrics = pd.DataFrame({
            'availability': resampled['success'].mean() * 100,
            'p50_latency': resampled['response_time_ms'].quantile(0.5),
            'p95_latency': resampled['response_time_ms'].quantile(0.95),
            'p99_latency': resampled['response_time_ms'].quantile(0.99),
            'error_rate': (1 - resampled['success'].mean()) * 100,
            'request_rate': resampled.size() / (pd.Timedelta(time_window).seconds / 60)  # requests per minute
        })
        
        return sli_metrics.round(2)
    
    def endpoint_analysis(self):
        """Analyze performance by endpoint"""
        
        endpoint_stats = self.df.groupby('endpoint').agg({
            'response_time_ms': ['mean', 'median', lambda x: x.quantile(0.95)],
            'success': lambda x: x.mean() * 100,
            'status_code': 'count'
        }).round(2)
        
        endpoint_stats.columns = ['avg_response', 'median_response', 'p95_response', 'success_rate', 'request_count']
        
        # Add error breakdown
        error_breakdown = self.df[~self.df['success']].groupby(['endpoint', 'status_code']).size().unstack(fill_value=0)
        
        return endpoint_stats, error_breakdown
    
    def identify_problematic_users(self, threshold_error_rate=20):
        """Identify users with high error rates"""
        
        user_stats = self.df.groupby('user_id').agg({
            'success': lambda x: (1 - x.mean()) * 100,  # error rate
            'response_time_ms': 'mean',
            'status_code': 'count'
        }).rename(columns={'success': 'error_rate', 'status_code': 'request_count'})
        
        problematic_users = user_stats[user_stats['error_rate'] > threshold_error_rate]
        
        return problematic_users.sort_values('error_rate', ascending=False)

# Use the analyzer
analyzer = APIPerformanceAnalyzer()
analyzer.load_api_logs()

# Get SLI metrics
sli_metrics = analyzer.calculate_sli_metrics('1H')
print("Hourly SLI Metrics (last 5 hours):")
print(sli_metrics.tail())

# Analyze endpoints
endpoint_stats, error_breakdown = analyzer.endpoint_analysis()
print("\nEndpoint Performance Summary:")
print(endpoint_stats)

# Find problematic users
problematic_users = analyzer.identify_problematic_users()
if len(problematic_users) > 0:
    print(f"\nUsers with >20% error rate: {len(problematic_users)}")
    print(problematic_users.head())
```

### Example 2: Kubernetes Pod Resource Analysis

```python
def analyze_k8s_metrics():
    """Analyze Kubernetes pod metrics"""
    
    # Simulate K8s pod metrics
    pods = []
    namespaces = ['default', 'production', 'staging', 'monitoring']
    
    for i in range(100):
        namespace = np.random.choice(namespaces)
        pods.append({
            'pod_name': f"{namespace}-app-{i:03d}",
            'namespace': namespace,
            'node': f"node-{np.random.randint(1, 11):02d}",
            'cpu_request_m': np.random.choice([100, 250, 500, 1000]),
            'cpu_usage_m': np.random.exponential(200),
            'memory_request_mi': np.random.choice([128, 256, 512, 1024]),
            'memory_usage_mi': np.random.normal(300, 100),
            'restart_count': np.random.poisson(0.5),
            'age_days': np.random.exponential(30),
            'status': np.random.choice(['Running', 'Pending', 'CrashLoopBackOff', 'Error'], 
                                     p=[0.9, 0.05, 0.03, 0.02])
        })
    
    df_pods = pd.DataFrame(pods)
    
    # Calculate resource efficiency
    df_pods['cpu_efficiency'] = (df_pods['cpu_usage_m'] / df_pods['cpu_request_m']) * 100
    df_pods['memory_efficiency'] = (df_pods['memory_usage_mi'] / df_pods['memory_request_mi']) * 100
    
    # Identify resource issues
    df_pods['over_cpu'] = df_pods['cpu_efficiency'] > 90
    df_pods['under_cpu'] = df_pods['cpu_efficiency'] < 10
    df_pods['over_memory'] = df_pods['memory_efficiency'] > 90
    df_pods['under_memory'] = df_pods['memory_efficiency'] < 10
    
    print("Kubernetes Resource Analysis:")
    print("-" * 50)
    
    # Namespace summary
    namespace_summary = df_pods.groupby('namespace').agg({
        'pod_name': 'count',
        'cpu_usage_m': 'sum',
        'memory_usage_mi': 'sum',
        'restart_count': 'sum',
        'status': lambda x: (x != 'Running').sum()
    }).rename(columns={
        'pod_name': 'pod_count',
        'cpu_usage_m': 'total_cpu_m',
        'memory_usage_mi': 'total_memory_mi',
        'restart_count': 'total_restarts',
        'status': 'unhealthy_pods'
    })
    
    print("\nNamespace Resource Usage:")
    print(namespace_summary)
    
    # Node distribution
    node_load = df_pods.groupby('node').agg({
        'pod_name': 'count',
        'cpu_usage_m': 'sum',
        'memory_usage_mi': 'sum'
    }).rename(columns={'pod_name': 'pod_count'})
    
    print("\nNode Load Distribution:")
    print(node_load.sort_values('cpu_usage_m', ascending=False).head())
    
    # Resource optimization opportunities
    print("\nResource Optimization Opportunities:")
    print(f"  Pods with <10% CPU usage: {df_pods['under_cpu'].sum()}")
    print(f"  Pods with >90% CPU usage: {df_pods['over_cpu'].sum()}")
    print(f"  Pods with <10% Memory usage: {df_pods['under_memory'].sum()}")
    print(f"  Pods with >90% Memory usage: {df_pods['over_memory'].sum()}")
    
    # Problem pods
    problem_pods = df_pods[
        (df_pods['status'] != 'Running') | 
        (df_pods['restart_count'] > 3) |
        (df_pods['over_cpu']) |
        (df_pods['over_memory'])
    ][['pod_name', 'namespace', 'status', 'restart_count', 'cpu_efficiency', 'memory_efficiency']]
    
    if len(problem_pods) > 0:
        print(f"\nProblem Pods ({len(problem_pods)} found):")
        print(problem_pods.head(10))
    
    return df_pods

# Run the analysis
k8s_metrics = analyze_k8s_metrics()
```

## Practice Exercises

1. **Log Aggregation Pipeline**: Build a pipeline that ingests logs from multiple sources, normalizes them, and creates a unified dashboard dataset.

2. **Capacity Forecasting**: Use historical data to predict when resources will need scaling based on growth trends.

3. **Incident Correlation**: Analyze incident data to find patterns and correlations between different types of failures.

4. **Cost Optimization Report**: Create a comprehensive cost analysis report that identifies optimization opportunities.

## References and Learning Resources

- [Pandas Official Documentation](https://pandas.pydata.org/docs/)
- [Pandas for Data Analysis (Wes McKinney)](https://www.oreilly.com/library/view/python-for-data/9781491957653/)
- [Real Python - Pandas DataFrames](https://realpython.com/pandas-dataframe/)
- [Pandas Cookbook](https://github.com/jvns/pandas-cookbook)
- [Time Series Analysis with Pandas](https://www.datacamp.com/tutorial/time-series-analysis-tutorial)
- [Log Analysis with Python and Pandas](https://www.elastic.co/blog/analyzing-logs-with-pandas)
- [DataCamp - Data Manipulation with Pandas](https://www.datacamp.com/courses/data-manipulation-with-pandas)
- [Effective Pandas (Matt Harrison)](https://store.metasnake.com/effective-pandas-book)