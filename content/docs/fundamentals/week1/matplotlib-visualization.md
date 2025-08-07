---
title: "Matplotlib for Monitoring Dashboards"
description: "Creating professional monitoring dashboards and infrastructure visualizations"
weight: 4
---

## Overview
Matplotlib is essential for creating monitoring dashboards, performance reports, and infrastructure visualizations in DevOps/SRE. It transforms metrics and logs into actionable insights through clear, professional visualizations.

## Core Concepts with DevOps Applications

### 1. Basic Monitoring Dashboards
**DevOps Context**: Real-time system metrics visualization

```python
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import matplotlib.dates as mdates
from matplotlib.gridspec import GridSpec

# Generate sample monitoring data
def generate_monitoring_data():
    """Generate realistic monitoring metrics"""
    hours = 24
    time_points = pd.date_range(start='2024-01-15', periods=hours*60, freq='1min')
    
    # CPU usage with spikes
    cpu = 40 + 10 * np.sin(np.linspace(0, 4*np.pi, len(time_points)))
    cpu += np.random.normal(0, 5, len(time_points))
    cpu[300:320] = 85 + np.random.normal(0, 5, 20)  # Spike
    
    # Memory usage with gradual increase
    memory = 50 + np.linspace(0, 20, len(time_points))
    memory += np.random.normal(0, 3, len(time_points))
    
    # Network I/O
    network_in = np.random.exponential(50, len(time_points))
    network_out = np.random.exponential(30, len(time_points))
    
    # Disk I/O
    disk_read = np.random.exponential(20, len(time_points))
    disk_write = np.random.exponential(15, len(time_points))
    
    return pd.DataFrame({
        'timestamp': time_points,
        'cpu': np.clip(cpu, 0, 100),
        'memory': np.clip(memory, 0, 100),
        'network_in': network_in,
        'network_out': network_out,
        'disk_read': disk_read,
        'disk_write': disk_write
    })

# Create monitoring dashboard
def create_monitoring_dashboard(df):
    """Create a comprehensive monitoring dashboard"""
    
    fig = plt.figure(figsize=(16, 10))
    fig.suptitle('System Monitoring Dashboard - Production Server', fontsize=16, fontweight='bold')
    
    gs = GridSpec(3, 2, figure=fig, hspace=0.3, wspace=0.3)
    
    # CPU Usage
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.plot(df['timestamp'], df['cpu'], color='#FF6B6B', linewidth=1.5)
    ax1.axhline(y=80, color='orange', linestyle='--', alpha=0.7, label='Warning (80%)')
    ax1.axhline(y=90, color='red', linestyle='--', alpha=0.7, label='Critical (90%)')
    ax1.fill_between(df['timestamp'], df['cpu'], alpha=0.3, color='#FF6B6B')
    ax1.set_title('CPU Usage (%)', fontweight='bold')
    ax1.set_ylabel('Usage (%)')
    ax1.set_ylim(0, 100)
    ax1.grid(True, alpha=0.3)
    ax1.legend(loc='upper right')
    
    # Memory Usage
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.plot(df['timestamp'], df['memory'], color='#4ECDC4', linewidth=1.5)
    ax2.axhline(y=85, color='orange', linestyle='--', alpha=0.7, label='Warning (85%)')
    ax2.axhline(y=95, color='red', linestyle='--', alpha=0.7, label='Critical (95%)')
    ax2.fill_between(df['timestamp'], df['memory'], alpha=0.3, color='#4ECDC4')
    ax2.set_title('Memory Usage (%)', fontweight='bold')
    ax2.set_ylabel('Usage (%)')
    ax2.set_ylim(0, 100)
    ax2.grid(True, alpha=0.3)
    ax2.legend(loc='upper right')
    
    # Network I/O
    ax3 = fig.add_subplot(gs[1, 0])
    ax3.plot(df['timestamp'], df['network_in'], label='Inbound', color='#45B7D1', linewidth=1)
    ax3.plot(df['timestamp'], df['network_out'], label='Outbound', color='#FFA07A', linewidth=1)
    ax3.set_title('Network I/O (MB/s)', fontweight='bold')
    ax3.set_ylabel('Throughput (MB/s)')
    ax3.grid(True, alpha=0.3)
    ax3.legend()
    
    # Disk I/O
    ax4 = fig.add_subplot(gs[1, 1])
    ax4.plot(df['timestamp'], df['disk_read'], label='Read', color='#98D8C8', linewidth=1)
    ax4.plot(df['timestamp'], df['disk_write'], label='Write', color='#F7DC6F', linewidth=1)
    ax4.set_title('Disk I/O (MB/s)', fontweight='bold')
    ax4.set_ylabel('Throughput (MB/s)')
    ax4.grid(True, alpha=0.3)
    ax4.legend()
    
    # Combined System Health Score
    ax5 = fig.add_subplot(gs[2, :])
    
    # Calculate health score (inverse of resource usage)
    health_score = 100 - (df['cpu'] * 0.4 + df['memory'] * 0.4 + 
                          np.clip(df['network_in'] + df['network_out'], 0, 100) * 0.1 +
                          np.clip(df['disk_read'] + df['disk_write'], 0, 100) * 0.1)
    
    ax5.plot(df['timestamp'], health_score, color='#2ECC71', linewidth=2)
    ax5.fill_between(df['timestamp'], health_score, alpha=0.3, color='#2ECC71')
    ax5.axhline(y=30, color='red', linestyle='--', alpha=0.7, label='Poor Health (<30)')
    ax5.axhline(y=60, color='orange', linestyle='--', alpha=0.7, label='Fair Health (30-60)')
    ax5.set_title('System Health Score', fontweight='bold')
    ax5.set_ylabel('Health Score')
    ax5.set_xlabel('Time')
    ax5.set_ylim(0, 100)
    ax5.grid(True, alpha=0.3)
    ax5.legend()
    
    # Format x-axis for all subplots
    for ax in [ax1, ax2, ax3, ax4, ax5]:
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
        ax.xaxis.set_major_locator(mdates.HourLocator(interval=4))
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45)
    
    plt.tight_layout()
    return fig

# Generate and visualize data
df_monitoring = generate_monitoring_data()
dashboard = create_monitoring_dashboard(df_monitoring)
plt.show()
```

### 2. SLA and Performance Reports
**DevOps Context**: Service Level Agreement compliance visualization

```python
def create_sla_report(service_name="API Gateway"):
    """Create SLA compliance visualization report"""
    
    # Generate sample SLA data
    days = pd.date_range(start='2024-01-01', end='2024-01-31', freq='D')
    
    # Daily metrics
    availability = np.random.normal(99.5, 0.5, len(days))
    availability[5] = 98.2  # Incident day
    availability[15] = 97.8  # Another incident
    availability = np.clip(availability, 95, 100)
    
    response_time_p99 = np.random.normal(200, 30, len(days))
    response_time_p99[5] = 450  # Spike
    response_time_p99[15] = 380
    
    error_rate = np.random.exponential(0.5, len(days))
    error_rate[5] = 3.2
    error_rate[15] = 2.8
    
    # Create figure with subplots
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(f'SLA Compliance Report - {service_name} (January 2024)', 
                 fontsize=16, fontweight='bold')
    
    # 1. Availability Trend
    ax1 = axes[0, 0]
    ax1.plot(days, availability, marker='o', color='#2ECC71', linewidth=2, markersize=4)
    ax1.axhline(y=99.9, color='gold', linestyle='--', label='SLA Target (99.9%)')
    ax1.fill_between(days, availability, 99.9, where=(availability >= 99.9), 
                     color='green', alpha=0.3, label='Above SLA')
    ax1.fill_between(days, availability, 99.9, where=(availability < 99.9), 
                     color='red', alpha=0.3, label='Below SLA')
    ax1.set_title('Daily Availability (%)', fontweight='bold')
    ax1.set_ylabel('Availability (%)')
    ax1.set_ylim(97, 100)
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    
    # 2. Response Time P99
    ax2 = axes[0, 1]
    bars = ax2.bar(days, response_time_p99, color=['red' if x > 300 else 'green' 
                                                   for x in response_time_p99], alpha=0.7)
    ax2.axhline(y=300, color='red', linestyle='--', label='SLA Limit (300ms)')
    ax2.set_title('P99 Response Time (ms)', fontweight='bold')
    ax2.set_ylabel('Response Time (ms)')
    ax2.legend()
    ax2.grid(True, alpha=0.3, axis='y')
    
    # 3. Error Rate Distribution
    ax3 = axes[1, 0]
    
    # Create box plot for weekly error rates
    weekly_errors = [error_rate[i:i+7] for i in range(0, len(error_rate), 7)]
    week_labels = [f'Week {i+1}' for i in range(len(weekly_errors))]
    
    bp = ax3.boxplot(weekly_errors, labels=week_labels, patch_artist=True)
    for patch in bp['boxes']:
        patch.set_facecolor('#FF6B6B')
        patch.set_alpha(0.7)
    
    ax3.axhline(y=1, color='orange', linestyle='--', label='Warning (1%)')
    ax3.axhline(y=2, color='red', linestyle='--', label='Critical (2%)')
    ax3.set_title('Weekly Error Rate Distribution (%)', fontweight='bold')
    ax3.set_ylabel('Error Rate (%)')
    ax3.legend()
    ax3.grid(True, alpha=0.3, axis='y')
    
    # 4. SLA Summary Gauge
    ax4 = axes[1, 1]
    ax4.axis('off')
    
    # Calculate monthly SLA compliance
    monthly_availability = np.mean(availability)
    sla_violations = np.sum(response_time_p99 > 300)
    avg_error_rate = np.mean(error_rate)
    
    # Create text summary
    summary_text = f"""
    Monthly SLA Summary
    {'='*30}
    
    Availability: {monthly_availability:.2f}%
    Status: {'✅ PASS' if monthly_availability >= 99.9 else '❌ FAIL'}
    
    P99 Response Time Violations: {sla_violations} days
    Status: {'✅ PASS' if sla_violations <= 2 else '⚠️ WARNING' if sla_violations <= 5 else '❌ FAIL'}
    
    Average Error Rate: {avg_error_rate:.2f}%
    Status: {'✅ PASS' if avg_error_rate < 1 else '⚠️ WARNING' if avg_error_rate < 2 else '❌ FAIL'}
    
    Overall Compliance: {'✅ COMPLIANT' if monthly_availability >= 99.9 and sla_violations <= 2 and avg_error_rate < 1 else '❌ NON-COMPLIANT'}
    """
    
    ax4.text(0.5, 0.5, summary_text, ha='center', va='center', fontsize=11,
            fontfamily='monospace', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    # Format dates on x-axis
    for ax in [ax1, ax2]:
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%m/%d'))
        ax.xaxis.set_major_locator(mdates.DayLocator(interval=5))
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45)
    
    plt.tight_layout()
    return fig

# Create SLA report
sla_report = create_sla_report()
plt.show()
```

### 3. Capacity Planning Visualizations
**DevOps Context**: Resource utilization trends and forecasting

```python
def create_capacity_planning_viz():
    """Visualize capacity trends and projections"""
    
    # Historical data (90 days)
    historical_days = pd.date_range(start='2023-11-01', end='2024-01-31', freq='D')
    
    # Storage growth pattern
    storage_base = 500  # GB
    storage_growth = np.linspace(0, 300, len(historical_days))
    storage_noise = np.random.normal(0, 20, len(historical_days))
    storage_used = storage_base + storage_growth + storage_noise
    
    # CPU usage pattern (weekly cycles)
    cpu_pattern = 40 + 20 * np.sin(np.linspace(0, 26*np.pi, len(historical_days)))
    cpu_noise = np.random.normal(0, 5, len(historical_days))
    cpu_used = cpu_pattern + cpu_noise
    
    # Future projection (30 days)
    future_days = pd.date_range(start='2024-02-01', end='2024-03-01', freq='D')
    
    # Linear regression for storage projection
    from sklearn.linear_model import LinearRegression
    X = np.arange(len(historical_days)).reshape(-1, 1)
    model = LinearRegression()
    model.fit(X, storage_used)
    
    X_future = np.arange(len(historical_days), len(historical_days) + len(future_days)).reshape(-1, 1)
    storage_projected = model.predict(X_future)
    
    # Create visualization
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('Capacity Planning Dashboard', fontsize=16, fontweight='bold')
    
    # 1. Storage Capacity Trend
    ax1 = axes[0, 0]
    ax1.plot(historical_days, storage_used, color='#3498DB', linewidth=2, label='Actual Usage')
    ax1.plot(future_days, storage_projected, color='#E74C3C', linewidth=2, 
            linestyle='--', label='Projected Usage')
    ax1.axhline(y=1000, color='red', linestyle='-', linewidth=2, label='Current Capacity (1TB)')
    ax1.fill_between(historical_days, storage_used, alpha=0.3, color='#3498DB')
    ax1.fill_between(future_days, storage_projected, alpha=0.3, color='#E74C3C')
    
    # Mark when capacity will be exceeded
    exceed_date = None
    for i, val in enumerate(storage_projected):
        if val > 1000:
            exceed_date = future_days[i]
            ax1.axvline(x=exceed_date, color='red', linestyle=':', alpha=0.7)
            ax1.text(exceed_date, 1050, f'Capacity Exceeded\n{exceed_date.strftime("%Y-%m-%d")}',
                    ha='center', fontsize=9, color='red')
            break
    
    ax1.set_title('Storage Capacity Planning', fontweight='bold')
    ax1.set_ylabel('Storage (GB)')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 2. CPU Usage Heatmap (by hour and day of week)
    ax2 = axes[0, 1]
    
    # Generate hourly CPU data
    hours = 24
    days_of_week = 7
    cpu_heatmap_data = np.random.normal(50, 15, (hours, days_of_week))
    
    # Business hours have higher usage
    for day in range(5):  # Monday to Friday
        for hour in range(9, 18):  # 9 AM to 6 PM
            cpu_heatmap_data[hour, day] += 20
    
    im = ax2.imshow(cpu_heatmap_data, cmap='YlOrRd', aspect='auto', vmin=0, vmax=100)
    ax2.set_title('CPU Usage Heatmap (Weekly Pattern)', fontweight='bold')
    ax2.set_xlabel('Day of Week')
    ax2.set_ylabel('Hour of Day')
    ax2.set_xticks(range(7))
    ax2.set_xticklabels(['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun'])
    ax2.set_yticks(range(0, 24, 3))
    ax2.set_yticklabels([f'{h:02d}:00' for h in range(0, 24, 3)])
    
    # Add colorbar
    cbar = plt.colorbar(im, ax=ax2)
    cbar.set_label('CPU Usage (%)', rotation=270, labelpad=15)
    
    # 3. Resource Utilization Comparison
    ax3 = axes[1, 0]
    
    resources = ['CPU', 'Memory', 'Storage', 'Network', 'GPU']
    current_usage = [65, 78, 82, 45, 92]
    projected_usage = [72, 85, 95, 52, 98]
    
    x = np.arange(len(resources))
    width = 0.35
    
    bars1 = ax3.bar(x - width/2, current_usage, width, label='Current', color='#2ECC71')
    bars2 = ax3.bar(x + width/2, projected_usage, width, label='Projected (30 days)', color='#E67E22')
    
    # Add capacity line
    ax3.axhline(y=85, color='red', linestyle='--', label='Recommended Max (85%)')
    
    # Add value labels on bars
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax3.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.0f}%', ha='center', va='bottom', fontsize=9)
    
    ax3.set_title('Resource Utilization Overview', fontweight='bold')
    ax3.set_ylabel('Utilization (%)')
    ax3.set_xticks(x)
    ax3.set_xticklabels(resources)
    ax3.legend()
    ax3.set_ylim(0, 110)
    ax3.grid(True, alpha=0.3, axis='y')
    
    # 4. Growth Rate Analysis
    ax4 = axes[1, 1]
    
    # Calculate growth rates
    metrics = ['Users', 'Requests/day', 'Storage', 'Bandwidth']
    growth_rates = [12.5, 18.3, 22.1, 15.7]  # Monthly growth %
    
    # Create circular progress indicators
    theta = np.linspace(0, 2*np.pi, 100)
    
    for i, (metric, rate) in enumerate(zip(metrics, growth_rates)):
        # Position for each gauge
        cx = 0.25 + (i % 2) * 0.5
        cy = 0.65 - (i // 2) * 0.5
        
        # Draw gauge background
        ax4.plot(cx + 0.15*np.cos(theta), cy + 0.15*np.sin(theta), 
                'lightgray', linewidth=10)
        
        # Draw gauge progress
        progress_theta = theta[:int(rate/30 * 100)]
        color = 'green' if rate < 15 else 'orange' if rate < 25 else 'red'
        ax4.plot(cx + 0.15*np.cos(progress_theta), cy + 0.15*np.sin(progress_theta),
                color=color, linewidth=10)
        
        # Add text
        ax4.text(cx, cy, f'{rate:.1f}%', ha='center', va='center', fontweight='bold')
        ax4.text(cx, cy-0.25, metric, ha='center', va='center', fontsize=9)
    
    ax4.set_title('Monthly Growth Rates', fontweight='bold')
    ax4.set_xlim(0, 1)
    ax4.set_ylim(0, 1)
    ax4.axis('off')
    
    # Format date axes
    for ax in [ax1]:
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
        ax.xaxis.set_major_locator(mdates.MonthLocator())
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45)
    
    plt.tight_layout()
    return fig

# Create capacity planning visualization
capacity_viz = create_capacity_planning_viz()
plt.show()
```

### 4. Incident and Alert Visualizations
**DevOps Context**: Incident patterns and alert frequency analysis

```python
def create_incident_analytics():
    """Create incident analysis visualizations"""
    
    # Generate incident data
    np.random.seed(42)
    
    # Incident times (more during business hours)
    incident_times = []
    for day in range(30):
        # Business hours incidents
        n_business = np.random.poisson(3)
        for _ in range(n_business):
            hour = np.random.uniform(9, 18)
            incident_times.append(day * 24 + hour)
        
        # Off-hours incidents
        n_offhours = np.random.poisson(1)
        for _ in range(n_offhours):
            hour = np.random.choice([np.random.uniform(0, 9), np.random.uniform(18, 24)])
            incident_times.append(day * 24 + hour)
    
    # Create figure
    fig = plt.figure(figsize=(16, 10))
    gs = GridSpec(3, 3, figure=fig, hspace=0.3, wspace=0.3)
    
    # 1. Incident Timeline
    ax1 = fig.add_subplot(gs[0, :])
    
    # Convert to datetime
    base_date = datetime(2024, 1, 1)
    incident_datetimes = [base_date + timedelta(hours=h) for h in incident_times]
    
    # Plot incidents as scatter
    severities = np.random.choice(['P1', 'P2', 'P3'], len(incident_datetimes), p=[0.1, 0.3, 0.6])
    colors = {'P1': 'red', 'P2': 'orange', 'P3': 'yellow'}
    
    for severity in ['P1', 'P2', 'P3']:
        mask = [s == severity for s in severities]
        times = [t for t, m in zip(incident_datetimes, mask) if m]
        y_values = [1] * len(times)
        ax1.scatter(times, y_values, c=colors[severity], s=100, alpha=0.7, label=severity)
    
    ax1.set_title('Incident Timeline (30 Days)', fontweight='bold')
    ax1.set_xlabel('Date')
    ax1.set_ylim(0.5, 1.5)
    ax1.set_yticks([])
    ax1.legend()
    ax1.grid(True, alpha=0.3, axis='x')
    ax1.xaxis.set_major_formatter(mdates.DateFormatter('%m/%d'))
    
    # 2. Incidents by Hour of Day
    ax2 = fig.add_subplot(gs[1, 0])
    
    hours = [t.hour for t in incident_datetimes]
    ax2.hist(hours, bins=24, color='#3498DB', alpha=0.7, edgecolor='black')
    ax2.axvspan(9, 18, alpha=0.2, color='yellow', label='Business Hours')
    ax2.set_title('Incidents by Hour of Day', fontweight='bold')
    ax2.set_xlabel('Hour')
    ax2.set_ylabel('Count')
    ax2.set_xticks(range(0, 24, 3))
    ax2.legend()
    ax2.grid(True, alpha=0.3, axis='y')
    
    # 3. Incidents by Day of Week
    ax3 = fig.add_subplot(gs[1, 1])
    
    days = [t.strftime('%A') for t in incident_datetimes]
    day_counts = pd.Series(days).value_counts()
    day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    day_counts = day_counts.reindex(day_order, fill_value=0)
    
    bars = ax3.bar(range(7), day_counts.values, color=['#E74C3C' if d in ['Saturday', 'Sunday'] 
                                                       else '#3498DB' for d in day_order])
    ax3.set_title('Incidents by Day of Week', fontweight='bold')
    ax3.set_xlabel('Day')
    ax3.set_ylabel('Count')
    ax3.set_xticks(range(7))
    ax3.set_xticklabels(['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun'])
    ax3.grid(True, alpha=0.3, axis='y')
    
    # 4. Service Impact Analysis
    ax4 = fig.add_subplot(gs[1, 2])
    
    services = ['API', 'Database', 'Auth', 'Cache', 'Queue']
    impacts = [15, 8, 12, 5, 3]
    
    # Create pie chart
    colors_pie = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#FFA07A', '#98D8C8']
    wedges, texts, autotexts = ax4.pie(impacts, labels=services, colors=colors_pie,
                                        autopct='%1.1f%%', startangle=90)
    ax4.set_title('Incidents by Service', fontweight='bold')
    
    # 5. MTTR Trend
    ax5 = fig.add_subplot(gs[2, 0])
    
    weeks = ['Week 1', 'Week 2', 'Week 3', 'Week 4']
    mttr_p1 = [45, 38, 42, 35]
    mttr_p2 = [120, 110, 95, 88]
    mttr_p3 = [180, 165, 150, 140]
    
    x = np.arange(len(weeks))
    width = 0.25
    
    ax5.bar(x - width, mttr_p1, width, label='P1', color='red', alpha=0.7)
    ax5.bar(x, mttr_p2, width, label='P2', color='orange', alpha=0.7)
    ax5.bar(x + width, mttr_p3, width, label='P3', color='yellow', alpha=0.7)
    
    ax5.set_title('Mean Time to Resolution (MTTR) Trend', fontweight='bold')
    ax5.set_xlabel('Week')
    ax5.set_ylabel('MTTR (minutes)')
    ax5.set_xticks(x)
    ax5.set_xticklabels(weeks)
    ax5.legend()
    ax5.grid(True, alpha=0.3, axis='y')
    
    # 6. Alert Noise Analysis
    ax6 = fig.add_subplot(gs[2, 1:])
    
    # Generate alert data
    hours_range = pd.date_range(start='2024-01-01', periods=168, freq='1H')  # 1 week
    alerts = np.random.poisson(5, len(hours_range))
    actionable = np.random.binomial(alerts, 0.3)  # 30% actionable
    
    ax6.fill_between(hours_range, alerts, alpha=0.3, color='red', label='Total Alerts')
    ax6.fill_between(hours_range, actionable, alpha=0.5, color='green', label='Actionable Alerts')
    ax6.set_title('Alert Noise Analysis (1 Week)', fontweight='bold')
    ax6.set_xlabel('Date')
    ax6.set_ylabel('Alert Count')
    ax6.legend()
    ax6.grid(True, alpha=0.3)
    ax6.xaxis.set_major_formatter(mdates.DateFormatter('%m/%d'))
    
    # Add noise ratio text
    noise_ratio = (1 - np.sum(actionable) / np.sum(alerts)) * 100
    ax6.text(0.02, 0.95, f'Noise Ratio: {noise_ratio:.1f}%', 
            transform=ax6.transAxes, fontsize=12, fontweight='bold',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    return fig

# Create incident analytics
incident_viz = create_incident_analytics()
plt.show()
```

### 5. Deployment and Release Visualizations
**DevOps Context**: Deployment frequency and success rates

```python
def create_deployment_dashboard():
    """Create deployment analytics dashboard"""
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle('Deployment Analytics Dashboard', fontsize=16, fontweight='bold')
    
    # 1. Deployment Frequency Over Time
    ax1 = axes[0, 0]
    
    dates = pd.date_range(start='2024-01-01', end='2024-01-31', freq='D')
    deployments_per_day = np.random.poisson(3, len(dates))
    deployments_per_day[5] = 0  # Weekend
    deployments_per_day[6] = 0
    deployments_per_day[12] = 0
    deployments_per_day[13] = 0
    
    ax1.bar(dates, deployments_per_day, color='#3498DB', alpha=0.7)
    ax1.axhline(y=np.mean(deployments_per_day), color='red', linestyle='--', 
               label=f'Average: {np.mean(deployments_per_day):.1f}')
    ax1.set_title('Daily Deployment Frequency', fontweight='bold')
    ax1.set_ylabel('Number of Deployments')
    ax1.legend()
    ax1.grid(True, alpha=0.3, axis='y')
    ax1.xaxis.set_major_formatter(mdates.DateFormatter('%m/%d'))
    plt.setp(ax1.xaxis.get_majorticklabels(), rotation=45)
    
    # 2. Success Rate by Environment
    ax2 = axes[0, 1]
    
    environments = ['Dev', 'Staging', 'Production']
    success_rates = [95, 92, 98]
    colors_env = ['#2ECC71', '#F39C12', '#E74C3C']
    
    bars = ax2.bar(environments, success_rates, color=colors_env, alpha=0.7)
    ax2.axhline(y=95, color='red', linestyle='--', label='Target: 95%')
    ax2.set_title('Deployment Success Rate by Environment', fontweight='bold')
    ax2.set_ylabel('Success Rate (%)')
    ax2.set_ylim(85, 100)
    ax2.legend()
    ax2.grid(True, alpha=0.3, axis='y')
    
    # Add value labels
    for bar, rate in zip(bars, success_rates):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                f'{rate}%', ha='center', fontweight='bold')
    
    # 3. Deployment Duration Distribution
    ax3 = axes[0, 2]
    
    durations = np.concatenate([
        np.random.normal(5, 1, 50),   # Quick deployments
        np.random.normal(15, 3, 30),  # Normal deployments
        np.random.normal(30, 5, 10)   # Long deployments
    ])
    
    ax3.hist(durations, bins=20, color='#9B59B6', alpha=0.7, edgecolor='black')
    ax3.axvline(x=np.median(durations), color='red', linestyle='--', 
               label=f'Median: {np.median(durations):.1f} min')
    ax3.set_title('Deployment Duration Distribution', fontweight='bold')
    ax3.set_xlabel('Duration (minutes)')
    ax3.set_ylabel('Frequency')
    ax3.legend()
    ax3.grid(True, alpha=0.3, axis='y')
    
    # 4. Rollback Analysis
    ax4 = axes[1, 0]
    
    services = ['API', 'Frontend', 'Backend', 'Database', 'Cache']
    rollback_counts = [2, 5, 3, 1, 0]
    total_deployments = [45, 52, 38, 15, 20]
    rollback_rates = [r/t*100 for r, t in zip(rollback_counts, total_deployments)]
    
    y_pos = np.arange(len(services))
    bars = ax4.barh(y_pos, rollback_rates, color=['red' if r > 5 else 'orange' if r > 2 else 'green' 
                                                   for r in rollback_rates])
    ax4.set_yticks(y_pos)
    ax4.set_yticklabels(services)
    ax4.set_title('Rollback Rate by Service', fontweight='bold')
    ax4.set_xlabel('Rollback Rate (%)')
    ax4.grid(True, alpha=0.3, axis='x')
    
    # Add value labels
    for i, (bar, rate) in enumerate(zip(bars, rollback_rates)):
        ax4.text(bar.get_width() + 0.5, bar.get_y() + bar.get_height()/2,
                f'{rate:.1f}%', va='center')
    
    # 5. Deployment Methods
    ax5 = axes[1, 1]
    
    methods = ['Blue-Green', 'Canary', 'Rolling', 'Recreate']
    method_counts = [35, 25, 30, 10]
    colors_methods = ['#3498DB', '#E67E22', '#2ECC71', '#E74C3C']
    
    wedges, texts, autotexts = ax5.pie(method_counts, labels=methods, colors=colors_methods,
                                        autopct='%1.1f%%', startangle=90)
    ax5.set_title('Deployment Methods Used', fontweight='bold')
    
    # 6. Lead Time Trend
    ax6 = axes[1, 2]
    
    weeks = pd.date_range(start='2024-01-01', periods=12, freq='W')
    lead_time = [5, 4.5, 4.2, 3.8, 3.5, 3.2, 3.0, 2.8, 2.7, 2.5, 2.4, 2.3]
    
    ax6.plot(weeks, lead_time, marker='o', color='#16A085', linewidth=2, markersize=8)
    ax6.fill_between(weeks, lead_time, alpha=0.3, color='#16A085')
    ax6.set_title('Lead Time Trend (Commit to Production)', fontweight='bold')
    ax6.set_ylabel('Lead Time (days)')
    ax6.grid(True, alpha=0.3)
    ax6.xaxis.set_major_formatter(mdates.DateFormatter('%m/%d'))
    plt.setp(ax6.xaxis.get_majorticklabels(), rotation=45)
    
    # Add trend annotation
    ax6.annotate(f'Improvement: {(lead_time[0] - lead_time[-1])/lead_time[0]*100:.1f}%',
                xy=(weeks[-1], lead_time[-1]), xytext=(weeks[-3], lead_time[-1] + 1),
                arrowprops=dict(arrowstyle='->', color='green'),
                fontsize=10, color='green', fontweight='bold')
    
    plt.tight_layout()
    return fig

# Create deployment dashboard
deployment_viz = create_deployment_dashboard()
plt.show()
```

## Real-World DevOps Examples

### Example 1: Multi-Service Performance Comparison

```python
def create_service_comparison_dashboard():
    """Compare performance metrics across multiple services"""
    
    services = ['Auth API', 'User API', 'Order API', 'Payment API', 'Notification API']
    
    # Generate metrics for each service
    metrics = {
        'Availability (%)': [99.95, 99.88, 99.92, 99.99, 99.85],
        'Avg Response (ms)': [45, 120, 85, 200, 35],
        'Error Rate (%)': [0.5, 1.2, 0.8, 0.1, 1.5],
        'Requests/sec': [500, 1200, 800, 300, 2000],
        'CPU Usage (%)': [35, 65, 55, 45, 70]
    }
    
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Normalize metrics for radar chart
    from math import pi
    
    categories = list(metrics.keys())
    N = len(categories)
    
    # Create angles for radar chart
    angles = [n / float(N) * 2 * pi for n in range(N)]
    angles += angles[:1]
    
    # Initialize the plot
    ax = plt.subplot(111, projection='polar')
    
    # Define colors for each service
    colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#FFA07A', '#98D8C8']
    
    for idx, service in enumerate(services):
        values = []
        for metric in metrics.keys():
            # Normalize values (0-100 scale)
            if 'Rate' in metric or 'Error' in metric:
                # Invert error metrics (lower is better)
                normalized = 100 - (metrics[metric][idx] / max(metrics[metric]) * 100)
            else:
                normalized = metrics[metric][idx] / max(metrics[metric]) * 100
            values.append(normalized)
        
        values += values[:1]
        
        # Plot
        ax.plot(angles, values, 'o-', linewidth=2, label=service, color=colors[idx])
        ax.fill(angles, values, alpha=0.25, color=colors[idx])
    
    # Fix axis labels
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories)
    ax.set_ylim(0, 100)
    
    # Add legend and title
    plt.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0))
    plt.title('Service Performance Comparison (Normalized)', size=14, fontweight='bold', pad=20)
    
    return fig

# Create service comparison
service_comparison = create_service_comparison_dashboard()
plt.show()
```

### Example 2: Cost Optimization Opportunities

```python
def visualize_cost_optimization():
    """Visualize cloud cost optimization opportunities"""
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Cloud Cost Optimization Analysis', fontsize=16, fontweight='bold')
    
    # 1. Unused Resources
    ax1 = axes[0, 0]
    
    resource_types = ['Unattached\nEBS', 'Idle\nLoad Balancers', 'Unused\nElastic IPs', 
                     'Old\nSnapshots', 'Stopped\nInstances']
    monthly_costs = [450, 280, 120, 350, 890]
    
    bars = ax1.bar(resource_types, monthly_costs, color='#E74C3C', alpha=0.7)
    ax1.set_title('Unused Resources - Monthly Cost', fontweight='bold')
    ax1.set_ylabel('Cost ($)')
    ax1.grid(True, alpha=0.3, axis='y')
    
    # Add total savings annotation
    total_savings = sum(monthly_costs)
    ax1.text(0.5, 0.95, f'Total Potential Savings: ${total_savings:,.0f}/month',
            transform=ax1.transAxes, ha='center', fontsize=11, fontweight='bold',
            bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.3))
    
    # 2. Right-sizing Opportunities
    ax2 = axes[0, 1]
    
    instance_types = ['Over-provisioned', 'Right-sized', 'Under-provisioned']
    instance_counts = [45, 120, 15]
    colors_sizing = ['#E74C3C', '#2ECC71', '#F39C12']
    
    wedges, texts, autotexts = ax2.pie(instance_counts, labels=instance_types, 
                                        colors=colors_sizing, autopct='%1.1f%%',
                                        startangle=90)
    ax2.set_title('EC2 Instance Sizing Analysis', fontweight='bold')
    
    # 3. Reserved vs On-Demand
    ax3 = axes[1, 0]
    
    months = pd.date_range(start='2024-01', periods=6, freq='M')
    on_demand_costs = [12000, 13000, 14000, 15000, 16000, 17000]
    reserved_costs = [8000, 8000, 8000, 8000, 8000, 8000]
    
    ax3.plot(months, on_demand_costs, marker='o', label='On-Demand', 
            color='#E74C3C', linewidth=2)
    ax3.plot(months, reserved_costs, marker='s', label='Reserved (1-year)', 
            color='#2ECC71', linewidth=2)
    ax3.fill_between(months, on_demand_costs, reserved_costs, 
                     alpha=0.3, color='yellow', label='Potential Savings')
    
    ax3.set_title('Reserved vs On-Demand Cost Comparison', fontweight='bold')
    ax3.set_ylabel('Monthly Cost ($)')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    ax3.xaxis.set_major_formatter(mdates.DateFormatter('%b'))
    
    # 4. Cost by Tag/Department
    ax4 = axes[1, 1]
    
    departments = ['Engineering', 'Data Science', 'QA', 'DevOps', 'Other']
    costs = [5500, 3200, 1800, 2500, 800]
    budgets = [5000, 3500, 2000, 2000, 1000]
    
    x = np.arange(len(departments))
    width = 0.35
    
    bars1 = ax4.bar(x - width/2, costs, width, label='Actual', color='#3498DB', alpha=0.7)
    bars2 = ax4.bar(x + width/2, budgets, width, label='Budget', color='#95A5A6', alpha=0.7)
    
    # Highlight over-budget departments
    for i, (cost, budget) in enumerate(zip(costs, budgets)):
        if cost > budget:
            ax4.plot(i, cost + 200, 'r^', markersize=10)
            ax4.text(i, cost + 400, f'+${cost-budget}', ha='center', color='red', fontweight='bold')
    
    ax4.set_title('Cost by Department vs Budget', fontweight='bold')
    ax4.set_ylabel('Cost ($)')
    ax4.set_xticks(x)
    ax4.set_xticklabels(departments, rotation=45, ha='right')
    ax4.legend()
    ax4.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    return fig

# Create cost optimization visualization
cost_viz = visualize_cost_optimization()
plt.show()
```

## Practice Exercises

1. **Custom Alerting Dashboard**: Create a dashboard that shows alert fatigue metrics, including false positive rates and alert response times.

2. **Database Performance Visualization**: Build visualizations for database metrics including query performance, connection pools, and slow query analysis.

3. **Network Traffic Flow**: Create a Sankey diagram or flow visualization showing traffic between services.

4. **CI/CD Pipeline Metrics**: Visualize build times, test coverage trends, and pipeline success rates.

## References and Learning Resources

- [Matplotlib Official Documentation](https://matplotlib.org/stable/contents.html)
- [Matplotlib Gallery](https://matplotlib.org/stable/gallery/index.html)
- [Seaborn for Statistical Visualizations](https://seaborn.pydata.org/)
- [Real Python - Matplotlib Guide](https://realpython.com/python-matplotlib-guide/)
- [Effective Data Visualization (Tufte)](https://www.edwardtufte.com/tufte/books_vdqi)
- [Python Graph Gallery](https://python-graph-gallery.com/)
- [Plotly for Interactive Dashboards](https://plotly.com/python/)
- [Grafana Python SDK](https://github.com/grafana/grafana-api-sdk)
- [Prometheus Python Client](https://github.com/prometheus/client_python)