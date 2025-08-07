---
title: "Python Basics for DevOps/SRE"
description: "Core Python concepts with real-world infrastructure automation examples"
weight: 1
---

## Overview
Python is the backbone of modern DevOps and SRE practices. This guide covers essential Python concepts with real-world infrastructure automation examples.

## Core Concepts

### 1. Variables and Data Types
**DevOps Context**: Configuration management and environment variables

```python
# Infrastructure configuration variables
server_name = "web-prod-01"
cpu_cores = 8
memory_gb = 16.5
is_production = True
tags = ["web", "production", "nginx"]

# Environment configuration
config = {
    "database_host": "db.example.com",
    "port": 5432,
    "max_connections": 100,
    "ssl_enabled": True
}
```

### 2. Control Flow
**DevOps Context**: Deployment logic and health checks

```python
# Deployment decision logic
def should_deploy(environment, tests_passed, approval_given):
    if environment == "production":
        if tests_passed and approval_given:
            return True
        else:
            print("Production deployment blocked: Tests or approval missing")
            return False
    elif environment in ["staging", "dev"]:
        return tests_passed
    else:
        return False

# Health check implementation
def check_service_health(service_url, timeout=5):
    import requests
    try:
        response = requests.get(service_url, timeout=timeout)
        if response.status_code == 200:
            return "healthy"
        elif response.status_code >= 500:
            return "critical"
        else:
            return "degraded"
    except requests.exceptions.Timeout:
        return "timeout"
    except Exception as e:
        return f"error: {str(e)}"
```

### 3. Functions and Modules
**DevOps Context**: Reusable automation scripts

```python
# system_monitor.py - Reusable monitoring module
import psutil
import datetime

def get_system_metrics():
    """Collect system metrics for monitoring"""
    return {
        "timestamp": datetime.datetime.now().isoformat(),
        "cpu_percent": psutil.cpu_percent(interval=1),
        "memory_percent": psutil.virtual_memory().percent,
        "disk_usage": psutil.disk_usage('/').percent,
        "network_io": psutil.net_io_counters()._asdict()
    }

def alert_if_critical(metrics, thresholds):
    """Send alerts if metrics exceed thresholds"""
    alerts = []
    if metrics["cpu_percent"] > thresholds.get("cpu", 80):
        alerts.append(f"High CPU: {metrics['cpu_percent']}%")
    if metrics["memory_percent"] > thresholds.get("memory", 90):
        alerts.append(f"High Memory: {metrics['memory_percent']}%")
    if metrics["disk_usage"] > thresholds.get("disk", 85):
        alerts.append(f"High Disk Usage: {metrics['disk_usage']}%")
    return alerts
```

### 4. Exception Handling
**DevOps Context**: Robust error handling in automation

```python
import subprocess
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def deploy_application(app_name, version):
    """Deploy application with proper error handling"""
    try:
        # Pre-deployment checks
        logger.info(f"Starting deployment of {app_name} v{version}")
        
        # Pull Docker image
        subprocess.run(
            ["docker", "pull", f"{app_name}:{version}"],
            check=True,
            capture_output=True
        )
        
        # Stop existing container
        try:
            subprocess.run(
                ["docker", "stop", app_name],
                check=True,
                capture_output=True,
                timeout=30
            )
        except subprocess.CalledProcessError:
            logger.warning(f"No existing container for {app_name}")
        
        # Start new container
        subprocess.run(
            ["docker", "run", "-d", "--name", app_name, f"{app_name}:{version}"],
            check=True,
            capture_output=True
        )
        
        logger.info(f"Successfully deployed {app_name} v{version}")
        return True
        
    except subprocess.TimeoutExpired:
        logger.error(f"Deployment timeout for {app_name}")
        return False
    except subprocess.CalledProcessError as e:
        logger.error(f"Deployment failed: {e.stderr.decode()}")
        return False
    except Exception as e:
        logger.error(f"Unexpected error during deployment: {str(e)}")
        return False
```

### 5. File Operations
**DevOps Context**: Configuration file management

```python
import json
import yaml
import os

class ConfigManager:
    """Manage application configurations"""
    
    def __init__(self, config_dir="/etc/myapp"):
        self.config_dir = config_dir
    
    def load_json_config(self, filename):
        """Load JSON configuration file"""
        filepath = os.path.join(self.config_dir, filename)
        try:
            with open(filepath, 'r') as f:
                return json.load(f)
        except FileNotFoundError:
            logger.error(f"Config file not found: {filepath}")
            return {}
        except json.JSONDecodeError as e:
            logger.error(f"Invalid JSON in {filepath}: {e}")
            return {}
    
    def load_yaml_config(self, filename):
        """Load YAML configuration file"""
        filepath = os.path.join(self.config_dir, filename)
        try:
            with open(filepath, 'r') as f:
                return yaml.safe_load(f)
        except FileNotFoundError:
            logger.error(f"Config file not found: {filepath}")
            return {}
        except yaml.YAMLError as e:
            logger.error(f"Invalid YAML in {filepath}: {e}")
            return {}
    
    def update_config(self, filename, updates):
        """Update configuration file"""
        filepath = os.path.join(self.config_dir, filename)
        
        # Backup existing config
        backup_path = f"{filepath}.backup"
        if os.path.exists(filepath):
            with open(filepath, 'r') as src, open(backup_path, 'w') as dst:
                dst.write(src.read())
        
        # Load existing config
        if filename.endswith('.json'):
            config = self.load_json_config(filename)
            config.update(updates)
            with open(filepath, 'w') as f:
                json.dump(config, f, indent=2)
        elif filename.endswith('.yaml') or filename.endswith('.yml'):
            config = self.load_yaml_config(filename)
            config.update(updates)
            with open(filepath, 'w') as f:
                yaml.dump(config, f, default_flow_style=False)
```

## Real-World DevOps Examples

### Example 1: Automated Server Health Check Script

```python
#!/usr/bin/env python3
"""
Server health check script that runs periodically via cron
"""

import requests
import smtplib
from email.mime.text import MIMEText
from datetime import datetime
import sys

SERVERS = [
    {"name": "web-01", "url": "http://web-01.internal:80/health"},
    {"name": "api-01", "url": "http://api-01.internal:8080/health"},
    {"name": "db-01", "url": "http://db-01.internal:5432/health"},
]

ALERT_EMAIL = "oncall@example.com"
SMTP_SERVER = "smtp.example.com"

def check_servers():
    """Check all servers and return status"""
    results = []
    for server in SERVERS:
        try:
            response = requests.get(server["url"], timeout=5)
            status = "UP" if response.status_code == 200 else "DOWN"
            results.append({
                "name": server["name"],
                "status": status,
                "code": response.status_code,
                "response_time": response.elapsed.total_seconds()
            })
        except requests.exceptions.RequestException as e:
            results.append({
                "name": server["name"],
                "status": "DOWN",
                "error": str(e)
            })
    return results

def send_alert(failed_servers):
    """Send email alert for failed servers"""
    message = f"""
    ALERT: Server Health Check Failed
    Time: {datetime.now()}
    
    Failed Servers:
    """
    for server in failed_servers:
        message += f"\n- {server['name']}: {server.get('error', 'HTTP ' + str(server.get('code')))}"
    
    msg = MIMEText(message)
    msg['Subject'] = 'Server Health Check Alert'
    msg['From'] = 'monitoring@example.com'
    msg['To'] = ALERT_EMAIL
    
    with smtplib.SMTP(SMTP_SERVER) as smtp:
        smtp.send_message(msg)

def main():
    results = check_servers()
    failed = [r for r in results if r["status"] != "UP"]
    
    if failed:
        print(f"CRITICAL: {len(failed)} servers are down")
        send_alert(failed)
        sys.exit(1)
    else:
        print(f"OK: All {len(results)} servers are healthy")
        sys.exit(0)

if __name__ == "__main__":
    main()
```

### Example 2: Log Rotation Script

```python
#!/usr/bin/env python3
"""
Log rotation script for application logs
"""

import os
import gzip
import shutil
from datetime import datetime, timedelta
import glob

LOG_DIR = "/var/log/myapp"
RETENTION_DAYS = 30
MAX_SIZE_MB = 100

def rotate_log(logfile):
    """Rotate a single log file"""
    # Generate timestamp for rotated file
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    rotated_name = f"{logfile}.{timestamp}"
    
    # Rename current log
    shutil.move(logfile, rotated_name)
    
    # Compress rotated log
    with open(rotated_name, 'rb') as f_in:
        with gzip.open(f"{rotated_name}.gz", 'wb') as f_out:
            shutil.copyfileobj(f_in, f_out)
    
    # Remove uncompressed rotated file
    os.remove(rotated_name)
    
    # Create new empty log file
    open(logfile, 'a').close()
    
    print(f"Rotated: {logfile} -> {rotated_name}.gz")

def cleanup_old_logs():
    """Remove logs older than retention period"""
    cutoff_date = datetime.now() - timedelta(days=RETENTION_DAYS)
    
    for compressed_log in glob.glob(f"{LOG_DIR}/*.gz"):
        file_time = datetime.fromtimestamp(os.path.getmtime(compressed_log))
        if file_time < cutoff_date:
            os.remove(compressed_log)
            print(f"Deleted old log: {compressed_log}")

def main():
    # Check each log file
    for logfile in glob.glob(f"{LOG_DIR}/*.log"):
        file_size_mb = os.path.getsize(logfile) / (1024 * 1024)
        
        if file_size_mb > MAX_SIZE_MB:
            rotate_log(logfile)
    
    # Clean up old logs
    cleanup_old_logs()

if __name__ == "__main__":
    main()
```

## Practice Exercises

1. **Service Restart Automation**: Write a Python script that monitors a service and automatically restarts it if it becomes unresponsive.

2. **Configuration Validator**: Create a script that validates YAML/JSON configuration files against a schema before deployment.

3. **Backup Script**: Implement a backup script that archives specified directories and uploads them to S3/cloud storage.

4. **Resource Monitor**: Build a script that monitors system resources and sends alerts when thresholds are exceeded.

## References and Learning Resources

- [Python Official Documentation](https://docs.python.org/3/)
- [Real Python - Python for DevOps](https://realpython.com/python-for-devops/)
- [Automate the Boring Stuff with Python](https://automatetheboringstuff.com/)
- [Python for DevOps (O'Reilly Book)](https://www.oreilly.com/library/view/python-for-devops/9781492057680/)
- [Google's Python Style Guide](https://google.github.io/styleguide/pyguide.html)
- [The Hitchhiker's Guide to Python](https://docs.python-guide.org/)
- [Python Requests Library Documentation](https://requests.readthedocs.io/)
- [psutil Documentation (System Monitoring)](https://psutil.readthedocs.io/)