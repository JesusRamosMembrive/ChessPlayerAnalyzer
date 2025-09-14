#!/usr/bin/env python3
"""
Performance Monitoring Integration for CI/CD Pipeline
Integrates with monitoring systems to track performance metrics over time.
"""

import json
import time
import sys
import logging
import requests
import os
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
from pathlib import Path
import argparse

# Configuration
PROMETHEUS_URL = os.getenv('PROMETHEUS_URL', 'http://localhost:9090')
GRAFANA_URL = os.getenv('GRAFANA_URL', 'http://localhost:3000')
GRAFANA_API_TOKEN = os.getenv('GRAFANA_API_TOKEN', '')
ALERT_WEBHOOK_URL = os.getenv('ALERT_WEBHOOK_URL', '')

# Performance thresholds
PERFORMANCE_THRESHOLDS = {
    'response_time_p95': 500,  # ms
    'response_time_avg': 100,  # ms
    'error_rate': 1.0,  # percentage
    'cpu_usage': 70.0,  # percentage
    'memory_usage': 80.0,  # percentage
    'speedup_factor': 2.0,  # minimum speedup to maintain
}

class PerformanceMonitor:
    """Performance monitoring integration for CI/CD pipeline."""

    def __init__(self, prometheus_url: str = PROMETHEUS_URL, grafana_url: str = GRAFANA_URL):
        self.prometheus_url = prometheus_url
        self.grafana_url = grafana_url
        self.session = requests.Session()
        self.session.timeout = 30

        # Configure logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger(__name__)

    def query_prometheus(self, query: str, start_time: Optional[datetime] = None,
                        end_time: Optional[datetime] = None, step: str = '1m') -> Dict:
        """Query Prometheus for metrics."""
        try:
            if start_time and end_time:
                # Range query
                params = {
                    'query': query,
                    'start': start_time.isoformat() + 'Z',
                    'end': end_time.isoformat() + 'Z',
                    'step': step
                }
                url = f"{self.prometheus_url}/api/v1/query_range"
            else:
                # Instant query
                params = {'query': query}
                url = f"{self.prometheus_url}/api/v1/query"

            response = self.session.get(url, params=params)
            response.raise_for_status()

            data = response.json()
            if data['status'] == 'success':
                return data['data']
            else:
                self.logger.error(f"Prometheus query failed: {data}")
                return {}

        except requests.RequestException as e:
            self.logger.error(f"Failed to query Prometheus: {e}")
            return {}

    def get_response_time_metrics(self, duration_minutes: int = 30) -> Dict[str, float]:
        """Get response time metrics for the last N minutes."""
        end_time = datetime.utcnow()
        start_time = end_time - timedelta(minutes=duration_minutes)

        # Query for response time percentiles
        queries = {
            'p95': 'histogram_quantile(0.95, rate(http_request_duration_seconds_bucket[5m]))',
            'p90': 'histogram_quantile(0.90, rate(http_request_duration_seconds_bucket[5m]))',
            'p50': 'histogram_quantile(0.50, rate(http_request_duration_seconds_bucket[5m]))',
            'avg': 'rate(http_request_duration_seconds_sum[5m]) / rate(http_request_duration_seconds_count[5m])'
        }

        metrics = {}
        for metric_name, query in queries.items():
            data = self.query_prometheus(query, start_time, end_time)
            if data and data.get('result'):
                # Get the latest value
                result = data['result'][0]
                if result.get('values'):
                    latest_value = float(result['values'][-1][1]) * 1000  # Convert to ms
                    metrics[f'response_time_{metric_name}'] = latest_value

        return metrics

    def get_error_rate_metrics(self, duration_minutes: int = 30) -> Dict[str, float]:
        """Get error rate metrics for the last N minutes."""
        end_time = datetime.utcnow()
        start_time = end_time - timedelta(minutes=duration_minutes)

        # Query for error rate
        query = '''
        (
            rate(http_requests_total{status=~"5.."}[5m]) /
            rate(http_requests_total[5m])
        ) * 100
        '''

        data = self.query_prometheus(query, start_time, end_time)
        metrics = {}

        if data and data.get('result'):
            result = data['result'][0]
            if result.get('values'):
                latest_value = float(result['values'][-1][1])
                metrics['error_rate'] = latest_value

        return metrics

    def get_resource_metrics(self, duration_minutes: int = 30) -> Dict[str, float]:
        """Get CPU and memory usage metrics."""
        end_time = datetime.utcnow()
        start_time = end_time - timedelta(minutes=duration_minutes)

        queries = {
            'cpu_usage': 'rate(process_cpu_seconds_total[5m]) * 100',
            'memory_usage': '(process_resident_memory_bytes / process_virtual_memory_max_bytes) * 100'
        }

        metrics = {}
        for metric_name, query in queries.items():
            data = self.query_prometheus(query, start_time, end_time)
            if data and data.get('result'):
                result = data['result'][0]
                if result.get('values'):
                    latest_value = float(result['values'][-1][1])
                    metrics[metric_name] = latest_value

        return metrics

    def get_custom_performance_metrics(self, duration_minutes: int = 30) -> Dict[str, float]:
        """Get custom performance metrics (speedup factors, optimization effectiveness)."""
        end_time = datetime.utcnow()
        start_time = end_time - timedelta(minutes=duration_minutes)

        # Custom metrics for our optimization work
        queries = {
            'analysis_speedup_factor': 'chess_analysis_speedup_ratio',
            'numpy_optimization_effectiveness': 'numpy_vs_pandas_performance_ratio',
            'quality_analysis_time': 'chess_quality_analysis_duration_seconds',
            'timing_analysis_time': 'chess_timing_analysis_duration_seconds',
            'longitudinal_analysis_time': 'chess_longitudinal_analysis_duration_seconds',
        }

        metrics = {}
        for metric_name, query in queries.items():
            data = self.query_prometheus(query, start_time, end_time)
            if data and data.get('result'):
                result = data['result'][0]
                if result.get('values'):
                    latest_value = float(result['values'][-1][1])
                    metrics[metric_name] = latest_value

        return metrics

    def check_performance_regression(self, current_metrics: Dict[str, float]) -> Dict[str, Any]:
        """Check for performance regressions against thresholds."""
        regression_results = {
            'passed': True,
            'violations': [],
            'warnings': [],
            'metrics': current_metrics,
            'thresholds': PERFORMANCE_THRESHOLDS,
            'timestamp': datetime.utcnow().isoformat()
        }

        # Check each metric against thresholds
        for metric, threshold in PERFORMANCE_THRESHOLDS.items():
            if metric not in current_metrics:
                regression_results['warnings'].append(f"Metric {metric} not available")
                continue

            current_value = current_metrics[metric]

            # Define threshold logic based on metric type
            if metric in ['response_time_p95', 'response_time_avg', 'error_rate', 'cpu_usage', 'memory_usage']:
                # Lower is better
                if current_value > threshold:
                    regression_results['violations'].append({
                        'metric': metric,
                        'current': current_value,
                        'threshold': threshold,
                        'severity': 'high' if current_value > threshold * 1.5 else 'medium'
                    })
                    regression_results['passed'] = False
            elif metric == 'speedup_factor':
                # Higher is better
                if current_value < threshold:
                    regression_results['violations'].append({
                        'metric': metric,
                        'current': current_value,
                        'threshold': threshold,
                        'severity': 'high' if current_value < threshold * 0.5 else 'medium'
                    })
                    regression_results['passed'] = False

        return regression_results

    def create_grafana_annotation(self, title: str, text: str, tags: List[str] = None) -> bool:
        """Create a Grafana annotation for deployment events."""
        if not GRAFANA_API_TOKEN:
            self.logger.warning("Grafana API token not provided, skipping annotation")
            return False

        headers = {
            'Authorization': f'Bearer {GRAFANA_API_TOKEN}',
            'Content-Type': 'application/json'
        }

        annotation_data = {
            'time': int(time.time() * 1000),
            'title': title,
            'text': text,
            'tags': tags or ['ci-cd', 'deployment']
        }

        try:
            response = self.session.post(
                f"{self.grafana_url}/api/annotations",
                headers=headers,
                json=annotation_data
            )
            response.raise_for_status()
            self.logger.info(f"Created Grafana annotation: {title}")
            return True
        except requests.RequestException as e:
            self.logger.error(f"Failed to create Grafana annotation: {e}")
            return False

    def send_alert(self, alert_data: Dict) -> bool:
        """Send alert to webhook endpoint."""
        if not ALERT_WEBHOOK_URL:
            self.logger.warning("Alert webhook URL not provided, skipping alert")
            return False

        try:
            response = self.session.post(ALERT_WEBHOOK_URL, json=alert_data)
            response.raise_for_status()
            self.logger.info("Alert sent successfully")
            return True
        except requests.RequestException as e:
            self.logger.error(f"Failed to send alert: {e}")
            return False

    def run_comprehensive_monitoring(self, duration_minutes: int = 30) -> Dict:
        """Run comprehensive performance monitoring check."""
        self.logger.info(f"Running comprehensive performance monitoring (last {duration_minutes} minutes)")

        # Collect all metrics
        all_metrics = {}

        # Response time metrics
        response_metrics = self.get_response_time_metrics(duration_minutes)
        all_metrics.update(response_metrics)

        # Error rate metrics
        error_metrics = self.get_error_rate_metrics(duration_minutes)
        all_metrics.update(error_metrics)

        # Resource metrics
        resource_metrics = self.get_resource_metrics(duration_minutes)
        all_metrics.update(resource_metrics)

        # Custom performance metrics
        custom_metrics = self.get_custom_performance_metrics(duration_minutes)
        all_metrics.update(custom_metrics)

        # Check for regressions
        regression_results = self.check_performance_regression(all_metrics)

        # Create comprehensive report
        report = {
            'monitoring_timestamp': datetime.utcnow().isoformat(),
            'duration_minutes': duration_minutes,
            'metrics': all_metrics,
            'regression_check': regression_results,
            'overall_status': 'PASS' if regression_results['passed'] else 'FAIL',
            'recommendations': self._generate_recommendations(regression_results)
        }

        return report

    def _generate_recommendations(self, regression_results: Dict) -> List[str]:
        """Generate recommendations based on regression results."""
        recommendations = []

        for violation in regression_results.get('violations', []):
            metric = violation['metric']
            severity = violation['severity']

            if metric.startswith('response_time'):
                if severity == 'high':
                    recommendations.append("Critical: Response time degradation detected. Consider scaling up resources or optimizing database queries.")
                else:
                    recommendations.append("Warning: Response time increase detected. Monitor closely and consider optimization.")

            elif metric == 'error_rate':
                recommendations.append("Critical: Error rate increase detected. Investigate logs and consider rollback.")

            elif metric in ['cpu_usage', 'memory_usage']:
                recommendations.append(f"Warning: High {metric} detected. Consider resource scaling.")

            elif metric == 'speedup_factor':
                if severity == 'high':
                    recommendations.append("Critical: Performance optimization regression detected. Investigation required.")
                else:
                    recommendations.append("Warning: Performance optimization effectiveness decreased.")

        if not recommendations:
            recommendations.append("All performance metrics are within acceptable thresholds.")

        return recommendations

def main():
    parser = argparse.ArgumentParser(description='Performance monitoring for CI/CD pipeline')
    parser.add_argument('--duration', '-d', type=int, default=30, help='Monitoring duration in minutes')
    parser.add_argument('--output', '-o', help='Output JSON file path', default='performance_monitoring_results.json')
    parser.add_argument('--annotate', action='store_true', help='Create Grafana annotation')
    parser.add_argument('--alert-on-failure', action='store_true', help='Send alert on performance regression')
    parser.add_argument('--verbose', '-v', action='store_true', help='Verbose output')

    args = parser.parse_args()

    # Configure logging level
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    # Initialize monitor
    monitor = PerformanceMonitor()

    try:
        # Run monitoring
        report = monitor.run_comprehensive_monitoring(args.duration)

        # Save results
        with open(args.output, 'w') as f:
            json.dump(report, f, indent=2)

        # Print summary
        print(f"Performance Monitoring Results:")
        print(f"Overall Status: {report['overall_status']}")
        print(f"Metrics Collected: {len(report['metrics'])}")
        print(f"Regression Violations: {len(report['regression_check']['violations'])}")

        if report['regression_check']['violations']:
            print(f"Violations:")
            for violation in report['regression_check']['violations']:
                print(f"  - {violation['metric']}: {violation['current']:.2f} (threshold: {violation['threshold']:.2f})")

        # Create Grafana annotation
        if args.annotate:
            title = f"CI/CD Performance Monitoring - {report['overall_status']}"
            text = f"Performance monitoring completed. Status: {report['overall_status']}, Violations: {len(report['regression_check']['violations'])}"
            monitor.create_grafana_annotation(title, text, ['ci-cd', 'performance-monitoring'])

        # Send alert if needed
        if args.alert_on_failure and report['overall_status'] == 'FAIL':
            alert_data = {
                'title': 'Performance Regression Detected',
                'message': f"Performance monitoring detected {len(report['regression_check']['violations'])} violations",
                'severity': 'high',
                'report': report
            }
            monitor.send_alert(alert_data)

        # Exit with appropriate code
        return 0 if report['overall_status'] == 'PASS' else 1

    except Exception as e:
        logging.error(f"Performance monitoring failed: {e}")
        return 1

if __name__ == '__main__':
    sys.exit(main())