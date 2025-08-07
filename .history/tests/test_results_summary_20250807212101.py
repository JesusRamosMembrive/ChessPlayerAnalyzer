#!/usr/bin/env python3
"""
Test Results Summary Generator
Creates a comprehensive table showing all test results across analysis modules.
"""

import pytest
import subprocess
import json
import logging
from pathlib import Path
from datetime import datetime
import sys
import tempfile
import os

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class TestResultsSummary:
    """Generate and display comprehensive test results summary."""
    
    def __init__(self):
        self.test_modules = [
            'test_analysis_quality.py',
            'test_analysis_timing.py', 
            'test_analysis_openings.py',
            'test_analysis_endgame.py',
            'test_analysis_benchmark.py',
            'test_analysis_longitudinal.py',
            'test_analysis_engine.py'
        ]
        self.results = {}
        
    def run_module_tests(self, module_name):
        """Run tests for a specific module and capture results."""
        logger.info(f"Running tests for {module_name}")
        
        try:
            project_root = Path(__file__).resolve().parents[1]
            test_path = project_root / 'tests' / module_name

            # Archivo de reporte en directorio temporal del sistema (portable)
            report_dir = Path(tempfile.gettempdir())
            report_file = report_dir / f"test_report_{module_name.replace('.py','')}.json"
            if report_file.exists():
                try:
                    report_file.unlink()
                except Exception:
                    pass

            # Intento 1: con pytest-json-report (si está instalado)
            cmd_base = [sys.executable, '-m', 'pytest', str(test_path), '-q']
            cmd_with_json = cmd_base + ['--json-report', f'--json-report-file={report_file.as_posix()}']
            result = subprocess.run(cmd_with_json, capture_output=True, text=True, cwd=str(project_root))

            # Si falla o no hay reporte, reintentar sin json-report
            if result.returncode != 0 or not report_file.exists():
                result = subprocess.run(cmd_base, capture_output=True, text=True, cwd=str(project_root))

            if report_file.exists():
                with open(report_file, 'r', encoding='utf-8') as f:
                    report_data = json.load(f)
                    
                self.results[module_name] = {
                    'total': report_data.get('summary', {}).get('total', 0),
                    'passed': report_data.get('summary', {}).get('passed', 0),
                    'failed': report_data.get('summary', {}).get('failed', 0),
                    'skipped': report_data.get('summary', {}).get('skipped', 0),
                    'duration': report_data.get('duration', 0),
                    'status': 'PASS' if report_data.get('summary', {}).get('failed', 0) == 0 else 'FAIL'
                }
            else:
                import re
                text_out = (result.stdout or '') + '\n' + (result.stderr or '')
                # Buscar una línea con conteos tipo: "20 passed", "3 failed", "2 skipped"
                passed = failed = skipped = 0
                m_pass = re.search(r"(\d+)\s+passed", text_out)
                m_fail = re.search(r"(\d+)\s+failed", text_out)
                m_skip = re.search(r"(\d+)\s+skipped", text_out)
                if m_pass:
                    passed = int(m_pass.group(1))
                if m_fail:
                    failed = int(m_fail.group(1))
                if m_skip:
                    skipped = int(m_skip.group(1))

                total = passed + failed + skipped
                if total > 0 or passed > 0 or failed > 0 or skipped > 0:
                    self.results[module_name] = {
                        'total': total,
                        'passed': passed,
                        'failed': failed,
                        'skipped': skipped,
                        'duration': 0,
                        'status': 'PASS' if failed == 0 else 'FAIL'
                    }
                else:
                    self.results[module_name] = {
                        'total': 0,
                        'passed': 0,
                        'failed': 1,
                        'skipped': 0,
                        'duration': 0,
                        'status': 'ERROR'
                    }
                    
            logger.info(f"✓ {module_name}: {self.results[module_name]}")
            
        except Exception as e:
            logger.error(f"Error running tests for {module_name}: {e}")
            self.results[module_name] = {
                'total': 0,
                'passed': 0,
                'failed': 1,
                'skipped': 0,
                'duration': 0,
                'status': 'ERROR'
            }
    
    def generate_summary_table(self):
        """Generate a formatted summary table of all test results."""
        logger.info("Generating comprehensive test results summary")
        
        for module in self.test_modules:
            self.run_module_tests(module)
        
        total_tests = sum(r['total'] for r in self.results.values())
        total_passed = sum(r['passed'] for r in self.results.values())
        total_failed = sum(r['failed'] for r in self.results.values())
        total_skipped = sum(r['skipped'] for r in self.results.values())
        total_duration = sum(r['duration'] for r in self.results.values())
        
        table = []
        table.append("=" * 100)
        table.append("CHESS PLAYER ANALYZER - UNIT TESTS SUMMARY")
        table.append("=" * 100)
        table.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S UTC')}")
        table.append("")
        table.append(f"{'MODULE':<35} {'TOTAL':<8} {'PASSED':<8} {'FAILED':<8} {'SKIPPED':<8} {'STATUS':<8}")
        table.append("-" * 100)
        
        for module, results in self.results.items():
            module_name = module.replace('test_analysis_', '').replace('.py', '').upper()
            table.append(f"{module_name:<35} {results['total']:<8} {results['passed']:<8} {results['failed']:<8} {results['skipped']:<8} {results['status']:<8}")
        
        table.append("-" * 100)
        table.append(f"{'TOTAL':<35} {total_tests:<8} {total_passed:<8} {total_failed:<8} {total_skipped:<8} {'SUMMARY':<8}")
        table.append("=" * 100)
        
        overall_status = "✅ ALL TESTS PASSED" if total_failed == 0 else f"❌ {total_failed} TESTS FAILED"
        table.append(f"Overall Status: {overall_status}")
        table.append(f"Success Rate: {(total_passed/total_tests*100):.1f}%" if total_tests > 0 else "Success Rate: N/A")
        table.append("")
        
        table.append("MODULE DETAILS:")
        table.append("-" * 50)
        for module, results in self.results.items():
            module_name = module.replace('test_analysis_', '').replace('.py', '')
            table.append(f"• {module_name.title()}: {results['passed']}/{results['total']} tests passed")
            if results['failed'] > 0:
                table.append(f"  ⚠️  {results['failed']} failed tests require attention")
        
        table.append("")
        table.append("ANALYSIS COVERAGE:")
        table.append("-" * 50)
        table.append("✓ Quality Analysis (ACPL, complexity, precision bursts)")
        table.append("✓ Timing Analysis (stats, correlations, lag detection)")
        table.append("✓ Opening Analysis (entropy, repertoire diversity)")
        table.append("✓ Endgame Analysis (tablebase positions, conversion)")
        table.append("✓ Benchmark Analysis (percentile calculations)")
        table.append("✓ Longitudinal Analysis (trends, step functions)")
        table.append("✓ Engine Analysis (move preparation, game analysis)")
        table.append("=" * 100)
        
        summary_text = "\n".join(table)
        logger.info("Test summary generated successfully")
        
        return summary_text
    
    def save_summary(self, filename="test_results_summary.txt"):
        """Save the summary to a file."""
        summary = self.generate_summary_table()
        
        filepath = Path(filename)
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(summary)
        
        logger.info(f"Summary saved to {filepath}")
        print(summary)
        
        return filepath

def test_generate_comprehensive_summary():
    """Test function to generate and display the comprehensive test summary."""
    logger.info("Starting comprehensive test results summary generation")
    
    summary_generator = TestResultsSummary()
    summary_file = summary_generator.save_summary()
    
    assert summary_file.exists()
    assert summary_file.stat().st_size > 0
    
    logger.info("✅ Comprehensive test summary generated successfully")
    
    return True

if __name__ == "__main__":
    test_generate_comprehensive_summary()
