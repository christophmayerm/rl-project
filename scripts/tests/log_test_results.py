"""
Log and track test results
"""

import json
import sys
from datetime import datetime
from pathlib import Path

def log_test_result(env_name, algo_name, passed, error_msg=None):
    """Log individual test result"""
    log_dir = Path("results/tests")
    log_dir.mkdir(parents=True, exist_ok=True)
    
    log_file = log_dir / "test_log.jsonl"
    
    result = {
        'timestamp': datetime.now().isoformat(),
        'environment': env_name,
        'algorithm': algo_name,
        'passed': passed,
        'error': error_msg
    }
    
    with open(log_file, 'a') as f:
        f.write(json.dumps(result) + '\n')
    
    return result

def get_test_summary():
    """Get summary of all tests"""
    log_file = Path("results/tests/test_log.jsonl")
    
    if not log_file.exists():
        return None
    
    results = []
    with open(log_file) as f:
        for line in f:
            results.append(json.loads(line))
    
    # Group by environment
    summary = {}
    for result in results:
        env = result['environment']
        if env not in summary:
            summary[env] = {'total': 0, 'passed': 0, 'failed': 0}
        
        summary[env]['total'] += 1
        if result['passed']:
            summary[env]['passed'] += 1
        else:
            summary[env]['failed'] += 1
    
    return summary

if __name__ == '__main__':
    summary = get_test_summary()
    
    if summary:
        print("\n" + "="*60)
        print("TEST SUMMARY")
        print("="*60)
        
        for env, stats in summary.items():
            status = "✅ PASS" if stats['failed'] == 0 else "❌ FAIL"
            print(f"{env:30s} {status} ({stats['passed']}/{stats['total']})")
        
        print("="*60)
    else:
        print("No test results found")