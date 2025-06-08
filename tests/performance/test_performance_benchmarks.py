#!/usr/bin/env python3
"""
Performance Benchmark Tests for Control Description Analyzer

Tests performance metrics as specified in testing strategy:
- Single control: < 100ms
- 10 controls: < 1 second  
- 100 controls: < 10 seconds
- 1000 controls: < 60 seconds
"""

import pytest
import sys
import os
import time
import statistics
from typing import List, Dict, Any

# Add project root to path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, project_root)

from src.core.analyzer import EnhancedControlAnalyzer


class TestAnalysisPerformance:
    """Test suite for analysis performance benchmarks"""

    @pytest.fixture(scope="class")
    def analyzer(self):
        """Create analyzer instance for performance testing"""
        return EnhancedControlAnalyzer('config/control_analyzer_updated.yaml')

    @pytest.fixture
    def sample_controls(self):
        """Generate sample controls for performance testing"""
        controls = [
            {
                'id': 'PERF_SIMPLE_001',
                'description': 'Finance Manager reviews transactions daily in SAP'
            },
            {
                'id': 'PERF_COMPLEX_001',
                'description': """
                The Senior Finance Manager reviews and validates all journal entries 
                daily in SAP to ensure compliance with SOX requirements and escalates 
                any discrepancies exceeding $10,000 to the Controller for resolution
                """
            },
            {
                'id': 'PERF_VAGUE_001',
                'description': 'Management periodically reviews various reports as appropriate'
            },
            {
                'id': 'PERF_MULTI_001',
                'description': """
                Manager reviews transactions, validates balances, approves entries,
                reconciles accounts, generates reports, and escalates exceptions
                """
            },
            {
                'id': 'PERF_LONG_001',
                'description': ' '.join(['The controller'] + ['reviews and approves'] * 50 + ['journal entries daily'])
            }
        ]
        return controls

    def test_single_control_performance(self, analyzer, benchmark):
        """Test single control analysis performance < 100ms"""
        control_description = "Finance Manager reviews and approves journal entries daily in SAP"
        
        # Warm up the analyzer (exclude from timing)
        analyzer.analyze_control('WARMUP', control_description)
        
        # Benchmark the analysis
        result = benchmark(analyzer.analyze_control, 'PERF_SINGLE_001', control_description)
        
        # Verify result is valid
        assert result is not None
        assert 'total_score' in result
        assert 'category' in result
        
        # Check performance threshold
        assert benchmark.stats['mean'] < 0.1, f"Single control should analyze in < 100ms, took {benchmark.stats['mean']*1000:.2f}ms"

    def test_single_control_variations(self, analyzer, sample_controls):
        """Test performance across different control types"""
        execution_times = []
        
        for control in sample_controls:
            start_time = time.perf_counter()
            result = analyzer.analyze_control(control['id'], control['description'])
            end_time = time.perf_counter()
            
            execution_time = (end_time - start_time) * 1000  # Convert to milliseconds
            execution_times.append(execution_time)
            
            # Each control should be under 100ms
            assert execution_time < 100, f"Control {control['id']} took {execution_time:.2f}ms (> 100ms threshold)"
            
            # Verify result validity
            assert result is not None
            assert result['control_id'] == control['id']
        
        # Report statistics
        avg_time = statistics.mean(execution_times)
        max_time = max(execution_times)
        min_time = min(execution_times)
        
        print(f"\nSingle Control Performance Statistics:")
        print(f"  Average: {avg_time:.2f}ms")
        print(f"  Min: {min_time:.2f}ms")
        print(f"  Max: {max_time:.2f}ms")
        
        assert avg_time < 50, f"Average single control time should be well under 100ms, got {avg_time:.2f}ms"

    def test_batch_10_controls_performance(self, analyzer, benchmark):
        """Test 10 controls batch analysis performance < 1 second"""
        # Generate 10 controls
        controls = []
        for i in range(10):
            controls.append({
                'id': f'BATCH10_{i:03d}',
                'description': f'Manager {i} reviews and approves transactions daily in system {i}'
            })
        
        def analyze_batch():
            results = []
            for control in controls:
                result = analyzer.analyze_control(control['id'], control['description'])
                results.append(result)
            return results
        
        # Warm up
        analyze_batch()
        
        # Benchmark
        results = benchmark(analyze_batch)
        
        # Verify results
        assert len(results) == 10
        for i, result in enumerate(results):
            assert result['control_id'] == f'BATCH10_{i:03d}'
        
        # Check performance threshold
        assert benchmark.stats['mean'] < 1.0, f"10 controls should analyze in < 1 second, took {benchmark.stats['mean']:.2f}s"

    def test_batch_100_controls_performance(self, analyzer):
        """Test 100 controls batch analysis performance < 10 seconds"""
        # Generate 100 varied controls
        controls = []
        templates = [
            "Finance Manager reviews {item} {frequency} in {system}",
            "{role} validates and approves {item} {frequency}",
            "The {role} performs {action} of {item} {frequency} to ensure {purpose}",
            "{dept} team {action} all {item} {frequency} using {system}",
            "Senior {role} {action} {item} and escalates exceptions to {escalation}"
        ]
        
        items = ["transactions", "journal entries", "invoices", "payments", "reconciliations"]
        frequencies = ["daily", "weekly", "monthly", "quarterly", "upon receipt"]
        systems = ["SAP", "Oracle", "QuickBooks", "Excel", "the ERP system"]
        roles = ["Analyst", "Manager", "Controller", "Accountant", "Supervisor"]
        actions = ["reviews", "validates", "approves", "reconciles", "processes"]
        depts = ["Finance", "Accounting", "Treasury", "Audit", "Compliance"]
        purposes = ["accuracy", "compliance", "completeness", "authorization", "validity"]
        escalations = ["Controller", "CFO", "Finance Director", "Audit Committee", "Management"]
        
        # Generate varied controls
        for i in range(100):
            template = templates[i % len(templates)]
            control_text = template.format(
                item=items[i % len(items)],
                frequency=frequencies[i % len(frequencies)],
                system=systems[i % len(systems)],
                role=roles[i % len(roles)],
                action=actions[i % len(actions)],
                dept=depts[i % len(depts)],
                purpose=purposes[i % len(purposes)],
                escalation=escalations[i % len(escalations)]
            )
            controls.append({
                'id': f'BATCH100_{i:03d}',
                'description': control_text
            })
        
        # Time the batch analysis
        start_time = time.perf_counter()
        results = []
        for control in controls:
            result = analyzer.analyze_control(control['id'], control['description'])
            results.append(result)
        end_time = time.perf_counter()
        
        execution_time = end_time - start_time
        
        # Verify results
        assert len(results) == 100
        for i, result in enumerate(results):
            assert result['control_id'] == f'BATCH100_{i:03d}'
            assert result['total_score'] >= 0
            assert result['category'] in ['Effective', 'Adequate', 'Needs Improvement']
        
        # Check performance threshold
        print(f"\n100 controls analyzed in {execution_time:.2f} seconds")
        assert execution_time < 10, f"100 controls should analyze in < 10 seconds, took {execution_time:.2f}s"

    @pytest.mark.slow
    def test_batch_1000_controls_performance(self, analyzer):
        """Test 1000 controls batch analysis performance < 60 seconds"""
        # Generate 1000 controls using a mix of patterns
        controls = []
        
        # Use multiple patterns to create realistic variety
        for i in range(1000):
            # Vary complexity
            if i % 10 == 0:
                # Complex control
                description = f"""
                The Senior {['Finance', 'Accounting', 'Audit'][i % 3]} Manager reviews and validates 
                all {['journal entries', 'transactions', 'reconciliations'][i % 3]} 
                {['daily', 'weekly', 'monthly'][i % 3]} in {['SAP', 'Oracle', 'QuickBooks'][i % 3]} 
                to ensure {['compliance', 'accuracy', 'completeness'][i % 3]} and escalates 
                any discrepancies exceeding ${(i % 10 + 1) * 1000} to the Controller
                """
            elif i % 5 == 0:
                # Medium control
                description = f"Manager {i} reviews {['reports', 'entries', 'balances'][i % 3]} {['daily', 'weekly', 'monthly'][i % 3]} and approves in system"
            else:
                # Simple control
                description = f"{['Analyst', 'Manager', 'Supervisor'][i % 3]} {i} validates data {['daily', 'weekly', 'monthly'][i % 3]}"
            
            controls.append({
                'id': f'BATCH1000_{i:04d}',
                'description': description
            })
        
        # Time the batch analysis
        start_time = time.perf_counter()
        results = []
        
        # Process in chunks to monitor progress
        chunk_size = 100
        for chunk_start in range(0, 1000, chunk_size):
            chunk_end = min(chunk_start + chunk_size, 1000)
            chunk_controls = controls[chunk_start:chunk_end]
            
            for control in chunk_controls:
                result = analyzer.analyze_control(control['id'], control['description'])
                results.append(result)
            
            # Progress indicator
            elapsed = time.perf_counter() - start_time
            progress = (chunk_end / 1000) * 100
            print(f"\rProgress: {progress:.0f}% ({chunk_end}/1000 controls) - {elapsed:.1f}s", end='')
        
        end_time = time.perf_counter()
        execution_time = end_time - start_time
        print(f"\n\n1000 controls analyzed in {execution_time:.2f} seconds")
        
        # Verify results
        assert len(results) == 1000
        
        # Sample verification (checking all would be slow)
        for i in range(0, 1000, 100):
            assert results[i]['control_id'] == f'BATCH1000_{i:04d}'
            assert results[i]['total_score'] >= 0
            assert results[i]['category'] in ['Effective', 'Adequate', 'Needs Improvement']
        
        # Calculate throughput
        controls_per_second = 1000 / execution_time
        print(f"Throughput: {controls_per_second:.2f} controls/second")
        
        # Check performance threshold
        assert execution_time < 60, f"1000 controls should analyze in < 60 seconds, took {execution_time:.2f}s"

    def test_performance_consistency(self, analyzer):
        """Test performance consistency across multiple runs"""
        control_description = "Finance Manager reviews and approves journal entries daily in SAP"
        execution_times = []
        
        # Run the same analysis multiple times
        runs = 20
        for i in range(runs):
            start_time = time.perf_counter()
            result = analyzer.analyze_control(f'CONSISTENCY_{i}', control_description)
            end_time = time.perf_counter()
            
            execution_time = (end_time - start_time) * 1000  # milliseconds
            execution_times.append(execution_time)
        
        # Calculate statistics
        avg_time = statistics.mean(execution_times)
        std_dev = statistics.stdev(execution_times)
        cv = (std_dev / avg_time) * 100  # Coefficient of variation
        
        print(f"\nPerformance Consistency Results ({runs} runs):")
        print(f"  Average: {avg_time:.2f}ms")
        print(f"  Std Dev: {std_dev:.2f}ms")
        print(f"  CV: {cv:.1f}%")
        
        # Performance should be consistent
        assert cv < 50, f"Performance variance too high: CV={cv:.1f}% (should be < 50%)"
        assert all(t < 100 for t in execution_times), "All runs should be under 100ms"

    def test_complexity_impact_on_performance(self, analyzer):
        """Test how control complexity affects performance"""
        # Test controls of increasing complexity
        controls = [
            # Simple: 5 words
            ("SIMPLE", "Manager reviews reports daily"),
            
            # Medium: ~25 words  
            ("MEDIUM", "The Finance Manager reviews and validates all journal entries daily in the SAP system to ensure accuracy and compliance"),
            
            # Complex: ~50 words
            ("COMPLEX", """
            The Senior Finance Manager comprehensively reviews, validates, and approves all journal 
            entries and financial transactions daily in the SAP ERP system, ensuring complete accuracy,
            regulatory compliance with SOX requirements, and proper authorization while maintaining
            detailed documentation and escalating any material discrepancies exceeding established
            thresholds to executive management
            """),
            
            # Very Complex: ~100 words
            ("VERY_COMPLEX", """
            The Senior Finance Manager, in coordination with the Accounting Team Lead and Internal Audit department,
            comprehensively reviews, analyzes, validates, and formally approves all journal entries, financial 
            transactions, account reconciliations, and adjusting entries on a daily basis within the SAP ERP system,
            ensuring complete accuracy, proper supporting documentation, appropriate authorization levels, full regulatory 
            compliance with SOX and GAAP requirements, adherence to company policies and procedures, while maintaining 
            detailed audit trails, investigating any anomalies or discrepancies, and immediately escalating any material 
            issues exceeding $10,000 or indicating potential fraud to the Controller and CFO for resolution
            """)
        ]
        
        print("\nComplexity Impact on Performance:")
        for complexity, description in controls:
            # Time each complexity level
            import time
            start_time = time.perf_counter()
            result = analyzer.analyze_control(f'COMPLEXITY_{complexity}', description)
            end_time = time.perf_counter()
            
            execution_time = (end_time - start_time) * 1000  # milliseconds
            
            # All should still meet 100ms threshold despite complexity
            assert execution_time < 100, f"{complexity} control should complete in < 100ms, took {execution_time:.2f}ms"
            
            print(f"  {complexity} control ({len(description.split())} words): {execution_time:.2f}ms")