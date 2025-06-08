#!/usr/bin/env python3
"""
Memory Usage Tests for Control Description Analyzer

Tests memory efficiency as specified in testing strategy:
- Single control: < 10MB
- 100 controls: < 100MB
- 1000 controls: < 500MB
"""

import pytest
import sys
import os
import gc
import psutil
import tracemalloc
from typing import List, Dict, Any

# Add project root to path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, project_root)

from src.core.analyzer import EnhancedControlAnalyzer


class TestMemoryEfficiency:
    """Test suite for memory usage efficiency"""

    @pytest.fixture(scope="class")
    def analyzer(self):
        """Create analyzer instance for memory testing"""
        return EnhancedControlAnalyzer('config/control_analyzer_updated.yaml')

    @pytest.fixture
    def process(self):
        """Get current process for memory monitoring"""
        return psutil.Process(os.getpid())

    def get_memory_usage_mb(self, process):
        """Get current memory usage in MB"""
        return process.memory_info().rss / 1024 / 1024

    def test_single_control_memory_usage(self, analyzer, process):
        """Test memory usage for single control analysis < 10MB"""
        # Force garbage collection and get baseline
        gc.collect()
        baseline_memory = self.get_memory_usage_mb(process)
        
        # Start memory tracing
        tracemalloc.start()
        
        # Analyze single control
        control_description = "Finance Manager reviews and approves journal entries daily in SAP"
        result = analyzer.analyze_control('MEM_SINGLE_001', control_description)
        
        # Get memory peak
        current, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        
        # Calculate memory used
        peak_mb = peak / 1024 / 1024
        
        # Force garbage collection
        gc.collect()
        after_memory = self.get_memory_usage_mb(process)
        
        # Calculate net memory increase
        memory_increase = after_memory - baseline_memory
        
        print(f"\nSingle Control Memory Usage:")
        print(f"  Peak traced memory: {peak_mb:.2f} MB")
        print(f"  Net memory increase: {memory_increase:.2f} MB")
        
        # Verify result is valid
        assert result is not None
        assert result['control_id'] == 'MEM_SINGLE_001'
        
        # Check memory threshold
        assert peak_mb < 10, f"Single control should use < 10MB, used {peak_mb:.2f} MB"

    def test_batch_100_controls_memory_usage(self, analyzer, process):
        """Test memory usage for 100 controls < 100MB"""
        # Force garbage collection and get baseline
        gc.collect()
        baseline_memory = self.get_memory_usage_mb(process)
        
        # Start memory tracing
        tracemalloc.start()
        
        # Generate and analyze 100 controls
        results = []
        for i in range(100):
            description = f"Manager {i} reviews and validates transactions daily in system {i % 10}"
            result = analyzer.analyze_control(f'MEM_100_{i:03d}', description)
            results.append(result)
            
            # Periodic garbage collection to simulate real usage
            if i % 20 == 0:
                gc.collect()
        
        # Get memory peak
        current, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        
        # Calculate memory used
        peak_mb = peak / 1024 / 1024
        
        # Force garbage collection
        gc.collect()
        after_memory = self.get_memory_usage_mb(process)
        
        # Calculate net memory increase
        memory_increase = after_memory - baseline_memory
        
        print(f"\n100 Controls Memory Usage:")
        print(f"  Peak traced memory: {peak_mb:.2f} MB")
        print(f"  Net memory increase: {memory_increase:.2f} MB")
        print(f"  Memory per control: {peak_mb / 100:.3f} MB")
        
        # Verify results
        assert len(results) == 100
        
        # Check memory threshold
        assert peak_mb < 100, f"100 controls should use < 100MB, used {peak_mb:.2f} MB"

    @pytest.mark.slow
    def test_batch_1000_controls_memory_usage(self, analyzer, process):
        """Test memory usage for 1000 controls < 500MB"""
        # Force garbage collection and get baseline
        gc.collect()
        baseline_memory = self.get_memory_usage_mb(process)
        
        # Start memory tracing
        tracemalloc.start()
        
        # Track memory usage at intervals
        memory_checkpoints = []
        
        # Generate and analyze 1000 controls
        results = []
        for i in range(1000):
            # Vary control complexity
            if i % 100 == 0:
                description = f"""
                The Senior Finance Manager {i} reviews, validates, and approves all journal entries
                daily in SAP to ensure compliance with SOX requirements and escalates exceptions
                """
            else:
                description = f"Analyst {i} validates transactions {['daily', 'weekly', 'monthly'][i % 3]}"
            
            result = analyzer.analyze_control(f'MEM_1000_{i:04d}', description)
            results.append(result)
            
            # Track memory at checkpoints
            if i % 100 == 99:
                current, _ = tracemalloc.get_traced_memory()
                memory_checkpoints.append((i + 1, current / 1024 / 1024))
                
                # Periodic garbage collection
                gc.collect()
                
                # Progress indicator
                print(f"\rProgress: {i + 1}/1000 controls - Memory: {current / 1024 / 1024:.1f} MB", end='')
        
        # Get final memory peak
        current, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        
        # Calculate memory used
        peak_mb = peak / 1024 / 1024
        
        # Force garbage collection
        gc.collect()
        after_memory = self.get_memory_usage_mb(process)
        
        # Calculate net memory increase
        memory_increase = after_memory - baseline_memory
        
        print(f"\n\n1000 Controls Memory Usage:")
        print(f"  Peak traced memory: {peak_mb:.2f} MB")
        print(f"  Net memory increase: {memory_increase:.2f} MB")
        print(f"  Memory per control: {peak_mb / 1000:.3f} MB")
        
        # Show memory growth pattern
        print("\nMemory Growth Pattern:")
        for controls, memory in memory_checkpoints:
            print(f"  After {controls} controls: {memory:.1f} MB")
        
        # Verify results
        assert len(results) == 1000
        
        # Check memory threshold
        assert peak_mb < 500, f"1000 controls should use < 500MB, used {peak_mb:.2f} MB"

    def test_memory_cleanup_efficiency(self, analyzer, process):
        """Test memory cleanup after analysis"""
        # Get baseline
        gc.collect()
        baseline_memory = self.get_memory_usage_mb(process)
        
        # Analyze many controls
        for i in range(50):
            description = f"Manager {i} reviews and approves transactions daily in SAP system"
            _ = analyzer.analyze_control(f'CLEANUP_{i:03d}', description)
        
        # Memory before cleanup
        before_cleanup = self.get_memory_usage_mb(process)
        
        # Force cleanup
        gc.collect()
        
        # Memory after cleanup
        after_cleanup = self.get_memory_usage_mb(process)
        
        # Calculate cleanup efficiency
        memory_freed = before_cleanup - after_cleanup
        cleanup_percentage = (memory_freed / (before_cleanup - baseline_memory)) * 100 if before_cleanup > baseline_memory else 0
        
        print(f"\nMemory Cleanup Efficiency:")
        print(f"  Baseline: {baseline_memory:.2f} MB")
        print(f"  Before cleanup: {before_cleanup:.2f} MB")
        print(f"  After cleanup: {after_cleanup:.2f} MB")
        print(f"  Memory freed: {memory_freed:.2f} MB ({cleanup_percentage:.1f}%)")
        
        # Should free most temporary memory
        assert cleanup_percentage > 50 or memory_freed < 5, "Should free significant memory or use minimal memory"

    def test_memory_leak_detection(self, analyzer):
        """Test for memory leaks in repeated analysis"""
        # Use tracemalloc to detect leaks
        tracemalloc.start()
        
        # Take initial snapshot
        gc.collect()
        snapshot1 = tracemalloc.take_snapshot()
        
        # Run many iterations
        iterations = 100
        for i in range(iterations):
            description = "Finance Manager reviews journal entries daily"
            _ = analyzer.analyze_control(f'LEAK_TEST_{i:03d}', description)
            
            # Periodic cleanup
            if i % 20 == 0:
                gc.collect()
        
        # Take final snapshot
        gc.collect()
        snapshot2 = tracemalloc.take_snapshot()
        
        # Compare snapshots
        top_stats = snapshot2.compare_to(snapshot1, 'lineno')
        
        # Check for significant memory growth
        total_growth = sum(stat.size_diff for stat in top_stats if stat.size_diff > 0)
        growth_per_iteration = total_growth / iterations / 1024  # KB per iteration
        
        print(f"\nMemory Leak Detection:")
        print(f"  Total iterations: {iterations}")
        print(f"  Total memory growth: {total_growth / 1024 / 1024:.2f} MB")
        print(f"  Growth per iteration: {growth_per_iteration:.2f} KB")
        
        # Show top memory allocations
        print("\nTop 5 memory growth areas:")
        for stat in top_stats[:5]:
            if stat.size_diff > 0:
                print(f"  {stat.traceback}: {stat.size_diff / 1024:.1f} KB")
        
        tracemalloc.stop()
        
        # Should not leak significant memory per iteration
        assert growth_per_iteration < 100, f"Memory leak detected: {growth_per_iteration:.2f} KB per iteration"

    def test_large_control_memory_impact(self, analyzer):
        """Test memory impact of very large control descriptions"""
        tracemalloc.start()
        
        # Create controls of increasing size
        sizes = [100, 500, 1000, 5000]  # words
        
        for word_count in sizes:
            gc.collect()
            
            # Create large control description
            description = ' '.join(['word' for _ in range(word_count)])
            
            # Measure memory before
            snapshot_before = tracemalloc.take_snapshot()
            
            # Analyze control
            result = analyzer.analyze_control(f'LARGE_{word_count}', description)
            
            # Measure memory after
            snapshot_after = tracemalloc.take_snapshot()
            
            # Calculate memory used
            stats = snapshot_after.compare_to(snapshot_before, 'lineno')
            memory_used = sum(stat.size_diff for stat in stats if stat.size_diff > 0) / 1024 / 1024
            
            print(f"\nControl with {word_count} words:")
            print(f"  Memory used: {memory_used:.2f} MB")
            print(f"  Memory per word: {memory_used * 1024 / word_count:.2f} KB")
            
            # Even large controls should use reasonable memory
            assert memory_used < 50, f"Large control ({word_count} words) should use < 50MB, used {memory_used:.2f} MB"
        
        tracemalloc.stop()