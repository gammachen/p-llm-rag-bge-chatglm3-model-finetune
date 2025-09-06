#!/usr/bin/env python3
"""
性能对比测试：原始版本 vs 优化版本
"""

import time
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))


def original_chunk_text(text, chunk_size=512, overlap=50):
    """原始版本 - 用于对比测试"""
    chunks = []
    start = 0
    while start < len(text):
        end = min(start + chunk_size, len(text))
        chunks.append(text[start:end])
        start = end - overlap
    return chunks


def optimized_chunk_text(text, chunk_size=512, overlap=50):
    """优化版本"""
    from document_loader import chunk_text
    return chunk_text(text, chunk_size, overlap)


def generate_test_text(size_mb=1):
    """生成测试文本"""
    base_text = "这是一个测试文本。包含中文和English内容。用于测试分块功能的性能和正确性。" * 100
    target_size = size_mb * 1024 * 1024  # MB to bytes
    repeat_count = max(1, target_size // len(base_text.encode('utf-8')))
    return base_text * repeat_count


def benchmark_function(func, text, chunk_size, overlap, iterations=100):
    """基准测试函数"""
    times = []
    
    for _ in range(iterations):
        start_time = time.perf_counter()
        try:
            result = func(text, chunk_size, overlap)
            end_time = time.perf_counter()
            times.append(end_time - start_time)
        except Exception as e:
            print(f"错误: {e}")
            return None, None, None
    
    if times:
        avg_time = sum(times) / len(times)
        min_time = min(times)
        max_time = max(times)
        return result, avg_time, min_time, max_time
    
    return None, None, None, None


def test_edge_cases():
    """测试边界情况"""
    print("🔍 边界情况测试")
    
    test_cases = [
        ("", "空文本"),
        ("a", "单字符"),
        ("ab", "双字符"),
        ("a" * 1000, "大文本"),
    ]
    
    for text, description in test_cases:
        print(f"\n{description}:")
        
        # 原始版本
        try:
            orig_result = original_chunk_text(text, chunk_size=10, overlap=5)
            print(f"  原始版本: {len(orig_result)}个块")
        except Exception as e:
            print(f"  原始版本: 错误 - {e}")
        
        # 优化版本
        try:
            opt_result = optimized_chunk_text(text, chunk_size=10, overlap=5)
            print(f"  优化版本: {len(opt_result)}个块")
        except Exception as e:
            print(f"  优化版本: 错误 - {e}")


def main():
    print("🚀 开始性能对比测试...")
    
    # 测试边界情况
    test_edge_cases()
    
    # 生成测试数据
    print("\n📊 生成测试数据...")
    test_text = generate_test_text(size_mb=0.1)  # 100KB for quick testing
    print(f"测试文本大小: {len(test_text.encode('utf-8')) / 1024:.1f}KB")
    
    # 测试参数组合
    test_params = [
        (512, 50),
        (256, 25),
        (1024, 100),
        (100, 10),
    ]
    
    print("\n⏱️  性能测试结果:")
    print("chunk_size | overlap | 版本 | 块数 | 平均时间(ms) | 最小时间 | 最大时间")
    print("-" * 70)
    
    for chunk_size, overlap in test_params:
        # 原始版本
        orig_result, orig_avg, orig_min, orig_max = benchmark_function(
            original_chunk_text, test_text, chunk_size, overlap, iterations=50
        )
        
        # 优化版本
        opt_result, opt_avg, opt_min, opt_max = benchmark_function(
            optimized_chunk_text, test_text, chunk_size, overlap, iterations=50
        )
        
        if orig_result and opt_result:
            print(f"{chunk_size:>10} | {overlap:>7} | 原始 | {len(orig_result):>4} | {orig_avg*1000:>10.2f} | {orig_min*1000:>8.2f} | {orig_max*1000:>8.2f}")
            print(f"{'':>10} | {'':>7} | 优化 | {len(opt_result):>4} | {opt_avg*1000:>10.2f} | {opt_min*1000:>8.2f} | {opt_max*1000:>8.2f}")
            print("-" * 70)
    
    # 测试极端情况
    print("\n⚠️ 极端情况测试:")
    extreme_params = [
        (5, 10),   # overlap > chunk_size
        (1, 0),    # 最小chunk_size
        (1000, 999),  # 接近chunk_size的overlap
    ]
    
    for chunk_size, overlap in extreme_params:
        print(f"\nchunk_size={chunk_size}, overlap={overlap}:")
        
        try:
            orig_result = original_chunk_text("测试文本", chunk_size, overlap)
            print(f"  原始版本: {len(orig_result)}个块")
        except Exception as e:
            print(f"  原始版本: 错误 - {e}")
        
        try:
            opt_result = optimized_chunk_text("测试文本", chunk_size, overlap)
            print(f"  优化版本: {len(opt_result)}个块")
        except Exception as e:
            print(f"  优化版本: 错误 - {e}")


if __name__ == "__main__":
    main()