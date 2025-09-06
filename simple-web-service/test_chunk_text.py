#!/usr/bin/env python3
"""
测试chunk_text函数的边界条件和死循环风险
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from document_loader import chunk_text


def test_empty_text():
    """测试空文本"""
    print("=== 测试空文本 ===")
    result = chunk_text("")
    print(f"结果: {result}")
    assert result == [], "空文本应返回空列表"


def test_whitespace_only():
    """测试仅空白字符"""
    print("=== 测试空白文本 ===")
    result = chunk_text("   \n\t  ")
    print(f"结果: {result}")
    assert result == [], "空白文本应返回空列表"


def test_short_text():
    """测试短文本"""
    print("=== 测试短文本 ===")
    text = "这是一个短文本"
    result = chunk_text(text, chunk_size=100)
    print(f"结果: {result}")
    assert len(result) == 1, "短文本应返回单个块"


def test_boundary_conditions():
    """测试边界条件"""
    print("=== 测试边界条件 ===")
    
    # 测试chunk_size <= overlap
    text = "a" * 100
    result = chunk_text(text, chunk_size=10, overlap=15)
    print(f"chunk_size=10, overlap=15 结果: {len(result)}个块")
    assert len(result) > 0, "应正常处理重叠大于块大小的情况"
    
    # 测试chunk_size为0或负数
    result = chunk_text("test", chunk_size=0)
    print(f"chunk_size=0 结果: {result}")
    assert len(result) == 1, "chunk_size为0时应返回原文本"


def test_large_text():
    """测试大文本"""
    print("=== 测试大文本 ===")
    text = "这是一个测试文本。" * 1000  # 20000字符
    result = chunk_text(text, chunk_size=512, overlap=50)
    print(f"大文本分块数量: {len(result)}")
    assert len(result) > 10, "大文本应生成多个块"
    
    # 验证重叠是否正确
    for i in range(1, len(result)):
        prev_chunk = result[i-1]
        curr_chunk = result[i]
        # 检查重叠部分是否存在
        overlap_found = any(
            prev_chunk[-j:] == curr_chunk[:j] 
            for j in range(1, min(51, len(prev_chunk), len(curr_chunk)))
        )
        if not overlap_found:
            print(f"警告: 块{i-1}和块{i}之间可能缺少重叠")


def test_progression_safety():
    """测试循环推进安全性"""
    print("=== 测试循环安全性 ===")
    
    # 极端情况测试
    text = "abc"
    
    # 测试极端重叠
    result = chunk_text(text, chunk_size=2, overlap=10)
    print(f"极端重叠测试结果: {result}")
    assert len(result) > 0, "应能处理极端重叠情况"
    
    # 验证不会出现无限循环
    for chunk_size in [1, 2, 5, 10, 100]:
        for overlap in [0, 1, 5, 10, 50, 100]:
            try:
                result = chunk_text(text, chunk_size=chunk_size, overlap=overlap)
                print(f"chunk_size={chunk_size}, overlap={overlap}: {len(result)}块")
            except Exception as e:
                print(f"错误: chunk_size={chunk_size}, overlap={overlap} 失败: {e}")
                raise


if __name__ == "__main__":
    print("🧪 开始测试chunk_text函数...")
    
    try:
        test_empty_text()
        test_whitespace_only()
        test_short_text()
        test_boundary_conditions()
        test_large_text()
        test_progression_safety()
        
        print("\n✅ 所有测试通过！")
        
    except Exception as e:
        print(f"\n❌ 测试失败: {e}")
        raise