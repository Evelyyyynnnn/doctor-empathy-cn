#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
运行所有测试
Run All Tests
"""

import sys
import os
import subprocess
import time

def run_test(test_file):
    """运行单个测试文件"""
    print(f"\n{'='*60}")
    print(f"🧪 运行测试: {test_file}")
    print(f"{'='*60}")
    
    start_time = time.time()
    
    try:
        # 运行测试文件
        result = subprocess.run([sys.executable, test_file], 
                              capture_output=True, 
                              text=True, 
                              cwd=os.path.dirname(os.path.abspath(__file__)))
        
        end_time = time.time()
        duration = end_time - start_time
        
        if result.returncode == 0:
            print(f"✅ 测试通过: {test_file}")
            print(f"⏱️  耗时: {duration:.2f}秒")
            if result.stdout:
                print("📤 输出:")
                print(result.stdout)
            return True
        else:
            print(f"❌ 测试失败: {test_file}")
            print(f"⏱️  耗时: {duration:.2f}秒")
            if result.stderr:
                print("🚨 错误信息:")
                print(result.stderr)
            if result.stdout:
                print("📤 输出:")
                print(result.stdout)
            return False
            
    except Exception as e:
        print(f"💥 运行测试时出现异常: {e}")
        return False

def main():
    """主函数"""
    print("🚀 开始运行所有测试...")
    
    # 获取当前目录下的所有测试文件
    test_dir = os.path.dirname(os.path.abspath(__file__))
    test_files = []
    
    for file in os.listdir(test_dir):
        if file.startswith('test_') and file.endswith('.py') and file != 'run_all_tests.py':
            test_files.append(file)
    
    if not test_files:
        print("❌ 没有找到测试文件")
        return
    
    print(f"📋 找到 {len(test_files)} 个测试文件:")
    for test_file in test_files:
        print(f"  - {test_file}")
    
    # 运行所有测试
    passed = 0
    failed = 0
    
    for test_file in test_files:
        test_path = os.path.join(test_dir, test_file)
        if run_test(test_path):
            passed += 1
        else:
            failed += 1
    
    # 打印测试结果摘要
    print(f"\n{'='*60}")
    print("📊 测试结果摘要")
    print(f"{'='*60}")
    print(f"✅ 通过: {passed}")
    print(f"❌ 失败: {failed}")
    print(f"📈 成功率: {passed/(passed+failed)*100:.1f}%")
    
    if failed == 0:
        print("\n🎉 所有测试都通过了！")
    else:
        print(f"\n⚠️  有 {failed} 个测试失败，请检查错误信息。")
    
    return failed == 0

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
