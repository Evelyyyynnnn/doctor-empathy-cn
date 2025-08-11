#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
测试词云生成修复
Test WordCloud Generation Fix
"""

import sys
import os
# 添加src目录到Python路径
sys.path.append(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'src'))

from empathy_analysis import EnhancedEmpathyAnalyzer
import json

def test_wordcloud_generation():
    """测试词云生成功能"""
    print("🔧 测试词云生成功能...")
    
    # 创建分析器实例
    analyzer = EnhancedEmpathyAnalyzer()
    
    # 加载现有的分析结果
    try:
        data_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'data', 'detailed_empathy_analysis.json')
        with open(data_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        analysis_results = [data]  # 将整个数据对象包装在列表中
        print(f"✅ 成功加载分析结果，包含 {data.get('total_cases', 0)} 个案例")
    except Exception as e:
        print(f"❌ 加载分析结果失败: {e}")
        return False
    
    # 测试词云生成
    try:
        print("\n📊 生成词云...")
        wordcloud = analyzer.generate_wordcloud(analysis_results)
        
        if wordcloud is not None:
            print("✅ 词云生成成功！")
            print("📁 词云图片已保存到: outputs/figures/enhanced_empathy_keywords_wordcloud.png")
            return True
        else:
            print("❌ 词云生成失败")
            return False
            
    except Exception as e:
        print(f"❌ 词云生成时出现错误: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """主函数"""
    print("🚀 开始测试词云生成修复...")
    
    # 确保输出目录存在
    output_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'outputs', 'figures')
    os.makedirs(output_dir, exist_ok=True)
    
    # 测试词云生成
    success = test_wordcloud_generation()
    
    if success:
        print("\n🎉 所有测试通过！词云生成修复成功。")
    else:
        print("\n💥 测试失败，请检查错误信息。")
    
    return success

if __name__ == "__main__":
    main() 