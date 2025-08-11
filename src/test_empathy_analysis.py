#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
测试同理心分析系统
"""

import sys
import os
sys.path.append('src')

from empathy_analysis import EmpathyAnalyzer

def test_empathy_recognition():
    """测试同理心识别功能"""
    print("=== 测试同理心识别系统 ===\n")
    
    # 创建分析器实例
    analyzer = EmpathyAnalyzer()
    
    # 测试样本文本（基于您的实际数据）
    test_texts = [
        "感谢您的信任，病情资料我已详细阅读。您目前的检查资料不齐全，需要补充以下检查：既往史",
        "能够理解，不行只能手术解决",
        "不要太着急，小孩毕竟还小很多事情呢，慢慢来啊，反正就要一个是要重视啊，好吧",
        "密切观察小儿的视功能发育，定期复查",
        "建议完善眼肌MRI及甲功检查",
        "注意休息，局部热敷按摩",
        "确实，这个情况需要重视",
        "我明白您的担心，但是不用太紧张",
        "建议您定期到医院检查，看看他那个白内障的那个变化情况",
        "您的眼睛原来有过什么其他情况"
    ]
    
    print("测试文本样本:")
    for i, text in enumerate(test_texts, 1):
        print(f"{i}. {text}")
    
    print("\n=== 同理心特征识别结果 ===\n")
    
    # 测试每个文本
    for i, text in enumerate(test_texts, 1):
        print(f"文本 {i}: {text}")
        print("-" * 50)
        
        # 识别同理心词汇
        empathy_words = analyzer._identify_empathy_words(text)
        
        if empathy_words:
            print(f"识别到的同理心词汇: {empathy_words}")
            
            # 按类别分类
            categorized_words = {}
            for word in empathy_words:
                for category, features in analyzer.empathy_features.items():
                    if word in features:
                        if category not in categorized_words:
                            categorized_words[category] = []
                        categorized_words[category].append(word)
                        break
            
            print("按类别分类:")
            for category, words in categorized_words.items():
                print(f"  {category}: {words}")
        else:
            print("未识别到同理心词汇")
        
        print()
    
    # 测试词云生成
    print("=== 测试词云生成 ===\n")
    
    # 创建模拟的分析结果
    mock_analysis_results = {
        'consultations': [
            {'doctor_speech': ' '.join(test_texts[:5])},
            {'doctor_speech': ' '.join(test_texts[5:])}
        ]
    }
    
    try:
        result = analyzer.generate_wordcloud(mock_analysis_results)
        if result:
            print(f"词云生成成功: {result}")
        else:
            print("词云生成失败")
    except Exception as e:
        print(f"词云生成出错: {e}")
    
    print("\n=== 测试完成 ===")

if __name__ == "__main__":
    test_empathy_recognition()
