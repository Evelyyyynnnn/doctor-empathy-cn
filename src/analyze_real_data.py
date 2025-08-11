#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
分析实际医疗对话数据中的同理心表达
"""

import sys
import os
import pandas as pd
import re
sys.path.append('src')

from empathy_analysis import EmpathyAnalyzer

def extract_doctor_speech_from_data(df):
    """从数据中提取医生话语"""
    doctor_speeches = []
    
    for idx, row in df.iterrows():
        # 获取对话列（第5列）
        conversation_col = df.columns[4] if len(df.columns) > 4 else df.columns[-1]
        conversation_text = str(row[conversation_col])
        
        # 提取医生话语（以"陈锦昌副主任医师:"开头的话语）
        doctor_pattern = r'陈锦昌副主任医师:([^患者]*?)(?=患者:|$)'
        matches = re.findall(doctor_pattern, conversation_text)
        
        for match in matches:
            # 清理文本
            cleaned_text = re.sub(r'\(\d{4}-\d{2}-\d{2}\)', '', match)  # 移除日期
            cleaned_text = re.sub(r'以上文字由机器转写，仅供参考', '', cleaned_text)  # 移除机器转写说明
            cleaned_text = re.sub(r'\d+["″]', '', cleaned_text)  # 移除时间标记
            cleaned_text = cleaned_text.strip()
            
            if cleaned_text and len(cleaned_text) > 5:  # 过滤太短的文本
                doctor_speeches.append(cleaned_text)
    
    return doctor_speeches

def analyze_empathy_patterns(doctor_speeches):
    """分析同理心表达模式"""
    analyzer = EmpathyAnalyzer()
    
    print("=== 医生话语同理心分析 ===\n")
    
    total_speeches = len(doctor_speeches)
    empathy_speeches = 0
    empathy_word_counts = {}
    category_counts = {}
    
    for i, speech in enumerate(doctor_speeches, 1):
        print(f"话语 {i}: {speech}")
        print("-" * 60)
        
        # 识别同理心词汇
        empathy_words = analyzer._identify_empathy_words(speech)
        
        if empathy_words:
            empathy_speeches += 1
            print(f"✓ 识别到同理心词汇: {empathy_words}")
            
            # 按类别分类
            categorized_words = {}
            for word in empathy_words:
                for category, features in analyzer.empathy_features.items():
                    if word in features:
                        if category not in categorized_words:
                            categorized_words[category] = []
                        categorized_words[category] = categorized_words.get(category, []) + [word]
                        
                        # 统计类别
                        category_counts[category] = category_counts.get(category, 0) + 1
                        break
            
            print("按类别分类:")
            for category, words in categorized_words.items():
                print(f"  {category}: {words}")
                
                # 统计词汇
                for word in words:
                    empathy_word_counts[word] = empathy_word_counts.get(word, 0) + 1
        else:
            print("✗ 未识别到同理心词汇")
        
        print()
    
    # 统计总结
    print("=== 分析总结 ===")
    print(f"总话语数: {total_speeches}")
    print(f"包含同理心的话语数: {empathy_speeches}")
    print(f"同理心话语比例: {empathy_speeches/total_speeches*100:.1f}%")
    
    print("\n同理心词汇频率统计:")
    sorted_words = sorted(empathy_word_counts.items(), key=lambda x: x[1], reverse=True)
    for word, count in sorted_words:
        print(f"  {word}: {count}")
    
    print("\n同理心类别统计:")
    sorted_categories = sorted(category_counts.items(), key=lambda x: x[1], reverse=True)
    for category, count in sorted_categories:
        print(f"  {category}: {count}")
    
    return empathy_word_counts, category_counts

def main():
    """主函数"""
    print("=== 医疗对话同理心分析系统 ===\n")
    
    try:
        # 读取数据
        print("正在读取数据...")
        df = pd.read_excel('data/Sample Data.xlsx')
        print(f"成功读取数据，共 {len(df)} 行")
        
        # 提取医生话语
        print("\n正在提取医生话语...")
        doctor_speeches = extract_doctor_speech_from_data(df)
        print(f"提取到 {len(doctor_speeches)} 条医生话语")
        
        if not doctor_speeches:
            print("未找到医生话语，请检查数据格式")
            return
        
        # 分析同理心模式
        empathy_words, category_counts = analyze_empathy_patterns(doctor_speeches)
        
        # 生成词云
        print("\n正在生成词云...")
        analyzer = EmpathyAnalyzer()
        
        # 创建分析结果数据
        analysis_results = {
            'consultations': [
                {'doctor_speech': speech} for speech in doctor_speeches
            ]
        }
        
        result = analyzer.generate_wordcloud(analysis_results)
        if result:
            print(f"词云生成成功: {result}")
        else:
            print("词云生成失败")
        
    except Exception as e:
        print(f"分析过程中出现错误: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
