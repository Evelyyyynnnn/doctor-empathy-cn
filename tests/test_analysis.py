#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
测试增强版同理心分析器
Test Enhanced Empathy Analyzer
"""

import sys
import os
# 添加src目录到Python路径
sys.path.append(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'src'))

from empathy_analysis import EnhancedEmpathyAnalyzer
import numpy as np

def test_basic_functionality():
    """测试基本功能"""
    print("\n🔧 测试基本功能...")
    
    analyzer = EnhancedEmpathyAnalyzer()
    
    # 测试特征提取
    test_text = "我理解您的担心，这种症状确实会让人焦虑"
    features = analyzer.extract_enhanced_features(test_text)
    empathy_features = analyzer.extract_empathy_features(test_text)
    
    print(f"增强特征数量: {len(features)}")
    print(f"同理心特征数量: {len(empathy_features)}")
    
    return len(features) > 0 and len(empathy_features) > 0

def test_enhanced_excel_functionality():
    """测试增强版Excel数据分析功能"""
    print("\n📊 测试增强版Excel数据分析功能...")
    
    analyzer = EnhancedEmpathyAnalyzer()
    
    # 测试医生话语提取
    test_conversation = "陈锦昌副主任医师:我理解您的担心(2024.01.01) 患者:我很焦虑 陈锦昌副主任医师:别担心，会好的(2024.01.01)"
    doctor_speech = analyzer.extract_doctor_speech(test_conversation)
    
    print(f"提取的医生话语: {doctor_speech}")
    
    # 测试增强版同理心评分
    test_text = "我理解您的担心，这种症状确实会让人焦虑"
    enhanced_result = analyzer.calculate_enhanced_empathy_score(test_text, use_enhanced_keywords=True)
    original_result = analyzer.calculate_enhanced_empathy_score(test_text, use_enhanced_keywords=False)
    
    print(f"增强版总分: {enhanced_result['total_score']:.2f}")
    print(f"原版总分: {original_result['total_score']:.2f}")
    print(f"各类别评分: {enhanced_result['category_scores']}")
    
    # 测试导出empathy_scores.csv功能
    print("\n📋 测试导出empathy_scores.csv功能...")
    
    # 创建模拟的分析结果数据
    mock_analysis_results = [
        {
            'case_id': 'Case_001',
            'doctor_name': '陈锦昌副主任医师',
            'patient_age': '45',
            'patient_gender': '女',
            'disease_category': '内科疾病',
            'consultation_date': '2024.01.01',
            'avg_empathy_score': 8.5,
            'empathy_density': 0.85,
            'word_count': 20,
            'dialogue_length': 50,
            'empathy_features_count': 5,
            'empathy_scores': {
                '感谢信任': 1.0,
                '理解认同': 2.0,
                '关心注意': 1.5,
                '安慰支持': 2.5,
                '倾听确认': 1.0,
                '耐心解释': 2.0
            },
            'doctor_speech': '我理解您的担心，这种症状确实会让人焦虑，别担心，会好的'
        }
    ]
    
    # 测试导出功能
    success = analyzer.export_empathy_scores_csv(mock_analysis_results, 'test_empathy_scores.csv')
    
    if success:
        print("✅ export_empathy_scores_csv 测试通过")
        # 清理测试文件
        import os
        if os.path.exists('test_empathy_scores.csv'):
            os.remove('test_empathy_scores.csv')
            print("🧹 测试文件已清理")
    else:
        print("❌ export_empathy_scores_csv 测试失败")
    
    return (len(doctor_speech) > 0 and 
            enhanced_result['total_score'] > 0 and 
            original_result['total_score'] > 0 and
            success)

def test_ml_functionality():
    """测试机器学习功能"""
    print("\n🤖 测试机器学习功能...")
    
    analyzer = EnhancedEmpathyAnalyzer()
    
    # 创建训练数据
    training_data = analyzer.create_synthetic_training_data()
    X, y = analyzer.prepare_training_data(training_data)
    
    print(f"训练数据特征维度: {X.shape}")
    print(f"训练数据标签分布: {np.sum(y, axis=0)}")
    
    # 测试交叉验证
    print("\n进行交叉验证...")
    cv_results = analyzer.cross_validate_models(X, y, cv_folds=3)
    
    for model_name, results in cv_results.items():
        print(f"{model_name}: CV F1 = {results['cv_mean']:.3f} ± {results['cv_std']:.3f}")
    
    # 测试模型训练
    print("\n训练机器学习模型...")
    ml_results = analyzer.train_ml_models(X, y)
    
    for model_name, results in ml_results.items():
        print(f"{model_name}: F1 Micro = {results['f1_micro']:.3f}, F1 Macro = {results['f1_macro']:.3f}")
    
    # 测试预测
    test_text = "我理解您的担心，这种症状确实会让人焦虑"
    print(f"\n测试预测文本: {test_text}")
    
    for model_name in ml_results.keys():
        try:
            result = analyzer.predict_empathy_ml(test_text, model_name)
            print(f"{model_name} 预测结果: 同理心总分 = {result['empathy_score']:.3f}")
        except Exception as e:
            print(f"{model_name} 预测失败: {e}")
    
    # 测试特征重要性分析
    print("\n分析特征重要性...")
    importance_df = analyzer.analyze_feature_importance('RandomForest')
    if importance_df is not None:
        print("特征重要性分析完成")
        print(f"前5个最重要特征:")
        for i, row in importance_df.head(5).iterrows():
            print(f"  {row['feature']}: {row['importance']:.4f}")
    
    # 测试集成预测
    print("\n测试集成预测...")
    try:
        ensemble_result = analyzer.ensemble_prediction(test_text)
        print(f"集成预测同理心总分: {ensemble_result['empathy_score']:.3f}")
        print("集成预测结果:")
        for label, pred in ensemble_result['ensemble_predictions'].items():
            prob = ensemble_result['ensemble_probabilities'].get(label, 0)
            print(f"  {label}: {'是' if pred else '否'} (概率: {prob:.3f})")
    except Exception as e:
        print(f"集成预测失败: {e}")
    
    return True

def test_model_persistence():
    """测试模型持久化"""
    print("\n💾 测试模型持久化...")
    
    analyzer = EnhancedEmpathyAnalyzer()
    
    # 创建训练数据并训练模型
    training_data = analyzer.create_synthetic_training_data()
    X, y = analyzer.prepare_training_data(training_data)
    
    # 训练模型
    ml_results = analyzer.train_ml_models(X, y)
    
    # 保存模型
    print("保存模型...")
    save_success = analyzer.save_models('test_models')
    
    if save_success:
        # 创建新的分析器实例
        new_analyzer = EnhancedEmpathyAnalyzer()
        
        # 加载模型
        print("加载模型...")
        load_success = new_analyzer.load_models('test_models')
        
        if load_success:
            # 测试加载的模型
            test_text = "我理解您的担心，这种症状确实会让人焦虑"
            try:
                result = new_analyzer.predict_empathy_ml(test_text, 'RandomForest')
                print(f"加载模型预测成功: 同理心总分 = {result['empathy_score']:.3f}")
                return True
            except Exception as e:
                print(f"加载模型预测失败: {e}")
                return False
        else:
            print("模型加载失败")
            return False
    else:
        print("模型保存失败")
        return False

def main():
    """主测试函数"""
    print("🧪 开始测试增强版同理心分析系统...")
    print("=" * 60)
    
    tests = [
        ("基本功能测试", test_basic_functionality),
        ("增强Excel功能测试", test_enhanced_excel_functionality),
        ("机器学习功能测试", test_ml_functionality),
        ("模型持久化测试", test_model_persistence)
    ]
    
    passed_tests = 0
    total_tests = len(tests)
    
    for test_name, test_func in tests:
        try:
            print(f"\n{'='*20} {test_name} {'='*20}")
            result = test_func()
            if result:
                print(f"✅ {test_name} 通过")
                passed_tests += 1
            else:
                print(f"❌ {test_name} 失败")
        except Exception as e:
            print(f"❌ {test_name} 出错: {e}")
            import traceback
            traceback.print_exc()
    
    print(f"\n{'='*60}")
    print(f"📊 测试结果汇总:")
    print(f"总测试数: {total_tests}")
    print(f"通过测试: {passed_tests}")
    print(f"失败测试: {total_tests - passed_tests}")
    print(f"成功率: {passed_tests/total_tests*100:.1f}%")
    
    if passed_tests == total_tests:
        print("🎉 所有测试都通过了！")
    else:
        print("⚠️ 部分测试失败，请检查相关功能")
    
    print("=" * 60)

if __name__ == "__main__":
    main()
