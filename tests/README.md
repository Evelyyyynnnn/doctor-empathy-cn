# 测试文件说明

## 📋 测试概览

本目录包含项目的所有测试文件，用于验证代码功能的正确性和稳定性。

## 🧪 测试文件

### 核心测试
- **[test_analysis.py](test_analysis.py)** - 主要分析功能的测试
- **[test_wordcloud_fix.py](test_wordcloud_fix.py)** - 词云生成修复的测试

### 测试运行
- **[run_all_tests.py](run_all_tests.py)** - 运行所有测试的主脚本
- **[__init__.py](__init__.py)** - Python包初始化文件

## 🚀 运行测试

### 运行所有测试
```bash
python tests/run_all_tests.py
```

### 运行单个测试
```bash
# 测试主要分析功能
python tests/test_analysis.py

# 测试词云修复
python tests/test_wordcloud_fix.py
```

### 从项目根目录运行
```bash
# 运行所有测试
python -m tests.run_all_tests

# 运行单个测试
python -m tests.test_analysis
python -m tests.test_wordcloud_fix
```

## 📊 测试覆盖

### 功能测试
- 同理心特征提取
- 文本预处理
- 评分计算
- 数据导出

### 修复测试
- 词云生成问题
- 中文字体显示
- 文件路径处理

## 🔧 测试环境

- Python 3.7+
- 相关依赖包（见requirements.txt）
- 测试数据文件

## 📝 添加新测试

如需添加新测试，请：
1. 创建新的测试文件（命名格式：`test_*.py`）
2. 继承适当的测试基类
3. 在`run_all_tests.py`中添加导入
4. 确保测试覆盖率

---

*最后更新：2024年*
