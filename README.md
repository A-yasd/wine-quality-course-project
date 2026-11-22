# 类别不平衡场景下的优质红酒筛选

深度学习课程设计（2026 秋季）小型研究项目。任务是把 UCI 红葡萄酒质量数据中的 `quality >= 7` 识别为稀有优质类，比较加权逻辑回归与不同正类权重设置的多层感知机。

## 主要结果

- 数据：1599 个样本、11 个理化特征，正类 217 个（13.57%）。
- 验证：5 折分层交叉验证 × 3 个随机种子，共 15 次外层评估。
- 无权重 MLP：Macro-F1 = 0.727 ± 0.045，PR-AUC = 0.538 ± 0.086。
- 与加权逻辑回归的 Macro-F1 配对差值为 0.030（p = 0.029）。
- 类权重会改变固定阈值下的召回/特异度权衡，但在阈值校准后没有稳定提升综合指标。

## 运行

```powershell
python -m pip install -r requirements.txt
python src/experiment.py --data data/winequality-red.csv --output . --repeats 3 --folds 5
python src/make_report.py
python src/make_pdf.py
```

实验在普通 CPU 上约 30 秒完成。报告由 `python-docx` 生成，并由 ReportLab 生成同步的 A4 PDF。

## 目录

- `data/winequality-red.csv`：UCI 公开数据。
- `src/experiment.py`：数据加载、嵌套验证、模型训练、指标、消融与绘图。
- `src/make_report.py`：读取实验结果并生成课程设计报告。
- `results/`：逐次指标、OOF 预测、汇总统计和置换重要性。
- `figures/`：报告使用的三张图。

## 数据与算法来源

- Cortez, P., Cerdeira, A., Almeida, F., Matos, T., & Reis, J. (2009). *Modeling wine preferences by data mining from physicochemical properties*. Decision Support Systems, 47(4), 547–553.
- UCI Machine Learning Repository: <https://archive.ics.uci.edu/dataset/186/wine+quality>

代码使用 NumPy 从零实现训练与指标计算。生成式 AI 用于代码组织、文字润色和排版辅助；实验数值均由本仓库脚本真实运行产生。
