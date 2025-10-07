# macao-concert-economy-analysis
澳門演藝經濟效益分析的機器學習模型
# 澳门演艺经济分析

本仓库包含论文《澳门文创产业发展策略研究—以演艺经济为驱动的实证分析与政策路径》的完整代码和数据。

## 文件结构

- `演唱会最终模型.ipynb`: 主分析文件，包含数据预处理、特征工程、模型训练、政策模拟等全部代码。
- `requirements.txt`: Python环境依赖包列表。
- `data/`: 数据目录（注意：因数据隐私，原始数据可能未包含，请参见数据说明部分）。
- `outputs/`: 运行代码后生成的图表和结果

  
核心代碼展示
数据预处理与特征工程
# 关键特征创建
df['演唱会_酒店交互'] = df['演唱会数量'] * df['酒店平均入住率_百分比']
df['观众消费潜力'] = df['观众人数'] * df['旅客人均消费_澳门元'] / 1000
df['是否旺季'] = df['季度'].isin([1, 4]).astype(int)

2. 模型训练
# 模型参数与训练
params = {
    'objective': 'regression',
    'num_leaves': 31,
    'learning_rate': 0.05,
    'random_state': 42
}

model = lgb.train(
    params, 
    train_data,
    num_boost_round=1000,
    valid_sets=[test_data],
    callbacks=[lgb.early_stopping(50)]
)

3. SHAP可解释性分析
# 特征重要性分析
explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X)
shap.summary_plot(shap_values, X, feature_names=feature_names)
# 可视化
shap.summary_plot(shap_values, X_test, feature_names=features)

4.政策效果模拟
# 政策情景预测
def simulate_policy(model, data, changes):
    scenario_data = data.copy()
    for feature, adjustment in changes.items():
        scenario_data[feature] *= (1 + adjustment/100)
    return model.predict(scenario_data).mean()

## 运行环境

建议使用Python 3.8及以上版本。安装依赖：

```bash
pip install -r requirements.txt
