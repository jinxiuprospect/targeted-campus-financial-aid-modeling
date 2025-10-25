# -*- coding: utf-8 -*-
"""
校园精准资助：从消费数据分析、异常识别、评分建模到资助等级划分
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib import rcParams
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, calinski_harabasz_score
import math

# 设置全局字体为宋体
rcParams['font.family'] = 'Songti SC'
plt.rcParams['axes.unicode_minus'] = False

# 一、加载清洗后数据
cleaned_data = pd.read_excel("1.xlsx")

# 二、数据预处理
cleaned_data = cleaned_data[cleaned_data['消费类型'] == '消费']
cleaned_data = cleaned_data.dropna(subset=['消费金额（元）'])
cleaned_data['消费时间'] = pd.to_datetime(cleaned_data['消费时间'])
cleaned_data.sort_values(by=['校园卡号', '消费时间'], inplace=True)

# 三、标记核心场所
core_places = ['第一食堂', '第二食堂', '第三食堂', '第四食堂', '第五食堂', '好利来食品店']
cleaned_data['场所类型'] = cleaned_data['消费地点'].apply(lambda x: '核心场所' if x in core_places else '辅助场所')

# 四、构建学生画像
def get_student_features(df):
    results = []
    for sid, group in df.groupby('校园卡号'):
        gender = group['性别'].iloc[0]
        total = group['消费金额（元）'].sum()
        count = group.shape[0]
        avg_once = total / count if count > 0 else 0
        daily = total / 30
        core_group = group[group['场所类型'] == '核心场所']
        core_ratio = len(core_group) / count if count > 0 else 0
        low_threshold = np.percentile(core_group['消费金额（元）'], 5) if len(core_group) > 0 else 0
        low_consume = (core_group['消费金额（元）'] < low_threshold).sum()
        low_ratio = low_consume / len(core_group) if len(core_group) > 0 else 0
        diversity = group['消费地点'].nunique()
        std = group['消费金额（元）'].std()
        cv = std / avg_once if avg_once > 0 else 0
        results.append([sid, gender, avg_once, count, low_ratio, diversity, core_ratio, daily, cv])
    return pd.DataFrame(results, columns=['校园卡号', '性别', '单次平均消费金额', '消费频次', '异常低消比例', '场所多样性', '核心场所消费比', '日均消费', '消费金额波动'])

student_features = get_student_features(cleaned_data)

# 五、可视化：单次平均消费金额直方图
plt.figure(figsize=(8, 5))
counts, bins, patches = plt.hist(student_features['单次平均消费金额'], bins=20, color='skyblue', edgecolor='black')
plt.title('单次平均消费金额分布')
plt.xlabel('元')
plt.ylabel('学生数量')
for count, patch in zip(counts, patches):
    plt.text(patch.get_x() + patch.get_width()/2, count, int(count), ha='center', va='bottom', fontsize=8)
plt.tight_layout()
plt.savefig("图1_单次平均消费金额分布.png", dpi=600)

# 六、构建评价指标体系并归一化
raw_scores = student_features.set_index('校园卡号')[['单次平均消费金额', '消费频次', '异常低消比例', '场所多样性', '核心场所消费比', '日均消费', '消费金额波动']].copy()
reverse_cols = ['单次平均消费金额', '场所多样性', '日均消费', '消费金额波动']
for col in reverse_cols:
    raw_scores[col] = raw_scores[col].max() - raw_scores[col]

scaler = MinMaxScaler()
normalized = pd.DataFrame(scaler.fit_transform(raw_scores), index=raw_scores.index, columns=raw_scores.columns)

# 七、计算指标相关性热力图
plt.figure(figsize=(8, 6))
corr = raw_scores.corr()
sns.heatmap(corr, annot=True, cmap='RdYlBu_r', fmt=".2f")
plt.title("评分指标间的相关性热力图")
plt.xticks(rotation=45, ha='right')
plt.yticks(rotation=0)
plt.tight_layout()
plt.savefig("图2_评分指标相关性热力图.png", dpi=600)

# 八、熵权法求权重
k = 1.0 / math.log(len(normalized))
P = normalized / normalized.sum()
E = -k * (P * np.log(P + 1e-8)).sum()
d = 1 - E
w = d / d.sum()

# 可视化权重柱状图
plt.figure(figsize=(8, 5))
bars = plt.bar(w.index, w.values)
plt.title("熵权法下各指标权重")
plt.ylabel("权重值")
plt.xticks(rotation=30)
for bar in bars:
    yval = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2, yval, f"{yval:.2f}", ha='center', va='bottom', fontsize=8)
plt.tight_layout()
plt.savefig("图3_熵权指标权重柱状图.png", dpi=600)

# 九、TOPSIS综合得分
Z = normalized * w
A_plus = Z.max()
A_minus = Z.min()
D_plus = np.linalg.norm(Z - A_plus, axis=1)
D_minus = np.linalg.norm(Z - A_minus, axis=1)
C = D_minus / (D_plus + D_minus)
raw_scores['贫困得分'] = C

# 可视化得分分布
plt.figure(figsize=(8, 5))
counts, bins, patches = plt.hist(C, bins=20, color='lightgreen', edgecolor='black')
plt.title('TOPSIS 综合贫困得分分布')
plt.xlabel('贫困得分')
plt.ylabel('学生数量')
for count, patch in zip(counts, patches):
    plt.text(patch.get_x() + patch.get_width()/2, count, int(count), ha='center', va='bottom', fontsize=8)
plt.tight_layout()
plt.savefig("图4_TOPSIS贫困得分分布图.png", dpi=600)

# 十、K-means聚类评估
selected = raw_scores.sort_values(by='贫困得分').head(50)
X = selected.drop(columns='贫困得分')
k_range = range(2, 8)
sse, sils, chs = [], [], []
for k in k_range:
    model = KMeans(n_clusters=k, random_state=0).fit(X)
    sse.append(model.inertia_)
    sils.append(silhouette_score(X, model.labels_))
    chs.append(calinski_harabasz_score(X, model.labels_))

# 肘部法则
plt.figure(figsize=(8, 5))
plt.plot(k_range, sse, marker='o')
for i, val in enumerate(sse):
    plt.text(k_range[i], val, f"{val:.0f}", ha='center', va='bottom', fontsize=8)
plt.title("肘部法则：聚类数与SSE")
plt.xlabel("K值")
plt.ylabel("SSE")
plt.tight_layout()
plt.savefig("图5_肘部法则_SSE曲线.png", dpi=600)

# 轮廓系数法
plt.figure(figsize=(8, 5))
plt.plot(k_range, sils, marker='^')
for i, val in enumerate(sils):
    plt.text(k_range[i], val, f"{val:.2f}", ha='center', va='bottom', fontsize=8)
plt.title("轮廓系数法：聚类数与轮廓系数")
plt.xlabel("K值")
plt.ylabel("轮廓系数")
plt.tight_layout()
plt.savefig("图6_轮廓系数法评估曲线.png", dpi=600)

# CH法
plt.figure(figsize=(8, 5))
plt.plot(k_range, chs, marker='s')
for i, val in enumerate(chs):
    plt.text(k_range[i], val, f"{val:.0f}", ha='center', va='bottom', fontsize=8)
plt.title("CH指数法：聚类数与CH值")
plt.xlabel("K值")
plt.ylabel("CH值")
plt.tight_layout()
plt.savefig("图7_CH值评估曲线.png", dpi=600)

# 十一、最终聚类 + 可视化
kmeans = KMeans(n_clusters=4, random_state=0).fit(X)
selected['资助等级'] = kmeans.labels_ + 1

plt.figure(figsize=(8, 6))
sns.scatterplot(x=selected['消费频次'], y=selected['单次平均消费金额'], hue=selected['资助等级'], palette='Set2')
plt.title('K-means聚类下的资助等级分布')
plt.xlabel('消费频次')
plt.ylabel('单次平均消费金额')
plt.tight_layout()
plt.savefig("图8_Kmeans聚类等级散点图.png", dpi=600)

plt.figure(figsize=(8, 5))
sns.boxplot(data=selected, x='资助等级', y='日均消费')
plt.title("不同资助等级学生的日均消费对比")
plt.xlabel("资助等级")
plt.ylabel("日均消费金额")
plt.tight_layout()
plt.savefig("图9_各等级日均消费箱型图.png", dpi=600)

# 输出Excel
to_save = selected.sort_values(by='资助等级')
to_save.to_excel("资助学生名单与等级.xlsx")

# 十二、金额区间测算
def compute_grant_ranges_refined(df, value_col='日均消费', level_col='资助等级'):
    base_days = 150
    grant_list = []

    # 找出Ⅰ类特困组的中位数，作为基础线
    base_group = df[df[level_col] == 1][value_col]
    base_median = base_group.median()
    base_need = base_median * base_days
    base_grant = base_need * 1.10  # 10%补贴

    # 补贴比例映射
    subsidy_ratios = {1: 1.10, 2: 1.00, 3: 0.90, 4: 0.80, 5: 0.70, 6: 0.60}

    for level in sorted(df[level_col].unique()):
        group = df[df[level_col] == level]
        median_daily = group[value_col].median()
        est_need = median_daily * base_days
        subsidy = subsidy_ratios.get(level, 0.60)
        suggested = base_need * subsidy
        low, high = round(suggested - 50, 2), round(suggested + 50, 2)

        grant_list.append({
            '资助等级': level,
            '人数': len(group),
            '日均消费中位数': round(median_daily, 2),
            '估算学期消费': round(est_need, 2),
            '建议资助金额': round(suggested, 2),
            '建议资助区间': f'{low} ~ {high} 元'
        })

    return pd.DataFrame(grant_list)

# 调用区间测算
grant_ranges_df = compute_grant_ranges_refined(selected)
grant_ranges_df.to_excel("不同等级资助金额区间测算表.xlsx", index=False)

# 可视化一：各等级建议金额对比
plt.figure(figsize=(8, 5))
bars = plt.bar(grant_ranges_df['资助等级'], grant_ranges_df['建议资助金额'], color='skyblue', edgecolor='black')
plt.title('各资助等级建议资助金额')
plt.xlabel('资助等级')
plt.ylabel('金额（元）')
for bar in bars:
    yval = bar.get_height()
    plt.text(bar.get_x() + bar.get_width() / 2, yval + 20, f"{int(yval)}", ha='center', fontsize=9)
plt.tight_layout()
plt.savefig("图10_各等级建议资助金额柱状图.png", dpi=600)

# 可视化二：预算总金额计算
grant_ranges_df['预算总额'] = grant_ranges_df['建议资助金额'] * grant_ranges_df['人数']
plt.figure(figsize=(8, 5))
bars2 = plt.bar(grant_ranges_df['资助等级'], grant_ranges_df['预算总额'], color='lightgreen', edgecolor='black')
plt.title('各等级资助预算总额')
plt.xlabel('资助等级')
plt.ylabel('总金额（元）')
for bar in bars2:
    yval = bar.get_height()
    plt.text(bar.get_x() + bar.get_width() / 2, yval + 500, f"{int(yval)}", ha='center', fontsize=9)
plt.tight_layout()
plt.savefig("图11_各等级预算总额柱状图.png", dpi=600)

# 汇总总预算输出
total_budget = grant_ranges_df['预算总额'].sum()
print(f"预计总资助预算：{round(total_budget, 2)} 元")


# 十三、针对第一问的消费趋势分析
# 图12：每日消费总额趋势图
cleaned_data['日期'] = cleaned_data['消费时间'].dt.date
by_day = cleaned_data.groupby('日期')['消费金额（元）'].sum()

plt.figure(figsize=(10, 5))
by_day.plot(marker='o', linestyle='-')
plt.title("每日消费总额趋势图")
plt.xlabel("日期")
plt.ylabel("消费总金额（元）")
plt.grid(True, linestyle='--', alpha=0.5)
plt.tight_layout()
plt.savefig("图12_每日消费趋势图.png", dpi=600)

# 图13：按小时消费频率直方图
cleaned_data['小时'] = cleaned_data['消费时间'].dt.hour
hour_counts = cleaned_data['小时'].value_counts().sort_index()

plt.figure(figsize=(10, 5))
bars = plt.bar(hour_counts.index, hour_counts.values, color='orange', edgecolor='black')
plt.title("各时段消费频率分布图")
plt.xlabel("小时（0-23）")
plt.ylabel("消费记录数")
plt.xticks(range(0, 24))
for i, bar in enumerate(bars):
    yval = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2, yval + 5, int(yval), ha='center', fontsize=9)
plt.tight_layout()
plt.savefig("图13_小时消费频率分布.png", dpi=600)

# 图14：一周内消费频率分布
cleaned_data['星期'] = cleaned_data['消费时间'].dt.dayofweek
weekday_map = ['周一', '周二', '周三', '周四', '周五', '周六', '周日']
cleaned_data['星期中文'] = cleaned_data['星期'].map(dict(enumerate(weekday_map)))
weekday_counts = cleaned_data['星期中文'].value_counts().reindex(weekday_map)

plt.figure(figsize=(8, 5))
bars2 = plt.bar(weekday_counts.index, weekday_counts.values, color='steelblue', edgecolor='black')
plt.title("周内消费频率分布图")
plt.xlabel("星期")
plt.ylabel("消费次数")
for bar in bars2:
    yval = bar.get_height()
    plt.text(bar.get_x() + bar.get_width() / 2, yval + 5, int(yval), ha='center', fontsize=9)
plt.tight_layout()
plt.savefig("图14_周内消费频率分布.png", dpi=600)
