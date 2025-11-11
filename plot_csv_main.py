# -*- coding: utf-8 -*-
import io
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

font_candidates = ['Noto Sans CJK SC', 'Source Han Sans SC', 'WenQuanYi Micro Hei', 'PingFang SC', 'Microsoft YaHei', 'SimHei', 'Arial Unicode MS']
plt.rcParams['font.sans-serif'] = font_candidates
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['axes.unicode_minus'] = False

import seaborn as sns
# sns.set(style='whitegrid', font_scale=1.0)


class SaveFig:
    def __init__(self, out_dir):
        self.out_dir_ = out_dir
        os.makedirs(out_dir, exist_ok=True)

    def save(self, name):
        path = os.path.join(self.out_dir_, f"{name}.png")
        plt.savefig(path, dpi=300, bbox_inches='tight')
        plt.close()


def annotate_points(ax, x, y, labels, dx=0.05):
    """
    在每个散点右侧标注队名的前5个字，最小字体，并倾斜30度。
    dx: 文本相对x的右移偏移量
    """
    for xi, yi, lab in zip(x, y, labels):
        txt = str(lab)[:5]
        ax.text(
            xi + dx, yi, txt,
            fontsize=6, va='center', ha='left', alpha=0.85,
            rotation=20, rotation_mode='anchor'
        )


def pearson_r(x, y):
    if len(x) < 2:
        return np.nan
    return float(pd.Series(x).corr(pd.Series(y), method='pearson'))


def main():
    # ========================================================
    # 解析为 DataFrame
    month = '202511'
    csv_path = os.path.join('input', f'{month}.csv')
    df = pd.read_csv(csv_path)

    sf = SaveFig(f"plot_{month}")
    # 标准化列名
    df.columns = ['评委A', '分数A', '评委B', '分数B', '队伍']

    # 转为数值
    df['分数A'] = pd.to_numeric(df['分数A'], errors='coerce')
    df['分数B'] = pd.to_numeric(df['分数B'], errors='coerce')

    # 去除全空或无效行
    df = df.dropna(subset=['队伍', '评委A', '评委B', '分数A', '分数B']).reset_index(drop=True)

    # 计算派生指标
    df['平均分'] = df[['分数A', '分数B']].mean(axis=1)
    df['分差'] = (df['分数A'] - df['分数B']).abs()

    # 构建“长表”便于分组统计
    long_df = pd.concat([
        df.rename(columns={'评委A':'评委', '分数A':'分数'})[['评委', '分数', '队伍']],
        df.rename(columns={'评委B':'评委', '分数B':'分数'})[['评委', '分数', '队伍']]
    ], axis=0, ignore_index=True)

    # 各评委给每队的平均分（若同一评委多次给某队分则平均）
    judge_team = long_df.groupby(['评委', '队伍'], as_index=False)['分数'].mean()

    # 各评委总体分布
    judge_stats = long_df.groupby('评委')['分数'].agg(['count','mean','std']).sort_values('mean', ascending=False)

    # 构建评委-队伍评分透视表（用于一致性相关系数）
    pivot = judge_team.pivot_table(index='队伍', columns='评委', values='分数')
    corr = pivot.corr(method='pearson')

    # 画图参数
    # sns.set(style='whitegrid', font_scale=1.0)
    palette = sns.color_palette('tab10')

    # 1) 每队平均分（柱状图）
    plt.figure(figsize=(12, max(6, len(df)//8)))
    avg_by_team = df.groupby('队伍')['平均分'].mean().sort_values(ascending=False)
    sns.barplot(x=avg_by_team.values, y=avg_by_team.index, color=palette[0])
    plt.title('每队平均分（两位评委均值）')
    plt.xlabel('平均分')
    plt.ylabel('队伍')
    plt.tight_layout()
    sf.save("01_avg_by_team")

    # 2) 每队评委分差（柱状图）
    plt.figure(figsize=(12, max(6, len(df)//8)))
    gap_by_team = df.groupby('队伍')['分差'].mean().sort_values(ascending=False)
    sns.barplot(x=gap_by_team.values, y=gap_by_team.index, color=palette[3])
    plt.title('每队两位评委分差（绝对值）')
    plt.xlabel('分差')
    plt.ylabel('队伍')
    plt.tight_layout()
    sf.save("02_gap_by_team")

    # 3) 两位评委散点图（汇总 + Top 评委组合）
    # 统计评委配对出现次数，选Top10组（不区分顺序）
    pair_counts = (
        df.assign(pair=df.apply(lambda r: tuple(sorted([r['评委A'], r['评委B']])), axis=1))
          .groupby('pair').size().sort_values(ascending=False)
    )
    top_pairs = pair_counts.head(10).index.tolist()

    # 统一坐标范围（基于全局数据）
    minv = min(df['分数A'].min(), df['分数B'].min())
    maxv = max(df['分数A'].max(), df['分数B'].max())
    pad = 0.2
    xmin, xmax = minv - pad, maxv + pad
    ymin, ymax = xmin, xmax

    # 动态子图网格：总图数 = 1（汇总） + len(top_pairs)
    total_plots = 1 + len(top_pairs)
    max_cols = 3  # 每行最多列数，可按需要改为 4
    ncols = min(max_cols, total_plots)
    nrows = int(np.ceil(total_plots / ncols))

    fig, axes = plt.subplots(nrows, ncols, figsize=(6*ncols, 6*nrows))
    # 将 axes 拉平成一维列表，统一索引；单个轴时也处理为列表
    if isinstance(axes, np.ndarray):
        axes = axes.ravel()
    else:
        axes = [axes]

    # 子图0：汇总（所有样本）
    ax = axes[0]
    x_all = df['分数A'].values
    y_all = df['分数B'].values
    ax.scatter(x_all, y_all, alpha=0.7, c=palette[1], s=28, edgecolors='none')
    ax.plot([xmin, xmax], [ymin, ymax], 'k--', linewidth=1)

    r_all = pearson_r(x_all, y_all)
    title_all = f'同队两位评委打分散点（汇总）  r={r_all:.2f}' if not np.isnan(r_all) else '同队两位评委打分散点（汇总）  r=NaN'
    ax.set_title(title_all)
    ax.set_xlabel('评委A分数')
    ax.set_ylabel('评委B分数')
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)
    ax.set_aspect('equal', adjustable='box')

    # 后续子图：Top 评委组合
    for idx, (a, b) in enumerate(top_pairs, start=1):
        ax = axes[idx]
        # 取出该评委组合的所有样本（不区分A/B位置，映射到 x: a 的分，y: b 的分）
        sub_ab = df[(df['评委A'] == a) & (df['评委B'] == b)][['分数A', '分数B', '队伍']]
        sub_ba = df[(df['评委A'] == b) & (df['评委B'] == a)][['分数A', '分数B', '队伍']]

        # 将 (b,a) 的行交换成 (a,b)
        if not sub_ba.empty:
            sub_ba = sub_ba.rename(columns={'分数A': '分数B', '分数B': '分数A'})

        sub = pd.concat([sub_ab, sub_ba], ignore_index=True)

        x = sub['分数A'].values
        y = sub['分数B'].values
        ax.scatter(x, y, alpha=0.9, c=palette[0], s=36, edgecolors='none')
        ax.plot([xmin, xmax], [ymin, ymax], 'k--', linewidth=1)

        r = pearson_r(x, y)
        r_txt = f'{r:.2f}' if not np.isnan(r) else 'NaN'
        ax.set_title(f'{a} vs {b}（N={len(sub)}，r={r_txt}）')
        ax.set_xlabel(f'{a} 分数')
        ax.set_ylabel(f'{b} 分数')
        ax.set_xlim(xmin, xmax)
        ax.set_ylim(ymin, ymax)
        ax.set_aspect('equal', adjustable='box')

        # 标注队名（前5字）在点右侧
        annotate_points(ax, x, y, sub['队伍'].values)

    # 关闭未使用的轴（如果有）
    for j in range(total_plots, len(axes)):
        axes[j].axis('off')

    plt.suptitle('同队两位评委打分散点（汇总 + Top评委组合）', y=0.98)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    sf.save("03_scatter_two_judges")

    # 4) 各评委打分分布（直方+KDE）
    plt.figure(figsize=(10,6))
    # 整体均值与中位数
    overall_mean = float(long_df['分数'].mean()) if not long_df.empty else np.nan
    overall_median = float(long_df['分数'].median()) if not long_df.empty else np.nan

    # 逐评委绘制分布
    for i, (judge, sub) in enumerate(long_df.groupby('评委')):
        # KDE
        sns.kdeplot(sub['分数'], fill=True, alpha=0.15, linewidth=1, label=judge)
        # 直方（不重复图例）
        sns.histplot(sub['分数'], bins=np.arange(-0.5, 5.5, 0.5),
                     stat='density', alpha=0.08, edgecolor=None, label=None)

    # 标题追加整体均值与中位数
    if np.isnan(overall_mean) or np.isnan(overall_median):
        plt.title('各评委打分分布（整体均值=NaN，中位数=NaN）')
    else:
        plt.title(f'各评委打分分布（整体均值={overall_mean:.2f}，中位数={overall_median:.2f}）')

    plt.xlabel('分数')
    plt.legend(frameon=False)
    plt.tight_layout()
    sf.save("04_judge_distributions")

    # 5) 各评委对各队打分箱线图
    plt.figure(figsize=(12,6))
    order = long_df.groupby('评委')['分数'].mean().sort_values(ascending=False).index
    sns.boxplot(data=long_df, x='评委', y='分数', order=order, palette='Set3', whis=(0, 100), showfliers=False)
    sns.stripplot(data=long_df, x='评委', y='分数', order=order, color='k', alpha=0.4, jitter=0.15, size=3)

    # 计算均值
    mean_by_judge = long_df.groupby('评委', sort=False)['分数'].mean().reindex(order)
    # 叠加均值点（菱形）
    plt.scatter(range(len(order)), mean_by_judge.values,
                marker='D', s=10, color='crimson', zorder=5, label='均值')

    # 可选：给均值加数值标注
    for i, v in enumerate(mean_by_judge.values):
        plt.text(i, v + 0.03, f'{v:.2f}', ha='center', va='bottom', fontsize=8, color='crimson')

    plt.title('各评委打分分布（箱线+散点）')
    plt.tight_layout()
    sf.save("05_box_strip_by_judge")

    # 6) 评委间一致性：皮尔逊相关热力图
    plt.figure(figsize=(8,6))
    sns.heatmap(corr, annot=True, fmt='.2f', cmap='RdYlBu', vmin=-1, vmax=1, linewidths=.5, cbar_kws={'label':'皮尔逊相关'})
    plt.title('评委间打分一致性（两两相关）')
    plt.tight_layout()
    sf.save("06_judge_corr")

    # 7) Top-N 队伍排行榜（平均分）
    TOP_N = 15
    plt.figure(figsize=(10, max(5, TOP_N*0.5)))
    topN = avg_by_team.head(TOP_N)[::-1]  # 为了好看自下而上
    sns.barplot(x=topN.values, y=topN.index, color=palette[2])
    for i, v in enumerate(topN.values):
        plt.text(v+0.02, i, f'{v:.2f}', va='center')
    plt.title(f'平均分Top{TOP_N}队伍')
    plt.xlabel('平均分')
    plt.ylabel('队伍')
    plt.tight_layout()
    sf.save("07_topN_avg")

    # 8) 各队平均分分布（直方图）
    plt.figure(figsize=(8,5))
    sns.histplot(avg_by_team, bins=np.linspace(avg_by_team.min()-0.25, avg_by_team.max()+0.25, 16), kde=True, color=palette[4])
    plt.title('各队平均分分布')
    plt.xlabel('平均分')
    plt.tight_layout()
    sf.save("08_avg_hist")

    # ========== 可选：导出汇总表 ==========
    summary = pd.DataFrame({
        '平均分': avg_by_team,
        '分差(平均)': gap_by_team
    }).sort_values('平均分', ascending=False)
    print('\n前20队汇总：')
    print(summary.head(20).round(3).to_string())
    print(f"\n图片已保存到目录：{os.path.abspath(sf.out_dir_)}")


if __name__ == "__main__":
    main()
