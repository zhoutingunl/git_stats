#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
团队评分比较分析工具
用于比较初评和终评进入的团队评分差异

作者：自动生成代码
日期：2024年
"""

import csv
import json
import statistics
from collections import defaultdict
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import pandas as pd
from scipy import stats

# 设置中文字体
font_candidates = ['Noto Sans CJK SC', 'Source Han Sans SC', 'WenQuanYi Micro Hei', 'PingFang SC', 'Microsoft YaHei', 'SimHei', 'Arial Unicode MS']
plt.rcParams['font.sans-serif'] = font_candidates
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['axes.unicode_minus'] = False

class TeamScoreAnalyzer:
    def __init__(self, initial_file='input/202510.csv', final_file='input/202510_fin.csv'):
        """
        初始化团队评分分析器

        Args:
            initial_file: 初评数据文件路径
            final_file: 终评数据文件路径
        """
        self.initial_file = initial_file
        self.final_file = final_file
        self.initial_data = {}
        self.final_data = {}
        self.comparison_data = []

    def safe_float(self, value):
        """安全转换浮点数"""
        try:
            if value == '' or value is None:
                return None
            return float(value)
        except (ValueError, TypeError):
            return None

    def load_initial_data(self):
        """加载初评数据"""
        print("正在加载初评数据...")
        try:
            team_scores = {}

            with open(self.initial_file, 'r', encoding='utf-8-sig') as f:
                reader = csv.reader(f)
                next(reader)  # 跳过表头

                for row in reader:
                    if len(row) < 5:
                        continue

                    judge1, score1_str, judge2, score2_str, team = row

                    score1 = self.safe_float(score1_str)
                    score2 = self.safe_float(score2_str)

                    # 计算平均分
                    scores = [s for s in [score1, score2] if s is not None]
                    if scores:
                        avg_score = sum(scores) / len(scores)
                        team_scores[team] = avg_score

            self.initial_data = team_scores
            print(f"初评数据加载完成，共 {len(team_scores)} 支队伍")

        except Exception as e:
            print(f"加载初评数据时出错: {e}")
            raise

    def load_final_data(self):
        """加载终评数据"""
        print("正在加载终评数据...")
        try:
            team_scores = {}

            with open(self.final_file, 'r', encoding='utf-8-sig') as f:
                # 尝试使用制表符分隔符
                reader = csv.reader(f, delimiter='\t')
                next(reader)  # 跳过表头

                for row in reader:
                    if len(row) < 2:
                        continue

                    team, score_str = row[:2]
                    team = team.strip()

                    if team == '' or team == '':
                        continue

                    score = self.safe_float(score_str)
                    if score is not None:
                        team_scores[team] = score

            self.final_data = team_scores
            print(f"终评数据加载完成，共 {len(team_scores)} 支队伍")

        except Exception as e:
            print(f"加载终评数据时出错: {e}")
            raise

    def match_teams(self):
        """匹配初评和终评的队伍"""
        print("正在匹配初评和终评队伍...")

        if not self.initial_data or not self.final_data:
            raise ValueError("请先加载初评和终评数据")

        matched_teams = []

        for team in self.final_data:
            if team in self.initial_data:
                initial_score = self.initial_data[team]
                final_score = self.final_data[team]

                score_diff = final_score - initial_score
                improvement_rate = (score_diff / initial_score * 100) if initial_score != 0 else 0

                matched_teams.append({
                    'team': team,
                    'initial_score': initial_score,
                    'final_score': final_score,
                    'score_diff': score_diff,
                    'improvement_rate': improvement_rate
                })

        # 按评分差异排序
        matched_teams.sort(key=lambda x: x['score_diff'], reverse=True)

        self.comparison_data = matched_teams
        print(f"成功匹配 {len(matched_teams)} 支进入终评的队伍")

        return matched_teams

    def analyze_score_changes(self):
        """分析评分变化"""
        if not self.comparison_data:
            self.match_teams()

        data = self.comparison_data

        print("\n=== 评分变化分析 ===")
        print(f"进入终评的队伍总数: {len(data)}")

        improved_teams = [t for t in data if t['score_diff'] > 0]
        declined_teams = [t for t in data if t['score_diff'] < 0]
        unchanged_teams = [t for t in data if t['score_diff'] == 0]

        print(f"评分提升的队伍: {len(improved_teams)} ({len(improved_teams)/len(data)*100:.1f}%)")
        print(f"评分下降的队伍: {len(declined_teams)} ({len(declined_teams)/len(data)*100:.1f}%)")
        print(f"评分不变的队伍: {len(unchanged_teams)}")

        # 计算统计信息
        initial_scores = [t['initial_score'] for t in data]
        final_scores = [t['final_score'] for t in data]
        score_diffs = [t['score_diff'] for t in data]

        print(f"\n评分统计:")
        print(f"初评平均分: {statistics.mean(initial_scores):.3f} ± {statistics.stdev(initial_scores):.3f}")
        print(f"终评平均分: {statistics.mean(final_scores):.3f} ± {statistics.stdev(final_scores):.3f}")
        print(f"平均评分差异: {statistics.mean(score_diffs):.3f} ± {statistics.stdev(score_diffs):.3f}")

        max_improvement = max(data, key=lambda x: x['score_diff'])
        max_decline = min(data, key=lambda x: x['score_diff'])

        print(f"最大提升: {max_improvement['score_diff']:.3f} (队伍: {max_improvement['team']})")
        print(f"最大下降: {max_decline['score_diff']:.3f} (队伍: {max_decline['team']})")

        return data

    def get_top_improvements(self, n=10):
        """获取评分提升最大的队伍"""
        if not self.comparison_data:
            self.match_teams()

        top_improvements = self.comparison_data[:n]
        print(f"\n=== 评分提升最大的 {n} 支队伍 ===")
        for team_data in top_improvements:
            print(f"{team_data['team']}: {team_data['initial_score']:.3f} → {team_data['final_score']:.3f} "
                  f"(+{team_data['score_diff']:.3f}, +{team_data['improvement_rate']:.1f}%)")

        return top_improvements

    def get_top_declines(self, n=10):
        """获取评分下降最大的队伍"""
        if not self.comparison_data:
            self.match_teams()

        # 按评分差异升序排序，取后n个
        sorted_data = sorted(self.comparison_data, key=lambda x: x['score_diff'])
        top_declines = sorted_data[:n]

        print(f"\n=== 评分下降最大的 {n} 支队伍 ===")
        for team_data in top_declines:
            print(f"{team_data['team']}: {team_data['initial_score']:.3f} → {team_data['final_score']:.3f} "
                  f"({team_data['score_diff']:.3f}, {team_data['improvement_rate']:.1f}%)")

        return top_declines

    def export_comparison_report(self, output_file='team_score_comparison.csv'):
        """导出比较报告"""
        if not self.comparison_data:
            self.match_teams()

        # 计算排名
        data = self.comparison_data[:]

        # 按初评分排序计算初评排名
        initial_sorted = sorted(data, key=lambda x: x['initial_score'], reverse=True)
        for i, team_data in enumerate(initial_sorted):
            team_data['initial_rank'] = i + 1

        # 按终评分排序计算终评排名
        final_sorted = sorted(data, key=lambda x: x['final_score'], reverse=True)
        for i, team_data in enumerate(final_sorted):
            team_data['final_rank'] = i + 1

        # 计算排名变化
        for team_data in data:
            team_data['rank_change'] = team_data['initial_rank'] - team_data['final_rank']

        # 导出CSV
        with open(output_file, 'w', encoding='utf-8-sig', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['队伍', '初评平均分', '初评排名', '终评分', '终评排名',
                           '评分差异', '评分提升率', '排名变化'])

            for team_data in data:
                writer.writerow([
                    team_data['team'],
                    f"{team_data['initial_score']:.3f}",
                    team_data['initial_rank'],
                    f"{team_data['final_score']:.3f}",
                    team_data['final_rank'],
                    f"{team_data['score_diff']:.3f}",
                    f"{team_data['improvement_rate']:.1f}",
                    team_data['rank_change']
                ])

        print(f"比较报告已导出至: {output_file}")
        return data

    def plot_score_comparison(self, output_dir='plots'):
        """绘制初评vs终评评分对比图表"""
        if not self.comparison_data:
            self.match_teams()

        # 创建输出目录
        Path(output_dir).mkdir(exist_ok=True)

        # 提取数据
        initial_scores = [t['initial_score'] for t in self.comparison_data]
        final_scores = [t['final_score'] for t in self.comparison_data]
        teams = [t['team'] for t in self.comparison_data]

        # 计算相关系数
        correlation = np.corrcoef(initial_scores, final_scores)[0, 1]
        p_value = stats.pearsonr(initial_scores, final_scores)[1]

        # 1. 散点图：初评vs终评
        plt.figure(figsize=(10, 8))
        plt.scatter(initial_scores, final_scores, alpha=0.7, s=60, edgecolors='black', linewidth=0.5)

        # 添加对角线（完全一致线）
        min_score = min(min(initial_scores), min(final_scores))
        max_score = max(max(initial_scores), max(final_scores))
        plt.plot([min_score, max_score], [min_score, max_score], 'r--', alpha=0.8, linewidth=2, label='完全一致线')

        # 添加趋势线
        z = np.polyfit(initial_scores, final_scores, 1)
        p = np.poly1d(z)
        plt.plot(initial_scores, p(initial_scores), 'b-', alpha=0.8, linewidth=2, label=f'趋势线 (y={z[0]:.3f}x+{z[1]:.3f})')

        plt.xlabel('初评平均分', fontsize=12)
        plt.ylabel('终评分', fontsize=12)
        plt.title(f'初评vs终评评分散点图\n相关系数: r={correlation:.3f} (p={p_value:.3f})', fontsize=14)
        plt.legend()
        plt.grid(True, alpha=0.3)

        # 添加部分队名标注（避免过于拥挤）
        for i, (x, y, team) in enumerate(zip(initial_scores, final_scores, teams)):
            if i % 5 == 0:  # 每5个标注一个
                plt.annotate(team[:8], (x, y), xytext=(5, 5), textcoords='offset points',
                           fontsize=8, alpha=0.7, rotation=15)

        plt.tight_layout()
        plt.savefig(f'{output_dir}/scatter_initial_vs_final.png', dpi=300, bbox_inches='tight')
        plt.close()

        # 2. 箱型图：初评vs终评分布对比
        plt.figure(figsize=(10, 6))

        # 准备箱型图数据
        box_data = [initial_scores, final_scores]
        box_labels = ['初评', '终评']

        bp = plt.boxplot(box_data, labels=box_labels, patch_artist=True)

        # 设置颜色
        colors = ['lightblue', 'lightcoral']
        for patch, color in zip(bp['boxes'], colors):
            patch.set_facecolor(color)

        # 添加散点图overlay
        for i, (scores, label) in enumerate(zip(box_data, box_labels)):
            x = np.random.normal(i+1, 0.04, len(scores))
            plt.scatter(x, scores, alpha=0.5, s=20)

        plt.ylabel('评分', fontsize=12)
        plt.title('初评vs终评分数分布箱型图', fontsize=14)
        plt.grid(True, alpha=0.3)

        # 添加统计信息
        initial_mean = np.mean(initial_scores)
        initial_std = np.std(initial_scores)
        final_mean = np.mean(final_scores)
        final_std = np.std(final_scores)

        plt.text(0.02, 0.98, f'初评: μ={initial_mean:.3f}, σ={initial_std:.3f}\n终评: μ={final_mean:.3f}, σ={final_std:.3f}',
                transform=plt.gca().transAxes, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

        plt.tight_layout()
        plt.savefig(f'{output_dir}/boxplot_score_comparison.png', dpi=300, bbox_inches='tight')
        plt.close()

        # 3. 评分差异分布图
        score_diffs = [t['score_diff'] for t in self.comparison_data]

        plt.figure(figsize=(12, 5))

        # 子图1：直方图
        plt.subplot(1, 2, 1)
        plt.hist(score_diffs, bins=20, alpha=0.7, edgecolor='black', color='skyblue')
        plt.axvline(x=0, color='red', linestyle='--', alpha=0.8, linewidth=2, label='无变化线')
        plt.axvline(x=np.mean(score_diffs), color='green', linestyle='-', alpha=0.8, linewidth=2, label=f'平均变化: {np.mean(score_diffs):.3f}')
        plt.xlabel('评分差异 (终评-初评)', fontsize=11)
        plt.ylabel('队伍数量', fontsize=11)
        plt.title('评分差异分布', fontsize=12)
        plt.legend()
        plt.grid(True, alpha=0.3)

        # 子图2：累积分布图
        plt.subplot(1, 2, 2)
        sorted_diffs = np.sort(score_diffs)
        cumulative_prob = np.arange(1, len(sorted_diffs) + 1) / len(sorted_diffs)
        plt.plot(sorted_diffs, cumulative_prob, linewidth=2, color='navy')
        plt.axvline(x=0, color='red', linestyle='--', alpha=0.8, linewidth=2, label='无变化线')
        plt.axhline(y=0.5, color='orange', linestyle=':', alpha=0.8, linewidth=2, label='50%分位线')
        plt.xlabel('评分差异 (终评-初评)', fontsize=11)
        plt.ylabel('累积概率', fontsize=11)
        plt.title('评分差异累积分布', fontsize=12)
        plt.legend()
        plt.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(f'{output_dir}/score_difference_distribution.png', dpi=300, bbox_inches='tight')
        plt.close()

        # 4. 相关性热力图（如果有CSV文件）
        try:
            # 读取CSV文件创建DataFrame
            df_data = []
            for team_data in self.comparison_data:
                df_data.append({
                    '队伍': team_data['team'],
                    '初评': team_data['initial_score'],
                    '终评': team_data['final_score'],
                    '差异': team_data['score_diff'],
                    '提升率': team_data['improvement_rate']
                })

            df = pd.DataFrame(df_data)
            correlation_matrix = df[['初评', '终评', '差异', '提升率']].corr()

            plt.figure(figsize=(8, 6))
            sns.heatmap(correlation_matrix, annot=True, fmt='.3f', cmap='RdBu_r',
                       center=0, square=True, cbar_kws={'label': '相关系数'})
            plt.title('评分指标相关性热力图', fontsize=14)
            plt.tight_layout()
            plt.savefig(f'{output_dir}/correlation_heatmap.png', dpi=300, bbox_inches='tight')
            plt.close()

        except Exception as e:
            print(f"生成相关性热力图时出错: {e}")

        print(f"所有图表已保存到 {output_dir}/ 目录")
        print(f"- 散点图: scatter_initial_vs_final.png")
        print(f"- 箱型图: boxplot_score_comparison.png")
        print(f"- 差异分布图: score_difference_distribution.png")
        print(f"- 相关性热力图: correlation_heatmap.png")
        print(f"相关系数: {correlation:.3f} (p值: {p_value:.3f})")

        return correlation, p_value

    def plot_from_csv(self, csv_file='team_score_comparison.csv', output_dir='plots'):
        """直接从CSV文件读取数据并生成图表"""
        try:
            # 读取CSV文件
            df = pd.read_csv(csv_file, encoding='utf-8-sig')

            # 创建输出目录
            Path(output_dir).mkdir(exist_ok=True)

            # 提取数据
            initial_scores = df['初评平均分'].values
            final_scores = df['终评分'].values
            teams = df['队伍'].values
            score_diffs = df['评分差异'].values

            # 计算相关系数
            correlation = np.corrcoef(initial_scores, final_scores)[0, 1]
            p_value = stats.pearsonr(initial_scores, final_scores)[1]

            # 1. 散点图：初评vs终评
            plt.figure(figsize=(10, 8))
            plt.scatter(initial_scores, final_scores, alpha=0.7, s=60, edgecolors='black', linewidth=0.5)

            # 添加对角线（完全一致线）
            min_score = min(min(initial_scores), min(final_scores))
            max_score = max(max(initial_scores), max(final_scores))
            plt.plot([min_score, max_score], [min_score, max_score], 'r--', alpha=0.8, linewidth=2, label='完全一致线')

            # 添加趋势线
            z = np.polyfit(initial_scores, final_scores, 1)
            p = np.poly1d(z)
            plt.plot(initial_scores, p(initial_scores), 'b-', alpha=0.8, linewidth=2, label=f'趋势线 (y={z[0]:.3f}x+{z[1]:.3f})')

            plt.xlabel('初评平均分', fontsize=12)
            plt.ylabel('终评分', fontsize=12)
            plt.title(f'初评vs终评评分散点图\n相关系数: r={correlation:.3f} (p={p_value:.3f})', fontsize=14)
            plt.legend()
            plt.grid(True, alpha=0.3)

            # 添加部分队名标注（避免过于拥挤）
            for i, (x, y, team) in enumerate(zip(initial_scores, final_scores, teams)):
                if i % 8 == 0:  # 每8个标注一个
                    plt.annotate(team[:8], (x, y), xytext=(5, 5), textcoords='offset points',
                               fontsize=8, alpha=0.7, rotation=15)

            plt.tight_layout()
            plt.savefig(f'{output_dir}/scatter_initial_vs_final.png', dpi=300, bbox_inches='tight')
            plt.close()

            # 2. 箱型图：初评vs终评分布对比
            plt.figure(figsize=(10, 6))

            # 准备箱型图数据
            box_data = [initial_scores, final_scores]
            box_labels = ['初评', '终评']

            bp = plt.boxplot(box_data, labels=box_labels, patch_artist=True)

            # 设置颜色
            colors = ['lightblue', 'lightcoral']
            for patch, color in zip(bp['boxes'], colors):
                patch.set_facecolor(color)

            # 添加散点图overlay
            for i, (scores, label) in enumerate(zip(box_data, box_labels)):
                x = np.random.normal(i+1, 0.04, len(scores))
                plt.scatter(x, scores, alpha=0.5, s=20)

            plt.ylabel('评分', fontsize=12)
            plt.title('初评vs终评分数分布箱型图', fontsize=14)
            plt.grid(True, alpha=0.3)

            # 添加统计信息
            initial_mean = np.mean(initial_scores)
            initial_std = np.std(initial_scores)
            final_mean = np.mean(final_scores)
            final_std = np.std(final_scores)

            plt.text(0.02, 0.98, f'初评: μ={initial_mean:.3f}, σ={initial_std:.3f}\n终评: μ={final_mean:.3f}, σ={final_std:.3f}',
                    transform=plt.gca().transAxes, verticalalignment='top',
                    bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

            plt.tight_layout()
            plt.savefig(f'{output_dir}/boxplot_score_comparison.png', dpi=300, bbox_inches='tight')
            plt.close()

            # 3. 评分差异分布图
            plt.figure(figsize=(12, 5))

            # 子图1：直方图
            plt.subplot(1, 2, 1)
            plt.hist(score_diffs, bins=20, alpha=0.7, edgecolor='black', color='skyblue')
            plt.axvline(x=0, color='red', linestyle='--', alpha=0.8, linewidth=2, label='无变化线')
            plt.axvline(x=np.mean(score_diffs), color='green', linestyle='-', alpha=0.8, linewidth=2, label=f'平均变化: {np.mean(score_diffs):.3f}')
            plt.xlabel('评分差异 (终评-初评)', fontsize=11)
            plt.ylabel('队伍数量', fontsize=11)
            plt.title('评分差异分布', fontsize=12)
            plt.legend()
            plt.grid(True, alpha=0.3)

            # 子图2：累积分布图
            plt.subplot(1, 2, 2)
            sorted_diffs = np.sort(score_diffs)
            cumulative_prob = np.arange(1, len(sorted_diffs) + 1) / len(sorted_diffs)
            plt.plot(sorted_diffs, cumulative_prob, linewidth=2, color='navy')
            plt.axvline(x=0, color='red', linestyle='--', alpha=0.8, linewidth=2, label='无变化线')
            plt.axhline(y=0.5, color='orange', linestyle=':', alpha=0.8, linewidth=2, label='50%分位线')
            plt.xlabel('评分差异 (终评-初评)', fontsize=11)
            plt.ylabel('累积概率', fontsize=11)
            plt.title('评分差异累积分布', fontsize=12)
            plt.legend()
            plt.grid(True, alpha=0.3)

            plt.tight_layout()
            plt.savefig(f'{output_dir}/score_difference_distribution.png', dpi=300, bbox_inches='tight')
            plt.close()

            # 4. 相关性热力图
            numeric_columns = ['初评平均分', '终评分', '评分差异', '评分提升率']
            correlation_matrix = df[numeric_columns].corr()

            plt.figure(figsize=(8, 6))
            sns.heatmap(correlation_matrix, annot=True, fmt='.3f', cmap='RdBu_r',
                       center=0, square=True, cbar_kws={'label': '相关系数'})
            plt.title('评分指标相关性热力图', fontsize=14)
            plt.tight_layout()
            plt.savefig(f'{output_dir}/correlation_heatmap.png', dpi=300, bbox_inches='tight')
            plt.close()

            print(f"从CSV文件 {csv_file} 生成的图表已保存到 {output_dir}/ 目录")
            print(f"- 散点图: scatter_initial_vs_final.png")
            print(f"- 箱型图: boxplot_score_comparison.png")
            print(f"- 差异分布图: score_difference_distribution.png")
            print(f"- 相关性热力图: correlation_heatmap.png")
            print(f"相关系数: {correlation:.3f} (p值: {p_value:.3f})")

            return correlation, p_value

        except Exception as e:
            print(f"从CSV文件生成图表时出错: {e}")
            return None, None

    def run_full_analysis(self):
        """运行完整分析"""
        print("开始完整的团队评分分析...")

        # 加载数据
        self.load_initial_data()
        self.load_final_data()

        # 匹配和分析
        self.match_teams()
        self.analyze_score_changes()

        # 获取排名
        self.get_top_improvements()
        self.get_top_declines()

        # 导出报告
        print("\n正在导出详细报告...")
        report = self.export_comparison_report()

        # 生成图表
        print("\n正在生成图表...")
        try:
            correlation, p_value = self.plot_score_comparison()
            print(f"\n=== 图表分析结果 ===")
            print(f"初评与终评相关系数: {correlation:.3f}")
            print(f"相关性显著性检验p值: {p_value:.3f}")
            if p_value < 0.05:
                print("相关性显著 (p < 0.05)")
            else:
                print("相关性不显著 (p ≥ 0.05)")
        except Exception as e:
            print(f"生成图表时出错: {e}")

        print("\n=== 分析完成 ===")
        return report


def main():
    """主函数"""
    print("团队评分比较分析工具")
    print("=" * 50)

    # 创建分析器
    analyzer = TeamScoreAnalyzer()

    try:
        # 运行完整分析
        report = analyzer.run_full_analysis()

        print("\n分析结果已生成:")
        print("1. team_score_comparison.csv - 详细报告")
        print("2. plots/ - 图表目录:")
        print("   - scatter_initial_vs_final.png - 初评vs终评散点图")
        print("   - boxplot_score_comparison.png - 初评vs终评箱型图")
        print("   - score_difference_distribution.png - 评分差异分布图")
        print("   - correlation_heatmap.png - 相关性热力图")

    except FileNotFoundError as e:
        print(f"文件未找到错误: {e}")
        print("请确保以下文件存在:")
        print("- input/202510.csv")
        print("- input/202510_fin.csv")
    except Exception as e:
        print(f"分析过程中出现错误: {e}")
        import traceback
        traceback.print_exc()


def plot_from_existing_csv():
    """直接从现有CSV文件生成图表（无需重新分析数据）"""
    print("团队评分图表生成工具")
    print("=" * 50)

    analyzer = TeamScoreAnalyzer()

    try:
        # 直接从CSV生成图表
        correlation, p_value = analyzer.plot_from_csv('team_score_comparison.csv', 'plots')

        if correlation is not None:
            print(f"\n=== 图表分析结果 ===")
            print(f"初评与终评相关系数: {correlation:.3f}")
            print(f"相关性显著性检验p值: {p_value:.3f}")
            if p_value < 0.05:
                print("相关性显著 (p < 0.05)")
            else:
                print("相关性不显著 (p ≥ 0.05)")

    except FileNotFoundError:
        print("未找到team_score_comparison.csv文件，正在运行完整分析...")
        main()
    except Exception as e:
        print(f"生成图表时出错: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    # 如果存在CSV文件，直接生成图表；否则运行完整分析
    import os
    if os.path.exists('team_score_comparison.csv'):
        plot_from_existing_csv()
    else:
        main()