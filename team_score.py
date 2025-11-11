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
        print("1. team_score_analysis.png - 分析图表")
        print("2. team_score_comparison.csv - 详细报告")

    except FileNotFoundError as e:
        print(f"文件未找到错误: {e}")
        print("请确保以下文件存在:")
        print("- input/202510.csv")
        print("- input/202510_fin.csv")
    except Exception as e:
        print(f"分析过程中出现错误: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()