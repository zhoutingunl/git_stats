#!/usr/bin/env python3
"""
遍历指定目录下的所有子目录(每个都是git仓库),
对每个子目录运行main.py的统计逻辑,并将结果汇总到一个MD文件中。
"""
import os
import sys
import subprocess
from pathlib import Path
from datetime import datetime
from collections import defaultdict

# 复用 main.py 中的函数和常量
LANG_KEYS = ["GO", "JAVA", "VUE", "KOTLIN", "JS", "JSON", "MD", "TEXT", "LOG", "Others"]

def run_git(args, cwd="."):
    """执行git命令"""
    try:
        result = subprocess.run(
            ["git"] + args,
            cwd=cwd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            check=True,
        )
        return result.stdout
    except subprocess.CalledProcessError as e:
        # 对于批量处理,不要exit,而是返回None
        print(f"Git command failed in {cwd}: git {' '.join(args)}", file=sys.stderr)
        print(e.stderr.strip(), file=sys.stderr)
        return None

def is_git_repo(path):
    """检查目录是否是git仓库"""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--git-dir"],
            cwd=path,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            check=True,
        )
        return result.returncode == 0
    except:
        return False

def classify_ext(filename: str) -> str:
    """分类文件扩展名"""
    if " -> " in filename:
        filename = filename.split(" -> ", 1)[1]
    fn = filename.lower()
    if fn.startswith('"') and fn.endswith('"') and len(fn) >= 2:
        fn = fn[1:-1]

    if fn.endswith(".go"):
        return "GO"
    if fn.endswith(".java"):
        return "JAVA"
    if fn.endswith(".vue"):
        return "VUE"
    if fn.endswith(".kt"):
        return "KOTLIN"
    if fn.endswith(".js") or fn.endswith(".mjs") or fn.endswith(".cjs"):
        return "JS"
    if fn.endswith(".json"):
        return "JSON"
    if fn.endswith(".log"):
        return "LOG"
    if fn.endswith(".txt"):
        return "TEXT"
    if fn.endswith(".md"):
        return "MD"
    return "Others"

def collect_commits(repo_path):
    """收集提交信息"""
    fmt = "%H%x09%an%x09%ad"
    log = run_git([
        "log",
        "--date=iso-strict",
        f"--pretty=format:{fmt}"
    ], cwd=repo_path)

    if log is None:
        return []

    commits = []
    for line in log.splitlines():
        if not line.strip():
            continue
        parts = line.split("\t")
        if len(parts) != 3:
            continue
        commit, author, date_iso = parts
        commits.append((commit, author, date_iso))
    return commits

def collect_added_by_commit(repo_path):
    """收集每个提交的新增行数统计"""
    log = run_git([
        "log",
        "--date=iso-strict",
        "--pretty=format:@@@%H",
        "--numstat",
    ], cwd=repo_path)

    if log is None:
        return {}, {}

    per_commit_added_total = {}
    per_commit_added_langs = {}
    current = None
    added_total = 0
    lang_added = defaultdict(int)

    def flush():
        if current is None:
            return
        per_commit_added_total[current] = added_total
        per_commit_added_langs[current] = dict(lang_added)

    for line in log.splitlines():
        if line.startswith("@@@"):
            if current is not None:
                flush()
            current = line[3:].strip()
            added_total = 0
            lang_added = defaultdict(int)
        elif line.strip():
            cols = line.split("\t")
            if len(cols) >= 3:
                a, _r, fname = cols[0], cols[1], cols[2]
                a_int = int(a) if a.isdigit() else 0
                added_total += a_int
                lang = classify_ext(fname)
                lang_added[lang] += a_int

    if current is not None:
        flush()

    return per_commit_added_total, per_commit_added_langs

def generate_stats_table(repo_path):
    """为指定的git仓库生成统计表格"""
    commits = collect_commits(repo_path)
    if not commits:
        return "无提交记录\n"

    per_commit_added_total, per_commit_added_langs = collect_added_by_commit(repo_path)

    # 按作者聚合
    stats = {}
    commits_by_author = defaultdict(list)

    for commit, author, date_iso in commits:
        commits_by_author[author].append((commit, date_iso))

    for author, items in commits_by_author.items():
        dates = []
        total_added = 0
        lang_sums = {k: 0 for k in LANG_KEYS}

        for commit, date_iso in items:
            try:
                dt = datetime.fromisoformat(date_iso.replace("Z", "+00:00"))
            except Exception:
                dt = date_iso
            dates.append(dt)

            total_added += per_commit_added_total.get(commit, 0)

            langs = per_commit_added_langs.get(commit, {})
            for k in LANG_KEYS:
                lang_sums[k] += langs.get(k, 0)

        if dates and isinstance(dates[0], datetime):
            first_dt = min(dates)
            last_dt = max(dates)
            first = first_dt.date().isoformat()
            last = last_dt.date().isoformat()
        else:
            dstrs = [d if isinstance(d, str) else d.isoformat() for d in dates]
            first = min(dstrs)
            last = max(dstrs)

        stats[author] = {
            "first": first,
            "last": last,
            "commits": len(items),
            "lines_added": total_added,
            "langs": lang_sums,
        }

    # 生成表格
    header = ["作者名", "首次提交日期", "最后一次提交日期", "提交次数", "新增行数"] + LANG_KEYS
    rows = []
    rows.append("|" + "|".join(header) + "|")
    rows.append("|" + "|".join(["---"] * len(header)) + "|")

    # 按新增行数降序排序
    for author in sorted(stats.keys(), key=lambda a: stats[a]["lines_added"], reverse=True):
        s = stats[author]
        row = [
            author,
            s["first"],
            s["last"],
            str(s["commits"]),
            str(s["lines_added"]),
        ] + [str(s["langs"].get(k, 0)) for k in LANG_KEYS]
        rows.append("|" + "|".join(row) + "|")

    return "\n".join(rows) + "\n"

def main():
    if len(sys.argv) < 2:
        print("用法: python batch_stats.py <父目录路径> [输出MD文件路径]")
        print("示例: python batch_stats.py /path/to/repos output.md")
        sys.exit(1)

    parent_dir = sys.argv[1]
    output_file = sys.argv[2] if len(sys.argv) > 2 else "git_stats_summary.md"

    if not os.path.isdir(parent_dir):
        print(f"错误: {parent_dir} 不是一个有效的目录", file=sys.stderr)
        sys.exit(1)

    # 遍历所有子目录
    subdirs = []
    for item in sorted(os.listdir(parent_dir)):
        item_path = os.path.join(parent_dir, item)
        if os.path.isdir(item_path) and is_git_repo(item_path):
            subdirs.append((item, item_path))

    if not subdirs:
        print(f"错误: 在 {parent_dir} 下没有找到任何git仓库", file=sys.stderr)
        sys.exit(1)

    print(f"找到 {len(subdirs)} 个git仓库:")
    for name, _ in subdirs:
        print(f"  - {name}")

    # 生成汇总MD文件
    with open(output_file, "w", encoding="utf-8") as f:
        f.write("# Git 仓库统计汇总\n\n")
        f.write(f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        f.write(f"扫描目录: `{parent_dir}`\n\n")
        f.write(f"仓库数量: {len(subdirs)}\n\n")
        f.write("---\n\n")

        for name, path in subdirs:
            print(f"\n正在处理: {name}")
            f.write(f"## {name}\n\n")
            f.write(f"路径: `{path}`\n\n")

            table = generate_stats_table(path)
            f.write(table)
            f.write("\n")

    print(f"\n✅ 统计完成! 结果已保存到: {output_file}")

if __name__ == "__main__":
    main()
