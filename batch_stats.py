#!/usr/bin/env python3
"""
遍历指定目录下的所有子目录(每个都是git仓库),
对每个子目录运行main.py的统计逻辑,并将结果汇总到一个MD文件中。
"""
import os
import sys
import subprocess
import json
from pathlib import Path
from datetime import datetime
from collections import defaultdict

# 复用 main.py 中的函数和常量
LANG_KEYS = ["Python", "GO", "JAVA", "VUE", "Kotlin", "JS", "TypeScript", "JSON", "MD", "Dockerfile", "TEXT", "LOG", "Others"]

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

    if fn.endswith(".py"):
        return "Python"
    if fn.endswith(".ts"):
        return "TypeScript"
    if fn.endswith(".go"):
        return "GO"
    if fn.endswith(".java"):
        return "JAVA"
    if fn.endswith(".vue"):
        return "VUE"
    if fn.endswith(".kt"):
        return "Kotlin"
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
    if fn.endswith("Dockerfile"):
        return "Dockerfile"

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

def run_tokei(repo_path):
    """运行tokei并返回统计数据"""
    try:
        # 尝试使用本地tokei或系统tokei
        tokei_cmd = "/workspace/tokei" if os.path.exists("/workspace/tokei") else "tokei"
        result = subprocess.run(
            [tokei_cmd, ".", "--output", "json"],
            cwd=repo_path,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            check=True,
        )
        return json.loads(result.stdout)
    except subprocess.CalledProcessError as e:
        print(f"Tokei command failed in {repo_path}: {e.stderr}", file=sys.stderr)
        return None
    except FileNotFoundError:
        print(f"Tokei not found. Please install tokei.", file=sys.stderr)
        return None
    except json.JSONDecodeError as e:
        print(f"Failed to parse tokei output: {e}", file=sys.stderr)
        return None

def generate_tokei_table(repo_path):
    """为指定的git仓库生成tokei统计表格"""
    tokei_data = run_tokei(repo_path)
    if not tokei_data:
        return "无法获取代码统计数据\n"

    # 收集所有语言的统计信息(排除Total)
    lang_stats = []
    for lang_name, lang_data in tokei_data.items():
        if lang_name == "Total":
            continue
        lang_stats.append({
            "language": lang_name,
            "files": len(lang_data.get("reports", [])),
            "lines": lang_data.get("code", 0) + lang_data.get("comments", 0) + lang_data.get("blanks", 0),
            "code": lang_data.get("code", 0),
            "comments": lang_data.get("comments", 0),
            "blanks": lang_data.get("blanks", 0),
        })

    if not lang_stats:
        return "无代码文件\n"

    # 按代码行数降序排序
    lang_stats.sort(key=lambda x: x["code"], reverse=True)

    # 生成表格
    header = ["语言", "文件数", "总行数", "代码行数", "注释行数", "空白行数"]
    rows = []
    rows.append("|" + "|".join(header) + "|")
    rows.append("|" + "|".join(["---"] * len(header)) + "|")

    for stat in lang_stats:
        row = [
            stat["language"],
            str(stat["files"]),
            str(stat["lines"]),
            str(stat["code"]),
            str(stat["comments"]),
            str(stat["blanks"]),
        ]
        rows.append("|" + "|".join(row) + "|")

    # 添加总计行
    if "Total" in tokei_data:
        total = tokei_data["Total"]
        total_lines = total.get("code", 0) + total.get("comments", 0) + total.get("blanks", 0)
        # 计算总文件数
        total_files = sum(stat["files"] for stat in lang_stats)
        total_row = [
            "**总计**",
            str(total_files),
            str(total_lines),
            str(total.get("code", 0)),
            str(total.get("comments", 0)),
            str(total.get("blanks", 0)),
        ]
        rows.append("|" + "|".join(total_row) + "|")

    return "\n".join(rows) + "\n"

def generate_daily_commits_table(repo_path):
    """生成每天提交的代码量表格"""
    cmd = """git log --numstat --date=short --pretty="%ad" | awk 'NF==1{day=$1;next} NF==3&&$1~/^[0-9]+$/{net[day]+=$1-$2} END{PROCINFO["sorted_in"]="@ind_str_asc"; print "date,net"; for(d in net) printf "%s,%d\\n", d, net[d]}'"""

    try:
        result = subprocess.run(
            cmd,
            cwd=repo_path,
            shell=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            check=True,
        )
        output = result.stdout.strip()
        if not output or output == "date,net":
            return "无每日提交数据\n"

        lines = output.split('\n')
        if len(lines) <= 1:
            return "无每日提交数据\n"

        # 生成表格
        rows = []
        rows.append("|日期|净增代码行数|")
        rows.append("|---|---|")

        for line in lines[1:]:  # 跳过表头
            if line.strip():
                parts = line.split(',')
                if len(parts) == 2:
                    rows.append(f"|{parts[0]}|{parts[1]}|")

        return "\n".join(rows) + "\n"
    except Exception as e:
        print(f"Failed to generate daily commits table in {repo_path}: {e}", file=sys.stderr)
        return "无法生成每日提交数据\n"

def generate_author_changes_table(repo_path):
    """生成每个作者增减的代码行数表格"""
    cmd = """git log --numstat --format="%aN" | awk 'NF==1{name=$0} NF==3{a[name]+=$1; d[name]+=$2} END{for(n in a) printf "%-25s +%-8d -%-8d\\n", n, a[n], d[n]}'"""

    try:
        result = subprocess.run(
            cmd,
            cwd=repo_path,
            shell=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            check=True,
        )
        output = result.stdout.strip()
        if not output:
            return "无作者提交数据\n"

        lines = output.split('\n')
        if not lines:
            return "无作者提交数据\n"

        # 生成表格
        rows = []
        rows.append("|作者名|新增行数|删除行数|")
        rows.append("|---|---|---|")

        for line in lines:
            if line.strip():
                # 解析格式: "作者名 +数字 -数字"
                parts = line.split()
                if len(parts) >= 3:
                    # 作者名可能包含空格，取除了最后两个元素外的所有部分
                    author = ' '.join(parts[:-2]).strip()
                    added = parts[-2].replace('+', '')
                    deleted = parts[-1].replace('-', '')
                    rows.append(f"|{author}|{added}|{deleted}|")

        return "\n".join(rows) + "\n"
    except Exception as e:
        print(f"Failed to generate author changes table in {repo_path}: {e}", file=sys.stderr)
        return "无法生成作者变更数据\n"

def generate_others_details_for_author(author, repo_path):
    """为指定作者生成Others分类的详细信息(MD格式)"""
    try:
        # 调用 list_others_by_author.py
        script_path = Path(__file__).parent / "list_others_by_author.py"
        result = subprocess.run(
            ["python3", str(script_path), author],
            cwd=repo_path,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            check=True,
        )

        # 读取生成的summary文件
        summary_file = Path(repo_path) / f"{author}_others_summary.tsv"
        if not summary_file.exists():
            return f"无法找到 {author} 的Others详细信息文件\n"

        with open(summary_file, 'r', encoding='utf-8') as f:
            lines = f.readlines()

        if len(lines) <= 1:
            return "无Others文件详情\n"

        # 转换TSV为MD表格
        md_lines = []
        md_lines.append(f"#### 作者 '{author}' 的 Others 分类详细文件列表\n")
        md_lines.append("|文件路径|新增行数总计|涉及提交数|")
        md_lines.append("|---|---|---|")

        # 跳过表头，处理数据行
        for line in lines[1:]:
            if line.strip():
                parts = line.strip().split('\t')
                if len(parts) == 3:
                    file_path, added_sum, commits_count = parts
                    md_lines.append(f"|{file_path}|{added_sum}|{commits_count}|")

        # 清理生成的TSV文件
        try:
            summary_file.unlink()
            detail_file = Path(repo_path) / f"{author}_others.tsv"
            if detail_file.exists():
                detail_file.unlink()
        except Exception as e:
            print(f"Warning: Failed to clean up TSV files: {e}", file=sys.stderr)

        return "\n".join(md_lines) + "\n"

    except subprocess.CalledProcessError as e:
        print(f"Failed to run list_others_by_author.py for {author} in {repo_path}: {e.stderr}", file=sys.stderr)
        return f"无法生成 {author} 的Others详细信息\n"
    except Exception as e:
        print(f"Error generating others details for {author}: {e}", file=sys.stderr)
        return f"处理 {author} 的Others信息时出错\n"

def generate_stats_table(repo_path):
    """为指定的git仓库生成统计表格，隐藏全为零的列"""
    commits = collect_commits(repo_path)
    if not commits:
        return "无提交记录\n", []

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

    # 检查哪些语言列全为零
    lang_has_data = {k: False for k in LANG_KEYS}
    for author_stats in stats.values():
        for k in LANG_KEYS:
            if author_stats["langs"].get(k, 0) > 0:
                lang_has_data[k] = True

    # 只保留有数据的语言列
    active_langs = [k for k in LANG_KEYS if lang_has_data[k]]

    # 生成表格
    header = ["作者名", "首次提交日期", "最后一次提交日期", "提交次数", "新增行数"] + active_langs
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
        ] + [str(s["langs"].get(k, 0)) for k in active_langs]
        rows.append("|" + "|".join(row) + "|")

    # 查找需要特殊处理的作者（Others类别超过2000行）
    authors_with_large_others = []
    for author, s in stats.items():
        others_lines = s["langs"].get("Others", 0)
        if others_lines > 2000:
            authors_with_large_others.append(author)
            print(f"  发现作者 '{author}' 的 'Others' 分类有 {others_lines} 行 (>2000), 将生成详细报告")

    return "\n".join(rows) + "\n", authors_with_large_others

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

            # Git统计表格
            f.write("### Git 统计\n\n")
            table, authors_with_large_others = generate_stats_table(path)
            f.write(table)
            f.write("\n")

            # 每天提交的代码量表格
            f.write("### 每天提交的代码量\n\n")
            daily_table = generate_daily_commits_table(path)
            f.write(daily_table)
            f.write("\n")

            # 每个作者增减的代码行数表格
            f.write("### 每个作者增减的代码行数\n\n")
            author_changes_table = generate_author_changes_table(path)
            f.write(author_changes_table)
            f.write("\n")

            # Tokei代码统计表格
            f.write("### 代码统计 (Tokei)\n\n")
            tokei_table = generate_tokei_table(path)
            f.write(tokei_table)
            f.write("\n")

            # 为Others分类超过2000行的作者生成详细报告
            if authors_with_large_others:
                for author in authors_with_large_others:
                    print(f"  正在为作者 '{author}' 生成 Others 详细报告...")
                    others_details = generate_others_details_for_author(author, path)
                    f.write(others_details)
                    f.write("\n")

    print(f"\n✅ 统计完成! 结果已保存到: {output_file}")

if __name__ == "__main__":
    main()
