#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import subprocess
import sys
from collections import defaultdict
from batch_stats import classify_ext


def run_git(cmd_args):
    try:
        res = subprocess.run(
            ["git"] + cmd_args,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            check=True,
        )
        return res.stdout
    except subprocess.CalledProcessError as e:
        print("Git command failed:", " ".join(cmd_args), file=sys.stderr)
        print(e.stderr.strip(), file=sys.stderr)
        sys.exit(2)

def main():
    if len(sys.argv) < 2:
        print("Usage: python3 list_others_by_author.py <author>")
        print('Example: python3 list_others_by_author.py "LIYIXIN"')
        sys.exit(1)

    author = sys.argv[1]

    # 取出作者的提交及其 numstat
    # 输出块结构：@@@<commit>\t<date>\n<numstat lines...>
    log_out = run_git([
        "log",
        "--author", author,
        "--date=iso-strict",
        "--numstat",
        '--pretty=@@@%H\t%ad'
    ])

    current_commit = None
    rows = []  # (commit, file, added_raw, deleted_raw, added_int)
    per_file_added = defaultdict(int)
    per_file_commits = defaultdict(set)

    for line in log_out.splitlines():
        if not line.strip():
            continue
        if line.startswith("@@@"):
            # 新的提交块
            # 格式 "@@@<hash>\t<date>"
            parts = line[3:].split("\t", 1)
            current_commit = parts[0].strip()
            continue

        # numstat 行：added<TAB>deleted<TAB>path
        parts = line.split("\t")
        if len(parts) != 3:
            continue

        added_raw, deleted_raw, path = parts[0], parts[1], parts[2]

        if classify_ext(path) != "Others":
            continue

        # 非数字（如 '-'）按 0 计
        added_int = int(added_raw) if added_raw.isdigit() else 0

        rows.append((current_commit, path, added_raw, deleted_raw, added_int))
        per_file_added[path] += added_int
        per_file_commits[path].add(current_commit)

    # 概览：按新增行数排序 Top 20
    print("Top 'Others' files by added lines:")
    if not per_file_added:
        print("(no files)")
    else:
        top = sorted(per_file_added.items(), key=lambda x: x[1], reverse=True)[:20]
        for total, path in sorted([(v, k) for k, v in per_file_added.items()], reverse=True)[:20]:
            print(f"{total}\t{path}")

    # 输出明细为 TSV
    out_tsv = f"{author}_others.tsv"
    with open(out_tsv, "w", encoding="utf-8") as f:
        f.write("commit\tfile\tadded\tdeleted\n")
        for commit, path, added_raw, deleted_raw, _added_int in rows:
            f.write(f"{commit}\t{path}\t{added_raw}\t{deleted_raw}\n")

    # 补充一个文件出现次数统计（涉及多少个提交）
    out_sum = f"{author}_others_summary.tsv"
    with open(out_sum, "w", encoding="utf-8") as f:
        f.write("file\tadded_sum\tcommits_count\n")
        for path in sorted(per_file_added.keys(), key=lambda p: per_file_added[p], reverse=True):
            f.write(f"{path}\t{per_file_added[path]}\t{len(per_file_commits[path])}\n")

    print(f"\nWritten: {out_tsv}")
    print(f"Written: {out_sum}")

    # 使用提示
    print("\nNext steps:")
    print("1) Inspect a specific file's diffs:")
    print('   git log -p --author "{}" -- -- <path/to/file>'.format(author))
    print("2) Show a particular commit and file diff:")
    print("   git show <commit> -- <path/to/file>")
    print("3) If diff shows 'Binary files differ', try exporting the blob:")
    print('   git show <commit>:"<path/to/file>" > out.bin')

if __name__ == "__main__":
    main()

