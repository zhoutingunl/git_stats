#!/usr/bin/env python3
import subprocess
import sys
from datetime import datetime
from collections import defaultdict

LANG_KEYS = ["GO", "JAVA", "VUE", "KOTLIN", "JS", "JSON", "MD", "TEXT", "LOG", "Others"]

def run_git(args, cwd="."):
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
        print("Git command failed:", "git " + " ".join(args), file=sys.stderr)
        print(e.stderr.strip(), file=sys.stderr)
        sys.exit(1)

def detect_repo_root():
    out = run_git(["rev-parse", "--show-toplevel"]).strip()
    return out

def collect_commits():
    # %H commit hash, %an author name, %ad author date ISO
    fmt = "%H%x09%an%x09%ad"
    log = run_git([
        "log",
        "--date=iso-strict",
        f"--pretty=format:{fmt}"
        # , "--no-merges"  # 可选：忽略合并提交
    ])
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

def classify_ext(filename: str) -> str:
    # 处理重命名 "old -> new"，取新文件名
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

def collect_added_by_commit():
    """
    返回:
      per_commit_added_total: {commit: added_total}
      per_commit_added_langs: {commit: {lang_key: added}}
    仅统计新增行数（added），忽略删除。
    """
    log = run_git([
        "log",
        "--date=iso-strict",
        "--pretty=format:@@@%H",
        "--numstat",
        # , "--no-merges"  # 可选
    ])
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
            # 存储上一提交
            if current is not None:
                flush()
            current = line[3:].strip()
            added_total = 0
            lang_added = defaultdict(int)
        elif line.strip():
            cols = line.split("\t")
            if len(cols) >= 3:
                a, _r, fname = cols[0], cols[1], cols[2]
                # 二进制等情况 a 可能为 '-'，视为 0
                a_int = int(a) if a.isdigit() else 0
                added_total += a_int
                lang = classify_ext(fname)
                lang_added[lang] += a_int

    if current is not None:
        flush()

    return per_commit_added_total, per_commit_added_langs

def main():
    # Ensure we are in a git repo
    try:
        _ = detect_repo_root()
    except SystemExit:
        return
    except Exception as e:
        print("Not a git repository?", e, file=sys.stderr)
        sys.exit(1)

    commits = collect_commits()
    if not commits:
        print("No commits found.")
        return

    per_commit_added_total, per_commit_added_langs = collect_added_by_commit()

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

    # 输出表头：将“代码行数”改为“新增行数”
    header = ["作者名", "首次提交日期", "最后一次提交日期", "提交次数", "新增行数"] + LANG_KEYS
    rows = []
    rows.append("|" + "|".join(header) + "|")
    rows.append("|" + "|".join(["---"] * len(header)) + "|")

    # 排序：按新增行数降序
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

    print("\n".join(rows))


if __name__ == "__main__":
    main()
