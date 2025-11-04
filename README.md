# Git Stats - Git仓库统计工具

这个项目提供了两个工具来统计Git仓库的提交信息和代码行数。

## 工具说明

### 1. main.py - 单仓库统计

统计单个Git仓库的作者提交信息,包括:
- 首次和最后一次提交日期
- 提交次数
- 新增代码行数
- 按语言分类的代码行数(GO, JAVA, VUE, KOTLIN, JS, JSON, MD, TEXT, LOG等)

#### 使用方法

```bash
cd /path/to/your/git/repo
python3 main.py
```

输出会直接打印到终端,是一个Markdown格式的表格。

### 2. batch_stats.py - 批量仓库统计

遍历指定目录下的所有子目录(每个子目录都应该是一个Git仓库),对每个仓库运行统计,并将所有结果汇总到一个Markdown文件中。

#### 使用方法

```bash
python3 batch_stats.py <父目录路径> [输出MD文件路径]
```

**参数说明:**
- `<父目录路径>`: 必需参数,包含多个Git仓库的父目录
- `[输出MD文件路径]`: 可选参数,输出的Markdown文件路径(默认为`git_stats_summary.md`)

**示例:**

```bash
# 统计 /home/user/projects 目录下的所有Git仓库
python3 batch_stats.py /home/user/projects

# 指定输出文件
python3 batch_stats.py /home/user/projects summary.md
```

#### 输出格式

生成的Markdown文件包含:
- 生成时间
- 扫描的目录路径
- 找到的仓库数量
- 每个仓库的详细统计表格(包含仓库名称和路径)

## 输出示例

```markdown
# Git 仓库统计汇总

生成时间: 2025-11-04 12:00:00

扫描目录: `/home/user/projects`

仓库数量: 3

---

## project1

路径: `/home/user/projects/project1`

|作者名|首次提交日期|最后一次提交日期|提交次数|新增行数|GO|JAVA|VUE|KOTLIN|JS|JSON|MD|TEXT|LOG|Others|
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
|张三|2024-01-01|2025-11-04|150|5000|3000|0|1000|0|500|300|200|0|0|0|
|李四|2024-02-01|2025-10-30|80|2000|1500|0|0|0|300|100|100|0|0|0|

## project2

...
```

## 环境要求

- Python 3.6+
- Git 命令行工具
- 需要在Git仓库中运行

## 许可

MIT License
