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
  - **Git统计表格**: 每个作者的提交信息和按语言分类的新增代码行数
  - **代码统计表格 (Tokei)**: 使用tokei工具统计的代码文件、总行数、代码行数、注释行数和空白行数

## 输出示例

```markdown
# Git 仓库统计汇总

生成时间: 2025-11-04 12:00:00

扫描目录: `/home/user/projects`

仓库数量: 3

---

## project1

路径: `/home/user/projects/project1`

### Git 统计

|作者名|首次提交日期|最后一次提交日期|提交次数|新增行数|Python|GO|JAVA|VUE|Kotlin|JS|TypeScript|JSON|MD|Dockerfile|TEXT|LOG|Others|
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
|张三|2024-01-01|2025-11-04|150|5000|0|3000|0|1000|0|500|0|300|200|0|0|0|0|
|李四|2024-02-01|2025-10-30|80|2000|0|1500|0|0|0|300|0|100|100|0|0|0|0|

### 代码统计 (Tokei)

|语言|文件数|总行数|代码行数|注释行数|空白行数|
|---|---|---|---|---|---|
|Go|50|15000|12000|1500|1500|
|JavaScript|30|8000|6500|800|700|
|TypeScript|20|5000|4000|500|500|
|**总计**|100|28000|22500|2800|2700|

## project2

...
```

## 环境要求

- Python 3.6+
- Git 命令行工具
- [Tokei](https://github.com/XAMPPRocky/tokei) - 代码统计工具 (用于batch_stats.py的代码统计功能)
  - 安装方法: `cargo install tokei` 或从 [releases](https://github.com/XAMPPRocky/tokei/releases) 下载预编译版本
- 需要在Git仓库中运行

## 许可

MIT License
