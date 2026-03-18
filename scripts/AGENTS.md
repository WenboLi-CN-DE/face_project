# scripts — 运维与工具脚本

**位置**: `scripts/`  
**职责**: 日志分析、阈值验证、SSH 配置、远程获取、手动测试等运维工具

## 结构

```
scripts/
├── log_parser.py               # 日志解析器
├── log_video_analyzer.py       # 视频日志分析器（16KB）
├── remote_fetch.py             # 远程数据获取工具（13KB）
├── ssh_config.py               # SSH 配置生成器
├── test_integration.py         # 集成测试脚本
├── test_threshold_validation.py# 阈值验证测试（9KB）
│
├── analyze2.py                 # CSV 分析工具
├── analyze_csv.py              # CSV 日志分析
├── export_docker_log.py        # Docker 日志导出
├── fetch_with_docker_log.py    # 带日志的远程获取
│
├── diagnose_ssh.sh             # SSH 连接诊断
├── setup_ssh_access.sh         # SSH 访问配置
│
├── verify_final.py             # 最终验证工具
├── verify_fix.py               # 修复验证
└── verify_full_fix.py          # 完整修复验证
│
└── manual_tests/               # 手动测试/诊断脚本（子目录）
    ├── batch_test_videos.py    # 批量视频动作测试
    ├── diagnose_video.py       # 视频诊断（输出 CSV）
    ├── diagnose_webm.py        # WebM 诊断
    ├── test_callback_network.py# 网络回调测试
    ├── test_frame_allocation.py# 帧分配测试
    └── remote_fetch_tests/     # 远程拉取视频测试（子目录）
        ├── test_all_videos.py  # 批量测试拉取视频
        ├── test_quality_fix.py # 质量分数修复测试
        ├── test_with_log_analysis.py # 带日志分析的测试
        ├── debug_is_liveness.py# 活体判定调试
        └── analyze_smoothing.py# 平滑分析
```

## 核心脚本

### 日志分析
| 脚本 | 用途 |
|------|------|
| `log_parser.py` | 解析活体检测日志，提取关键指标 |
| `log_video_analyzer.py` | 视频级日志分析，逐帧统计动作/分数 |
| `analyze_csv.py` | CSV 格式日志分析 |

### 阈值验证
| 脚本 | 用途 |
|------|------|
| `test_threshold_validation.py` | 验证阈值自动调整功能 |
| `verify_final.py` | 最终验证脚本 |

### SSH 配置
| 脚本 | 用途 |
|------|------|
| `ssh_config.py` | 生成 SSH 配置文件 |
| `setup_ssh_access.sh` | 一键配置 SSH 访问 |
| `diagnose_ssh.sh` | SSH 连接诊断 |

### 远程获取
| 脚本 | 用途 |
|------|------|
| `remote_fetch.py` | 从远程服务器获取数据 |
| `fetch_with_docker_log.py` | 带 Docker 日志的远程获取 |
| `export_docker_log.py` | 导出 Docker 容器日志 |

### 测试验证
| 脚本 | 用途 |
|------|------|
| `test_integration.py` | 集成测试脚本 |
| `verify_fix.py` | 修复验证 |
| `verify_full_fix.py` | 完整修复验证 |

---

### 手动测试脚本（manual_tests/）

#### 基础测试脚本
| 脚本 | 用途 | 输出 |
|------|------|------|
| `batch_test_videos.py` | 批量测试视频动作检测 | `output/results/batch_test_result.txt` |
| `diagnose_video.py` | 视频诊断，记录每帧参数 | `output/results/*.csv` |
| `diagnose_webm.py` | WebM 格式诊断 | - |
| `test_callback_network.py` | 测试回调网络连接 | - |
| `test_frame_allocation.py` | 测试帧分配修复效果 | - |

#### 远程拉取测试（remote_fetch_tests/）
| 脚本 | 用途 | 说明 |
|------|------|------|
| `test_all_videos.py` | 批量测试远程拉取视频 | 使用 `output/remote_fetch/videos/` 中的视频 |
| `test_quality_fix.py` | 测试质量分数修复效果 | 对比修复前后的质量分数 |
| `test_with_log_analysis.py` | 带日志分析的视频测试 | 结合日志和视频分析 |
| `debug_is_liveness.py` | 活体判定调试工具 | 调试 `is_liveness` 判断逻辑（逐帧分析） |
| `analyze_smoothing.py` | 平滑分析工具 | 分析阈值平滑效果 |

**使用示例**:
```bash
# 批量测试视频
uv run python scripts/manual_tests/batch_test_videos.py --video-dir 15

# 诊断单个视频
uv run python scripts/manual_tests/diagnose_video.py 15/test.webm shake_head_diagnosis

# 测试帧分配
uv run python scripts/manual_tests/test_frame_allocation.py --video test.mp4 --actions nod mouth_open

# 测试远程拉取视频
uv run python scripts/manual_tests/remote_fetch_tests/test_all_videos.py
```

### 日志分析

| 脚本 | 用途 |
|------|------|
| `log_parser.py` | 解析活体检测日志，提取关键指标 |
| `log_video_analyzer.py` | 视频级日志分析，逐帧统计动作/分数 |
| `analyze_csv.py` | CSV 格式日志分析 |

### 阈值验证

| 脚本 | 用途 |
|------|------|
| `test_threshold_validation.py` | 验证阈值自动调整功能 |
| `verify_final.py` | 最终验证脚本 |

### SSH 配置

| 脚本 | 用途 |
|------|------|
| `ssh_config.py` | 生成 SSH 配置文件 |
| `setup_ssh_access.sh` | 一键配置 SSH 访问 |
| `diagnose_ssh.sh` | SSH 连接诊断 |

### 远程获取

| 脚本 | 用途 |
|------|------|
| `remote_fetch.py` | 从远程服务器获取数据 |
| `fetch_with_docker_log.py` | 带 Docker 日志的远程获取 |
| `export_docker_log.py` | 导出 Docker 容器日志 |

### 测试验证

| 脚本 | 用途 |
|------|------|
| `test_integration.py` | 集成测试脚本 |
| `verify_fix.py` | 修复验证 |
| `verify_full_fix.py` | 完整修复验证 |

## 使用示例

### 日志解析
```bash
# 解析活体检测日志
uv run python scripts/log_parser.py --input output/xxx.log

# 视频日志分析
uv run python scripts/log_video_analyzer.py --input batch_test_output.log
```

### SSH 配置
```bash
# 生成 SSH 配置
uv run python scripts/ssh_config.py

# 设置 SSH 访问
bash scripts/setup_ssh_access.sh

# 诊断 SSH 连接
bash scripts/diagnose_ssh.sh
```

### 远程获取
```bash
# 从远程服务器获取数据
uv run python scripts/remote_fetch.py --host <server> --path <remote_path>
```

## 输出目录结构

```
output/
├── results/          # 测试结果、CSV 文件、分析报告
├── logs/             # 日志文件（vrlface.log 等）
├── remote_fetch/     # 远程获取的数据和测试
│   ├── videos/       # 拉取的视频文件
│   ├── 动作对应.txt   # 视频动作配置
│   └── archive/      # 历史报告归档
└── temp/             # 临时文件
```

**分类规则**:
- CSV 结果、测试报告 → `output/results/`
- 日志文件 → `output/logs/`
- 临时文件 → `output/temp/`
- 远程拉取视频 → `output/remote_fetch/videos/`
- 远程测试报告 → `output/remote_fetch/archive/`

## 注意事项

- 所有脚本应使用 `uv run python` 运行以确保依赖正确
- Shell 脚本需要执行权限：`chmod +x scripts/*.sh`
- 输出文件应保存到 `output/` 对应子目录，避免污染根目录
- 远程获取脚本需要 SSH 密钥认证
- manual_tests/ 中的脚本用于手动诊断，不属于自动化测试套件

## 相关文档

- `docs/ssh-config-usage.md` — SSH 配置使用指南
- `docs/remote_fetch.md` — 远程获取工具文档
- `docs/docker-log-workflow.md` — Docker 日志工作流
