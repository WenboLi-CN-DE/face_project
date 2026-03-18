# tests — 测试套件

**位置**: `tests/`  
**职责**: 单元测试与集成测试（镜像源码结构）

## 结构

```
tests/
├── __init__.py
├── test_log_parser.py          # 日志解析器测试
├── face/                       # 人脸识别测试
│   ├── __init__.py
│   └── test_face.py            # 配置/检测/比对/搜索测试
└── liveness/                   # 活体检测测试
    ├── __init__.py
    ├── test_liveness.py        # 配置/检测器/融合引擎测试
    ├── test_benchmark.py       # 基准帧校准测试
    ├── test_async_processor.py # 异步任务处理器测试
    ├── test_callback.py        # HTTP 回调客户端测试
    ├── test_liveness_api.py    # 活体 API 接口测试
    └── test_fix.py             # 修复验证测试
```

## 测试组织

**镜像源码结构**:
- `tests/face/` ↔ `vrlFace/face/`
- `tests/liveness/` ↔ `vrlFace/liveness/`

**按功能划分**:
- 每个核心组件有独立测试文件
- 活体检测测试更复杂（6 个测试文件 vs 1 个）

## 运行命令

```bash
# 运行所有测试
pytest tests/ -v

# 仅人脸识别测试
pytest tests/face/ -v

# 仅活体检测测试
pytest tests/liveness/ -v

# 单个测试文件
pytest tests/liveness/test_benchmark.py -v
```

## 测试覆盖

| 模块 | 测试文件 | 覆盖内容 |
|------|----------|----------|
| **face** | `test_face.py` | 配置加载、人脸检测、1:1 比对、1:N 搜索 |
| **liveness** | `test_liveness.py` | 配置预设、检测器初始化、融合引擎 |
| **liveness** | `test_benchmark.py` | 基准帧采集、校准验证、防替换攻击 |
| **liveness** | `test_async_processor.py` | 异步任务队列、后台处理 |
| **liveness** | `test_callback.py` | HTTP 回调、HMAC 签名验证 |
| **liveness** | `test_liveness_api.py` | FastAPI 接口测试 |
| **scripts** | `test_log_parser.py` | 日志解析功能 |

## 测试配置

**pyproject.toml**:
```toml
[tool.pytest.ini_options]
testpaths = ["tests"]
python_files = "test_*.py"
python_classes = "Test*"
python_functions = "test_*"
addopts = "-v"
```

## CI/CD 集成

GitHub Actions 自动运行：
```yaml
- name: Run tests
  run: pytest tests/ -v

- name: Flake8 check
  run: flake8 .
```

## 测试约定

**命名规范**:
- 测试文件：`test_*.py`
- 测试类：`Test*`
- 测试函数：`test_*`

**Fixtures 使用**:
- `tmp_path` — 临时目录
- `capsys` — 捕获输出
- `monkeypatch` — 环境变量修改
- 无自定义 fixtures（每个测试文件自包含）

**导入模式**:
```python
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
```

---

## 手动测试脚本

**位置**: `scripts/manual_tests/`

| 脚本 | 用途 |
|------|------|
| `batch_test_videos.py` | 批量视频动作测试 |
| `test_frame_allocation.py` | 帧分配修复测试 |
| `test_callback_network.py` | 网络回调连接测试 |
| `diagnose_video.py` | 视频诊断（输出 CSV） |
| `debug_is_liveness.py` | 活体判定调试（逐帧分析） |
| `remote_fetch_tests/test_all_videos.py` | 远程拉取视频批量测试 |

---

## 注意事项

- 测试文件必须以 `test_` 开头
- 测试类必须以 `Test` 开头
- 测试函数必须以 `test_` 开头
- 活体检测测试可能需要模型文件
