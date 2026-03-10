# AGENTS.md - 编程助手指南

> 本文档为 AI 编程助手提供开发规范、代码风格和测试指南

## 🌏 语言设置

**重要：请全程使用简体中文进行对话和交流**

- 所有对话、说明、解释使用中文
- 代码注释使用中文
- 提交信息使用中文
- 文档编写使用中文
- 保持专业、简洁的风格，直接回答问题

## 📚 文档导航

- **README.md** - 项目介绍、快速开始
- **AGENTS.md** (本文档) - 开发规范、代码风格、测试命令
- **PROJECT_ROADMAP.md** - 项目路线图

---

## 构建/检查/测试命令

### 环境设置
```bash
# 使用 uv 管理项目
uv sync                    # 同步依赖
uv sync --extra dev        # 同步开发依赖
uv venv                    # 创建虚拟环境
```

### 运行命令
```bash
uv run python main.py                    # 运行主程序
uv run python script.py                  # 运行任意脚本
uv run python tests/test_liveness.py     # 运行单个测试脚本
```

### 测试实践
```bash
# 直接运行测试脚本 (不使用 pytest)
uv run python tests/test_module.py

# 运行示例/演示脚本
uv run python demo.py
uv run python vrlFace/liveness_example.py
```

### 代码检查
```bash
uv run flake8 .      # 代码风格检查
uv run mypy .        # 类型检查
uv run pylint .      # 代码质量 (可选)
```

### 格式化
```bash
uv run black .         # 格式化代码
uv run isort .         # 排序导入
uv run black --check . # 检查格式
```

## 代码风格指南

### 通用原则
- 遵循 PEP 8 风格指南
- 最大行长度：100 字符 (Black 配置)
- 所有函数签名使用类型提示
- 使用双引号 `"` 作为字符串引号

### 导入规范
```python
import os
import numpy as np
from insightface.app import FaceAnalysis
from . import local_module
# 避免：from module import *
```

### 命名约定
- 变量和函数：`snake_case` (如 `user_name`)
- 类：`CamelCase` (如 `FaceVerification`)
- 常量：`ALL_CAPS` (如 `MAX_RETRIES`)
- 保护成员：`_single_underscore`
- 私有成员：`__double_underscore`

### 错误处理
```python
try:
    result = risky_operation()
except SpecificError as e:
    logger.error(f"操作失败：{e}")
    raise CustomError("有意义的消息") from e
# 避免裸 except，使用 else 子句处理正常流程
```

### 类型注解
```python
def greet(name: str, age: int) -> str: ...
def process(items: list[str]) -> dict[str, int]: ...
def find(user_id: int) -> Optional[User]: ...
```

### 文档字符串
- 所有公共函数/类使用 Google 风格文档字符串
```python
def calculate(a: int, b: int) -> int:
    """计算两个数的和
    
    参数:
        a: 第一个数字
        b: 第二个数字
        
    返回:
        a 和 b 的和
    """
    return a + b
```

### 测试实践
- 测试文件镜像源结构
- 测试脚本命名：`test_*.py`
- 使用 Arrange-Act-Assert 模式
- 每个测试只验证一个行为

### 安全准则
- 绝不硬编码密钥，使用 `os.getenv()` 或 `.env`
- 验证所有用户输入
- 使用 `pathlib.Path` 处理文件路径

### 性能优化
- 优化前先分析
- 使用 NumPy 向量化操作
- 避免过早优化

## 项目特定约定

### 核心依赖
- `insightface` - 人脸识别引擎
- `opencv-python` (cv2) - 图像处理
- `numpy` - 数值计算
- `onnxruntime-gpu` - GPU 加速推理

### 目录约定
- 模型存放在 `models/buffalo_l/` 目录
- 测试数据存放在 `data/` 或 `tests/data/` 目录
- 输出结果存放在 `output/` 目录
- 日志文件存放在 `logs/` 目录

### 配置管理
```python
# 相似度阈值推荐
THRESHOLD_STRICT = 0.70    # 严格模式
THRESHOLD_NORMAL = 0.55    # 正常模式
THRESHOLD_LOOSE = 0.40     # 宽松模式

# 检测参数
DET_SIZE = (640, 640)      # 平衡速度与精度
CTX_ID = 0                 # GPU ID, -1 为 CPU
```

### 关键决策
1. **模型选择**: buffalo_l 模型，精度高适合生产
2. **GPU 加速**: 推荐使用 onnxruntime-gpu
3. **代码引用**: 使用 `file_path:line_number` 格式 (如 `face_verification.py:42`)

