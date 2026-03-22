"""
静默活体检测多路径支持测试

测试场景:
1. 标准路径 /data/videos -> 直接使用
2. 远程路径 /opt/test -> 映射到 /data/videos
3. 远程路径 /opt/test2026 -> 映射到 /data/videos
4. 未授权路径 -> 拒绝访问
"""

import sys
import os

# 添加项目根目录到路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from vrlFace.silent_liveness.config import SilentConfig, get_config


def test_path_mapping():
    """测试路径映射功能"""
    print("\n" + "=" * 60)
    print("测试 1: 路径映射功能")
    print("=" * 60)

    config = SilentConfig()

    # 测试用例
    test_cases = [
        # (输入路径，期望输出路径)
        ("/data/videos/test.jpg", "/data/videos/test.jpg"),  # 无需映射
        (
            "/opt/test/test.jpg",
            "/data/videos/test.jpg",
        ),  # 映射 /opt/test -> /data/videos
        (
            "/opt/test2026/face.jpg",
            "/data/videos/face.jpg",
        ),  # 映射 /opt/test2026 -> /data/videos
        ("/opt/test/2026/03/15/test.jpg", "/data/videos/2026/03/15/test.jpg"),  # 子目录
    ]

    all_passed = True
    for input_path, expected in test_cases:
        result = config.resolve_path(input_path)
        status = "✅" if result == expected else "❌"
        print(f"{status} {input_path} -> {result}")
        if result != expected:
            print(f"   期望：{expected}")
            all_passed = False

    if all_passed:
        print("\n✅ 所有路径映射测试通过")
    else:
        print("\n❌ 部分路径映射测试失败")

    return all_passed


def test_path_allowed():
    """测试路径前缀校验功能"""
    print("\n" + "=" * 60)
    print("测试 2: 路径前缀校验（安全限制）")
    print("=" * 60)

    config = SilentConfig()

    # 测试用例
    test_cases = [
        ("/data/videos/test.jpg", True),  # 允许
        ("/opt/test/test.jpg", True),  # 允许
        ("/opt/test2026/face.jpg", True),  # 允许
        ("/etc/passwd", False),  # 拒绝
        ("/tmp/malicious.jpg", False),  # 拒绝
        ("/home/user/photo.jpg", False),  # 拒绝
    ]

    all_passed = True
    for input_path, expected in test_cases:
        result = config.is_path_allowed(input_path)
        status = "✅" if result == expected else "❌"
        print(f"{status} {input_path} -> {'允许' if result else '拒绝'}")
        if result != expected:
            print(f"   期望：{'允许' if expected else '拒绝'}")
            all_passed = False

    if all_passed:
        print("\n✅ 所有路径校验测试通过")
    else:
        print("\n❌ 部分路径校验测试失败")

    return all_passed


def test_config_display():
    """显示当前配置"""
    print("\n" + "=" * 60)
    print("当前配置")
    print("=" * 60)

    config = get_config()

    print("\n路径映射配置:")
    for external, internal in config.path_mapping.items():
        print(f"  {external} -> {internal}")

    print("\n允许的路径前缀:")
    for prefix in config.allowed_prefixes:
        print(f"  {prefix}")


def test_env_override():
    """测试环境变量覆盖"""
    print("\n" + "=" * 60)
    print("测试 3: 环境变量覆盖配置")
    print("=" * 60)

    # 设置测试环境变量
    os.environ["SILENT_PICTURE_PATHS"] = "/custom/path=/data/videos;/another=/tmp"
    os.environ["SILENT_ALLOWED_PATH_PREFIXES"] = "/data/videos,/custom/path,/another"

    # 重新创建配置（注意：实际运行时配置是单例，这里仅演示）
    new_config = SilentConfig()

    print("环境变量:")
    print(f"  SILENT_PICTURE_PATHS={os.environ['SILENT_PICTURE_PATHS']}")
    print(
        f"  SILENT_ALLOWED_PATH_PREFIXES={os.environ['SILENT_ALLOWED_PATH_PREFIXES']}"
    )

    print("\n解析后的配置:")
    print("  路径映射:")
    for external, internal in new_config.path_mapping.items():
        print(f"    {external} -> {internal}")

    print("  允许前缀:")
    for prefix in new_config.allowed_prefixes:
        print(f"    {prefix}")

    # 清理环境变量
    del os.environ["SILENT_PICTURE_PATHS"]
    del os.environ["SILENT_ALLOWED_PATH_PREFIXES"]

    print("\n✅ 环境变量覆盖测试完成")


if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("静默活体检测多路径支持测试")
    print("路径映射 + 安全校验")
    print("=" * 60)

    # 显示当前配置
    test_config_display()

    # 运行测试
    results = []

    results.append(("路径映射", test_path_mapping()))
    results.append(("路径校验", test_path_allowed()))
    test_env_override()

    # 汇总结果
    print("\n" + "=" * 60)
    print("测试汇总")
    print("=" * 60)

    for name, passed in results:
        status = "✅ 通过" if passed else "❌ 失败"
        print(f"{status}: {name}")

    all_passed = all(result[1] for result in results)

    print("\n" + "=" * 60)
    if all_passed:
        print("✅ 所有测试通过")
    else:
        print("❌ 部分测试失败")
    print("=" * 60)

    sys.exit(0 if all_passed else 1)
