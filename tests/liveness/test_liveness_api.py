"""
活体检测 API 测试

运行:
    pytest tests/liveness/test_liveness_api.py -v
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))


def test_schemas_import():
    """测试 schema 模型可以正常导入"""
    from vrlFace.liveness.schemas import (
        MoveLivenessAsyncRequest,
        ThresholdConfig,
        ActionConfig,
    )

    req = MoveLivenessAsyncRequest(
        request_id="test_001",
        task_id="task_001",
        video_path="/tmp/test.mp4",
        actions=["blink", "mouth_open"],
        callback_url="http://example.com/callback",
        threshold_config=ThresholdConfig(liveness_threshold=0.5, action_threshold=0.85),
        action_config=ActionConfig(max_video_duration=6, per_action_timeout=2),
    )
    assert req.request_id == "test_001"
    assert len(req.actions) == 2
    assert req.threshold_config.liveness_threshold == 0.5
    assert req.action_config.max_video_duration == 6
    assert req.callback_url == "http://example.com/callback"


def test_threshold_config_defaults():
    """测试阈值配置默认值"""
    from vrlFace.liveness.schemas import ThresholdConfig

    cfg = ThresholdConfig()
    assert cfg.liveness_threshold == 0.5
    assert cfg.action_threshold == 0.85


def test_video_analyzer_import():
    """测试分析器模块可以正常导入"""
    from vrlFace.liveness.video_analyzer import (
        ACTION_ALIASES,
    )

    assert "blink" in ACTION_ALIASES
    assert "mouth_open" in ACTION_ALIASES
    assert "shake_head" in ACTION_ALIASES


def test_action_aliases_coverage():
    """测试动作别名覆盖所有支持动作"""
    from vrlFace.liveness.video_analyzer import ACTION_ALIASES

    SUPPORTED = {
        "blink",
        "mouth_open",
        "shake_head",
        "nod",
        "nod_down",
        "nod_up",
        "turn_left",
        "turn_right",
    }
    for action in SUPPORTED:
        assert action in ACTION_ALIASES, f"动作 {action} 缺少别名映射"


def test_error_result_on_missing_video():
    """测试视频不存在时返回错误结果"""
    from vrlFace.liveness.video_analyzer import VideoLivenessAnalyzer

    analyzer = VideoLivenessAnalyzer()
    result = analyzer.analyze(
        video_path="/nonexistent/path/video.mp4",
        actions=["blink", "mouth_open"],
    )
    assert result.is_liveness == 0
    assert result.is_face_exist == 0
    assert result.action_verify.passed is False
    assert len(result.action_verify.action_details) == 2


def test_action_result_fields():
    """测试 ActionResult 数据类字段"""
    from vrlFace.liveness.video_analyzer import ActionResult

    r = ActionResult(action="blink", passed=True, confidence=0.95, msg="检测到有效眨眼")
    assert r.action == "blink"
    assert r.passed is True
    assert r.confidence == 0.95


def test_face_info_fields():
    """测试 FaceInfo 数据类字段"""
    from vrlFace.liveness.video_analyzer import FaceInfo

    fi = FaceInfo(confidence=0.96, quality_score=0.88)
    assert fi.confidence == 0.96
    assert fi.quality_score == 0.88


def test_action_cn_mapping():
    """测试中文映射函数"""
    from vrlFace.liveness.video_analyzer import _action_cn

    assert _action_cn("blink") == "眨眼"
    assert _action_cn("mouth_open") == "张嘴"
    assert _action_cn("shake_head") == "摇头"
    assert _action_cn("unknown") == "unknown"


def test_liveness_api_router_import():
    """测试活体 API 路由可以正常导入"""
    from vrlFace.liveness.api import router
    from fastapi.routing import APIRoute

    assert router is not None
    routes = [r.path for r in router.routes if isinstance(r, APIRoute)]
    assert "/vrlMoveLiveness" in routes


def test_main_fastapi_has_liveness_route():
    """测试主 app 注册了活体路由"""
    from vrlFace.main_fastapi import app
    from fastapi.routing import APIRoute

    paths = [r.path for r in app.routes if isinstance(r, APIRoute)]
    assert "/vrlMoveLiveness" in paths
    assert "/healthz" in paths


def test_api_missing_video_returns_400():
    """测试视频不存在时 API 返回 400"""
    try:
        from fastapi.testclient import TestClient
        from vrlFace.main_fastapi import app

        client = TestClient(app)
        resp = client.post(
            "/vrlMoveLiveness",
            json={
                "request_id": "test_req_001",
                "task_id": "task_001",
                "video_path": "/nonexistent/video.mp4",
                "actions": ["blink"],
                "callback_url": "http://example.com/callback",
                "threshold_config": {
                    "liveness_threshold": 0.5,
                    "action_threshold": 0.85,
                },
            },
        )
        assert resp.status_code == 400
        assert "不存在" in resp.json()["detail"]
    except ImportError:
        import pytest

        pytest.skip("需要 httpx: pip install httpx")


def test_api_invalid_action_returns_400():
    """测试非法动作时 API 返回 400"""
    try:
        from fastapi.testclient import TestClient
        from vrlFace.main_fastapi import app

        client = TestClient(app)
        resp = client.post(
            "/vrlMoveLiveness",
            json={
                "request_id": "test_req_002",
                "task_id": "task_002",
                "video_path": "/nonexistent/video.mp4",
                "actions": ["fly"],
                "callback_url": "http://example.com/callback",
                "threshold_config": {
                    "liveness_threshold": 0.5,
                    "action_threshold": 0.85,
                },
            },
        )
        assert resp.status_code == 400
        assert "不支持的动作" in resp.json()["detail"]
    except ImportError:
        import pytest

        pytest.skip("需要 httpx: pip install httpx")


def test_healthz():
    """测试健康检查接口"""
    try:
        from fastapi.testclient import TestClient
        from vrlFace.main_fastapi import app

        client = TestClient(app)
        resp = client.get("/healthz")
        assert resp.status_code == 200
        assert resp.json()["status"] == "healthy"
    except ImportError:
        import pytest

        pytest.skip("需要 httpx: pip install httpx")
