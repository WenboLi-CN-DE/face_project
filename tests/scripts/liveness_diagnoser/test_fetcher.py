#!/usr/bin/env python3
"""
RemoteVideoFetcher 单元测试
"""

import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch, mock_open

from scripts.liveness_diagnoser.fetcher import RemoteVideoFetcher
from scripts.liveness_diagnoser.models import FetchConfig
from scripts.ssh_config import SSHConfig


class TestRemoteVideoFetcherInit(unittest.TestCase):
    """测试 RemoteVideoFetcher 初始化"""

    def test_init_with_config(self):
        """测试使用配置对象初始化"""
        config = FetchConfig(
            host="192.168.1.100",
            port=22,
            username="deploy",
            key_filename="/path/to/key.pem",
            remote_log_path="/var/log/app.log",
            output_dir="/tmp/output",
        )

        fetcher = RemoteVideoFetcher(config)

        self.assertEqual(fetcher.config.host, "192.168.1.100")
        self.assertEqual(fetcher.config.port, 22)
        self.assertEqual(fetcher.config.username, "deploy")
        self.assertEqual(fetcher.config.key_filename, "/path/to/key.pem")
        self.assertIsNone(fetcher.ssh)
        self.assertIsNone(fetcher.sftp)
        self.assertIsNone(fetcher.video_entry)


class TestRemoteVideoFetcherConnect(unittest.TestCase):
    """测试 RemoteVideoFetcher.connect 方法"""

    @patch("scripts.liveness_diagnoser.fetcher.SSHClient")
    def test_connect_success(self, mock_ssh_client):
        """测试 SSH 连接成功场景"""
        config = FetchConfig(
            host="192.168.1.100",
            port=22,
            username="deploy",
            key_filename="/path/to/key.pem",
            remote_log_path="/var/log/app.log",
            output_dir="/tmp/output",
        )

        fetcher = RemoteVideoFetcher(config)

        mock_ssh = MagicMock()
        mock_sftp = MagicMock()
        mock_ssh_client.return_value = mock_ssh
        mock_ssh.open_sftp.return_value = mock_sftp

        fetcher.connect()

        mock_ssh_client.assert_called_once()
        mock_ssh.set_missing_host_key_policy.assert_called_once()
        mock_ssh.connect.assert_called_once_with(
            hostname="192.168.1.100",
            port=22,
            username="deploy",
            key_filename="/path/to/key.pem",
            timeout=30,
            allow_agent=True,
            look_for_keys=True,
        )
        mock_ssh.open_sftp.assert_called_once()
        self.assertEqual(fetcher.ssh, mock_ssh)
        self.assertEqual(fetcher.sftp, mock_sftp)

    @patch("scripts.liveness_diagnoser.fetcher.SSHClient")
    def test_connect_failure(self, mock_ssh_client):
        """测试 SSH 连接失败场景"""
        config = FetchConfig(
            host="192.168.1.100",
            port=22,
            username="deploy",
            key_filename="/path/to/key.pem",
            remote_log_path="/var/log/app.log",
            output_dir="/tmp/output",
        )

        fetcher = RemoteVideoFetcher(config)

        mock_ssh = MagicMock()
        mock_ssh.connect.side_effect = Exception("Connection refused")
        mock_ssh_client.return_value = mock_ssh

        with self.assertRaises(Exception):
            fetcher.connect()


class TestRemoteVideoFetcherFindVideo(unittest.TestCase):
    """测试 RemoteVideoFetcher.find_video_by_task_id 方法"""

    @patch("scripts.liveness_diagnoser.fetcher.LogParser")
    @patch("scripts.liveness_diagnoser.fetcher.Path")
    def test_find_video_found(self, mock_path_class, mock_log_parser_class):
        """测试找到视频条目"""
        config = FetchConfig(
            host="192.168.1.100",
            port=22,
            username="deploy",
            key_filename="/path/to/key.pem",
            remote_log_path="/var/log/app.log",
            output_dir="/tmp/output",
        )

        fetcher = RemoteVideoFetcher(config)
        fetcher.sftp = MagicMock()

        mock_temp_log = MagicMock()
        mock_temp_log.exists.return_value = True
        mock_path_class.return_value = mock_temp_log

        mock_parser = MagicMock()
        mock_entry = MagicMock()
        mock_entry.task_id = "task123"
        mock_entry.video_filename = "test.webm"
        mock_parser.parse_file.return_value = [mock_entry]
        mock_log_parser_class.return_value = mock_parser

        result = fetcher.find_video_by_task_id("task123")

        self.assertEqual(result, mock_entry)
        self.assertEqual(fetcher.video_entry, mock_entry)
        fetcher.sftp.get.assert_called_once()
        mock_parser.parse_file.assert_called_once()

    @patch("scripts.liveness_diagnoser.fetcher.LogParser")
    @patch("scripts.liveness_diagnoser.fetcher.Path")
    def test_find_video_not_found(self, mock_path_class, mock_log_parser_class):
        """测试未找到视频条目"""
        config = FetchConfig(
            host="192.168.1.100",
            port=22,
            username="deploy",
            key_filename="/path/to/key.pem",
            remote_log_path="/var/log/app.log",
            output_dir="/tmp/output",
        )

        fetcher = RemoteVideoFetcher(config)
        fetcher.sftp = MagicMock()

        mock_temp_log = MagicMock()
        mock_temp_log.exists.return_value = True
        mock_path_class.return_value = mock_temp_log

        mock_parser = MagicMock()
        mock_entry = MagicMock()
        mock_entry.task_id = "task456"
        mock_parser.parse_file.return_value = [mock_entry]
        mock_log_parser_class.return_value = mock_parser

        result = fetcher.find_video_by_task_id("task123")

        self.assertIsNone(result)
        self.assertIsNone(fetcher.video_entry)


class TestRemoteVideoFetcherFromSshConfig(unittest.TestCase):
    """测试 RemoteVideoFetcher.from_ssh_config 类方法"""

    @patch("scripts.liveness_diagnoser.fetcher.get_ssh_config")
    def test_from_ssh_config_success(self, mock_get_config):
        """测试从 SSH 配置成功创建"""
        mock_ssh_config = SSHConfig(
            name="test-server",
            host="192.168.1.100",
            user="deploy",
            pem_key="/path/to/key.pem",
            port=22,
            remote_log="/var/log/app.log",
        )
        mock_get_config.return_value = mock_ssh_config

        fetcher = RemoteVideoFetcher.from_ssh_config("test-server", "task123")

        self.assertIsInstance(fetcher, RemoteVideoFetcher)
        self.assertEqual(fetcher.config.host, "192.168.1.100")
        self.assertEqual(fetcher.config.username, "deploy")
        self.assertEqual(fetcher.config.task_id, "task123")

    @patch("scripts.liveness_diagnoser.fetcher.get_ssh_config")
    def test_from_ssh_config_not_found(self, mock_get_config):
        """测试 SSH 配置不存在"""
        mock_get_config.return_value = None

        with self.assertRaises(ValueError) as context:
            RemoteVideoFetcher.from_ssh_config("nonexistent", "task123")

        self.assertIn("SSH 配置不存在", str(context.exception))

    @patch("scripts.liveness_diagnoser.fetcher.get_ssh_config")
    def test_from_ssh_config_missing_remote_log(self, mock_get_config):
        """测试 SSH 配置缺少 remote_log"""
        mock_ssh_config = SSHConfig(
            name="test-server",
            host="192.168.1.100",
            user="deploy",
            pem_key="/path/to/key.pem",
            port=22,
            remote_log=None,
        )
        mock_get_config.return_value = mock_ssh_config

        with self.assertRaises(ValueError) as context:
            RemoteVideoFetcher.from_ssh_config("test-server", "task123")

        self.assertIn("缺少 remote-log", str(context.exception))


if __name__ == "__main__":
    unittest.main()
