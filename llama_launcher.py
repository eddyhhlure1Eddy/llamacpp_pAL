"""
Llama.cpp Server Launcher GUI
A comprehensive GUI application for managing llama-server instances.
Author: eddy
"""

import tkinter as tk
from tkinter import ttk, messagebox, scrolledtext, filedialog
import subprocess
import threading
import json
import os
import sys
import time
import signal
import webbrowser
import logging
import re
from pathlib import Path
from datetime import datetime
from typing import Optional, Dict, List, Any
import queue

try:
    import requests
    HAS_REQUESTS = True
except ImportError:
    HAS_REQUESTS = False


APP_NAME = "Llama.cpp Server Launcher"
APP_VERSION = "3.0.0"
DEFAULT_LLAMA_DIR = os.path.dirname(os.path.abspath(__file__))
CONFIG_FILENAME = "launcher_config.json"
LOG_FILENAME = "launcher.log"

# Remote server default settings
DEFAULT_REMOTE_HOST = "192.168.31.123"
DEFAULT_REMOTE_USER = "Administrator"
DEFAULT_REMOTE_LLAMA_DIR = "E:/llama.cpp2/llama.cpp2"
DEFAULT_REMOTE_MODEL_DIR = "E:/llama.cpp2/llama.cpp2/models"

# MoE expert tensor name pattern (same as llama.cpp's LLM_FFN_EXPS_REGEX)
LLM_FFN_EXPS_REGEX = "\\.ffn_(up|down|gate)_(ch|)exps"


def generate_moe_ot_rules(total_layers: int, gpu_count: int) -> list:
    """Generate -ot rules for evenly distributing MoE experts across GPUs.

    Uses the same pattern format as llama.cpp's built-in -ncmoe:
        blk\\.N\\.ffn_(up|down|gate)_(ch|)exps
    Returns a list of (pattern, device) tuples.
    """
    if total_layers <= 0 or gpu_count <= 0:
        return []
    layers_per_gpu = total_layers / gpu_count
    rules = []
    for gpu_idx in range(gpu_count):
        start = int(gpu_idx * layers_per_gpu)
        end = int((gpu_idx + 1) * layers_per_gpu) - 1
        if gpu_idx == gpu_count - 1:
            end = total_layers - 1
        device = f"CUDA{gpu_idx}"
        for layer in range(start, end + 1):
            rules.append((f"blk\\.{layer}{LLM_FFN_EXPS_REGEX}", device))
    return rules


def generate_moe_weighted_rules(total_layers: int, ratios: list) -> list:
    """Generate -ot rules for distributing MoE experts across GPUs by weight ratios.

    Args:
        total_layers: Total number of MoE layers
        ratios: Weight ratios per GPU, e.g. [2, 1, 1] = GPU0 gets 50%, GPU1/2 get 25% each
    Returns a list of (pattern, device) tuples.
    """
    if total_layers <= 0 or not ratios or sum(ratios) <= 0:
        return []
    total_weight = sum(ratios)
    rules = []
    current_layer = 0
    for gpu_idx, weight in enumerate(ratios):
        if weight <= 0:
            continue
        if gpu_idx == len(ratios) - 1:
            num_layers = total_layers - current_layer
        else:
            num_layers = round(total_layers * weight / total_weight)
        if num_layers <= 0:
            continue
        device = f"CUDA{gpu_idx}"
        for layer in range(current_layer, current_layer + num_layers):
            rules.append((f"blk\\.{layer}{LLM_FFN_EXPS_REGEX}", device))
        current_layer += num_layers
    return rules


class UIText:
    """UI text constants for localization."""

    WINDOW_TITLE = "Llama.cpp 服务器启动器"

    MODEL_CONFIG = "模型配置"
    LLAMA_DIR = "Llama.cpp 目录:"
    BROWSE = "浏览"
    MAIN_MODEL = "主模型:"
    MMPROJ = "视觉模型:"
    REFRESH_MODELS = "刷新模型"
    OPEN_MODELS_FOLDER = "打开模型目录"

    SERVER_PARAMS = "服务器参数"
    HOST = "主机:"
    PORT = "端口:"
    CONTEXT = "上下文:"
    GPU_LAYERS = "GPU层数:"
    MAIN_GPU = "主GPU:"
    BATCH_SIZE = "批大小:"
    UBATCH_SIZE = "物理批:"
    TEMP = "温度:"
    TOP_P = "Top-P:"
    TOP_K = "Top-K:"
    REP_PEN = "重复惩罚:"
    CACHE_TYPE_K = "KV-K类型:"
    CACHE_TYPE_V = "KV-V类型:"
    USE_JINJA = "启用 Jinja"
    FLASH_ATTN = "Flash Attention"
    SPLIT_MODE_ROW = "张量并行"
    NO_MMAP = "完全加载"
    CUSTOM_ARGS = "自定义参数:"

    SERVER_CONTROL = "服务器控制"
    STATUS = "状态:"
    STATUS_RUNNING = "运行中"
    STATUS_STOPPED = "已停止"
    STATUS_LOADING = "加载中"
    START_SERVER = "启动服务器"
    STOP_SERVER = "停止服务器"
    OPEN_WEBUI = "打开 WebUI"
    TEST_API = "测试 API"
    SAVE_CONFIG = "保存配置"
    RESET_DEFAULTS = "重置默认"

    SERVER_OUTPUT = "服务器输出"
    CLEAR_LOG = "清除日志"
    COPY_LOG = "复制日志"
    AUTO_SCROLL = "自动滚动"

    READY = "就绪"
    CONFIG_SAVED = "配置已保存"
    LOG_COPIED = "日志已复制到剪贴板"

    MSG_SELECT_MODEL = "请先选择模型文件"
    MSG_STARTING = "正在启动服务器..."
    MSG_STARTED = "服务器启动成功"
    MSG_START_FAILED = "服务器启动失败"
    MSG_STOPPING = "正在停止服务器..."
    MSG_STOPPED = "服务器已停止"
    MSG_STOP_WARNING = "警告: 服务器可能未完全停止"
    MSG_TESTING_API = "正在测试 API..."
    MSG_API_OK = "API 测试: 正常"
    MSG_API_FAILED = "API 测试失败"
    MSG_MODEL = "模型:"
    MSG_VISION = "视觉:"
    MSG_ENABLED = "已启用"
    MSG_DISABLED = "已禁用"
    MSG_FOUND_MODELS = "找到 {} 个模型和 {} 个视觉模型文件"
    MSG_DIR_NOT_FOUND = "目录不存在:"
    MSG_SCAN_ERROR = "扫描模型时出错:"
    MSG_OPENED_WEBUI = "已打开 WebUI:"
    MSG_RESET_CONFIRM = "确定要重置所有设置为默认值吗?"
    MSG_SETTINGS_RESET = "设置已重置为默认值"
    MSG_CONFIRM = "确认"
    MSG_ERROR = "错误"
    MSG_EXIT_CONFIRM = "确认退出"
    MSG_SERVER_RUNNING_EXIT = "服务器仍在运行，是否停止服务器并退出?"
    MSG_UNEXPECTED_TERMINATE = "服务器进程意外终止"

    # Remote server UI text
    REMOTE_SERVER = "远程服务器"
    REMOTE_CONNECTION = "远程连接设置"
    REMOTE_HOST = "远程主机:"
    REMOTE_USER = "用户名:"
    REMOTE_PASSWORD = "密码:"
    REMOTE_TEST_CONNECTION = "测试连接"
    REMOTE_CONNECTED = "已连接"
    REMOTE_DISCONNECTED = "未连接"
    REMOTE_SERVER_PARAMS = "远程服务器参数"
    REMOTE_LLAMA_DIR = "Llama目录:"
    REMOTE_MODEL_DIR = "模型目录:"
    REMOTE_MODEL = "模型文件:"
    REMOTE_MMPROJ = "视觉模型:"
    REMOTE_PARALLEL = "并发数:"
    REMOTE_PORT = "端口:"
    REMOTE_CONTEXT = "上下文:"
    REMOTE_GPU_LAYERS = "GPU层数:"
    REMOTE_MAIN_GPU = "主GPU:"
    REMOTE_TEMP = "温度:"
    REMOTE_TOP_P = "Top-P:"
    REMOTE_TOP_K = "Top-K:"
    REMOTE_REP_PEN = "重复惩罚:"
    REMOTE_BATCH_SIZE = "批大小:"
    REMOTE_UBATCH_SIZE = "物理批:"
    REMOTE_CACHE_TYPE_K = "KV-K类型:"
    REMOTE_CACHE_TYPE_V = "KV-V类型:"
    REMOTE_CACHE_REUSE = "缓存复用:"
    REMOTE_CACHE_RAM = "缓存大小(MB):"
    REMOTE_SLOT_SAVE = "保存KV缓存"
    REMOTE_CUDA_GRAPH_OPT = "CUDA Graph优化"
    REMOTE_FIT = "Fit(Auto)"
    REMOTE_FIT_TARGET = "预留空间(MiB):"
    REMOTE_NO_KV_OFFLOAD = "KV放CPU"
    REMOTE_CUSTOM_ARGS = "自定义参数:"
    REMOTE_SCAN_MODELS = "扫描模型"
    REMOTE_CONTROL = "远程服务器控制"
    REMOTE_STATUS = "远程状态:"
    REMOTE_START = "启动远程服务器"
    REMOTE_STOP = "停止远程服务器"
    REMOTE_CHECK_STATUS = "检查状态"
    REMOTE_FETCH_LOG = "获取日志"
    REMOTE_OPEN_WEBUI = "打开远程WebUI"
    REMOTE_STARTING = "正在启动远程服务器..."
    REMOTE_STARTED = "远程服务器启动成功"
    REMOTE_START_FAILED = "远程服务器启动失败"
    REMOTE_STOPPING = "正在停止远程服务器..."
    REMOTE_STOPPED = "远程服务器已停止"
    REMOTE_CONNECTING = "正在连接..."
    REMOTE_CONNECTION_OK = "远程连接成功"
    REMOTE_CONNECTION_FAILED = "远程连接失败"
    REMOTE_SCANNING = "正在扫描远程模型..."
    REMOTE_FOUND_MODELS = "找到 {} 个模型文件"

    # Multi-instance UI text
    MULTI_INSTANCE = "多实例模式"
    MULTI_ENABLED = "启用多实例"
    NUM_INSTANCES = "实例数量:"
    START_PORT_LABEL = "起始端口:"
    INSTANCE_LIST = "实例状态"
    START_ALL = "全部启动"
    STOP_ALL = "全部停止"
    COPY_URLS = "复制 URLs"
    MSG_STARTING_INSTANCES = "正在启动 {} 个实例..."
    MSG_ALL_STARTED = "所有 {} 个实例已启动"
    MSG_STOPPING_INSTANCES = "正在停止所有实例..."
    MSG_ALL_STOPPED = "所有实例已停止"
    MSG_INSTANCE_STARTED = "端口 {} 的实例已启动"
    MSG_INSTANCE_STOPPED = "端口 {} 的实例已停止"
    MSG_URLS_COPIED = "服务器 URLs 已复制到剪贴板"
    MSG_INSTANCE_FAILED = "端口 {} 的实例启动失败"
    INSTANCE_PORT = "端口"
    INSTANCE_STATUS_COL = "状态"
    INSTANCE_PID = "PID"
    STATUS_STARTING = "启动中"

    # Remote multi-instance UI text
    REMOTE_MULTI_INSTANCE = "远程多实例模式"
    REMOTE_MULTI_ENABLED = "启用多实例"
    REMOTE_NUM_INSTANCES = "实例数量:"
    REMOTE_START_PORT_LABEL = "起始端口:"
    REMOTE_START_ALL = "全部启动"
    REMOTE_STOP_ALL = "全部停止"
    REMOTE_COPY_URLS = "复制 URLs"
    MSG_REMOTE_STARTING_INSTANCES = "正在启动 {} 个远程实例..."
    MSG_REMOTE_ALL_STARTED = "所有 {} 个远程实例已启动"
    MSG_REMOTE_STOPPING_INSTANCES = "正在停止所有远程实例..."
    MSG_REMOTE_ALL_STOPPED = "所有远程实例已停止"
    MSG_REMOTE_INSTANCE_STARTED = "远程端口 {} 的实例已启动"
    MSG_REMOTE_INSTANCE_STOPPED = "远程端口 {} 的实例已停止"
    MSG_REMOTE_INSTANCE_FAILED = "远程端口 {} 的实例启动失败"

    # Multi-model instance UI text
    INSTANCE_MODEL = "模型"
    INSTANCE_MMPROJ = "视觉模型"
    INSTANCE_GPU = "GPU"
    DOUBLE_CLICK_SELECT_MODEL = "双击选择模型"
    ADD_INSTANCE = "添加实例"
    REMOVE_INSTANCE = "删除实例"
    MSG_SELECT_MODEL_FOR_INSTANCE = "请为端口 {} 的实例选择模型"
    MSG_NO_INSTANCE_SELECTED = "请先选择一个实例"


def setup_logging(log_dir: str) -> logging.Logger:
    """Setup application logging."""
    logger = logging.getLogger("LlamaLauncher")
    logger.setLevel(logging.DEBUG)

    log_path = os.path.join(log_dir, LOG_FILENAME)
    file_handler = logging.FileHandler(log_path, encoding="utf-8")
    file_handler.setLevel(logging.DEBUG)

    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)

    formatter = logging.Formatter(
        "%(asctime)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)

    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    return logger


class ServerConfig:
    """Configuration manager for server settings."""

    DEFAULT_CONFIG = {
        "llama_dir": DEFAULT_LLAMA_DIR,
        "model_file": "",
        "mmproj_file": "",
        "port": 8080,
        "host": "0.0.0.0",
        "context_size": 32768,
        "gpu_layers": 99,
        "main_gpu": -1,
        "temperature": 0.8,
        "top_p": 0.6,
        "top_k": 2,
        "repeat_penalty": 1.1,
        "use_jinja": True,
        "use_flash_attn": False,
        "split_mode_row": False,
        "no_mmap": False,
        "cache_type_k": "f16",
        "cache_type_v": "f16",
        "threads": 0,
        "batch_size": 512,
        "ubatch_size": 512,
        "parallel": 1,
        "fit": False,
        "fit_target": 1024,
        "no_kv_offload": False,
        "cuda_graph_opt": False,
        "cache_reuse": 0,
        "cache_ram": -1,
        "slot_save": False,
        "moe_mode": "不分配",
        "moe_layers": 62,
        "moe_gpu_count": 3,
        "moe_ratio0": 2,
        "moe_ratio1": 1,
        "moe_ratio2": 1,
        "moe_cpu_total": 0,
        "moe_cpu_layers": 0,
        "moe_ts": "",
        "override_tensor": "",
        "lookup_enabled": False,
        "lookup_cache": "lookup_cache.bin",
        "lookup_static": "",
        "draft_max": 16,
        "draft_min": 2,
        "custom_args": "",
        "auto_start": False,
        "minimize_to_tray": False,
        "window_geometry": "1150x700",
        "remote_host": DEFAULT_REMOTE_HOST,
        "remote_user": DEFAULT_REMOTE_USER,
        "remote_password": "admin",
        "remote_llama_dir": DEFAULT_REMOTE_LLAMA_DIR,
        "remote_model_dir": DEFAULT_REMOTE_MODEL_DIR,
        "remote_model": "",
        "remote_mmproj": "",
        "remote_port": 8080,
        "remote_context": 32768,
        "remote_gpu_layers": 99,
        "remote_parallel": 4,
        "remote_main_gpu": "Auto",
        "remote_temp": 0.8,
        "remote_top_p": 0.6,
        "remote_top_k": 2,
        "remote_repeat_penalty": 1.1,
        "remote_batch_size": 512,
        "remote_ubatch_size": 512,
        "remote_cache_type_k": "f16",
        "remote_cache_type_v": "f16",
        "remote_cache_reuse": 0,
        "remote_cache_ram": 8192,
        "remote_slot_save": False,
        "remote_cuda_graph_opt": False,
        "remote_fit": True,
        "remote_fit_target": 1024,
        "remote_no_kv_offload": False,
        "remote_custom_args": "",
        "multi_instance_enabled": False,
        "multi_instance_count": 3,
        "multi_start_port": 8080,
        "remote_multi_instance_enabled": False,
        "remote_multi_start_port": 8080,
        "remote_multi_instance_configs": {},
    }

    def __init__(self, config_path: str):
        self.config_path = config_path
        self.config = self.DEFAULT_CONFIG.copy()
        self.load()

    def load(self) -> None:
        """Load configuration from file."""
        if os.path.exists(self.config_path):
            try:
                with open(self.config_path, "r", encoding="utf-8") as f:
                    saved_config = json.load(f)
                    self.config.update(saved_config)
            except (json.JSONDecodeError, IOError) as e:
                print(f"Error loading config: {e}")

    def save(self) -> None:
        """Save configuration to file."""
        try:
            with open(self.config_path, "w", encoding="utf-8") as f:
                json.dump(self.config, f, indent=2, ensure_ascii=False)
        except IOError as e:
            print(f"Error saving config: {e}")

    def get(self, key: str, default: Any = None) -> Any:
        """Get configuration value."""
        return self.config.get(key, default)

    def set(self, key: str, value: Any) -> None:
        """Set configuration value."""
        self.config[key] = value

    def update(self, data: Dict[str, Any]) -> None:
        """Update multiple configuration values."""
        self.config.update(data)


class ProcessManager:
    """Manager for llama-server process lifecycle."""

    def __init__(self, logger: logging.Logger):
        self.logger = logger
        self.process: Optional[subprocess.Popen] = None
        self.output_queue: queue.Queue = queue.Queue()
        self.reader_thread: Optional[threading.Thread] = None
        self.running = False

    def build_command(self, config: ServerConfig) -> List[str]:
        """Build the command line for llama-server."""
        llama_dir = config.get("llama_dir")
        exe_path = os.path.join(llama_dir, "build", "bin", "Release", "llama-server.exe")

        if not os.path.exists(exe_path):
            raise FileNotFoundError(f"llama-server.exe not found at: {exe_path}")

        model_path = os.path.join(llama_dir, "models", config.get("model_file"))
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found: {model_path}")

        cmd = [exe_path]
        cmd.extend(["-m", model_path])

        mmproj_file = config.get("mmproj_file")
        if mmproj_file:
            mmproj_path = os.path.join(llama_dir, "models", mmproj_file)
            if os.path.exists(mmproj_path):
                cmd.extend(["--mmproj", mmproj_path])

        cmd.extend(["--host", config.get("host", "0.0.0.0")])
        cmd.extend(["--port", str(config.get("port", 8080))])
        cmd.extend(["-c", str(config.get("context_size", 32768))])
        cmd.extend(["-ngl", str(config.get("gpu_layers", 99))])

        parallel = config.get("parallel", 1)
        if parallel > 1:
            cmd.extend(["-np", str(parallel)])

        main_gpu = config.get("main_gpu", "Auto")
        if isinstance(main_gpu, int):
            main_gpu = "Auto" if main_gpu < 0 else str(main_gpu)
        if main_gpu and main_gpu != "Auto":
            if "," in str(main_gpu):
                cuda_devs = ",".join([f"CUDA{d.strip()}" for d in str(main_gpu).split(",")])
                cmd.extend(["-dev", cuda_devs])
            else:
                cmd.extend(["--split-mode", "none", "-mg", str(main_gpu)])

        cmd.extend(["--temp", str(config.get("temperature", 0.8))])
        cmd.extend(["--top-p", str(config.get("top_p", 0.6))])
        cmd.extend(["--top-k", str(config.get("top_k", 2))])
        cmd.extend(["--repeat-penalty", str(config.get("repeat_penalty", 1.1))])

        threads = config.get("threads", 0)
        if threads > 0:
            cmd.extend(["-t", str(threads)])

        batch_size = config.get("batch_size", 512)
        if batch_size > 0:
            cmd.extend(["-b", str(batch_size)])

        ubatch_size = config.get("ubatch_size", 512)
        if ubatch_size > 0:
            cmd.extend(["-ub", str(ubatch_size)])

        if config.get("use_jinja", True):
            cmd.append("--jinja")

        if config.get("use_flash_attn", False):
            cmd.extend(["--flash-attn", "on"])

        if config.get("split_mode_row", False):
            cmd.extend(["--split-mode", "row"])

        if config.get("no_mmap", False):
            cmd.append("--no-mmap")

        if config.get("fit", False):
            cmd.extend(["--fit", "on"])
            fit_target = config.get("fit_target", 1024)
            if fit_target > 0:
                cmd.extend(["--fit-target", str(fit_target)])

        if config.get("no_kv_offload", False):
            cmd.append("-nkvo")

        cache_type_k = config.get("cache_type_k", "f16")
        if cache_type_k and cache_type_k != "f16":
            cmd.extend(["--cache-type-k", cache_type_k])

        cache_type_v = config.get("cache_type_v", "f16")
        if cache_type_v and cache_type_v != "f16":
            cmd.extend(["--cache-type-v", cache_type_v])

        cache_reuse = config.get("cache_reuse", 0)
        if cache_reuse > 0:
            cmd.extend(["--cache-reuse", str(cache_reuse)])

        cache_ram = config.get("cache_ram", -1)
        if cache_ram >= 0:
            cmd.extend(["--cache-ram", str(cache_ram)])

        if config.get("slot_save", False):
            llama_dir = config.get("llama_dir")
            slot_save_path = os.path.join(llama_dir, "cache")
            cmd.extend(["--slot-save-path", slot_save_path])

        # MoE expert GPU allocation
        override_tensor = config.get("override_tensor", "")
        moe_ts = ""
        if override_tensor.startswith("__CPU_MOE__"):
            cmd.append("--cpu-moe")
            if "|" in override_tensor:
                moe_ts = override_tensor.split("|", 1)[1]
        elif override_tensor.startswith("__NCPU_MOE_"):
            moe_part = override_tensor.split("|")[0]
            n = moe_part[len("__NCPU_MOE_"):-2]
            cmd.extend(["--n-cpu-moe", n])
            if "|" in override_tensor:
                moe_ts = override_tensor.split("|", 1)[1]
        if moe_ts:
            cmd.extend(["-ts", moe_ts])
            cmd.extend(["--split-mode", "layer"])

        moe_ratios = [
            config.get("moe_ratio0", 0),
            config.get("moe_ratio1", 0),
            config.get("moe_ratio2", 0),
        ]
        moe_gpu_count = config.get("moe_gpu_count", 0)
        fit = config.get("fit", False)
        if override_tensor == "__WEIGHTED_SPLIT__" and any(r > 0 for r in moe_ratios) and not fit:
            ratio_str = ",".join(str(r) for r in moe_ratios)
            cmd.extend(["-ts", ratio_str])
            cmd.extend(["--split-mode", "layer"])
        elif override_tensor == "__AUTO_SPLIT__" and moe_gpu_count > 0 and not fit:
            cmd.extend(["-ts", ",".join(["1"] * moe_gpu_count)])
            cmd.extend(["--split-mode", "layer"])
        elif override_tensor and not override_tensor.startswith("__"):
            for rule in override_tensor.split(","):
                rule = rule.strip()
                if rule and "=" in rule:
                    cmd.extend(["-ot", rule])

        # Lookup decoding (speculative)
        if config.get("lookup_enabled", False):
            llama_dir = config.get("llama_dir")
            cache_dir = os.path.join(llama_dir, "cache")
            lookup_cache = config.get("lookup_cache", "")
            if lookup_cache:
                cmd.extend(["-lcd", os.path.join(cache_dir, lookup_cache)])
            lookup_static = config.get("lookup_static", "")
            if lookup_static:
                cmd.extend(["-lcs", os.path.join(cache_dir, lookup_static)])
            draft_max = config.get("draft_max", 0)
            if draft_max > 0:
                cmd.extend(["--draft-max", str(draft_max)])
            draft_min = config.get("draft_min", 0)
            if draft_min > 0:
                cmd.extend(["--draft-min", str(draft_min)])

        cmd.append("--verbose")

        custom_args = config.get("custom_args", "").strip()
        if custom_args:
            cmd.extend(custom_args.split())

        return cmd

    def start(self, config: ServerConfig) -> bool:
        """Start the llama-server process."""
        if self.running:
            self.logger.warning("Server is already running")
            return False

        try:
            cmd = self.build_command(config)
            self.logger.info(f"Starting server with command: {' '.join(cmd)}")

            startupinfo = subprocess.STARTUPINFO()
            startupinfo.dwFlags |= subprocess.STARTF_USESHOWWINDOW
            startupinfo.wShowWindow = subprocess.SW_HIDE

            env = os.environ.copy()
            if config.get("cuda_graph_opt", False):
                env["GGML_CUDA_GRAPH_OPT"] = "1"

            self.process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                stdin=subprocess.PIPE,
                bufsize=1,
                universal_newlines=True,
                startupinfo=startupinfo,
                creationflags=subprocess.CREATE_NO_WINDOW,
                env=env,
                cwd=config.get("llama_dir")
            )

            self.running = True
            self.reader_thread = threading.Thread(
                target=self._read_output,
                daemon=True
            )
            self.reader_thread.start()

            self.logger.info(f"Server started with PID: {self.process.pid}")
            return True

        except Exception as e:
            self.logger.error(f"Failed to start server: {e}")
            self.running = False
            return False

    def stop(self) -> bool:
        """Stop the llama-server process."""
        if not self.running or not self.process:
            self.logger.warning("Server is not running")
            return False

        try:
            self.logger.info("Stopping server...")
            self.running = False

            self.process.terminate()

            try:
                self.process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self.logger.warning("Server did not terminate, killing...")
                self.process.kill()
                self.process.wait(timeout=3)

            self.process = None
            self.logger.info("Server stopped successfully")
            return True

        except Exception as e:
            self.logger.error(f"Failed to stop server: {e}")
            return False

    def is_running(self) -> bool:
        """Check if the server process is running."""
        if self.process is None:
            return False

        poll_result = self.process.poll()
        if poll_result is not None:
            self.running = False
            return False

        return self.running

    def get_output(self) -> Optional[str]:
        """Get pending output from the server."""
        lines = []
        while True:
            try:
                line = self.output_queue.get_nowait()
                lines.append(line)
            except queue.Empty:
                break

        return "".join(lines) if lines else None

    def _read_output(self) -> None:
        """Read output from the server process."""
        try:
            while self.running and self.process:
                line = self.process.stdout.readline()
                if line:
                    self.output_queue.put(line)
                elif self.process.poll() is not None:
                    break
        except Exception as e:
            self.logger.error(f"Error reading output: {e}")


class MultiInstanceProcessManager:
    """Manager for multiple llama-server process instances on different ports."""

    def __init__(self, logger: logging.Logger):
        self.logger = logger
        self.processes: Dict[int, subprocess.Popen] = {}
        self.output_queues: Dict[int, queue.Queue] = {}
        self.reader_threads: Dict[int, threading.Thread] = {}
        self.running_ports: set = set()

    def build_command(self, config: ServerConfig, port: int) -> List[str]:
        """Build the command line for llama-server with specific port."""
        llama_dir = config.get("llama_dir")
        exe_path = os.path.join(llama_dir, "build", "bin", "Release", "llama-server.exe")

        if not os.path.exists(exe_path):
            raise FileNotFoundError(f"llama-server.exe not found at: {exe_path}")

        model_path = os.path.join(llama_dir, "models", config.get("model_file"))
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found: {model_path}")

        cmd = [exe_path]
        cmd.extend(["-m", model_path])

        mmproj_file = config.get("mmproj_file")
        if mmproj_file:
            mmproj_path = os.path.join(llama_dir, "models", mmproj_file)
            if os.path.exists(mmproj_path):
                cmd.extend(["--mmproj", mmproj_path])

        cmd.extend(["--host", config.get("host", "0.0.0.0")])
        cmd.extend(["--port", str(port)])
        cmd.extend(["-c", str(config.get("context_size", 32768))])
        cmd.extend(["-ngl", str(config.get("gpu_layers", 99))])

        main_gpu = config.get("main_gpu", -1)
        if main_gpu >= 0:
            cmd.extend(["--main-gpu", str(main_gpu)])
            cmd.extend(["--split-mode", "none"])

        cmd.extend(["--temp", str(config.get("temperature", 0.8))])
        cmd.extend(["--top-p", str(config.get("top_p", 0.6))])
        cmd.extend(["--top-k", str(config.get("top_k", 2))])
        cmd.extend(["--repeat-penalty", str(config.get("repeat_penalty", 1.1))])

        threads = config.get("threads", 0)
        if threads > 0:
            cmd.extend(["-t", str(threads)])

        batch_size = config.get("batch_size", 512)
        if batch_size > 0:
            cmd.extend(["-b", str(batch_size)])

        ubatch_size = config.get("ubatch_size", 512)
        if ubatch_size > 0:
            cmd.extend(["-ub", str(ubatch_size)])

        if config.get("use_jinja", True):
            cmd.append("--jinja")

        if config.get("use_flash_attn", False):
            cmd.extend(["--flash-attn", "on"])

        if config.get("split_mode_row", False):
            cmd.extend(["--split-mode", "row"])

        if config.get("no_mmap", False):
            cmd.append("--no-mmap")

        cache_type_k = config.get("cache_type_k", "f16")
        if cache_type_k and cache_type_k != "f16":
            cmd.extend(["--cache-type-k", cache_type_k])

        cache_type_v = config.get("cache_type_v", "f16")
        if cache_type_v and cache_type_v != "f16":
            cmd.extend(["--cache-type-v", cache_type_v])

        cmd.append("--verbose")

        custom_args = config.get("custom_args", "").strip()
        if custom_args:
            cmd.extend(custom_args.split())

        return cmd

    def start_instance(self, config: ServerConfig, port: int) -> bool:
        """Start a single llama-server instance on specified port."""
        if port in self.running_ports:
            self.logger.warning(f"Instance on port {port} is already running")
            return False

        try:
            cmd = self.build_command(config, port)
            self.logger.info(f"Starting instance on port {port}: {' '.join(cmd)}")

            startupinfo = subprocess.STARTUPINFO()
            startupinfo.dwFlags |= subprocess.STARTF_USESHOWWINDOW
            startupinfo.wShowWindow = subprocess.SW_HIDE

            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                stdin=subprocess.PIPE,
                bufsize=1,
                universal_newlines=True,
                startupinfo=startupinfo,
                creationflags=subprocess.CREATE_NO_WINDOW,
                cwd=config.get("llama_dir")
            )

            self.processes[port] = process
            self.output_queues[port] = queue.Queue()
            self.running_ports.add(port)

            reader_thread = threading.Thread(
                target=self._read_output,
                args=(port,),
                daemon=True
            )
            self.reader_threads[port] = reader_thread
            reader_thread.start()

            self.logger.info(f"Instance on port {port} started with PID: {process.pid}")
            return True

        except Exception as e:
            self.logger.error(f"Failed to start instance on port {port}: {e}")
            return False

    def start_all(self, config: ServerConfig, count: int, start_port: int,
                  progress_callback=None) -> Dict[int, bool]:
        """Start multiple instances starting from start_port."""
        results = {}
        for i in range(count):
            port = start_port + i
            success = self.start_instance(config, port)
            results[port] = success
            if progress_callback:
                progress_callback(port, success, i + 1, count)
            if i < count - 1:
                time.sleep(2)
        return results

    def stop_instance(self, port: int) -> bool:
        """Stop a specific instance by port."""
        if port not in self.processes:
            self.logger.warning(f"No instance running on port {port}")
            return False

        try:
            self.logger.info(f"Stopping instance on port {port}...")
            process = self.processes[port]

            process.terminate()
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self.logger.warning(f"Instance on port {port} did not terminate, killing...")
                process.kill()
                process.wait(timeout=3)

            del self.processes[port]
            if port in self.output_queues:
                del self.output_queues[port]
            if port in self.reader_threads:
                del self.reader_threads[port]
            self.running_ports.discard(port)

            self.logger.info(f"Instance on port {port} stopped successfully")
            return True

        except Exception as e:
            self.logger.error(f"Failed to stop instance on port {port}: {e}")
            return False

    def stop_all(self) -> Dict[int, bool]:
        """Stop all running instances."""
        results = {}
        ports_to_stop = list(self.running_ports)
        for port in ports_to_stop:
            results[port] = self.stop_instance(port)
        return results

    def is_instance_running(self, port: int) -> bool:
        """Check if a specific instance is running."""
        if port not in self.processes:
            return False

        process = self.processes[port]
        poll_result = process.poll()
        if poll_result is not None:
            self.running_ports.discard(port)
            return False

        return port in self.running_ports

    def get_running_instances(self) -> List[Dict[str, Any]]:
        """Get list of running instances with their status."""
        instances = []
        ports_to_check = list(self.running_ports)

        for port in ports_to_check:
            if port in self.processes:
                process = self.processes[port]
                poll_result = process.poll()

                if poll_result is None:
                    instances.append({
                        "port": port,
                        "status": "running",
                        "pid": process.pid
                    })
                else:
                    self.running_ports.discard(port)
                    instances.append({
                        "port": port,
                        "status": "stopped",
                        "pid": None
                    })

        return instances

    def get_running_count(self) -> int:
        """Get count of currently running instances."""
        count = 0
        for port in list(self.running_ports):
            if self.is_instance_running(port):
                count += 1
        return count

    def get_all_output(self) -> Dict[int, str]:
        """Get pending output from all instances."""
        outputs = {}
        for port, q in self.output_queues.items():
            lines = []
            while True:
                try:
                    line = q.get_nowait()
                    lines.append(line)
                except queue.Empty:
                    break
            if lines:
                outputs[port] = "".join(lines)
        return outputs

    def get_instance_pid(self, port: int) -> Optional[int]:
        """Get PID of instance on specified port."""
        if port in self.processes and self.is_instance_running(port):
            return self.processes[port].pid
        return None

    def get_server_urls(self, host: str = "localhost") -> List[str]:
        """Get list of server URLs for all running instances."""
        urls = []
        for port in sorted(self.running_ports):
            if self.is_instance_running(port):
                urls.append(f"http://{host}:{port}")
        return urls

    def _read_output(self, port: int) -> None:
        """Read output from a specific instance."""
        try:
            process = self.processes.get(port)
            q = self.output_queues.get(port)

            while port in self.running_ports and process and q:
                line = process.stdout.readline()
                if line:
                    q.put(f"[Port {port}] {line}")
                elif process.poll() is not None:
                    break
        except Exception as e:
            self.logger.error(f"Error reading output from port {port}: {e}")


class RemoteProcessManager:
    """Manager for remote llama-server process via WinRM."""

    def __init__(self, logger: logging.Logger):
        self.logger = logger
        self.connected = False
        self.remote_host = ""
        self.remote_user = ""
        self.remote_password = "admin"

    def set_credentials(self, host: str, user: str, password: str) -> None:
        """Set remote connection credentials."""
        self.remote_host = host
        self.remote_user = user
        self.remote_password = password

    def _run_remote_command(self, script: str, timeout: int = 60) -> tuple:
        """Execute PowerShell command on remote host via WinRM."""
        ps_command = f'''
$secpasswd = ConvertTo-SecureString "{self.remote_password}" -AsPlainText -Force
$cred = New-Object System.Management.Automation.PSCredential ("{self.remote_user}", $secpasswd)
Invoke-Command -ComputerName {self.remote_host} -Credential $cred -ScriptBlock {{ {script} }}
'''
        try:
            startupinfo = subprocess.STARTUPINFO()
            startupinfo.dwFlags |= subprocess.STARTF_USESHOWWINDOW
            startupinfo.wShowWindow = subprocess.SW_HIDE

            result = subprocess.run(
                ["powershell", "-NoProfile", "-Command", ps_command],
                capture_output=True,
                text=True,
                timeout=timeout,
                startupinfo=startupinfo,
                creationflags=subprocess.CREATE_NO_WINDOW
            )
            stdout = result.stdout.strip() if result.stdout else ""
            stderr = result.stderr.strip() if result.stderr else ""
            return result.returncode == 0, stdout, stderr
        except subprocess.TimeoutExpired:
            return False, "", "Command timed out"
        except Exception as e:
            return False, "", str(e)

    def test_connection(self) -> tuple:
        """Test remote connection."""
        success, stdout, stderr = self._run_remote_command("hostname; Get-Date", timeout=30)
        if success and stdout:
            self.connected = True
            return True, stdout
        self.connected = False
        return False, stderr or "Connection failed"

    def scan_models(self, model_dir: str) -> tuple:
        """Scan for model files on remote server.
        
        Shows first shard only: *00001-of-* or single files.
        Filters out non-model gguf files (imatrix, mtmd, tokenizer, etc.)
        """
        script = f'''
Get-ChildItem "{model_dir}" -Filter "*.gguf" -ErrorAction SilentlyContinue |
Where-Object {{ 
    $name = $_.Name.ToLower()
    $isModel = ($_.Name -notmatch "-of-" -or $_.Name -match "00001-of-") -and
               -not ($name -match "imatrix" -or $name -match "mtmd" -or $name -match "tokenizer" -or $name -match "vocab" -or $name -match "encoder" -or $name -match "decoder")
    if ($isModel) {{ $_.Name }}
}} |
Select-Object -ExpandProperty Name
'''
        success, stdout, stderr = self._run_remote_command(script, timeout=60)
        if success:
            models = [m.strip() for m in stdout.split('\n') if m.strip() and m.strip().endswith('.gguf')]
            return True, models
        return False, []

    def scan_mmproj(self, model_dir: str) -> tuple:
        """Scan for mmproj files on remote server."""
        script = f'''
Get-ChildItem "{model_dir}" -Filter "*mmproj*.gguf" -ErrorAction SilentlyContinue |
Select-Object -ExpandProperty Name
'''
        success, stdout, stderr = self._run_remote_command(script, timeout=60)
        if success:
            models = [m.strip() for m in stdout.split('\n') if m.strip() and m.strip().endswith('.gguf')]
            return True, models
        return False, []

    def start_server(self, llama_dir: str, model_dir: str, model: str, mmproj: str,
                     port: int, context: int, gpu_layers: int, parallel: int,
                     use_jinja: bool = True, use_flash_attn: bool = False,
                     split_mode_row: bool = False, no_mmap: bool = False,
                     fit: bool = False, fit_target: int = 1024, no_kv_offload: bool = False,
                     gpu_select: str = "Auto", batch_size: int = 512, ubatch_size: int = 512,
                     cache_type_k: str = "f16", cache_type_v: str = "f16",
                     cache_reuse: int = 0, cache_ram: int = 8192,
                     slot_save: bool = False, cuda_graph_opt: bool = False,
                     temp: float = 0.8, top_p: float = 0.6, top_k: int = 2,
                     repeat_penalty: float = 1.1, custom_args: str = "",
                     override_tensor: str = "",
                     moe_layers: int = 0, moe_gpu_count: int = 0,
                     moe_target_gpu: int = -1,
                     moe_ratios: list = None,
                     lookup_cache_dynamic: str = "",
                     lookup_cache_static: str = "",
                     draft_max: int = 0,
                     draft_min: int = 0) -> tuple:
        """Start llama-server on remote host using Windows Scheduled Task.

        Args:
            gpu_select: GPU selection - "Auto", single GPU "0"/"1", or multi-GPU "0,1"/"0,1,2"
            override_tensor: MoE mode - "__TARGET_GPU__", "__AUTO_SPLIT__", "__WEIGHTED_SPLIT__", custom rules, or empty
            moe_target_gpu: Target GPU index for __TARGET_GPU__ mode
            moe_ratios: Weight ratios per GPU for __WEIGHTED_SPLIT__ mode, e.g. [2, 1, 1]
            lookup_cache_dynamic: Path for dynamic lookup cache file (-lcd)
            lookup_cache_static: Path for static lookup cache file (-lcs)
            draft_max: Max tokens to draft per speculative step (--draft-max)
            draft_min: Min tokens to draft per speculative step (--draft-min)
        """
        llama_dir_win = llama_dir.replace("/", "\\")
        model_dir_win = model_dir.replace("/", "\\")

        exe_path = f"{llama_dir_win}\\build\\bin\\Release\\llama-server.exe"
        model_path = f"{model_dir_win}\\{model}"
        batch_path = f"{llama_dir_win}\\start_server.bat"

        args_list = [
            f'-m "{model_path}"',
            f'-ngl {gpu_layers if not fit else -1}',
            f'-c {context}',
            '--host 0.0.0.0',
            f'-np {parallel}',
            f'-b {batch_size}',
            f'-ub {ubatch_size}',
            f'--temp {temp:.1f}',
            f'--top-p {top_p:.1f}',
            f'--top-k {top_k}',
            f'--repeat-penalty {repeat_penalty:.2f}',
            '--verbose'
        ]

        # GPU selection: Auto, single GPU (0/1/2), or multi-GPU (0,1/0,1,2)
        if gpu_select and gpu_select != "Auto":
            if "," in gpu_select:
                # Multi-GPU: use -dev parameter with CUDA device names
                # Convert "0,1" to "CUDA0,CUDA1"
                cuda_devs = ",".join([f"CUDA{d.strip()}" for d in gpu_select.split(",")])
                args_list.append(f'-dev {cuda_devs}')
            else:
                # Single GPU: use --split-mode none -mg
                args_list.append(f'--split-mode none -mg {gpu_select}')

        if use_jinja:
            args_list.append('--jinja')

        if use_flash_attn:
            args_list.append('--flash-attn on')

        if fit:
            args_list.append('--fit on')
            if fit_target > 0:
                args_list.append(f'--fit-target {fit_target}')

        if no_kv_offload:
            args_list.append('-nkvo')

        # MoE expert CPU offload (format: __CPU_MOE__|ts or __NCPU_MOE_N__|ts)
        moe_cpu_ts = ""
        if override_tensor.startswith("__CPU_MOE__"):
            args_list.append('--cpu-moe')
            if "|" in override_tensor:
                moe_cpu_ts = override_tensor.split("|", 1)[1]
        elif override_tensor.startswith("__NCPU_MOE_"):
            moe_part = override_tensor.split("|")[0]
            n = moe_part[len("__NCPU_MOE_"):-2]
            args_list.append(f'--n-cpu-moe {n}')
            if "|" in override_tensor:
                moe_cpu_ts = override_tensor.split("|", 1)[1]
        if moe_cpu_ts:
            args_list.append(f'-ts {moe_cpu_ts}')
            args_list.append('--split-mode layer')

        # When using weighted/auto GPU split, force --split-mode layer (overrides row mode)
        if override_tensor == "__WEIGHTED_SPLIT__" and moe_ratios and not fit:
            ratio_str = ",".join(str(r) for r in moe_ratios)
            args_list.append(f'-ts {ratio_str}')
            args_list.append('--split-mode layer')
        elif override_tensor == "__AUTO_SPLIT__" and moe_gpu_count > 0 and not fit:
            args_list.append(f'-ts {",".join(["1"] * moe_gpu_count)}')
            args_list.append('--split-mode layer')
        elif split_mode_row and not fit:
            args_list.append('--split-mode row')

        if no_mmap:
            args_list.append('--no-mmap')

        if cache_type_k:
            args_list.append(f'--cache-type-k {cache_type_k}')

        if cache_type_v:
            args_list.append(f'--cache-type-v {cache_type_v}')

        if cache_reuse > 0:
            args_list.append(f'--cache-reuse {cache_reuse}')

        if cache_ram >= 0:
            args_list.append(f'--cache-ram {cache_ram}')

        if slot_save:
            slot_save_path = f"{llama_dir_win}\\cache"
            args_list.append(f'--slot-save-path "{slot_save_path}"')

        if mmproj:
            mmproj_path = f"{model_dir_win}\\{mmproj}"
            args_list.append(f'--mmproj "{mmproj_path}"')

        if override_tensor and not override_tensor.startswith("__"):
            for rule in override_tensor.split(","):
                rule = rule.strip()
                if rule and "=" in rule:
                    args_list.append(f'-ot "{rule}"')

        # Lookup decoding (cache-based speculative decoding)
        if lookup_cache_dynamic:
            cache_dir = f"{llama_dir_win}\\cache"
            lcd_path = f"{cache_dir}\\{lookup_cache_dynamic}"
            args_list.append(f'-lcd "{lcd_path}"')
        if lookup_cache_static:
            cache_dir = f"{llama_dir_win}\\cache"
            lcs_path = f"{cache_dir}\\{lookup_cache_static}"
            args_list.append(f'-lcs "{lcs_path}"')
        if draft_max > 0:
            args_list.append(f'--draft-max {draft_max}')
        if draft_min > 0:
            args_list.append(f'--draft-min {draft_min}')

        if custom_args:
            args_list.append(custom_args)

        args = ' '.join(args_list)

        # CUDA Graph optimization environment variable
        cuda_env_line = "set GGML_CUDA_GRAPH_OPT=1" if cuda_graph_opt else ""

        script = f'''
# Only stop the main instance (by port), not multi-instances
$existingProc = Get-NetTCPConnection -LocalPort {port} -ErrorAction SilentlyContinue |
    Select-Object -ExpandProperty OwningProcess -First 1
if ($existingProc) {{
    Stop-Process -Id $existingProc -Force -ErrorAction SilentlyContinue
    Start-Sleep -Seconds 2
}}

# Remove old scheduled task for main instance only
Unregister-ScheduledTask -TaskName "LlamaServer_Main" -Confirm:$false -ErrorAction SilentlyContinue

        # Debug: Show the command that will be executed
Write-Output "=== DEBUG: Command to execute ==="
Write-Output "Exe: {exe_path}"
Write-Output "Args: {args}"
Write-Output "Fit: {fit} FitTarget: {fit_target}MiB"
Write-Output "============================="

# Create cache directory if needed
if (-not (Test-Path "{llama_dir_win}\\cache")) {{
    New-Item -ItemType Directory -Path "{llama_dir_win}\\cache" -Force | Out-Null
}}

# Create batch file - redirect all output to log file
$logFile = "{llama_dir_win}\\llama_server.log"
$batchContent = @"
@echo off
cd /d {llama_dir_win}
if not exist "{llama_dir_win}\\cache" mkdir "{llama_dir_win}\\cache"
set PATH=%PATH%;C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v13.1\\bin;{llama_dir_win}\\build\\bin
{cuda_env_line}
echo [%date% %time%] Starting llama-server... > "$logFile"
echo Command: {exe_path} {args} >> "$logFile"
echo ============================================ >> "$logFile"
{exe_path} {args} >> "$logFile" 2>&1
echo [%date% %time%] Server exited with code %errorlevel% >> "$logFile"
"@
$batchContent | Out-File -FilePath "{batch_path}" -Encoding ASCII

# Clear old log
if (Test-Path $logFile) {{ Remove-Item $logFile -Force }}

# Create scheduled task
$action = New-ScheduledTaskAction -Execute "{batch_path}" -WorkingDirectory "{llama_dir_win}"
$trigger = New-ScheduledTaskTrigger -Once -At (Get-Date).AddSeconds(2)
$settings = New-ScheduledTaskSettingsSet -AllowStartIfOnBatteries -DontStopIfGoingOnBatteries -StartWhenAvailable -ExecutionTimeLimit (New-TimeSpan -Hours 24)
$principal = New-ScheduledTaskPrincipal -UserId "Administrator" -LogonType Interactive -RunLevel Highest

Register-ScheduledTask -TaskName "LlamaServer_Main" -Action $action -Trigger $trigger -Settings $settings -Principal $principal -Force | Out-Null
Start-ScheduledTask -TaskName "LlamaServer_Main"

# Poll for server to start (large MoE models need time to load)
$maxWait = 180
$waited = 0
$procId = $null
while ($waited -lt $maxWait) {{
    Start-Sleep -Seconds 5
    $waited += 5
    $procId = Get-NetTCPConnection -LocalPort {port} -ErrorAction SilentlyContinue |
        Select-Object -ExpandProperty OwningProcess -First 1
    if ($procId) {{ break }}
    # Check if process crashed
    $llProc = Get-Process -Name "llama-server" -ErrorAction SilentlyContinue
    if (-not $llProc) {{
        $taskState = (Get-ScheduledTask -TaskName "LlamaServer_Main" -ErrorAction SilentlyContinue).State
        if ($taskState -ne "Running") {{ break }}
    }}
    "Waiting for model to load... (${{waited}}s/${{maxWait}}s)"
}}

if ($procId) {{
    "Started PID: $procId on port {port} (loaded in ${{waited}}s)"
}} else {{
    "FAILED: Server did not start within ${{maxWait}}s"
    "--- Batch file ---"
    if (Test-Path "{batch_path}") {{ Get-Content "{batch_path}" }}
    "--- Server log (last 50 lines) ---"
    if (Test-Path $logFile) {{
        Get-Content $logFile -Tail 50
    }} else {{
        "(no log file found - server may not have started at all)"
    }}
}}
'''
        success, stdout, stderr = self._run_remote_command(script, timeout=240)
        combined = (stdout or "") + (stderr or "")
        if "Started PID" in combined:
            return True, combined
        return False, combined if combined else "Failed to start server"

    def stop_server(self, port: int = 8080) -> tuple:
        """Stop llama-server on remote host by port and clean up scheduled task."""
        script = f'''
# Stop the main instance only (by port)
$procId = Get-NetTCPConnection -LocalPort {port} -ErrorAction SilentlyContinue |
    Select-Object -ExpandProperty OwningProcess -First 1
if ($procId) {{
    Stop-Process -Id $procId -Force -ErrorAction SilentlyContinue
    "Stopped process PID: $procId on port {port}"
}} else {{
    "No process found on port {port}"
}}

# Remove the main scheduled task only
Unregister-ScheduledTask -TaskName "LlamaServer_Main" -Confirm:$false -ErrorAction SilentlyContinue
"Scheduled task LlamaServer_Main removed"
'''
        success, stdout, stderr = self._run_remote_command(script, timeout=30)
        return success, stdout or stderr

    def check_server_status(self, port: int) -> tuple:
        """Check if server is running on remote host by port."""
        script = f'''
$procId = Get-NetTCPConnection -LocalPort {port} -ErrorAction SilentlyContinue |
    Select-Object -ExpandProperty OwningProcess -First 1
if ($procId) {{
    $proc = Get-Process -Id $procId -ErrorAction SilentlyContinue
    if ($proc) {{
        $mem = [math]::Round($proc.WorkingSet64/1GB, 2)
        "Running|PID:$procId|Mem:$($mem)GB|Port:{port}"
    }} else {{
        "Running|PID:$procId|Port:{port}"
    }}
}} else {{
    "Stopped"
}}
'''
        success, stdout, stderr = self._run_remote_command(script, timeout=15)
        if success:
            return True, stdout.strip()
        return False, "Unknown"

    def get_remote_log(self, llama_dir: str, tail_lines: int = 100) -> tuple:
        """Get remote server log file content."""
        llama_dir_win = llama_dir.replace("/", "\\")
        log_path = f"{llama_dir_win}\\llama_server.log"

        script = f'''
$logPath = "{log_path}"
if (Test-Path $logPath) {{
    $content = Get-Content $logPath -Tail {tail_lines} -ErrorAction SilentlyContinue
    if ($content) {{
        $content -join "`n"
    }} else {{
        "[Log file is empty]"
    }}
}} else {{
    "[Log file not found: {log_path}]"
}}
'''
        success, stdout, stderr = self._run_remote_command(script, timeout=15)
        if success:
            return True, stdout
        return False, stderr or "Failed to read log"

    def clear_remote_log(self, llama_dir: str) -> tuple:
        """Clear remote server log file."""
        llama_dir_win = llama_dir.replace("/", "\\")
        log_path = f"{llama_dir_win}\\llama_server.log"

        script = f'''
$logPath = "{log_path}"
if (Test-Path $logPath) {{
    Remove-Item $logPath -Force
    "Log file cleared"
}} else {{
    "Log file not found"
}}
'''
        success, stdout, stderr = self._run_remote_command(script, timeout=10)
        return success, stdout or stderr

    def get_gguf_block_count(self, model_dir: str, model_file: str) -> tuple:
        """Read block_count (number of layers) from GGUF file metadata on remote host."""
        model_dir_win = model_dir.replace("/", "\\")
        model_path = f"{model_dir_win}\\{model_file}"

        script = f'''
$path = "{model_path}"
if (-not (Test-Path $path)) {{ Write-Output "FILE_NOT_FOUND"; exit }}
try {{
    $fs = [System.IO.File]::OpenRead($path)
    $br = New-Object System.IO.BinaryReader($fs)
    $magic = [System.Text.Encoding]::ASCII.GetString($br.ReadBytes(4))
    if ($magic -ne "GGUF") {{ Write-Output "NOT_GGUF"; $br.Close(); exit }}
    $version = $br.ReadUInt32()
    $tensor_count = $br.ReadUInt64()
    $metadata_count = $br.ReadUInt64()
    for ($i = 0; $i -lt [Math]::Min($metadata_count, 500); $i++) {{
        $key_len = $br.ReadUInt64()
        $key = [System.Text.Encoding]::UTF8.GetString($br.ReadBytes([int]$key_len))
        $vtype = $br.ReadUInt32()
        if ($key -match "block_count$") {{
            if ($vtype -eq 4) {{ Write-Output ("BLOCK_COUNT=" + $br.ReadUInt32()) }}
            elseif ($vtype -eq 5) {{ Write-Output ("BLOCK_COUNT=" + $br.ReadInt32()) }}
            elseif ($vtype -eq 10) {{ Write-Output ("BLOCK_COUNT=" + $br.ReadUInt64()) }}
            else {{ Write-Output "UNEXPECTED_TYPE" }}
            $br.Close(); exit
        }}
        switch ($vtype) {{
            0  {{ [void]$br.ReadByte() }}
            1  {{ [void]$br.ReadSByte() }}
            2  {{ [void]$br.ReadUInt16() }}
            3  {{ [void]$br.ReadInt16() }}
            4  {{ [void]$br.ReadUInt32() }}
            5  {{ [void]$br.ReadInt32() }}
            6  {{ [void]$br.ReadSingle() }}
            7  {{ [void]$br.ReadByte() }}
            8  {{ $slen = $br.ReadUInt64(); [void]$br.ReadBytes([int]$slen) }}
            9  {{
                $atype = $br.ReadUInt32(); $acount = $br.ReadUInt64()
                for ($j = 0; $j -lt $acount; $j++) {{
                    switch ($atype) {{
                        0  {{ [void]$br.ReadByte() }}
                        1  {{ [void]$br.ReadSByte() }}
                        2  {{ [void]$br.ReadUInt16() }}
                        3  {{ [void]$br.ReadInt16() }}
                        4  {{ [void]$br.ReadUInt32() }}
                        5  {{ [void]$br.ReadInt32() }}
                        6  {{ [void]$br.ReadSingle() }}
                        7  {{ [void]$br.ReadByte() }}
                        8  {{ $slen2 = $br.ReadUInt64(); [void]$br.ReadBytes([int]$slen2) }}
                        10 {{ [void]$br.ReadUInt64() }}
                        11 {{ [void]$br.ReadInt64() }}
                        12 {{ [void]$br.ReadDouble() }}
                    }}
                }}
            }}
            10 {{ [void]$br.ReadUInt64() }}
            11 {{ [void]$br.ReadInt64() }}
            12 {{ [void]$br.ReadDouble() }}
        }}
    }}
    Write-Output "NOT_FOUND"
    $br.Close()
}} catch {{
    Write-Output ("ERROR:" + $_.Exception.Message)
}}
'''
        success, stdout, stderr = self._run_remote_command(script, timeout=30)
        if success and stdout:
            stdout = stdout.strip()
            if stdout.startswith("BLOCK_COUNT="):
                try:
                    count = int(stdout.split("=")[1])
                    return True, count
                except ValueError:
                    pass
            return False, stdout
        return False, stderr or "Failed to read GGUF metadata"

    def start_multi_instances(self, llama_dir: str, model_dir: str, model: str, mmproj: str,
                               start_port: int, count: int, context: int, gpu_layers: int,
                               parallel: int, use_jinja: bool = True, use_flash_attn: bool = False,
                               split_mode_row: bool = False, no_mmap: bool = False,
                               main_gpu: int = -1, batch_size: int = 512,
                               cache_type_k: str = "f16", cache_type_v: str = "f16",
                               cuda_graph_opt: bool = False,
                               temp: float = 0.8, top_p: float = 0.6, top_k: int = 2,
                               repeat_penalty: float = 1.1, custom_args: str = "",
                               override_tensor: str = "",
                               moe_layers: int = 0, moe_gpu_count: int = 0,
                               moe_target_gpu: int = -1,
                               moe_ratios: list = None,
                               lookup_cache_dynamic: str = "",
                               lookup_cache_static: str = "",
                               draft_max: int = 0,
                               draft_min: int = 0) -> tuple:
        """Start multiple llama-server instances on remote host.

        Creates separate batch files and scheduled tasks for each instance.
        """
        llama_dir_win = llama_dir.replace("/", "\\")
        model_dir_win = model_dir.replace("/", "\\")

        exe_path = f"{llama_dir_win}\\build\\bin\\Release\\llama-server.exe"
        model_path = f"{model_dir_win}\\{model}"
        mmproj_path = f"{model_dir_win}\\{mmproj}" if mmproj else ""

        # Pre-format float parameters for PowerShell
        temp_str = f"{temp:.1f}"
        top_p_str = f"{top_p:.1f}"
        repeat_penalty_str = f"{repeat_penalty:.2f}"

        # Build the script to start multiple instances
        script = f'''
# Stop all existing llama-server processes
Stop-Process -Name "llama-server" -Force -ErrorAction SilentlyContinue
Start-Sleep -Seconds 2

# Remove old scheduled tasks
for ($i = 0; $i -lt {count}; $i++) {{
    $taskName = "LlamaServer_$i"
    Unregister-ScheduledTask -TaskName $taskName -Confirm:$false -ErrorAction SilentlyContinue
}}
Unregister-ScheduledTask -TaskName "LlamaServer" -Confirm:$false -ErrorAction SilentlyContinue

$results = @()

for ($i = 0; $i -lt {count}; $i++) {{
    $port = {start_port} + $i
    $taskName = "LlamaServer_$i"
    $batchPath = "{llama_dir_win}\\start_server_$i.bat"

    # Build arguments list
    $argList = @(
        '-m "{model_path}"',
        '-ngl {gpu_layers}',
        '-c {context}',
        '--host 0.0.0.0',
        '-np {parallel}',
        '-b {batch_size}',
        '--temp {temp_str}',
        '--top-p {top_p_str}',
        '--top-k {top_k}',
        '--repeat-penalty {repeat_penalty_str}',
        '--verbose'
    )
'''
        if main_gpu >= 0:
            script += f'''
    $args += ' -mg {main_gpu}'
'''
        if use_jinja:
            script += '''
    $args += ' --jinja'
'''
        if use_flash_attn:
            script += '''
    $args += '-fa on'
'''
        moe_cpu_ts = ""
        if override_tensor.startswith("__CPU_MOE__"):
            script += '''
    $args += ' --cpu-moe'
'''
            if "|" in override_tensor:
                moe_cpu_ts = override_tensor.split("|", 1)[1]
        elif override_tensor.startswith("__NCPU_MOE_"):
            moe_part = override_tensor.split("|")[0]
            n = moe_part[len("__NCPU_MOE_"):-2]
            script += f'''
    $args += ' --n-cpu-moe {n}'
'''
            if "|" in override_tensor:
                moe_cpu_ts = override_tensor.split("|", 1)[1]
        if moe_cpu_ts:
            script += f'''
    $args += ' -ts {moe_cpu_ts} --split-mode layer'
'''
        if override_tensor == "__WEIGHTED_SPLIT__" and moe_ratios:
            ratio_str = ",".join(str(r) for r in moe_ratios)
            script += f'''
    $args += ' -ts {ratio_str} --split-mode layer'
'''
        elif override_tensor == "__AUTO_SPLIT__" and moe_gpu_count > 0:
            equal_str = ",".join(["1"] * moe_gpu_count)
            script += f'''
    $args += ' -ts {equal_str} --split-mode layer'
'''
        elif split_mode_row:
            script += '''
    $args += ' --split-mode row'
'''
        if no_mmap:
            script += '''
    $args += ' --no-mmap'
'''
        if cache_type_k:
            script += f'''
    $args += ' --cache-type-k {cache_type_k}'
'''
        if cache_type_v:
            script += f'''
    $args += ' --cache-type-v {cache_type_v}'
'''
        if mmproj:
            script += f'''
    $args += ' --mmproj "{mmproj_path}"'
'''
        if override_tensor and not override_tensor.startswith("__"):
            for rule in override_tensor.split(","):
                rule = rule.strip()
                if rule and "=" in rule:
                    script += f'''
    $args += ' -ot "{rule}"'
'''
        # Lookup decoding
        if lookup_cache_dynamic:
            lcd_path = f"{llama_dir_win}\\cache\\{lookup_cache_dynamic}"
            script += f'''
    $args += ' -lcd "{lcd_path}"'
'''
        if lookup_cache_static:
            lcs_path = f"{llama_dir_win}\\cache\\{lookup_cache_static}"
            script += f'''
    $args += ' -lcs "{lcs_path}"'
'''
        if draft_max > 0:
            script += f'''
    $args += ' --draft-max {draft_max}'
'''
        if draft_min > 0:
            script += f'''
    $args += ' --draft-min {draft_min}'
'''
        if custom_args:
            script += f'''
    $args += ' {custom_args}'
'''

        # CUDA Graph optimization environment variable
        cuda_env_line = "set GGML_CUDA_GRAPH_OPT=1" if cuda_graph_opt else ""

        script += f'''
    # Create batch file
    $batchContent = @"
@echo off
cd /d {llama_dir_win}
set PATH=%PATH%;C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v13.1\\bin;{llama_dir_win}\\build\\bin
{cuda_env_line}
{exe_path} $args
"@
    $batchContent | Out-File -FilePath $batchPath -Encoding ASCII

    # Create and start scheduled task
    $action = New-ScheduledTaskAction -Execute $batchPath -WorkingDirectory "{llama_dir_win}"
    $trigger = New-ScheduledTaskTrigger -Once -At (Get-Date).AddSeconds(2)
    $settings = New-ScheduledTaskSettingsSet -AllowStartIfOnBatteries -DontStopIfGoingOnBatteries -StartWhenAvailable -ExecutionTimeLimit (New-TimeSpan -Hours 24)
    $principal = New-ScheduledTaskPrincipal -UserId "Administrator" -LogonType Interactive -RunLevel Highest

    Register-ScheduledTask -TaskName $taskName -Action $action -Trigger $trigger -Settings $settings -Principal $principal -Force | Out-Null
    Start-ScheduledTask -TaskName $taskName

    $results += "Started task for port $port"

    # Wait between starts to avoid resource conflicts
    Start-Sleep -Seconds 3
}}

# Wait for all processes to start
Start-Sleep -Seconds 5

# Check results
$procs = Get-Process -Name "llama-server" -ErrorAction SilentlyContinue
if ($procs) {{
    $count = @($procs).Count
    "SUCCESS: $count instance(s) running"
    foreach ($p in $procs) {{
        "  PID: $($p.Id)"
    }}
}} else {{
    "ERROR: No instances running"
}}
'''
        success, stdout, stderr = self._run_remote_command(script, timeout=120)
        if success and "SUCCESS" in stdout:
            return True, stdout
        error_msg = stderr if stderr else stdout if stdout else "Failed to start instances"
        return False, error_msg

    def start_multi_model_instances(self, llama_dir: str, model_dir: str,
                                     instances: List[Dict[str, Any]],
                                     context: int, gpu_layers: int,
                                     parallel: int, use_jinja: bool = True,
                                     use_flash_attn: bool = False,
                                     fit: bool = False,
                                     split_mode_row: bool = False, no_mmap: bool = False,
                                     batch_size: int = 512,
                                     cache_type_k: str = "f16", cache_type_v: str = "f16",
                                     cuda_graph_opt: bool = False,
                                     temp: float = 0.8, top_p: float = 0.6, top_k: int = 2,
                                     repeat_penalty: float = 1.1, custom_args: str = "",
                                     override_tensor: str = "",
                                     moe_layers: int = 0, moe_gpu_count: int = 0,
                                     moe_target_gpu: int = -1,
                                     moe_ratios: list = None,
                                     lookup_cache_dynamic: str = "",
                                     lookup_cache_static: str = "",
                                     draft_max: int = 0,
                                     draft_min: int = 0) -> tuple:
        """Start multiple llama-server instances with different models on remote host.

        Args:
            instances: List of dicts with {"port": int, "model": str, "mmproj": str, "main_gpu": int}
        """
        llama_dir_win = llama_dir.replace("/", "\\")
        model_dir_win = model_dir.replace("/", "\\")
        exe_path = f"{llama_dir_win}\\build\\bin\\Release\\llama-server.exe"

        # Pre-format float parameters for PowerShell
        temp_str = f"{temp:.1f}"
        top_p_str = f"{top_p:.1f}"
        repeat_penalty_str = f"{repeat_penalty:.2f}"

        # CUDA Graph optimization environment variable
        cuda_env_line = "set GGML_CUDA_GRAPH_OPT=1" if cuda_graph_opt else ""

        # Build base optional args string (without main_gpu, which is per-instance)
        base_optional_args = ""
        if use_jinja:
            base_optional_args += " --jinja"
        if use_flash_attn:
            base_optional_args += " -fa on"
        if fit:
            base_optional_args += " --fit on"
        moe_cpu_ts = ""
        if override_tensor.startswith("__CPU_MOE__") and not fit:
            base_optional_args += " --cpu-moe"
            if "|" in override_tensor:
                moe_cpu_ts = override_tensor.split("|", 1)[1]
        elif override_tensor.startswith("__NCPU_MOE_") and not fit:
            moe_part = override_tensor.split("|")[0]
            n = moe_part[len("__NCPU_MOE_"):-2]
            base_optional_args += f" --n-cpu-moe {n}"
            if "|" in override_tensor:
                moe_cpu_ts = override_tensor.split("|", 1)[1]
        if moe_cpu_ts:
            base_optional_args += f" -ts {moe_cpu_ts} --split-mode layer"
        if override_tensor == "__WEIGHTED_SPLIT__" and moe_ratios and not fit:
            ratio_str = ",".join(str(r) for r in moe_ratios)
            base_optional_args += f' -ts {ratio_str} --split-mode layer'
        elif override_tensor == "__AUTO_SPLIT__" and moe_gpu_count > 0 and not fit:
            equal_str = ",".join(["1"] * moe_gpu_count)
            base_optional_args += f' -ts {equal_str} --split-mode layer'
        elif split_mode_row and not fit:
            base_optional_args += " --split-mode row"
        if no_mmap:
            base_optional_args += " --no-mmap"
        if cache_type_k:
            base_optional_args += f" --cache-type-k {cache_type_k}"
        if cache_type_v:
            base_optional_args += f" --cache-type-v {cache_type_v}"
        if override_tensor and not override_tensor.startswith("__"):
            for rule in override_tensor.split(","):
                rule = rule.strip()
                if rule and "=" in rule:
                    base_optional_args += f' -ot "{rule}"'
        # Lookup decoding
        if lookup_cache_dynamic:
            lcd_path = f"{llama_dir_win}\\cache\\{lookup_cache_dynamic}"
            base_optional_args += f' -lcd "{lcd_path}"'
        if lookup_cache_static:
            lcs_path = f"{llama_dir_win}\\cache\\{lookup_cache_static}"
            base_optional_args += f' -lcs "{lcs_path}"'
        if draft_max > 0:
            base_optional_args += f" --draft-max {draft_max}"
        if draft_min > 0:
            base_optional_args += f" --draft-min {draft_min}"
        if custom_args:
            base_optional_args += f" {custom_args}"

        count = len(instances)

        # Build the script to start multiple instances with different models
        script = f'''
# Stop all existing llama-server processes
Stop-Process -Name "llama-server" -Force -ErrorAction SilentlyContinue
Start-Sleep -Seconds 2

# Remove old scheduled tasks
for ($i = 0; $i -lt 10; $i++) {{
    $taskName = "LlamaServer_$i"
    Unregister-ScheduledTask -TaskName $taskName -Confirm:$false -ErrorAction SilentlyContinue
}}
Unregister-ScheduledTask -TaskName "LlamaServer" -Confirm:$false -ErrorAction SilentlyContinue

$results = @()
'''

        # Add each instance configuration
        for idx, inst in enumerate(instances):
            port = inst["port"]
            model = inst["model"]
            mmproj = inst.get("mmproj", "")
            inst_main_gpu = inst.get("main_gpu", -1)

            model_path = f"{model_dir_win}\\{model}"
            mmproj_arg = f' --mmproj "{model_dir_win}\\{mmproj}"' if mmproj else ""

            # Build per-instance optional args (including main_gpu)
            optional_args = ""
            if inst_main_gpu >= 0:
                # When using specific GPU, must set split-mode to none
                optional_args += f" --split-mode none -mg {inst_main_gpu}"
            optional_args += base_optional_args

            # Use port value directly in batch file, not PowerShell variable
            script += f'''
# Instance {idx}: Port {port}, Model: {model}, GPU: {inst_main_gpu if inst_main_gpu >= 0 else "Auto"}
$taskName_{idx} = "LlamaServer_{idx}"
$batchPath_{idx} = "{llama_dir_win}\\start_server_{idx}.bat"

$batchContent_{idx} = @"
@echo off
cd /d {llama_dir_win}
set PATH=%PATH%;C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v13.1\\bin;{llama_dir_win}\\build\\bin
{cuda_env_line}
{exe_path} -m "{model_path}" -ngl {gpu_layers} -c {context} --host 0.0.0.0 --port {port} -np {parallel} -b {batch_size} --temp {temp_str} --top-p {top_p_str} --top-k {top_k} --repeat-penalty {repeat_penalty_str} --verbose{optional_args}{mmproj_arg}
"@
$batchContent_{idx} | Out-File -FilePath $batchPath_{idx} -Encoding ASCII

$action_{idx} = New-ScheduledTaskAction -Execute $batchPath_{idx} -WorkingDirectory "{llama_dir_win}"
$trigger_{idx} = New-ScheduledTaskTrigger -Once -At (Get-Date).AddSeconds(2)
$settings_{idx} = New-ScheduledTaskSettingsSet -AllowStartIfOnBatteries -DontStopIfGoingOnBatteries -StartWhenAvailable -ExecutionTimeLimit (New-TimeSpan -Hours 24)
$principal_{idx} = New-ScheduledTaskPrincipal -UserId "Administrator" -LogonType Interactive -RunLevel Highest

Register-ScheduledTask -TaskName $taskName_{idx} -Action $action_{idx} -Trigger $trigger_{idx} -Settings $settings_{idx} -Principal $principal_{idx} -Force | Out-Null
Start-ScheduledTask -TaskName $taskName_{idx}

$results += "Started instance {idx} on port {port} with model {model}"

# Wait for model to fully load before starting next instance
# Check if the server is responding on its port
$maxWait = 120  # Maximum 120 seconds
$waited = 0
while ($waited -lt $maxWait) {{
    Start-Sleep -Seconds 5
    $waited += 5
    try {{
        $response = Invoke-WebRequest -Uri "http://localhost:{port}/health" -TimeoutSec 3 -ErrorAction SilentlyContinue
        if ($response.StatusCode -eq 200) {{
            "Instance {idx} on port {port} is ready"
            break
        }}
    }} catch {{
        # Server not ready yet, continue waiting
    }}
}}
if ($waited -ge $maxWait) {{
    "Warning: Instance {idx} on port {port} did not respond within $maxWait seconds"
}}
'''

        script += f'''
# Wait for all processes to start
Start-Sleep -Seconds 10

# Check results
$procs = Get-Process -Name "llama-server" -ErrorAction SilentlyContinue
if ($procs) {{
    $count = @($procs).Count
    "SUCCESS: $count instance(s) running"
    foreach ($p in $procs) {{
        "  PID: $($p.Id)"
    }}
}} else {{
    "ERROR: No instances running"
}}
'''

        success, stdout, stderr = self._run_remote_command(script, timeout=180)
        if success and "SUCCESS" in stdout:
            return True, stdout
        error_msg = stderr if stderr else stdout if stdout else "Failed to start instances"
        return False, error_msg

    def stop_all_instances(self) -> tuple:
        """Stop all llama-server instances on remote host."""
        script = '''
# Stop all llama-server processes
$procs = Get-Process -Name "llama-server" -ErrorAction SilentlyContinue
if ($procs) {
    $count = @($procs).Count
    Stop-Process -Name "llama-server" -Force -ErrorAction SilentlyContinue
    "Stopped $count instance(s)"
} else {
    "No running instances found"
}

# Remove all scheduled tasks
$tasks = Get-ScheduledTask -TaskName "LlamaServer*" -ErrorAction SilentlyContinue
foreach ($task in $tasks) {
    Unregister-ScheduledTask -TaskName $task.TaskName -Confirm:$false -ErrorAction SilentlyContinue
}
"Scheduled tasks removed"
'''
        success, stdout, stderr = self._run_remote_command(script, timeout=30)
        return success, stdout or stderr

    def check_multi_instances_status(self) -> tuple:
        """Check status of all llama-server instances on remote host."""
        script = '''
$procs = Get-Process -Name "llama-server" -ErrorAction SilentlyContinue
if ($procs) {
    $results = @()
    foreach ($p in $procs) {
        $mem = [math]::Round($p.WorkingSet64/1GB, 2)
        $results += "PID:$($p.Id)|Mem:$($mem)GB"
    }
    "Running|Count:" + @($procs).Count + "|" + ($results -join ";")
} else {
    "Stopped|Count:0"
}
'''
        success, stdout, stderr = self._run_remote_command(script, timeout=15)
        if success:
            return True, stdout.strip()
        return False, "Unknown"

    def get_remote_server_urls(self, start_port: int, count: int) -> List[str]:
        """Get list of server URLs for all instances."""
        urls = []
        for i in range(count):
            port = start_port + i
            urls.append(f"http://{self.remote_host}:{port}")
        return urls


class StatusMonitor:
    """Monitor server health and status."""

    def __init__(self, logger: logging.Logger):
        self.logger = logger
        self.last_status = None
        self.last_check_time = None

    def check_health(self, host: str, port: int) -> Dict[str, Any]:
        """Check server health via API."""
        if not HAS_REQUESTS:
            return {"status": "unknown", "error": "requests module not installed"}

        try:
            url = f"http://{host}:{port}/health"
            response = requests.get(url, timeout=3)

            if response.status_code == 200:
                data = response.json()
                self.last_status = data
                self.last_check_time = datetime.now()
                return {"status": "ok", "data": data}
            else:
                return {"status": "error", "code": response.status_code}

        except requests.ConnectionError:
            return {"status": "offline", "error": "Connection refused"}
        except requests.Timeout:
            return {"status": "timeout", "error": "Request timeout"}
        except Exception as e:
            return {"status": "error", "error": str(e)}

    def get_props(self, host: str, port: int) -> Optional[Dict[str, Any]]:
        """Get server properties."""
        if not HAS_REQUESTS:
            return None

        try:
            url = f"http://{host}:{port}/props"
            response = requests.get(url, timeout=5)

            if response.status_code == 200:
                return response.json()
        except Exception as e:
            self.logger.debug(f"Error getting props: {e}")

        return None


class LauncherGUI:
    """Main GUI application class."""

    def __init__(self):
        self.root = tk.Tk()
        self.root.title(f"{UIText.WINDOW_TITLE} v{APP_VERSION}")
        self.root.geometry("1000x720")
        self.root.minsize(900, 600)

        self.llama_dir = DEFAULT_LLAMA_DIR
        config_path = os.path.join(self.llama_dir, CONFIG_FILENAME)

        self.logger = setup_logging(self.llama_dir)
        self.config = ServerConfig(config_path)
        self.process_manager = ProcessManager(self.logger)
        self.multi_process_manager = MultiInstanceProcessManager(self.logger)
        self.remote_process_manager = RemoteProcessManager(self.logger)
        self.status_monitor = StatusMonitor(self.logger)

        self.model_files: List[str] = []
        self.remote_model_files: List[str] = []
        self.remote_mmproj_files: List[str] = []
        self.mmproj_files: List[str] = []

        self._create_styles()
        self._create_widgets()
        self._load_config_to_ui()
        self._scan_models()

        self.status_update_job = None
        self.output_update_job = None
        self._start_periodic_updates()

        self.root.protocol("WM_DELETE_WINDOW", self._on_closing)

        self.logger.info(f"{APP_NAME} v{APP_VERSION} started")

    def _create_styles(self) -> None:
        """Create custom ttk styles."""
        style = ttk.Style()
        style.theme_use("clam")

        style.configure("Title.TLabel", font=("Microsoft YaHei UI", 11, "bold"))
        style.configure("Status.TLabel", font=("Microsoft YaHei UI", 10))
        style.configure("Start.TButton", font=("Microsoft YaHei UI", 10, "bold"))
        style.configure("Stop.TButton", font=("Microsoft YaHei UI", 10, "bold"))

        style.configure(
            "Running.TLabel",
            foreground="green",
            font=("Microsoft YaHei UI", 10, "bold")
        )
        style.configure(
            "Stopped.TLabel",
            foreground="red",
            font=("Microsoft YaHei UI", 10, "bold")
        )
        style.configure(
            "Loading.TLabel",
            foreground="orange",
            font=("Microsoft YaHei UI", 10, "bold")
        )

    def _create_widgets(self) -> None:
        """Create all GUI widgets."""
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.pack(fill=tk.BOTH, expand=True)

        self.notebook = ttk.Notebook(main_frame)
        self.notebook.pack(fill=tk.BOTH, expand=True)

        local_tab = ttk.Frame(self.notebook, padding="10")
        self.notebook.add(local_tab, text="  Local Server  ")

        remote_tab = ttk.Frame(self.notebook, padding="10")
        self.notebook.add(remote_tab, text="  Remote Server  ")

        self._create_local_tab(local_tab)
        self._create_remote_tab(remote_tab)

        self._create_status_bar(main_frame)

    def _create_local_tab(self, parent: ttk.Frame) -> None:
        """Create local server tab content."""
        top_frame = ttk.Frame(parent)
        top_frame.pack(fill=tk.X, pady=(0, 10))

        self._create_model_section(top_frame)
        self._create_params_section(top_frame)
        self._create_control_section(top_frame)

        # Advanced options row: MoE + Speculative Decoding
        advanced_frame = ttk.Frame(parent)
        advanced_frame.pack(fill=tk.X, pady=(0, 5))
        self._create_local_moe_section(advanced_frame)
        self._create_local_lookup_section(advanced_frame)

        multi_frame = ttk.Frame(parent)
        multi_frame.pack(fill=tk.X, pady=(0, 10))
        self._create_multi_instance_section(multi_frame)

        self._create_log_section(parent)

    def _create_model_section(self, parent: ttk.Frame) -> None:
        """Create model selection section."""
        frame = ttk.LabelFrame(parent, text=UIText.MODEL_CONFIG, padding="10")
        frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 5))

        dir_frame = ttk.Frame(frame)
        dir_frame.pack(fill=tk.X, pady=(0, 10))

        ttk.Label(dir_frame, text=UIText.LLAMA_DIR).pack(side=tk.LEFT)
        self.dir_var = tk.StringVar(value=self.llama_dir)
        dir_entry = ttk.Entry(dir_frame, textvariable=self.dir_var, width=35)
        dir_entry.pack(side=tk.LEFT, padx=5, fill=tk.X, expand=True)

        ttk.Button(
            dir_frame,
            text=UIText.BROWSE,
            command=self._browse_directory,
            width=8
        ).pack(side=tk.LEFT)

        model_frame = ttk.Frame(frame)
        model_frame.pack(fill=tk.X, pady=5)

        ttk.Label(model_frame, text=UIText.MAIN_MODEL, width=12).pack(side=tk.LEFT)
        self.model_var = tk.StringVar()
        self.model_combo = ttk.Combobox(
            model_frame,
            textvariable=self.model_var,
            state="readonly",
            width=40
        )
        self.model_combo.pack(side=tk.LEFT, padx=5, fill=tk.X, expand=True)

        mmproj_frame = ttk.Frame(frame)
        mmproj_frame.pack(fill=tk.X, pady=5)

        ttk.Label(mmproj_frame, text=UIText.MMPROJ, width=12).pack(side=tk.LEFT)
        self.mmproj_var = tk.StringVar()
        self.mmproj_combo = ttk.Combobox(
            mmproj_frame,
            textvariable=self.mmproj_var,
            state="readonly",
            width=40
        )
        self.mmproj_combo.pack(side=tk.LEFT, padx=5, fill=tk.X, expand=True)

        btn_frame = ttk.Frame(frame)
        btn_frame.pack(fill=tk.X, pady=(10, 0))

        ttk.Button(
            btn_frame,
            text=UIText.REFRESH_MODELS,
            command=self._scan_models,
            width=15
        ).pack(side=tk.LEFT)

        ttk.Button(
            btn_frame,
            text=UIText.OPEN_MODELS_FOLDER,
            command=self._open_models_folder,
            width=18
        ).pack(side=tk.LEFT, padx=5)

    def _create_params_section(self, parent: ttk.Frame) -> None:
        """Create parameters configuration section."""
        frame = ttk.LabelFrame(parent, text=UIText.SERVER_PARAMS, padding="10")
        frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=5)

        network_frame = ttk.Frame(frame)
        network_frame.pack(fill=tk.X, pady=2)

        ttk.Label(network_frame, text=UIText.HOST, width=10).pack(side=tk.LEFT)
        self.host_var = tk.StringVar(value="0.0.0.0")
        ttk.Entry(network_frame, textvariable=self.host_var, width=15).pack(
            side=tk.LEFT, padx=2
        )

        ttk.Label(network_frame, text=UIText.PORT, width=6).pack(
            side=tk.LEFT, padx=(10, 0)
        )
        self.port_var = tk.IntVar(value=8080)
        ttk.Spinbox(
            network_frame,
            textvariable=self.port_var,
            from_=1,
            to=65535,
            width=8
        ).pack(side=tk.LEFT, padx=2)

        ttk.Label(network_frame, text=UIText.REMOTE_PARALLEL, width=8).pack(
            side=tk.LEFT, padx=(10, 0)
        )
        self.parallel_var = tk.IntVar(value=1)
        ttk.Spinbox(
            network_frame,
            textvariable=self.parallel_var,
            from_=1,
            to=64,
            width=5
        ).pack(side=tk.LEFT, padx=2)

        ctx_frame = ttk.Frame(frame)
        ctx_frame.pack(fill=tk.X, pady=2)

        ttk.Label(ctx_frame, text=UIText.CONTEXT, width=10).pack(side=tk.LEFT)
        self.context_var = tk.IntVar(value=32768)
        ctx_combo = ttk.Combobox(
            ctx_frame,
            textvariable=self.context_var,
            values=[2048, 4096, 8192, 16384, 32768, 65536, 131072],
            width=10
        )
        ctx_combo.pack(side=tk.LEFT, padx=2)

        ttk.Label(ctx_frame, text=UIText.GPU_LAYERS, width=10).pack(
            side=tk.LEFT, padx=(10, 0)
        )
        self.gpu_layers_var = tk.IntVar(value=99)
        self.gpu_layers_spinbox = ttk.Spinbox(
            ctx_frame,
            textvariable=self.gpu_layers_var,
            from_=0,
            to=999,
            width=6
        )
        self.gpu_layers_spinbox.pack(side=tk.LEFT, padx=2)

        ttk.Label(ctx_frame, text=UIText.MAIN_GPU, width=8).pack(
            side=tk.LEFT, padx=(10, 0)
        )
        self.main_gpu_var = tk.StringVar(value="Auto")
        ttk.Combobox(
            ctx_frame,
            textvariable=self.main_gpu_var,
            values=["Auto", "0", "1", "2", "3", "0,1", "0,2", "1,2", "0,1,2", "0,1,2,3"],
            width=8
        ).pack(side=tk.LEFT, padx=2)

        sampling_frame = ttk.Frame(frame)
        sampling_frame.pack(fill=tk.X, pady=2)

        ttk.Label(sampling_frame, text=UIText.TEMP, width=10).pack(side=tk.LEFT)
        self.temp_var = tk.DoubleVar(value=0.8)
        ttk.Spinbox(
            sampling_frame,
            textvariable=self.temp_var,
            from_=0.0,
            to=2.0,
            increment=0.1,
            width=6
        ).pack(side=tk.LEFT, padx=2)

        ttk.Label(sampling_frame, text=UIText.TOP_P, width=6).pack(
            side=tk.LEFT, padx=(10, 0)
        )
        self.top_p_var = tk.DoubleVar(value=0.6)
        ttk.Spinbox(
            sampling_frame,
            textvariable=self.top_p_var,
            from_=0.0,
            to=1.0,
            increment=0.1,
            width=6
        ).pack(side=tk.LEFT, padx=2)

        sampling2_frame = ttk.Frame(frame)
        sampling2_frame.pack(fill=tk.X, pady=2)

        ttk.Label(sampling2_frame, text=UIText.TOP_K, width=10).pack(side=tk.LEFT)
        self.top_k_var = tk.IntVar(value=2)
        ttk.Spinbox(
            sampling2_frame,
            textvariable=self.top_k_var,
            from_=0,
            to=100,
            width=6
        ).pack(side=tk.LEFT, padx=2)

        ttk.Label(sampling2_frame, text=UIText.REP_PEN, width=10).pack(
            side=tk.LEFT, padx=(10, 0)
        )
        self.repeat_penalty_var = tk.DoubleVar(value=1.1)
        ttk.Spinbox(
            sampling2_frame,
            textvariable=self.repeat_penalty_var,
            from_=1.0,
            to=2.0,
            increment=0.05,
            width=6
        ).pack(side=tk.LEFT, padx=2)

        ttk.Label(sampling2_frame, text=UIText.BATCH_SIZE, width=8).pack(
            side=tk.LEFT, padx=(10, 0)
        )
        self.batch_size_var = tk.IntVar(value=512)
        batch_combo = ttk.Combobox(
            sampling2_frame,
            textvariable=self.batch_size_var,
            values=[128, 256, 512, 1024, 2048, 4096],
            width=6
        )
        batch_combo.pack(side=tk.LEFT, padx=2)

        ttk.Label(sampling2_frame, text=UIText.UBATCH_SIZE, width=8).pack(
            side=tk.LEFT, padx=(10, 0)
        )
        self.ubatch_size_var = tk.IntVar(value=512)
        ubatch_combo = ttk.Combobox(
            sampling2_frame,
            textvariable=self.ubatch_size_var,
            values=[128, 256, 512, 1024, 2048, 4096],
            width=6
        )
        ubatch_combo.pack(side=tk.LEFT, padx=2)

        ttk.Label(sampling2_frame, text=UIText.CACHE_TYPE_K, width=8).pack(
            side=tk.LEFT, padx=(10, 0)
        )
        self.cache_type_k_var = tk.StringVar(value="f16")
        ttk.Combobox(
            sampling2_frame,
            textvariable=self.cache_type_k_var,
            values=["f16", "f32", "bf16", "q8_0", "q4_0", "q4_1", "iq4_nl", "q5_0", "q5_1"],
            width=7, state="readonly"
        ).pack(side=tk.LEFT, padx=2)

        ttk.Label(sampling2_frame, text=UIText.CACHE_TYPE_V, width=8).pack(
            side=tk.LEFT, padx=(5, 0)
        )
        self.cache_type_v_var = tk.StringVar(value="f16")
        ttk.Combobox(
            sampling2_frame,
            textvariable=self.cache_type_v_var,
            values=["f16", "f32", "bf16", "q8_0", "q4_0", "q4_1", "iq4_nl", "q5_0", "q5_1"],
            width=7, state="readonly"
        ).pack(side=tk.LEFT, padx=2)

        options_frame = ttk.Frame(frame)
        options_frame.pack(fill=tk.X, pady=(5, 2))

        self.jinja_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(
            options_frame,
            text=UIText.USE_JINJA,
            variable=self.jinja_var
        ).pack(side=tk.LEFT)

        self.flash_attn_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(
            options_frame,
            text=UIText.FLASH_ATTN,
            variable=self.flash_attn_var
        ).pack(side=tk.LEFT, padx=10)

        self.split_mode_row_var = tk.BooleanVar(value=False)
        self.split_mode_check = ttk.Checkbutton(
            options_frame,
            text=UIText.SPLIT_MODE_ROW,
            variable=self.split_mode_row_var
        )
        self.split_mode_check.pack(side=tk.LEFT, padx=10)

        self.no_mmap_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(
            options_frame,
            text=UIText.NO_MMAP,
            variable=self.no_mmap_var
        ).pack(side=tk.LEFT, padx=10)

        self.cuda_graph_opt_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(
            options_frame,
            text=UIText.REMOTE_CUDA_GRAPH_OPT,
            variable=self.cuda_graph_opt_var
        ).pack(side=tk.LEFT, padx=10)

        # Options row 2: Fit, KV offload, cache options
        options_frame2 = ttk.Frame(frame)
        options_frame2.pack(fill=tk.X, pady=(2, 2))

        self.fit_var = tk.BooleanVar(value=False)
        self.fit_check = ttk.Checkbutton(
            options_frame2,
            text=UIText.REMOTE_FIT,
            variable=self.fit_var,
            command=self._on_local_fit_toggle
        )
        self.fit_check.pack(side=tk.LEFT)

        ttk.Label(options_frame2, text=UIText.REMOTE_FIT_TARGET).pack(
            side=tk.LEFT, padx=(5, 0)
        )
        self.fit_target_var = tk.IntVar(value=1024)
        self.fit_target_spin = ttk.Spinbox(
            options_frame2,
            textvariable=self.fit_target_var,
            from_=128,
            to=8192,
            increment=128,
            width=6
        )
        self.fit_target_spin.pack(side=tk.LEFT, padx=2)

        self.no_kv_offload_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(
            options_frame2,
            text=UIText.REMOTE_NO_KV_OFFLOAD,
            variable=self.no_kv_offload_var
        ).pack(side=tk.LEFT, padx=10)

        # Cache optimization options
        ttk.Label(options_frame2, text=UIText.REMOTE_CACHE_REUSE, width=8).pack(
            side=tk.LEFT, padx=(10, 0)
        )
        self.cache_reuse_var = tk.IntVar(value=0)
        ttk.Combobox(
            options_frame2,
            textvariable=self.cache_reuse_var,
            values=[0, 64, 128, 256, 512, 1024],
            width=6
        ).pack(side=tk.LEFT, padx=2)

        ttk.Label(options_frame2, text=UIText.REMOTE_CACHE_RAM, width=12).pack(
            side=tk.LEFT, padx=(8, 0)
        )
        self.cache_ram_var = tk.IntVar(value=-1)
        ttk.Combobox(
            options_frame2,
            textvariable=self.cache_ram_var,
            values=[-1, 0, 2048, 4096, 8192, 16384, 32768],
            width=6
        ).pack(side=tk.LEFT, padx=2)

        self.slot_save_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(
            options_frame2,
            text=UIText.REMOTE_SLOT_SAVE,
            variable=self.slot_save_var
        ).pack(side=tk.LEFT, padx=(10, 0))

        custom_frame = ttk.Frame(frame)
        custom_frame.pack(fill=tk.X, pady=(5, 0))

        ttk.Label(custom_frame, text=UIText.CUSTOM_ARGS).pack(side=tk.LEFT)
        self.custom_args_var = tk.StringVar()
        ttk.Entry(
            custom_frame,
            textvariable=self.custom_args_var,
            width=25
        ).pack(side=tk.LEFT, padx=5, fill=tk.X, expand=True)

    def _create_local_moe_section(self, parent: ttk.Frame) -> None:
        """Create local MoE expert GPU allocation section."""
        moe_frame = ttk.LabelFrame(parent, text="MoE专家GPU分配", padding="5")
        moe_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 5))

        # Row 1: mode selector
        moe_row1 = ttk.Frame(moe_frame)
        moe_row1.pack(fill=tk.X)

        self.local_moe_mode_var = tk.StringVar(value="不分配")
        ttk.Label(moe_row1, text="专家放置:").pack(side=tk.LEFT)
        local_moe_mode_combo = ttk.Combobox(
            moe_row1,
            textvariable=self.local_moe_mode_var,
            values=["不分配", "专家放CPU", "按比例分配", "均分到GPU", "自定义"],
            state="readonly", width=12
        )
        local_moe_mode_combo.pack(side=tk.LEFT, padx=5)
        local_moe_mode_combo.bind("<<ComboboxSelected>>", self._on_local_moe_mode_changed)

        # Total layers
        self.local_moe_layers_label = ttk.Label(moe_row1, text="总层数:")
        self.local_moe_layers_var = tk.IntVar(value=62)
        self.local_moe_layers_spin = ttk.Spinbox(
            moe_row1, textvariable=self.local_moe_layers_var,
            from_=1, to=200, width=4,
            command=self._update_local_moe_preview
        )
        self.local_moe_layers_spin.bind("<KeyRelease>", lambda e: self._update_local_moe_preview())

        # GPU count (for 均分)
        self.local_moe_gpu_label = ttk.Label(moe_row1, text="GPU数:")
        self.local_moe_gpu_count_var = tk.IntVar(value=3)
        self.local_moe_gpu_combo = ttk.Combobox(
            moe_row1,
            textvariable=self.local_moe_gpu_count_var,
            values=[2, 3, 4, 5, 6, 7, 8],
            state="readonly", width=3
        )
        self.local_moe_gpu_combo.bind("<<ComboboxSelected>>", lambda e: self._update_local_moe_preview())

        # Row 2: weighted ratio (for "按比例分配")
        self.local_moe_ratio_row = ttk.Frame(moe_frame)
        ttk.Label(self.local_moe_ratio_row, text="GPU0:").pack(side=tk.LEFT)
        self.local_moe_ratio0_var = tk.IntVar(value=2)
        r0 = ttk.Spinbox(
            self.local_moe_ratio_row, textvariable=self.local_moe_ratio0_var,
            from_=0, to=10, width=3, command=self._update_local_moe_preview
        )
        r0.pack(side=tk.LEFT, padx=2)
        r0.bind("<KeyRelease>", lambda e: self._update_local_moe_preview())
        ttk.Label(self.local_moe_ratio_row, text="GPU1:").pack(side=tk.LEFT, padx=(8, 0))
        self.local_moe_ratio1_var = tk.IntVar(value=1)
        r1 = ttk.Spinbox(
            self.local_moe_ratio_row, textvariable=self.local_moe_ratio1_var,
            from_=0, to=10, width=3, command=self._update_local_moe_preview
        )
        r1.pack(side=tk.LEFT, padx=2)
        r1.bind("<KeyRelease>", lambda e: self._update_local_moe_preview())
        ttk.Label(self.local_moe_ratio_row, text="GPU2:").pack(side=tk.LEFT, padx=(8, 0))
        self.local_moe_ratio2_var = tk.IntVar(value=1)
        r2 = ttk.Spinbox(
            self.local_moe_ratio_row, textvariable=self.local_moe_ratio2_var,
            from_=0, to=10, width=3, command=self._update_local_moe_preview
        )
        r2.pack(side=tk.LEFT, padx=2)
        r2.bind("<KeyRelease>", lambda e: self._update_local_moe_preview())

        # Row for "专家放CPU" mode
        self.local_moe_cpu_row = ttk.Frame(moe_frame)
        ttk.Label(self.local_moe_cpu_row, text="总层数:").pack(side=tk.LEFT)
        self.local_moe_cpu_total_var = tk.IntVar(value=0)
        cpu_total_spin = ttk.Spinbox(
            self.local_moe_cpu_row, textvariable=self.local_moe_cpu_total_var,
            from_=1, to=200, width=4,
            command=self._update_local_moe_cpu_preview
        )
        cpu_total_spin.pack(side=tk.LEFT, padx=2)
        cpu_total_spin.bind("<KeyRelease>", lambda e: self._update_local_moe_cpu_preview())

        ttk.Label(self.local_moe_cpu_row, text="放CPU:").pack(side=tk.LEFT, padx=(10, 0))
        self.local_moe_cpu_layers_var = tk.IntVar(value=0)
        cpu_layers_spin = ttk.Spinbox(
            self.local_moe_cpu_row, textvariable=self.local_moe_cpu_layers_var,
            from_=0, to=200, width=4,
            command=self._update_local_moe_cpu_preview
        )
        cpu_layers_spin.pack(side=tk.LEFT, padx=2)
        cpu_layers_spin.bind("<KeyRelease>", lambda e: self._update_local_moe_cpu_preview())

        ttk.Label(self.local_moe_cpu_row, text="GPU比例:").pack(side=tk.LEFT, padx=(10, 0))
        self.local_moe_ts_var = tk.StringVar(value="")
        ts_entry = ttk.Entry(
            self.local_moe_cpu_row, textvariable=self.local_moe_ts_var, width=10
        )
        ts_entry.pack(side=tk.LEFT, padx=2)
        ts_entry.bind("<KeyRelease>", lambda e: self._update_local_moe_cpu_preview())

        # Custom rule entry (for "自定义")
        self.local_moe_custom_row = ttk.Frame(moe_frame)
        self.local_override_tensor_var = tk.StringVar()
        ttk.Label(self.local_moe_custom_row, text="规则:").pack(side=tk.LEFT)
        self.local_override_tensor_entry = ttk.Entry(
            self.local_moe_custom_row,
            textvariable=self.local_override_tensor_var,
            width=50
        )
        self.local_override_tensor_entry.pack(side=tk.LEFT, padx=5, fill=tk.X, expand=True)

        # Preview label
        self.local_moe_preview_var = tk.StringVar()
        self.local_moe_preview_label = ttk.Label(
            moe_frame, textvariable=self.local_moe_preview_var,
            foreground="gray", wraplength=450, justify=tk.LEFT
        )

        # Initialize visibility
        self._on_local_moe_mode_changed()

    def _create_local_lookup_section(self, parent: ttk.Frame) -> None:
        """Create local speculative decoding (Lookup) section."""
        spec_frame = ttk.LabelFrame(parent, text="推测解码 (Speculative Decoding)", padding="5")
        spec_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(5, 0))

        spec_row1 = ttk.Frame(spec_frame)
        spec_row1.pack(fill=tk.X)

        self.local_lookup_enabled_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(
            spec_row1,
            text="缓存预测(Lookup)",
            variable=self.local_lookup_enabled_var,
            command=self._on_local_lookup_toggle
        ).pack(side=tk.LEFT)

        ttk.Label(spec_row1, text="draft-max:").pack(side=tk.LEFT, padx=(15, 0))
        self.local_draft_max_var = tk.IntVar(value=16)
        ttk.Spinbox(
            spec_row1, textvariable=self.local_draft_max_var,
            from_=2, to=128, width=4
        ).pack(side=tk.LEFT, padx=2)

        ttk.Label(spec_row1, text="draft-min:").pack(side=tk.LEFT, padx=(10, 0))
        self.local_draft_min_var = tk.IntVar(value=2)
        ttk.Spinbox(
            spec_row1, textvariable=self.local_draft_min_var,
            from_=1, to=64, width=4
        ).pack(side=tk.LEFT, padx=2)

        # Lookup cache file path row
        self.local_spec_cache_row = ttk.Frame(spec_frame)
        ttk.Label(self.local_spec_cache_row, text="动态缓存:").pack(side=tk.LEFT)
        self.local_lookup_cache_var = tk.StringVar(value="lookup_cache.bin")
        ttk.Entry(
            self.local_spec_cache_row,
            textvariable=self.local_lookup_cache_var,
            width=25
        ).pack(side=tk.LEFT, padx=5, fill=tk.X, expand=True)
        ttk.Button(
            self.local_spec_cache_row, text="按模型", width=6,
            command=self._auto_local_lookup_cache_name
        ).pack(side=tk.LEFT, padx=(0, 5))

        ttk.Label(self.local_spec_cache_row, text="静态缓存:").pack(side=tk.LEFT, padx=(10, 0))
        self.local_lookup_static_var = tk.StringVar(value="")
        ttk.Entry(
            self.local_spec_cache_row,
            textvariable=self.local_lookup_static_var,
            width=15
        ).pack(side=tk.LEFT, padx=5)

        # Initialize visibility
        self._on_local_lookup_toggle()

    def _create_control_section(self, parent: ttk.Frame) -> None:
        """Create control buttons section."""
        frame = ttk.LabelFrame(parent, text=UIText.SERVER_CONTROL, padding="10")
        frame.pack(side=tk.LEFT, fill=tk.BOTH, padx=(5, 0))

        status_frame = ttk.Frame(frame)
        status_frame.pack(fill=tk.X, pady=(0, 10))

        ttk.Label(status_frame, text=UIText.STATUS, width=8).pack(side=tk.LEFT)
        self.status_label = ttk.Label(
            status_frame,
            text=UIText.STATUS_STOPPED,
            style="Stopped.TLabel",
            width=10
        )
        self.status_label.pack(side=tk.LEFT, padx=5)

        self.start_btn = ttk.Button(
            frame,
            text=UIText.START_SERVER,
            command=self._start_server,
            style="Start.TButton",
            width=15
        )
        self.start_btn.pack(fill=tk.X, pady=2)

        self.stop_btn = ttk.Button(
            frame,
            text=UIText.STOP_SERVER,
            command=self._stop_server,
            style="Stop.TButton",
            width=15,
            state=tk.DISABLED
        )
        self.stop_btn.pack(fill=tk.X, pady=2)

        ttk.Separator(frame, orient=tk.HORIZONTAL).pack(fill=tk.X, pady=10)

        ttk.Button(
            frame,
            text=UIText.OPEN_WEBUI,
            command=self._open_webui,
            width=15
        ).pack(fill=tk.X, pady=2)

        ttk.Button(
            frame,
            text=UIText.TEST_API,
            command=self._test_api,
            width=15
        ).pack(fill=tk.X, pady=2)

        ttk.Separator(frame, orient=tk.HORIZONTAL).pack(fill=tk.X, pady=10)

        ttk.Button(
            frame,
            text=UIText.SAVE_CONFIG,
            command=self._save_config,
            width=15
        ).pack(fill=tk.X, pady=2)

        ttk.Button(
            frame,
            text=UIText.RESET_DEFAULTS,
            command=self._reset_defaults,
            width=15
        ).pack(fill=tk.X, pady=2)

    def _create_multi_instance_section(self, parent: ttk.Frame) -> None:
        """Create multi-instance control section."""
        frame = ttk.LabelFrame(parent, text=UIText.MULTI_INSTANCE, padding="10")
        frame.pack(fill=tk.X)

        top_row = ttk.Frame(frame)
        top_row.pack(fill=tk.X, pady=(0, 10))

        left_frame = ttk.Frame(top_row)
        left_frame.pack(side=tk.LEFT, fill=tk.X, expand=True)

        self.multi_enabled_var = tk.BooleanVar(value=False)
        enable_check = ttk.Checkbutton(
            left_frame,
            text=UIText.MULTI_ENABLED,
            variable=self.multi_enabled_var,
            command=self._on_multi_mode_toggle
        )
        enable_check.pack(side=tk.LEFT)

        params_frame = ttk.Frame(left_frame)
        params_frame.pack(side=tk.LEFT, padx=(20, 0))

        ttk.Label(params_frame, text=UIText.NUM_INSTANCES).pack(side=tk.LEFT)
        self.multi_count_var = tk.IntVar(value=3)
        self.multi_count_spin = ttk.Spinbox(
            params_frame,
            textvariable=self.multi_count_var,
            from_=2,
            to=16,
            width=5,
            state=tk.DISABLED
        )
        self.multi_count_spin.pack(side=tk.LEFT, padx=(5, 15))

        ttk.Label(params_frame, text=UIText.START_PORT_LABEL).pack(side=tk.LEFT)
        self.multi_start_port_var = tk.IntVar(value=8080)
        self.multi_start_port_spin = ttk.Spinbox(
            params_frame,
            textvariable=self.multi_start_port_var,
            from_=1024,
            to=65000,
            width=8,
            state=tk.DISABLED
        )
        self.multi_start_port_spin.pack(side=tk.LEFT, padx=5)

        btn_frame = ttk.Frame(top_row)
        btn_frame.pack(side=tk.RIGHT)

        self.multi_start_all_btn = ttk.Button(
            btn_frame,
            text=UIText.START_ALL,
            command=self._start_all_instances,
            width=12,
            state=tk.DISABLED
        )
        self.multi_start_all_btn.pack(side=tk.LEFT, padx=2)

        self.multi_stop_all_btn = ttk.Button(
            btn_frame,
            text=UIText.STOP_ALL,
            command=self._stop_all_instances,
            width=12,
            state=tk.DISABLED
        )
        self.multi_stop_all_btn.pack(side=tk.LEFT, padx=2)

        self.multi_copy_urls_btn = ttk.Button(
            btn_frame,
            text=UIText.COPY_URLS,
            command=self._copy_server_urls,
            width=12,
            state=tk.DISABLED
        )
        self.multi_copy_urls_btn.pack(side=tk.LEFT, padx=2)

        list_frame = ttk.Frame(frame)
        list_frame.pack(fill=tk.X)

        columns = ("port", "status", "pid")
        self.instance_tree = ttk.Treeview(
            list_frame,
            columns=columns,
            show="headings",
            height=4
        )

        self.instance_tree.heading("port", text=UIText.INSTANCE_PORT)
        self.instance_tree.heading("status", text=UIText.INSTANCE_STATUS_COL)
        self.instance_tree.heading("pid", text=UIText.INSTANCE_PID)

        self.instance_tree.column("port", width=100, anchor=tk.CENTER)
        self.instance_tree.column("status", width=150, anchor=tk.CENTER)
        self.instance_tree.column("pid", width=100, anchor=tk.CENTER)

        scrollbar = ttk.Scrollbar(list_frame, orient=tk.VERTICAL, command=self.instance_tree.yview)
        self.instance_tree.configure(yscrollcommand=scrollbar.set)

        self.instance_tree.pack(side=tk.LEFT, fill=tk.X, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

        running_label_frame = ttk.Frame(frame)
        running_label_frame.pack(fill=tk.X, pady=(5, 0))

        self.multi_running_label = ttk.Label(
            running_label_frame,
            text="Running: 0 / 0",
            style="Status.TLabel"
        )
        self.multi_running_label.pack(side=tk.LEFT)

    def _on_multi_mode_toggle(self) -> None:
        """Handle multi-instance mode toggle."""
        enabled = self.multi_enabled_var.get()
        state = tk.NORMAL if enabled else tk.DISABLED

        self.multi_count_spin.config(state=state)
        self.multi_start_port_spin.config(state=state)
        self.multi_start_all_btn.config(state=state if enabled else tk.DISABLED)

        if enabled:
            self._update_instance_tree_preview()
        else:
            for item in self.instance_tree.get_children():
                self.instance_tree.delete(item)
            self.multi_running_label.config(text="Running: 0 / 0")

    def _update_instance_tree_preview(self) -> None:
        """Update instance tree with preview of ports to be used."""
        for item in self.instance_tree.get_children():
            self.instance_tree.delete(item)

        count = self.multi_count_var.get()
        start_port = self.multi_start_port_var.get()

        for i in range(count):
            port = start_port + i
            running = self.multi_process_manager.is_instance_running(port)
            pid = self.multi_process_manager.get_instance_pid(port)

            status = UIText.STATUS_RUNNING if running else UIText.STATUS_STOPPED
            pid_str = str(pid) if pid else "-"

            self.instance_tree.insert("", tk.END, values=(port, status, pid_str))

        running_count = self.multi_process_manager.get_running_count()
        self.multi_running_label.config(text=f"Running: {running_count} / {count}")

    def _start_all_instances(self) -> None:
        """Start all instances in multi-instance mode."""
        if not self.model_var.get():
            messagebox.showerror(UIText.MSG_ERROR, UIText.MSG_SELECT_MODEL)
            return

        self._save_ui_to_config()

        count = self.multi_count_var.get()
        start_port = self.multi_start_port_var.get()

        self._log_message(UIText.MSG_STARTING_INSTANCES.format(count))
        self.multi_start_all_btn.config(state=tk.DISABLED)
        self.multi_stop_all_btn.config(state=tk.DISABLED)

        for item in self.instance_tree.get_children():
            self.instance_tree.delete(item)

        for i in range(count):
            port = start_port + i
            self.instance_tree.insert("", tk.END, values=(port, UIText.STATUS_STARTING, "-"))

        def start_thread():
            def progress_callback(port, success, current, total):
                self.root.after(0, lambda: self._on_instance_progress(port, success, current, total))

            results = self.multi_process_manager.start_all(
                self.config, count, start_port, progress_callback
            )
            self.root.after(0, lambda: self._on_all_instances_started(results))

        threading.Thread(target=start_thread, daemon=True).start()

    def _on_instance_progress(self, port: int, success: bool, current: int, total: int) -> None:
        """Callback for instance start progress."""
        for item in self.instance_tree.get_children():
            values = self.instance_tree.item(item, "values")
            if int(values[0]) == port:
                if success:
                    pid = self.multi_process_manager.get_instance_pid(port)
                    self.instance_tree.item(item, values=(port, UIText.STATUS_RUNNING, pid or "-"))
                    self._log_message(UIText.MSG_INSTANCE_STARTED.format(port))
                else:
                    self.instance_tree.item(item, values=(port, UIText.STATUS_STOPPED, "-"))
                    self._log_message(UIText.MSG_INSTANCE_FAILED.format(port))
                break

        running_count = self.multi_process_manager.get_running_count()
        self.multi_running_label.config(text=f"Running: {running_count} / {total}")

    def _on_all_instances_started(self, results: Dict[int, bool]) -> None:
        """Callback when all instances have been started."""
        success_count = sum(1 for v in results.values() if v)
        total = len(results)

        self._log_message(UIText.MSG_ALL_STARTED.format(f"{success_count}/{total}"))

        self.multi_start_all_btn.config(state=tk.DISABLED)
        self.multi_stop_all_btn.config(state=tk.NORMAL)
        self.multi_copy_urls_btn.config(state=tk.NORMAL)

        running_count = self.multi_process_manager.get_running_count()
        self.multi_running_label.config(text=f"Running: {running_count} / {total}")

    def _stop_all_instances(self) -> None:
        """Stop all running instances."""
        self._log_message(UIText.MSG_STOPPING_INSTANCES)
        self.multi_stop_all_btn.config(state=tk.DISABLED)

        def stop_thread():
            results = self.multi_process_manager.stop_all()
            self.root.after(0, lambda: self._on_all_instances_stopped(results))

        threading.Thread(target=stop_thread, daemon=True).start()

    def _on_all_instances_stopped(self, results: Dict[int, bool]) -> None:
        """Callback when all instances have been stopped."""
        self._log_message(UIText.MSG_ALL_STOPPED)

        for item in self.instance_tree.get_children():
            values = self.instance_tree.item(item, "values")
            port = values[0]
            self.instance_tree.item(item, values=(port, UIText.STATUS_STOPPED, "-"))
            self._log_message(UIText.MSG_INSTANCE_STOPPED.format(port))

        self.multi_start_all_btn.config(state=tk.NORMAL)
        self.multi_stop_all_btn.config(state=tk.DISABLED)
        self.multi_copy_urls_btn.config(state=tk.DISABLED)

        count = self.multi_count_var.get()
        self.multi_running_label.config(text=f"Running: 0 / {count}")

    def _copy_server_urls(self) -> None:
        """Copy all running server URLs to clipboard."""
        urls = self.multi_process_manager.get_server_urls()
        if urls:
            url_text = "\n".join(urls)
            self.root.clipboard_clear()
            self.root.clipboard_append(url_text)
            self._log_message(UIText.MSG_URLS_COPIED)
            self._log_message(f"URLs: {', '.join(urls)}")
            self.statusbar_label.config(text=UIText.MSG_URLS_COPIED)
        else:
            self._log_message("No running instances to copy")

    def _create_log_section(self, parent: ttk.Frame) -> None:
        """Create log output section."""
        frame = ttk.LabelFrame(parent, text=UIText.SERVER_OUTPUT, padding="5")
        frame.pack(fill=tk.BOTH, expand=True, pady=(0, 5))

        self.log_text = scrolledtext.ScrolledText(
            frame,
            wrap=tk.WORD,
            font=("Consolas", 9),
            height=15,
            state=tk.DISABLED,
            bg="#1e1e1e",
            fg="#d4d4d4",
            insertbackground="white"
        )
        self.log_text.pack(fill=tk.BOTH, expand=True)

        btn_frame = ttk.Frame(frame)
        btn_frame.pack(fill=tk.X, pady=(5, 0))

        ttk.Button(
            btn_frame,
            text=UIText.CLEAR_LOG,
            command=self._clear_log,
            width=12
        ).pack(side=tk.LEFT)

        ttk.Button(
            btn_frame,
            text=UIText.COPY_LOG,
            command=self._copy_log,
            width=12
        ).pack(side=tk.LEFT, padx=5)

        self.auto_scroll_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(
            btn_frame,
            text=UIText.AUTO_SCROLL,
            variable=self.auto_scroll_var
        ).pack(side=tk.LEFT, padx=10)

    def _create_remote_tab(self, parent: ttk.Frame) -> None:
        """Create remote server control tab content."""
        top_frame = ttk.Frame(parent)
        top_frame.pack(fill=tk.X, pady=(0, 10))

        self._create_remote_connection_section(top_frame)
        self._create_remote_params_section(top_frame)
        self._create_remote_control_section(top_frame)

        multi_frame = ttk.Frame(parent)
        multi_frame.pack(fill=tk.X, pady=(0, 10))
        self._create_remote_multi_instance_section(multi_frame)

        self._create_remote_log_section(parent)

    def _create_remote_connection_section(self, parent: ttk.Frame) -> None:
        """Create remote connection settings section."""
        frame = ttk.LabelFrame(parent, text=UIText.REMOTE_CONNECTION, padding="10")
        frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 5))

        host_frame = ttk.Frame(frame)
        host_frame.pack(fill=tk.X, pady=2)

        ttk.Label(host_frame, text=UIText.REMOTE_HOST, width=10).pack(side=tk.LEFT)
        self.remote_host_var = tk.StringVar(value=DEFAULT_REMOTE_HOST)
        ttk.Entry(host_frame, textvariable=self.remote_host_var, width=18).pack(
            side=tk.LEFT, padx=5, fill=tk.X, expand=True
        )

        user_frame = ttk.Frame(frame)
        user_frame.pack(fill=tk.X, pady=2)

        ttk.Label(user_frame, text=UIText.REMOTE_USER, width=10).pack(side=tk.LEFT)
        self.remote_user_var = tk.StringVar(value=DEFAULT_REMOTE_USER)
        ttk.Entry(user_frame, textvariable=self.remote_user_var, width=18).pack(
            side=tk.LEFT, padx=5, fill=tk.X, expand=True
        )

        pass_frame = ttk.Frame(frame)
        pass_frame.pack(fill=tk.X, pady=2)

        ttk.Label(pass_frame, text=UIText.REMOTE_PASSWORD, width=10).pack(side=tk.LEFT)
        self.remote_pass_var = tk.StringVar()
        ttk.Entry(pass_frame, textvariable=self.remote_pass_var, show="*", width=18).pack(
            side=tk.LEFT, padx=5, fill=tk.X, expand=True
        )

        conn_status_frame = ttk.Frame(frame)
        conn_status_frame.pack(fill=tk.X, pady=(5, 2))

        self.remote_conn_status = ttk.Label(
            conn_status_frame,
            text=UIText.REMOTE_DISCONNECTED,
            style="Stopped.TLabel"
        )
        self.remote_conn_status.pack(side=tk.LEFT)

        btn_frame = ttk.Frame(frame)
        btn_frame.pack(fill=tk.X, pady=(10, 0))

        ttk.Button(
            btn_frame,
            text=UIText.REMOTE_TEST_CONNECTION,
            command=self._test_remote_connection,
            width=15
        ).pack(side=tk.LEFT)

    def _create_remote_params_section(self, parent: ttk.Frame) -> None:
        """Create remote server parameters section."""
        frame = ttk.LabelFrame(parent, text=UIText.REMOTE_SERVER_PARAMS, padding="10")
        frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=5)

        dir_frame = ttk.Frame(frame)
        dir_frame.pack(fill=tk.X, pady=2)

        ttk.Label(dir_frame, text=UIText.REMOTE_LLAMA_DIR, width=10).pack(side=tk.LEFT)
        self.remote_llama_dir_var = tk.StringVar(value=DEFAULT_REMOTE_LLAMA_DIR)
        ttk.Entry(dir_frame, textvariable=self.remote_llama_dir_var, width=25).pack(
            side=tk.LEFT, padx=5, fill=tk.X, expand=True
        )

        model_dir_frame = ttk.Frame(frame)
        model_dir_frame.pack(fill=tk.X, pady=2)

        ttk.Label(model_dir_frame, text=UIText.REMOTE_MODEL_DIR, width=10).pack(side=tk.LEFT)
        self.remote_model_dir_var = tk.StringVar(value=DEFAULT_REMOTE_MODEL_DIR)
        ttk.Entry(model_dir_frame, textvariable=self.remote_model_dir_var, width=25).pack(
            side=tk.LEFT, padx=5, fill=tk.X, expand=True
        )

        model_frame = ttk.Frame(frame)
        model_frame.pack(fill=tk.X, pady=2)

        ttk.Label(model_frame, text=UIText.REMOTE_MODEL, width=10).pack(side=tk.LEFT)
        self.remote_model_var = tk.StringVar()
        self.remote_model_combo = ttk.Combobox(
            model_frame,
            textvariable=self.remote_model_var,
            state="readonly",
            width=30
        )
        self.remote_model_combo.pack(side=tk.LEFT, padx=5, fill=tk.X, expand=True)

        mmproj_frame = ttk.Frame(frame)
        mmproj_frame.pack(fill=tk.X, pady=2)

        ttk.Label(mmproj_frame, text=UIText.REMOTE_MMPROJ, width=10).pack(side=tk.LEFT)
        self.remote_mmproj_var = tk.StringVar()
        self.remote_mmproj_combo = ttk.Combobox(
            mmproj_frame,
            textvariable=self.remote_mmproj_var,
            state="readonly",
            width=30
        )
        self.remote_mmproj_combo.pack(side=tk.LEFT, padx=5, fill=tk.X, expand=True)

        params_row1 = ttk.Frame(frame)
        params_row1.pack(fill=tk.X, pady=2)

        ttk.Label(params_row1, text=UIText.REMOTE_PORT, width=6).pack(side=tk.LEFT)
        self.remote_port_var = tk.IntVar(value=8080)
        ttk.Spinbox(
            params_row1,
            textvariable=self.remote_port_var,
            from_=1,
            to=65535,
            width=7
        ).pack(side=tk.LEFT, padx=2)

        ttk.Label(params_row1, text=UIText.REMOTE_PARALLEL, width=8).pack(
            side=tk.LEFT, padx=(10, 0)
        )
        self.remote_parallel_var = tk.IntVar(value=4)
        ttk.Spinbox(
            params_row1,
            textvariable=self.remote_parallel_var,
            from_=1,
            to=64,
            width=5
        ).pack(side=tk.LEFT, padx=2)

        params_row2 = ttk.Frame(frame)
        params_row2.pack(fill=tk.X, pady=2)

        ttk.Label(params_row2, text=UIText.REMOTE_CONTEXT, width=8).pack(side=tk.LEFT)
        self.remote_context_var = tk.IntVar(value=32768)
        ctx_combo = ttk.Combobox(
            params_row2,
            textvariable=self.remote_context_var,
            values=[2048, 4096, 8192, 16384, 32768, 65536, 131072],
            width=8
        )
        ctx_combo.pack(side=tk.LEFT, padx=2)

        ttk.Label(params_row2, text=UIText.REMOTE_GPU_LAYERS, width=8).pack(
            side=tk.LEFT, padx=(10, 0)
        )
        self.remote_gpu_layers_var = tk.IntVar(value=99)
        self.remote_gpu_layers_spinbox = ttk.Spinbox(
            params_row2,
            textvariable=self.remote_gpu_layers_var,
            from_=0,
            to=999,
            width=5
        )
        self.remote_gpu_layers_spinbox.pack(side=tk.LEFT, padx=2)

        ttk.Label(params_row2, text=UIText.REMOTE_MAIN_GPU, width=6).pack(
            side=tk.LEFT, padx=(10, 0)
        )
        self.remote_main_gpu_var = tk.StringVar(value="Auto")
        # Allow both preset and custom GPU selection (e.g. "0,1" for multi-GPU)
        ttk.Combobox(
            params_row2,
            textvariable=self.remote_main_gpu_var,
            values=["Auto", "0", "1", "2", "3", "0,1", "0,2", "1,2", "0,1,2", "0,1,2,3"],
            width=8
        ).pack(side=tk.LEFT, padx=2)

        # Sampling parameters row
        sampling_row = ttk.Frame(frame)
        sampling_row.pack(fill=tk.X, pady=2)

        ttk.Label(sampling_row, text=UIText.REMOTE_TEMP, width=5).pack(side=tk.LEFT)
        self.remote_temp_var = tk.DoubleVar(value=0.8)
        ttk.Spinbox(
            sampling_row,
            textvariable=self.remote_temp_var,
            from_=0.0,
            to=2.0,
            increment=0.1,
            width=5
        ).pack(side=tk.LEFT, padx=2)

        ttk.Label(sampling_row, text=UIText.REMOTE_TOP_P, width=5).pack(
            side=tk.LEFT, padx=(8, 0)
        )
        self.remote_top_p_var = tk.DoubleVar(value=0.6)
        ttk.Spinbox(
            sampling_row,
            textvariable=self.remote_top_p_var,
            from_=0.0,
            to=1.0,
            increment=0.1,
            width=5
        ).pack(side=tk.LEFT, padx=2)

        ttk.Label(sampling_row, text=UIText.REMOTE_TOP_K, width=5).pack(
            side=tk.LEFT, padx=(8, 0)
        )
        self.remote_top_k_var = tk.IntVar(value=2)
        ttk.Spinbox(
            sampling_row,
            textvariable=self.remote_top_k_var,
            from_=0,
            to=100,
            width=4
        ).pack(side=tk.LEFT, padx=2)

        ttk.Label(sampling_row, text=UIText.REMOTE_REP_PEN, width=7).pack(
            side=tk.LEFT, padx=(8, 0)
        )
        self.remote_repeat_penalty_var = tk.DoubleVar(value=1.1)
        ttk.Spinbox(
            sampling_row,
            textvariable=self.remote_repeat_penalty_var,
            from_=1.0,
            to=2.0,
            increment=0.05,
            width=5
        ).pack(side=tk.LEFT, padx=2)

        # Batch and KV Cache row
        batch_kv_row = ttk.Frame(frame)
        batch_kv_row.pack(fill=tk.X, pady=2)

        ttk.Label(batch_kv_row, text=UIText.REMOTE_BATCH_SIZE, width=6).pack(side=tk.LEFT)
        self.remote_batch_size_var = tk.IntVar(value=512)
        ttk.Combobox(
            batch_kv_row,
            textvariable=self.remote_batch_size_var,
            values=[128, 256, 512, 1024, 2048, 4096],
            width=6
        ).pack(side=tk.LEFT, padx=2)

        ttk.Label(batch_kv_row, text=UIText.REMOTE_UBATCH_SIZE, width=6).pack(side=tk.LEFT, padx=(8, 0))
        self.remote_ubatch_size_var = tk.IntVar(value=512)
        ttk.Combobox(
            batch_kv_row,
            textvariable=self.remote_ubatch_size_var,
            values=[128, 256, 512, 1024, 2048, 4096],
            width=6
        ).pack(side=tk.LEFT, padx=2)

        ttk.Label(batch_kv_row, text=UIText.REMOTE_CACHE_TYPE_K, width=8).pack(
            side=tk.LEFT, padx=(8, 0)
        )
        self.remote_cache_type_k_var = tk.StringVar(value="f16")
        ttk.Combobox(
            batch_kv_row,
            textvariable=self.remote_cache_type_k_var,
            values=["f16", "f32", "bf16", "q8_0", "q4_0", "q4_1", "iq4_nl", "q5_0", "q5_1", "mxfp4"],
            width=8, state="readonly"
        ).pack(side=tk.LEFT, padx=2)

        ttk.Label(batch_kv_row, text="KV-V:").pack(side=tk.LEFT, padx=(10, 0))
        self.remote_cache_type_v_var = tk.StringVar(value="f16")
        ttk.Combobox(
            batch_kv_row,
            textvariable=self.remote_cache_type_v_var,
            values=["f16", "f32", "bf16", "q8_0", "q4_0", "q4_1", "iq4_nl", "q5_0", "q5_1", "mxfp4"],
            width=8, state="readonly"
        ).pack(side=tk.LEFT, padx=2)

        # Options row
        options_row = ttk.Frame(frame)
        options_row.pack(fill=tk.X, pady=(5, 2))

        self.remote_jinja_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(
            options_row,
            text=UIText.USE_JINJA,
            variable=self.remote_jinja_var
        ).pack(side=tk.LEFT)

        self.remote_flash_attn_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(
            options_row,
            text=UIText.FLASH_ATTN,
            variable=self.remote_flash_attn_var
        ).pack(side=tk.LEFT, padx=10)

        self.remote_fit_var = tk.BooleanVar(value=True)
        fit_check = ttk.Checkbutton(
            options_row,
            text=UIText.REMOTE_FIT,
            variable=self.remote_fit_var,
            command=self._on_fit_toggle
        )
        fit_check.pack(side=tk.LEFT, padx=10)

        # Fit target (预留显存空间)
        ttk.Label(options_row, text=UIText.REMOTE_FIT_TARGET).pack(side=tk.LEFT, padx=(5, 0))
        self.remote_fit_target_var = tk.IntVar(value=1024)
        fit_target_spin = ttk.Spinbox(
            options_row,
            textvariable=self.remote_fit_target_var,
            from_=128,
            to=8192,
            increment=128,
            width=6
        )
        fit_target_spin.pack(side=tk.LEFT, padx=2)

        self.remote_no_kv_offload_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(
            options_row,
            text=UIText.REMOTE_NO_KV_OFFLOAD,
            variable=self.remote_no_kv_offload_var
        ).pack(side=tk.LEFT, padx=10)

        self.remote_split_mode_row_var = tk.BooleanVar(value=False)
        self.remote_split_mode_check = ttk.Checkbutton(
            options_row,
            text=UIText.SPLIT_MODE_ROW,
            variable=self.remote_split_mode_row_var
        )
        self.remote_split_mode_check.pack(side=tk.LEFT, padx=10)

        self.remote_no_mmap_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(
            options_row,
            text=UIText.NO_MMAP,
            variable=self.remote_no_mmap_var
        ).pack(side=tk.LEFT, padx=10)

        self.remote_cuda_graph_opt_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(
            options_row,
            text=UIText.REMOTE_CUDA_GRAPH_OPT,
            variable=self.remote_cuda_graph_opt_var
        ).pack(side=tk.LEFT, padx=10)

        # Cache optimization row
        cache_opt_row = ttk.Frame(frame)
        cache_opt_row.pack(fill=tk.X, pady=(5, 2))

        ttk.Label(cache_opt_row, text=UIText.REMOTE_CACHE_REUSE, width=8).pack(
            side=tk.LEFT
        )
        self.remote_cache_reuse_var = tk.IntVar(value=0)
        ttk.Combobox(
            cache_opt_row,
            textvariable=self.remote_cache_reuse_var,
            values=[0, 64, 128, 256, 512, 1024],
            width=6
        ).pack(side=tk.LEFT, padx=2)

        ttk.Label(cache_opt_row, text=UIText.REMOTE_CACHE_RAM, width=12).pack(
            side=tk.LEFT, padx=(8, 0)
        )
        self.remote_cache_ram_var = tk.IntVar(value=8192)
        ttk.Combobox(
            cache_opt_row,
            textvariable=self.remote_cache_ram_var,
            values=[-1, 0, 2048, 4096, 8192, 16384, 32768],
            width=6
        ).pack(side=tk.LEFT, padx=2)

        self.remote_slot_save_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(
            cache_opt_row,
            text=UIText.REMOTE_SLOT_SAVE,
            variable=self.remote_slot_save_var
        ).pack(side=tk.LEFT, padx=(15, 0))

        # MoE expert GPU allocation section
        moe_frame = ttk.LabelFrame(frame, text="MoE专家GPU分配", padding="5")
        moe_frame.pack(fill=tk.X, pady=(5, 0))

        # Row 1: mode selector
        moe_row1 = ttk.Frame(moe_frame)
        moe_row1.pack(fill=tk.X)

        self.remote_moe_mode_var = tk.StringVar(value="不分配")
        ttk.Label(moe_row1, text="专家放置:").pack(side=tk.LEFT)
        moe_mode_combo = ttk.Combobox(
            moe_row1,
            textvariable=self.remote_moe_mode_var,
            values=["不分配", "专家放CPU", "按比例分配", "均分到GPU", "自定义"],
            state="readonly", width=12
        )
        moe_mode_combo.pack(side=tk.LEFT, padx=5)
        moe_mode_combo.bind("<<ComboboxSelected>>", self._on_moe_mode_changed)

        # Total layers (shared by 按比例/均分)
        self.moe_layers_label = ttk.Label(moe_row1, text="总层数:")
        self.remote_moe_layers_var = tk.IntVar(value=62)
        self.moe_layers_spin = ttk.Spinbox(
            moe_row1, textvariable=self.remote_moe_layers_var,
            from_=1, to=200, width=4,
            command=self._update_moe_preview
        )
        self.moe_layers_spin.bind("<KeyRelease>", lambda e: self._update_moe_preview())
        self.moe_detect_btn = ttk.Button(moe_row1, text="自动", width=4,
                                          command=self._detect_moe_layers)

        # GPU count (for 均分)
        self.moe_gpu_label = ttk.Label(moe_row1, text="GPU数:")
        self.remote_moe_gpu_count_var = tk.IntVar(value=3)
        self.moe_gpu_combo = ttk.Combobox(
            moe_row1,
            textvariable=self.remote_moe_gpu_count_var,
            values=[2, 3, 4, 5, 6, 7, 8],
            state="readonly", width=3
        )
        self.moe_gpu_combo.bind("<<ComboboxSelected>>", lambda e: self._update_moe_preview())

        # Row 2: weighted ratio (for "按比例分配")
        # GPU0(x16):GPU1(x4):GPU2(x4) ratio
        self.moe_ratio_row = ttk.Frame(moe_frame)
        ttk.Label(self.moe_ratio_row, text="GPU0(x16):").pack(side=tk.LEFT)
        self.remote_moe_ratio0_var = tk.IntVar(value=2)
        ratio0_spin = ttk.Spinbox(
            self.moe_ratio_row, textvariable=self.remote_moe_ratio0_var,
            from_=0, to=10, width=3, command=self._update_moe_preview
        )
        ratio0_spin.pack(side=tk.LEFT, padx=2)
        ratio0_spin.bind("<KeyRelease>", lambda e: self._update_moe_preview())
        ttk.Label(self.moe_ratio_row, text="GPU1(x4):").pack(side=tk.LEFT, padx=(8, 0))
        self.remote_moe_ratio1_var = tk.IntVar(value=1)
        ratio1_spin = ttk.Spinbox(
            self.moe_ratio_row, textvariable=self.remote_moe_ratio1_var,
            from_=0, to=10, width=3, command=self._update_moe_preview
        )
        ratio1_spin.pack(side=tk.LEFT, padx=2)
        ratio1_spin.bind("<KeyRelease>", lambda e: self._update_moe_preview())
        ttk.Label(self.moe_ratio_row, text="GPU2(x4):").pack(side=tk.LEFT, padx=(8, 0))
        self.remote_moe_ratio2_var = tk.IntVar(value=1)
        ratio2_spin = ttk.Spinbox(
            self.moe_ratio_row, textvariable=self.remote_moe_ratio2_var,
            from_=0, to=10, width=3, command=self._update_moe_preview
        )
        ratio2_spin.pack(side=tk.LEFT, padx=2)
        ratio2_spin.bind("<KeyRelease>", lambda e: self._update_moe_preview())

        # Row for "专家放CPU" mode: total layers + cpu layers
        self.moe_cpu_row = ttk.Frame(moe_frame)
        ttk.Label(self.moe_cpu_row, text="总层数:").pack(side=tk.LEFT)
        self.remote_moe_cpu_total_var = tk.IntVar(value=0)
        self.moe_cpu_total_spin = ttk.Spinbox(
            self.moe_cpu_row, textvariable=self.remote_moe_cpu_total_var,
            from_=1, to=200, width=4,
            command=self._update_moe_cpu_preview
        )
        self.moe_cpu_total_spin.pack(side=tk.LEFT, padx=2)
        self.moe_cpu_total_spin.bind("<KeyRelease>", lambda e: self._update_moe_cpu_preview())
        self.moe_cpu_detect_btn = ttk.Button(
            self.moe_cpu_row, text="自动", width=4,
            command=self._detect_moe_cpu_layers
        )
        self.moe_cpu_detect_btn.pack(side=tk.LEFT, padx=(0, 10))

        ttk.Label(self.moe_cpu_row, text="放CPU:").pack(side=tk.LEFT)
        self.remote_moe_cpu_layers_var = tk.IntVar(value=0)
        self.moe_cpu_layers_spin = ttk.Spinbox(
            self.moe_cpu_row, textvariable=self.remote_moe_cpu_layers_var,
            from_=0, to=200, width=4,
            command=self._update_moe_cpu_preview
        )
        self.moe_cpu_layers_spin.pack(side=tk.LEFT, padx=2)
        self.moe_cpu_layers_spin.bind("<KeyRelease>", lambda e: self._update_moe_cpu_preview())

        ttk.Label(self.moe_cpu_row, text="GPU比例:").pack(side=tk.LEFT, padx=(10, 0))
        self.remote_moe_ts_var = tk.StringVar(value="")
        self.moe_ts_entry = ttk.Entry(
            self.moe_cpu_row, textvariable=self.remote_moe_ts_var, width=10
        )
        self.moe_ts_entry.pack(side=tk.LEFT, padx=2)
        self.moe_ts_entry.bind("<KeyRelease>", lambda e: self._update_moe_cpu_preview())

        # Custom rule entry (for "自定义")
        self.moe_custom_row = ttk.Frame(moe_frame)
        self.remote_override_tensor_var = tk.StringVar()
        ttk.Label(self.moe_custom_row, text="规则:").pack(side=tk.LEFT)
        self.remote_override_tensor_entry = ttk.Entry(
            self.moe_custom_row,
            textvariable=self.remote_override_tensor_var,
            width=50
        )
        self.remote_override_tensor_entry.pack(side=tk.LEFT, padx=5, fill=tk.X, expand=True)

        # Preview label
        self.moe_preview_var = tk.StringVar()
        self.moe_preview_label = ttk.Label(
            moe_frame, textvariable=self.moe_preview_var,
            foreground="gray", wraplength=550, justify=tk.LEFT
        )

        # Initialize visibility
        self._on_moe_mode_changed()

        # Speculative decoding section (Lookup Decoding)
        spec_frame = ttk.LabelFrame(frame, text="推测解码 (Speculative Decoding)", padding="5")
        spec_frame.pack(fill=tk.X, pady=(5, 0))

        spec_row1 = ttk.Frame(spec_frame)
        spec_row1.pack(fill=tk.X)

        self.remote_lookup_enabled_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(
            spec_row1,
            text="缓存预测(Lookup)",
            variable=self.remote_lookup_enabled_var,
            command=self._on_lookup_toggle
        ).pack(side=tk.LEFT)

        ttk.Label(spec_row1, text="draft-max:").pack(side=tk.LEFT, padx=(15, 0))
        self.remote_draft_max_var = tk.IntVar(value=16)
        self.remote_draft_max_spin = ttk.Spinbox(
            spec_row1, textvariable=self.remote_draft_max_var,
            from_=2, to=128, width=4
        )
        self.remote_draft_max_spin.pack(side=tk.LEFT, padx=2)

        ttk.Label(spec_row1, text="draft-min:").pack(side=tk.LEFT, padx=(10, 0))
        self.remote_draft_min_var = tk.IntVar(value=2)
        self.remote_draft_min_spin = ttk.Spinbox(
            spec_row1, textvariable=self.remote_draft_min_var,
            from_=1, to=64, width=4
        )
        self.remote_draft_min_spin.pack(side=tk.LEFT, padx=2)

        # Lookup cache file path row (shown when enabled)
        self.spec_cache_row = ttk.Frame(spec_frame)
        ttk.Label(self.spec_cache_row, text="动态缓存:").pack(side=tk.LEFT)
        self.remote_lookup_cache_var = tk.StringVar(value="lookup_cache.bin")
        ttk.Entry(
            self.spec_cache_row,
            textvariable=self.remote_lookup_cache_var,
            width=30
        ).pack(side=tk.LEFT, padx=5, fill=tk.X, expand=True)
        ttk.Button(
            self.spec_cache_row, text="按模型", width=6,
            command=self._auto_lookup_cache_name
        ).pack(side=tk.LEFT, padx=(0, 5))
        ttk.Label(
            self.spec_cache_row,
            text="(llama_dir/cache/)",
            foreground="gray"
        ).pack(side=tk.LEFT)

        # Static cache path (optional)
        ttk.Label(self.spec_cache_row, text="静态缓存:").pack(side=tk.LEFT, padx=(10, 0))
        self.remote_lookup_static_var = tk.StringVar(value="")
        ttk.Entry(
            self.spec_cache_row,
            textvariable=self.remote_lookup_static_var,
            width=20
        ).pack(side=tk.LEFT, padx=5)

        # Initialize lookup visibility
        self._on_lookup_toggle()

        # Custom args row
        custom_row = ttk.Frame(frame)
        custom_row.pack(fill=tk.X, pady=(5, 0))

        ttk.Label(custom_row, text=UIText.REMOTE_CUSTOM_ARGS).pack(side=tk.LEFT)
        self.remote_custom_args_var = tk.StringVar()
        ttk.Entry(
            custom_row,
            textvariable=self.remote_custom_args_var,
            width=30
        ).pack(side=tk.LEFT, padx=5, fill=tk.X, expand=True)

        # Scan button row
        scan_btn_frame = ttk.Frame(frame)
        scan_btn_frame.pack(fill=tk.X, pady=(10, 0))

        ttk.Button(
            scan_btn_frame,
            text=UIText.REMOTE_SCAN_MODELS,
            command=self._scan_remote_models,
            width=12
        ).pack(side=tk.LEFT)

    def _create_remote_control_section(self, parent: ttk.Frame) -> None:
        """Create remote server control buttons section."""
        frame = ttk.LabelFrame(parent, text=UIText.REMOTE_CONTROL, padding="10")
        frame.pack(side=tk.LEFT, fill=tk.BOTH, padx=(5, 0))

        status_frame = ttk.Frame(frame)
        status_frame.pack(fill=tk.X, pady=(0, 10))

        ttk.Label(status_frame, text=UIText.REMOTE_STATUS, width=10).pack(side=tk.LEFT)
        self.remote_status_label = ttk.Label(
            status_frame,
            text=UIText.STATUS_STOPPED,
            style="Stopped.TLabel",
            width=12
        )
        self.remote_status_label.pack(side=tk.LEFT, padx=5)

        self.remote_start_btn = ttk.Button(
            frame,
            text=UIText.REMOTE_START,
            command=self._start_remote_server,
            style="Start.TButton",
            width=18
        )
        self.remote_start_btn.pack(fill=tk.X, pady=2)

        self.remote_stop_btn = ttk.Button(
            frame,
            text=UIText.REMOTE_STOP,
            command=self._stop_remote_server,
            style="Stop.TButton",
            width=18,
            state=tk.DISABLED
        )
        self.remote_stop_btn.pack(fill=tk.X, pady=2)

        ttk.Separator(frame, orient=tk.HORIZONTAL).pack(fill=tk.X, pady=10)

        ttk.Button(
            frame,
            text=UIText.REMOTE_CHECK_STATUS,
            command=self._check_remote_status,
            width=18
        ).pack(fill=tk.X, pady=2)

        ttk.Button(
            frame,
            text=UIText.REMOTE_FETCH_LOG,
            command=self._fetch_remote_log,
            width=18
        ).pack(fill=tk.X, pady=2)

        ttk.Button(
            frame,
            text=UIText.REMOTE_OPEN_WEBUI,
            command=self._open_remote_webui,
            width=18
        ).pack(fill=tk.X, pady=2)

    def _create_remote_multi_instance_section(self, parent: ttk.Frame) -> None:
        """Create remote multi-instance control section."""
        frame = ttk.LabelFrame(parent, text=UIText.REMOTE_MULTI_INSTANCE, padding="10")
        frame.pack(fill=tk.X)

        # Initialize instance configurations storage
        self.remote_instance_configs = {}  # {port: {"model": str, "mmproj": str}}

        top_row = ttk.Frame(frame)
        top_row.pack(fill=tk.X, pady=(0, 10))

        left_frame = ttk.Frame(top_row)
        left_frame.pack(side=tk.LEFT, fill=tk.X, expand=True)

        self.remote_multi_enabled_var = tk.BooleanVar(value=False)
        enable_check = ttk.Checkbutton(
            left_frame,
            text=UIText.REMOTE_MULTI_ENABLED,
            variable=self.remote_multi_enabled_var,
            command=self._on_remote_multi_mode_toggle
        )
        enable_check.pack(side=tk.LEFT)

        params_frame = ttk.Frame(left_frame)
        params_frame.pack(side=tk.LEFT, padx=(20, 0))

        ttk.Label(params_frame, text=UIText.REMOTE_START_PORT_LABEL).pack(side=tk.LEFT)
        self.remote_multi_start_port_var = tk.IntVar(value=8080)
        self.remote_multi_start_port_spin = ttk.Spinbox(
            params_frame,
            textvariable=self.remote_multi_start_port_var,
            from_=1024,
            to=65000,
            width=8,
            state=tk.DISABLED
        )
        self.remote_multi_start_port_spin.pack(side=tk.LEFT, padx=5)

        btn_frame = ttk.Frame(top_row)
        btn_frame.pack(side=tk.RIGHT)

        self.remote_multi_add_btn = ttk.Button(
            btn_frame,
            text=UIText.ADD_INSTANCE,
            command=self._add_remote_instance,
            width=10,
            state=tk.DISABLED
        )
        self.remote_multi_add_btn.pack(side=tk.LEFT, padx=2)

        self.remote_multi_remove_btn = ttk.Button(
            btn_frame,
            text=UIText.REMOVE_INSTANCE,
            command=self._remove_remote_instance,
            width=10,
            state=tk.DISABLED
        )
        self.remote_multi_remove_btn.pack(side=tk.LEFT, padx=2)

        self.remote_multi_start_all_btn = ttk.Button(
            btn_frame,
            text=UIText.REMOTE_START_ALL,
            command=self._start_all_remote_instances,
            width=12,
            state=tk.DISABLED
        )
        self.remote_multi_start_all_btn.pack(side=tk.LEFT, padx=2)

        self.remote_multi_stop_all_btn = ttk.Button(
            btn_frame,
            text=UIText.REMOTE_STOP_ALL,
            command=self._stop_all_remote_instances,
            width=12,
            state=tk.DISABLED
        )
        self.remote_multi_stop_all_btn.pack(side=tk.LEFT, padx=2)

        self.remote_multi_copy_urls_btn = ttk.Button(
            btn_frame,
            text=UIText.REMOTE_COPY_URLS,
            command=self._copy_remote_server_urls,
            width=12,
            state=tk.DISABLED
        )
        self.remote_multi_copy_urls_btn.pack(side=tk.LEFT, padx=2)

        # Hint label
        hint_frame = ttk.Frame(frame)
        hint_frame.pack(fill=tk.X, pady=(0, 5))
        ttk.Label(hint_frame, text=UIText.DOUBLE_CLICK_SELECT_MODEL,
                  font=("", 9, "italic")).pack(side=tk.LEFT)

        list_frame = ttk.Frame(frame)
        list_frame.pack(fill=tk.BOTH, expand=True)

        columns = ("port", "model", "mmproj", "gpu", "status")
        self.remote_instance_tree = ttk.Treeview(
            list_frame,
            columns=columns,
            show="headings",
            height=5
        )

        self.remote_instance_tree.heading("port", text=UIText.INSTANCE_PORT)
        self.remote_instance_tree.heading("model", text=UIText.INSTANCE_MODEL)
        self.remote_instance_tree.heading("mmproj", text=UIText.INSTANCE_MMPROJ)
        self.remote_instance_tree.heading("gpu", text=UIText.INSTANCE_GPU)
        self.remote_instance_tree.heading("status", text=UIText.INSTANCE_STATUS_COL)

        self.remote_instance_tree.column("port", width=60, anchor=tk.CENTER)
        self.remote_instance_tree.column("model", width=220, anchor=tk.W)
        self.remote_instance_tree.column("mmproj", width=130, anchor=tk.W)
        self.remote_instance_tree.column("gpu", width=60, anchor=tk.CENTER)
        self.remote_instance_tree.column("status", width=70, anchor=tk.CENTER)

        # Bind double-click to select model
        self.remote_instance_tree.bind("<Double-1>", self._on_remote_instance_double_click)

        scrollbar = ttk.Scrollbar(list_frame, orient=tk.VERTICAL, command=self.remote_instance_tree.yview)
        self.remote_instance_tree.configure(yscrollcommand=scrollbar.set)

        self.remote_instance_tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

        running_label_frame = ttk.Frame(frame)
        running_label_frame.pack(fill=tk.X, pady=(5, 0))

        self.remote_multi_running_label = ttk.Label(
            running_label_frame,
            text="Running: 0 / 0",
            style="Status.TLabel"
        )
        self.remote_multi_running_label.pack(side=tk.LEFT)

    def _on_remote_multi_mode_toggle(self) -> None:
        """Handle remote multi-instance mode toggle."""
        enabled = self.remote_multi_enabled_var.get()
        state = tk.NORMAL if enabled else tk.DISABLED

        self.remote_multi_start_port_spin.config(state=state)
        self.remote_multi_add_btn.config(state=state)
        self.remote_multi_remove_btn.config(state=state)
        self.remote_multi_start_all_btn.config(state=state if enabled else tk.DISABLED)

        if enabled:
            # Add default instances if tree is empty
            if not self.remote_instance_tree.get_children():
                self._add_default_remote_instances()
        else:
            for item in self.remote_instance_tree.get_children():
                self.remote_instance_tree.delete(item)
            self.remote_instance_configs.clear()
            self.remote_multi_running_label.config(text="Running: 0 / 0")

    def _add_default_remote_instances(self) -> None:
        """Add default instances when enabling multi-instance mode."""
        start_port = self.remote_multi_start_port_var.get()
        default_model = self.remote_model_var.get() or ""
        default_mmproj = self.remote_mmproj_var.get() or "(None)"
        default_gpu = self._main_gpu_str_to_int(self.remote_main_gpu_var.get())

        # Add 2 default instances
        for i in range(2):
            port = start_port + i
            self.remote_instance_configs[port] = {
                "model": default_model,
                "mmproj": default_mmproj if default_mmproj != "(None)" else "",
                "main_gpu": default_gpu
            }
            gpu_display = "Auto" if default_gpu < 0 else f"GPU {default_gpu}"
            self.remote_instance_tree.insert(
                "", tk.END,
                values=(port, default_model, default_mmproj, gpu_display, UIText.STATUS_STOPPED)
            )

        self._update_remote_instance_count_label()

    def _add_remote_instance(self) -> None:
        """Add a new instance to the multi-instance list."""
        # Find the next available port
        existing_ports = [
            int(self.remote_instance_tree.item(item, "values")[0])
            for item in self.remote_instance_tree.get_children()
        ]

        if existing_ports:
            next_port = max(existing_ports) + 1
        else:
            next_port = self.remote_multi_start_port_var.get()

        default_model = self.remote_model_var.get() or ""
        default_mmproj = self.remote_mmproj_var.get() or "(None)"
        default_gpu = self._main_gpu_str_to_int(self.remote_main_gpu_var.get())

        self.remote_instance_configs[next_port] = {
            "model": default_model,
            "mmproj": default_mmproj if default_mmproj != "(None)" else "",
            "main_gpu": default_gpu
        }
        gpu_display = "Auto" if default_gpu < 0 else f"GPU {default_gpu}"
        self.remote_instance_tree.insert(
            "", tk.END,
            values=(next_port, default_model, default_mmproj, gpu_display, UIText.STATUS_STOPPED)
        )

        self._update_remote_instance_count_label()

    def _remove_remote_instance(self) -> None:
        """Remove selected instance from the multi-instance list."""
        selected = self.remote_instance_tree.selection()
        if not selected:
            messagebox.showwarning(UIText.MSG_ERROR, UIText.MSG_NO_INSTANCE_SELECTED)
            return

        for item in selected:
            values = self.remote_instance_tree.item(item, "values")
            port = int(values[0])
            if port in self.remote_instance_configs:
                del self.remote_instance_configs[port]
            self.remote_instance_tree.delete(item)

        self._update_remote_instance_count_label()

    def _on_remote_instance_double_click(self, event) -> None:
        """Handle double-click on instance row to select model."""
        region = self.remote_instance_tree.identify_region(event.x, event.y)
        if region != "cell":
            return

        column_id = self.remote_instance_tree.identify_column(event.x)
        item = self.remote_instance_tree.identify_row(event.y)

        if not item:
            return

        values = self.remote_instance_tree.item(item, "values")
        port = int(values[0])

        # Map column index to column name
        # columns = ("port", "model", "mmproj", "gpu", "status")
        # #1=port, #2=model, #3=mmproj, #4=gpu, #5=status
        column_map = {"#1": "port", "#2": "model", "#3": "mmproj", "#4": "gpu", "#5": "status"}
        column_name = column_map.get(column_id, "")

        if column_name == "model":
            self._show_model_selection_dialog(item, port, "model")
        elif column_name == "mmproj":
            self._show_model_selection_dialog(item, port, "mmproj")
        elif column_name == "gpu":
            self._show_gpu_selection_dialog(item, port)

    def _show_model_selection_dialog(self, tree_item: str, port: int, model_type: str) -> None:
        """Show dialog to select model for an instance."""
        # Get model lists
        if model_type == "model":
            models = getattr(self, 'remote_model_files', [])
            if not models:
                messagebox.showwarning(
                    "提示",
                    "请先点击\"扫描模型\"按钮获取远程模型列表"
                )
                return
        else:
            mmproj_files = getattr(self, 'remote_mmproj_files', [])
            models = ["(None)"] + mmproj_files

        dialog = tk.Toplevel(self.root)
        dialog.title(f"选择{'模型' if model_type == 'model' else '视觉模型'} - 端口 {port}")
        dialog.geometry("500x400")
        dialog.transient(self.root)
        dialog.grab_set()

        # Center the dialog
        dialog.update_idletasks()
        x = self.root.winfo_x() + (self.root.winfo_width() - 500) // 2
        y = self.root.winfo_y() + (self.root.winfo_height() - 400) // 2
        dialog.geometry(f"+{x}+{y}")

        title_text = f"为端口 {port} 选择{'模型' if model_type == 'model' else '视觉模型'}:"
        if model_type == "mmproj":
            title_text += f"\n(找到 {len(models) - 1} 个视觉模型文件)"
        else:
            title_text += f"\n(找到 {len(models)} 个模型文件)"

        ttk.Label(dialog, text=title_text,
                  font=("", 10, "bold")).pack(pady=10)

        # Model listbox
        list_frame = ttk.Frame(dialog)
        list_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)

        listbox = tk.Listbox(list_frame, font=("Consolas", 10))
        scrollbar = ttk.Scrollbar(list_frame, orient=tk.VERTICAL, command=listbox.yview)
        listbox.configure(yscrollcommand=scrollbar.set)

        listbox.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

        for model in models:
            listbox.insert(tk.END, model)

        # Current selection
        current_value = self.remote_instance_configs.get(port, {}).get(model_type, "")
        if model_type == "mmproj" and not current_value:
            current_value = "(None)"

        for i, model in enumerate(models):
            if model == current_value:
                listbox.selection_set(i)
                listbox.see(i)
                break

        def on_select():
            selection = listbox.curselection()
            if selection:
                selected_model = listbox.get(selection[0])

                # Update config
                if port not in self.remote_instance_configs:
                    self.remote_instance_configs[port] = {"model": "", "mmproj": ""}

                if model_type == "mmproj":
                    self.remote_instance_configs[port]["mmproj"] = "" if selected_model == "(None)" else selected_model
                else:
                    self.remote_instance_configs[port]["model"] = selected_model

                # Update tree
                values = list(self.remote_instance_tree.item(tree_item, "values"))
                if model_type == "model":
                    values[1] = selected_model
                else:
                    values[2] = selected_model
                self.remote_instance_tree.item(tree_item, values=values)

            dialog.destroy()

        btn_frame = ttk.Frame(dialog)
        btn_frame.pack(fill=tk.X, padx=10, pady=10)

        ttk.Button(btn_frame, text="确定", command=on_select, width=10).pack(side=tk.LEFT, padx=5)
        ttk.Button(btn_frame, text="取消", command=dialog.destroy, width=10).pack(side=tk.LEFT, padx=5)

        # Double-click to select
        listbox.bind("<Double-1>", lambda e: on_select())

    def _show_gpu_selection_dialog(self, tree_item: str, port: int) -> None:
        """Show dialog to select GPU for an instance."""
        # GPU options: Auto, GPU 0, GPU 1, GPU 2, GPU 3
        gpu_options = ["Auto", "GPU 0", "GPU 1", "GPU 2", "GPU 3"]

        dialog = tk.Toplevel(self.root)
        dialog.title(f"选择GPU - 端口 {port}")
        dialog.geometry("300x280")
        dialog.transient(self.root)
        dialog.grab_set()

        # Center the dialog
        dialog.update_idletasks()
        x = self.root.winfo_x() + (self.root.winfo_width() - 300) // 2
        y = self.root.winfo_y() + (self.root.winfo_height() - 280) // 2
        dialog.geometry(f"+{x}+{y}")

        ttk.Label(dialog, text=f"为端口 {port} 选择GPU:",
                  font=("", 10, "bold")).pack(pady=10)

        # GPU listbox
        list_frame = ttk.Frame(dialog)
        list_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)

        listbox = tk.Listbox(list_frame, font=("Consolas", 11), height=6)
        listbox.pack(fill=tk.BOTH, expand=True)

        for gpu in gpu_options:
            listbox.insert(tk.END, gpu)

        # Current selection
        current_gpu = self.remote_instance_configs.get(port, {}).get("main_gpu", -1)
        current_value = "Auto" if current_gpu < 0 else f"GPU {current_gpu}"

        for i, gpu in enumerate(gpu_options):
            if gpu == current_value:
                listbox.selection_set(i)
                listbox.see(i)
                break

        def on_select():
            selection = listbox.curselection()
            if selection:
                selected_gpu = listbox.get(selection[0])

                # Convert to int value
                if selected_gpu == "Auto":
                    gpu_value = -1
                else:
                    gpu_value = int(selected_gpu.replace("GPU ", ""))

                # Update config
                if port not in self.remote_instance_configs:
                    self.remote_instance_configs[port] = {"model": "", "mmproj": "", "main_gpu": -1}
                self.remote_instance_configs[port]["main_gpu"] = gpu_value

                # Update tree
                values = list(self.remote_instance_tree.item(tree_item, "values"))
                values[3] = selected_gpu  # gpu column is index 3
                self.remote_instance_tree.item(tree_item, values=values)

            dialog.destroy()

        btn_frame = ttk.Frame(dialog)
        btn_frame.pack(fill=tk.X, padx=10, pady=10)

        ttk.Button(btn_frame, text="确定", command=on_select, width=10).pack(side=tk.LEFT, padx=5)
        ttk.Button(btn_frame, text="取消", command=dialog.destroy, width=10).pack(side=tk.LEFT, padx=5)

        # Double-click to select
        listbox.bind("<Double-1>", lambda e: on_select())

    def _update_remote_instance_count_label(self) -> None:
        """Update the running instances count label."""
        total = len(self.remote_instance_tree.get_children())
        running = 0
        for item in self.remote_instance_tree.get_children():
            status = self.remote_instance_tree.item(item, "values")[4]  # status is now index 4
            if status == UIText.STATUS_RUNNING:
                running += 1
        self.remote_multi_running_label.config(text=f"Running: {running} / {total}")

    def _start_all_remote_instances(self) -> None:
        """Start all remote instances in multi-instance mode with different models."""
        if not self.remote_process_manager.connected:
            self._remote_log_message("Please test connection first")
            return

        # Build instance list from tree
        instances = []
        for item in self.remote_instance_tree.get_children():
            values = self.remote_instance_tree.item(item, "values")
            port = int(values[0])
            model = values[1]
            mmproj = values[2] if values[2] != "(None)" else ""
            gpu_str = values[3]  # GPU column

            # Convert GPU display string to int
            if gpu_str == "Auto":
                main_gpu = -1
            else:
                main_gpu = int(gpu_str.replace("GPU ", ""))

            if not model:
                messagebox.showerror(
                    UIText.MSG_ERROR,
                    UIText.MSG_SELECT_MODEL_FOR_INSTANCE.format(port)
                )
                return

            instances.append({
                "port": port,
                "model": model,
                "mmproj": mmproj,
                "main_gpu": main_gpu
            })

        if not instances:
            messagebox.showerror(UIText.MSG_ERROR, "请先添加实例")
            return

        count = len(instances)
        self._remote_log_message(UIText.MSG_REMOTE_STARTING_INSTANCES.format(count))
        self.remote_multi_start_all_btn.config(state=tk.DISABLED)
        self.remote_multi_stop_all_btn.config(state=tk.DISABLED)
        self.remote_multi_add_btn.config(state=tk.DISABLED)
        self.remote_multi_remove_btn.config(state=tk.DISABLED)

        # Update tree to show starting status
        for item in self.remote_instance_tree.get_children():
            values = list(self.remote_instance_tree.item(item, "values"))
            values[4] = UIText.STATUS_STARTING  # status is now index 4
            self.remote_instance_tree.item(item, values=values)

        def start_thread():
            success, msg = self.remote_process_manager.start_multi_model_instances(
                llama_dir=self.remote_llama_dir_var.get(),
                model_dir=self.remote_model_dir_var.get(),
                instances=instances,
                context=self.remote_context_var.get(),
                gpu_layers=self.remote_gpu_layers_var.get(),
                parallel=self.remote_parallel_var.get(),
                use_jinja=self.remote_jinja_var.get(),
                use_flash_attn=self.remote_flash_attn_var.get(),
                fit=self.remote_fit_var.get(),
                split_mode_row=self.remote_split_mode_row_var.get(),
                no_mmap=self.remote_no_mmap_var.get(),
                batch_size=self.remote_batch_size_var.get(),
                cache_type_k=self.remote_cache_type_k_var.get(),
                cache_type_v=self.remote_cache_type_v_var.get(),
                cuda_graph_opt=self.remote_cuda_graph_opt_var.get(),
                temp=self.remote_temp_var.get(),
                top_p=self.remote_top_p_var.get(),
                top_k=self.remote_top_k_var.get(),
                repeat_penalty=self.remote_repeat_penalty_var.get(),
                custom_args=self.remote_custom_args_var.get(),
                override_tensor=self.remote_override_tensor_var.get(),
                moe_layers=self.remote_moe_layers_var.get() if self.remote_moe_mode_var.get() in ("均分到GPU", "按比例分配") else 0,
                moe_gpu_count=self.remote_moe_gpu_count_var.get() if self.remote_moe_mode_var.get() == "均分到GPU" else 0,
                moe_ratios=[self.remote_moe_ratio0_var.get(), self.remote_moe_ratio1_var.get(), self.remote_moe_ratio2_var.get()] if self.remote_moe_mode_var.get() == "按比例分配" else None,
                lookup_cache_dynamic=self.remote_lookup_cache_var.get() if self.remote_lookup_enabled_var.get() else "",
                lookup_cache_static=self.remote_lookup_static_var.get() if self.remote_lookup_enabled_var.get() else "",
                draft_max=self.remote_draft_max_var.get() if self.remote_lookup_enabled_var.get() else 0,
                draft_min=self.remote_draft_min_var.get() if self.remote_lookup_enabled_var.get() else 0,
            )
            self.root.after(0, lambda: self._on_all_remote_instances_started(success, msg, count))

        threading.Thread(target=start_thread, daemon=True).start()

    def _on_all_remote_instances_started(self, success: bool, msg: str, count: int) -> None:
        """Callback when all remote instances have been started."""
        if success:
            self._remote_log_message(f"{UIText.MSG_REMOTE_ALL_STARTED.format(count)}")
            self._remote_log_message(msg)

            for item in self.remote_instance_tree.get_children():
                values = list(self.remote_instance_tree.item(item, "values"))
                values[4] = UIText.STATUS_RUNNING  # status is now index 4
                self.remote_instance_tree.item(item, values=values)

            self.remote_multi_start_all_btn.config(state=tk.DISABLED)
            self.remote_multi_stop_all_btn.config(state=tk.NORMAL)
            self.remote_multi_copy_urls_btn.config(state=tk.NORMAL)
            self.remote_multi_add_btn.config(state=tk.DISABLED)
            self.remote_multi_remove_btn.config(state=tk.DISABLED)
            self.remote_multi_running_label.config(text=f"Running: {count} / {count}")
        else:
            self._remote_log_message(f"{UIText.MSG_REMOTE_START_FAILED}: {msg}")

            for item in self.remote_instance_tree.get_children():
                values = list(self.remote_instance_tree.item(item, "values"))
                values[4] = UIText.STATUS_STOPPED  # status is now index 4
                self.remote_instance_tree.item(item, values=values)

            self.remote_multi_start_all_btn.config(state=tk.NORMAL)
            self.remote_multi_stop_all_btn.config(state=tk.DISABLED)
            self.remote_multi_copy_urls_btn.config(state=tk.DISABLED)
            self.remote_multi_add_btn.config(state=tk.NORMAL)
            self.remote_multi_remove_btn.config(state=tk.NORMAL)
            self.remote_multi_running_label.config(text=f"Running: 0 / {count}")

    def _stop_all_remote_instances(self) -> None:
        """Stop all remote running instances."""
        self._remote_log_message(UIText.MSG_REMOTE_STOPPING_INSTANCES)
        self.remote_multi_stop_all_btn.config(state=tk.DISABLED)

        def stop_thread():
            success, msg = self.remote_process_manager.stop_all_instances()
            self.root.after(0, lambda: self._on_all_remote_instances_stopped(success, msg))

        threading.Thread(target=stop_thread, daemon=True).start()

    def _on_all_remote_instances_stopped(self, success: bool, msg: str) -> None:
        """Callback when all remote instances have been stopped."""
        self._remote_log_message(UIText.MSG_REMOTE_ALL_STOPPED)
        self._remote_log_message(msg)

        for item in self.remote_instance_tree.get_children():
            values = list(self.remote_instance_tree.item(item, "values"))
            values[4] = UIText.STATUS_STOPPED  # status is now index 4
            self.remote_instance_tree.item(item, values=values)

        self.remote_multi_start_all_btn.config(state=tk.NORMAL)
        self.remote_multi_stop_all_btn.config(state=tk.DISABLED)
        self.remote_multi_copy_urls_btn.config(state=tk.DISABLED)
        self.remote_multi_add_btn.config(state=tk.NORMAL)
        self.remote_multi_remove_btn.config(state=tk.NORMAL)

        count = len(self.remote_instance_tree.get_children())
        self.remote_multi_running_label.config(text=f"Running: 0 / {count}")

    def _copy_remote_server_urls(self) -> None:
        """Copy all remote server URLs to clipboard."""
        # Get ports from tree
        ports = []
        for item in self.remote_instance_tree.get_children():
            values = self.remote_instance_tree.item(item, "values")
            ports.append(int(values[0]))

        urls = [f"http://{self.remote_process_manager.remote_host}:{port}" for port in ports]

        if urls:
            url_text = "\n".join(urls)
            self.root.clipboard_clear()
            self.root.clipboard_append(url_text)
            self._remote_log_message(UIText.MSG_URLS_COPIED)
            self._remote_log_message(f"URLs: {', '.join(urls)}")
            self.statusbar_label.config(text=UIText.MSG_URLS_COPIED)
        else:
            self._remote_log_message("No URLs to copy")

    def _create_remote_log_section(self, parent: ttk.Frame) -> None:
        """Create remote server log output section."""
        frame = ttk.LabelFrame(parent, text=UIText.SERVER_OUTPUT, padding="5")
        frame.pack(fill=tk.BOTH, expand=True, pady=(0, 5))

        self.remote_log_text = scrolledtext.ScrolledText(
            frame,
            wrap=tk.WORD,
            font=("Consolas", 9),
            height=15,
            state=tk.DISABLED,
            bg="#1e1e1e",
            fg="#d4d4d4",
            insertbackground="white"
        )
        self.remote_log_text.pack(fill=tk.BOTH, expand=True)

        btn_frame = ttk.Frame(frame)
        btn_frame.pack(fill=tk.X, pady=(5, 0))

        ttk.Button(
            btn_frame,
            text=UIText.CLEAR_LOG,
            command=self._clear_remote_log,
            width=12
        ).pack(side=tk.LEFT)

        ttk.Button(
            btn_frame,
            text=UIText.COPY_LOG,
            command=self._copy_remote_log,
            width=12
        ).pack(side=tk.LEFT, padx=5)

    def _test_remote_connection(self) -> None:
        """Test connection to remote server."""
        self._remote_log_message(UIText.REMOTE_CONNECTING)
        self.remote_conn_status.config(text=UIText.REMOTE_CONNECTING, style="Loading.TLabel")

        def test_thread():
            self.remote_process_manager.set_credentials(
                self.remote_host_var.get(),
                self.remote_user_var.get(),
                self.remote_pass_var.get()
            )
            success, msg = self.remote_process_manager.test_connection()
            self.root.after(0, lambda: self._on_remote_connection_tested(success, msg))

        threading.Thread(target=test_thread, daemon=True).start()

    def _on_remote_connection_tested(self, success: bool, msg: str) -> None:
        """Callback when remote connection test completes."""
        if success:
            self.remote_conn_status.config(text=UIText.REMOTE_CONNECTED, style="Running.TLabel")
            self._remote_log_message(f"{UIText.REMOTE_CONNECTION_OK}: {msg}")
            # Auto-save credentials on successful connection
            self._save_ui_to_config()
            self.config.save()
        else:
            self.remote_conn_status.config(text=UIText.REMOTE_DISCONNECTED, style="Stopped.TLabel")
            self._remote_log_message(f"{UIText.REMOTE_CONNECTION_FAILED}: {msg}")

    def _scan_remote_models(self) -> None:
        """Scan for models on remote server."""
        if not self.remote_process_manager.connected:
            self._test_remote_connection()
            self._remote_log_message("Please test connection first")
            return

        self._remote_log_message(UIText.REMOTE_SCANNING)

        def scan_thread():
            model_dir = self.remote_model_dir_var.get()
            success_models, models = self.remote_process_manager.scan_models(model_dir)
            success_mmproj, mmprojs = self.remote_process_manager.scan_mmproj(model_dir)
            self.root.after(0, lambda: self._on_remote_models_scanned(
                success_models, models, success_mmproj, mmprojs
            ))

        threading.Thread(target=scan_thread, daemon=True).start()

    def _on_remote_models_scanned(
        self,
        success_models: bool,
        models: List[str],
        success_mmproj: bool,
        mmprojs: List[str]
    ) -> None:
        """Callback when remote model scan completes."""
        if success_models:
            self.remote_model_files = models
            self.remote_model_combo["values"] = models
            if models and not self.remote_model_var.get():
                self.remote_model_var.set(models[0])

        if success_mmproj:
            self.remote_mmproj_files = mmprojs
            self.remote_mmproj_combo["values"] = ["(None)"] + mmprojs
            if mmprojs and not self.remote_mmproj_var.get():
                self.remote_mmproj_var.set(mmprojs[0])

        total_found = len(self.remote_model_files)
        self._remote_log_message(UIText.REMOTE_FOUND_MODELS.format(total_found))

    def _start_remote_server(self) -> None:
        """Start llama-server on remote host."""
        if not self.remote_model_var.get():
            messagebox.showerror(UIText.MSG_ERROR, UIText.MSG_SELECT_MODEL)
            return

        if not self.remote_process_manager.connected:
            self._remote_log_message("Please test connection first")
            return

        self._remote_log_message(UIText.REMOTE_STARTING)
        self._update_remote_status(UIText.STATUS_LOADING, "Loading.TLabel")
        self.remote_start_btn.config(state=tk.DISABLED)

        def start_thread():
            mmproj = self.remote_mmproj_var.get()
            if mmproj == "(None)":
                mmproj = ""

            success, msg = self.remote_process_manager.start_server(
                llama_dir=self.remote_llama_dir_var.get(),
                model_dir=self.remote_model_dir_var.get(),
                model=self.remote_model_var.get(),
                mmproj=mmproj,
                port=self.remote_port_var.get(),
                context=self.remote_context_var.get(),
                gpu_layers=self.remote_gpu_layers_var.get(),
                parallel=self.remote_parallel_var.get(),
                use_jinja=self.remote_jinja_var.get(),
                use_flash_attn=self.remote_flash_attn_var.get(),
                fit=self.remote_fit_var.get(),
                fit_target=self.remote_fit_target_var.get(),
                no_kv_offload=self.remote_no_kv_offload_var.get(),
                split_mode_row=self.remote_split_mode_row_var.get(),
                no_mmap=self.remote_no_mmap_var.get(),
                gpu_select=self.remote_main_gpu_var.get(),
                batch_size=self.remote_batch_size_var.get(),
                ubatch_size=self.remote_ubatch_size_var.get(),
                cache_type_k=self.remote_cache_type_k_var.get(),
                cache_type_v=self.remote_cache_type_v_var.get(),
                cache_reuse=self.remote_cache_reuse_var.get(),
                cache_ram=self.remote_cache_ram_var.get(),
                slot_save=self.remote_slot_save_var.get(),
                cuda_graph_opt=self.remote_cuda_graph_opt_var.get(),
                temp=self.remote_temp_var.get(),
                top_p=self.remote_top_p_var.get(),
                top_k=self.remote_top_k_var.get(),
                repeat_penalty=self.remote_repeat_penalty_var.get(),
                custom_args=self.remote_custom_args_var.get(),
                override_tensor=self.remote_override_tensor_var.get(),
                moe_layers=self.remote_moe_layers_var.get() if self.remote_moe_mode_var.get() in ("均分到GPU", "按比例分配") else 0,
                moe_gpu_count=self.remote_moe_gpu_count_var.get() if self.remote_moe_mode_var.get() == "均分到GPU" else 0,
                moe_ratios=[self.remote_moe_ratio0_var.get(), self.remote_moe_ratio1_var.get(), self.remote_moe_ratio2_var.get()] if self.remote_moe_mode_var.get() == "按比例分配" else None,
                lookup_cache_dynamic=self.remote_lookup_cache_var.get() if self.remote_lookup_enabled_var.get() else "",
                lookup_cache_static=self.remote_lookup_static_var.get() if self.remote_lookup_enabled_var.get() else "",
                draft_max=self.remote_draft_max_var.get() if self.remote_lookup_enabled_var.get() else 0,
                draft_min=self.remote_draft_min_var.get() if self.remote_lookup_enabled_var.get() else 0,
            )
            self.root.after(0, lambda: self._on_remote_server_started(success, msg))

        threading.Thread(target=start_thread, daemon=True).start()

    def _on_remote_server_started(self, success: bool, msg: str) -> None:
        """Callback when remote server start attempt completes."""
        if success:
            self._remote_log_message(f"{UIText.REMOTE_STARTED}: {msg}")
            self._update_remote_status(UIText.STATUS_RUNNING, "Running.TLabel")
            self.remote_start_btn.config(state=tk.DISABLED)
            self.remote_stop_btn.config(state=tk.NORMAL)
        else:
            self._remote_log_message(f"{UIText.REMOTE_START_FAILED}: {msg}")
            self._update_remote_status(UIText.STATUS_STOPPED, "Stopped.TLabel")
            self.remote_start_btn.config(state=tk.NORMAL)
            self.remote_stop_btn.config(state=tk.DISABLED)

    def _stop_remote_server(self) -> None:
        """Stop llama-server on remote host."""
        self._remote_log_message(UIText.REMOTE_STOPPING)
        port = self.remote_port_var.get()

        def stop_thread():
            success, msg = self.remote_process_manager.stop_server(port)
            self.root.after(0, lambda: self._on_remote_server_stopped(success, msg))

        threading.Thread(target=stop_thread, daemon=True).start()

    def _on_remote_server_stopped(self, success: bool, msg: str) -> None:
        """Callback when remote server stop attempt completes."""
        if success:
            self._remote_log_message(f"{UIText.REMOTE_STOPPED}: {msg}")
        else:
            self._remote_log_message(f"Stop failed: {msg}")

        self._update_remote_status(UIText.STATUS_STOPPED, "Stopped.TLabel")
        self.remote_start_btn.config(state=tk.NORMAL)
        self.remote_stop_btn.config(state=tk.DISABLED)

    def _check_remote_status(self) -> None:
        """Check remote server status."""
        if not self.remote_process_manager.connected:
            self._remote_log_message("Please test connection first")
            return

        def check_thread():
            success, status = self.remote_process_manager.check_server_status(
                self.remote_port_var.get()
            )
            self.root.after(0, lambda: self._on_remote_status_checked(success, status))

        threading.Thread(target=check_thread, daemon=True).start()

    def _on_remote_status_checked(self, success: bool, status: str) -> None:
        """Callback when remote status check completes."""
        self._remote_log_message(f"Remote server status: {status}")

        if "Running" in status:
            self._update_remote_status(UIText.STATUS_RUNNING, "Running.TLabel")
            self.remote_start_btn.config(state=tk.DISABLED)
            self.remote_stop_btn.config(state=tk.NORMAL)
        else:
            self._update_remote_status(UIText.STATUS_STOPPED, "Stopped.TLabel")
            self.remote_start_btn.config(state=tk.NORMAL)
            self.remote_stop_btn.config(state=tk.DISABLED)

    def _fetch_remote_log(self) -> None:
        """Fetch and display remote server log."""
        if not self.remote_process_manager.connected:
            self._remote_log_message("Please test connection first")
            return

        self._remote_log_message("Fetching remote server log...")

        def fetch_thread():
            success, log_content = self.remote_process_manager.get_remote_log(
                self.remote_llama_dir_var.get(),
                tail_lines=200
            )
            self.root.after(0, lambda: self._on_remote_log_fetched(success, log_content))

        threading.Thread(target=fetch_thread, daemon=True).start()

    def _on_remote_log_fetched(self, success: bool, log_content: str) -> None:
        """Callback when remote log fetch completes."""
        if success and log_content:
            self._remote_log_message("=== Remote Server Log ===")
            # Split log content and add each line
            for line in log_content.split('\n'):
                if line.strip():
                    self._remote_log_message(line)
            self._remote_log_message("=== End of Log ===")
        else:
            self._remote_log_message(f"Failed to fetch log: {log_content}")

    def _open_remote_webui(self) -> None:
        """Open remote server WebUI in browser."""
        host = self.remote_host_var.get()
        port = self.remote_port_var.get()
        url = f"http://{host}:{port}"
        webbrowser.open(url)
        self._remote_log_message(f"{UIText.MSG_OPENED_WEBUI} {url}")

    def _update_remote_status(self, text: str, style: str) -> None:
        """Update remote status label."""
        self.remote_status_label.config(text=text, style=style)

    def _on_moe_mode_changed(self, event=None) -> None:
        """Handle MoE mode selection change."""
        mode = self.remote_moe_mode_var.get()

        # Hide all conditional widgets first
        self.moe_layers_label.pack_forget()
        self.moe_layers_spin.pack_forget()
        self.moe_detect_btn.pack_forget()
        self.moe_gpu_label.pack_forget()
        self.moe_gpu_combo.pack_forget()
        self.moe_ratio_row.pack_forget()
        self.moe_cpu_row.pack_forget()
        self.moe_custom_row.pack_forget()
        self.moe_preview_label.pack_forget()

        if mode == "不分配":
            self.remote_override_tensor_var.set("")
            self.moe_preview_var.set("")
        elif mode == "专家放CPU":
            self.remote_override_tensor_var.set("__CPU_MOE__")
            self.moe_cpu_row.pack(fill=tk.X, pady=(3, 0))
            self.moe_preview_label.pack(fill=tk.X, pady=(3, 0))
            self._update_moe_cpu_preview()
        elif mode == "按比例分配":
            self.moe_ratio_row.pack(fill=tk.X, pady=(3, 0))
            self.moe_preview_label.pack(fill=tk.X, pady=(3, 0))
            self._update_moe_preview()
        elif mode == "均分到GPU":
            self.moe_gpu_label.pack(side=tk.LEFT, padx=(10, 0))
            self.moe_gpu_combo.pack(side=tk.LEFT, padx=2)
            self.moe_preview_label.pack(fill=tk.X, pady=(3, 0))
            self._update_moe_preview()
        elif mode == "自定义":
            if self.remote_override_tensor_var.get() in ("", "__AUTO_SPLIT__", "__WEIGHTED_SPLIT__"):
                self.remote_override_tensor_var.set("")
            self.moe_custom_row.pack(fill=tk.X, pady=(3, 0))
            self.moe_preview_var.set("")

    def _update_moe_preview(self) -> None:
        """Update MoE preview when ratio/GPU count changes."""
        mode = self.remote_moe_mode_var.get()

        if mode == "按比例分配":
            try:
                ratios = [
                    self.remote_moe_ratio0_var.get(),
                    self.remote_moe_ratio1_var.get(),
                    self.remote_moe_ratio2_var.get(),
                ]
            except (tk.TclError, ValueError):
                return
            total_weight = sum(ratios)
            if total_weight <= 0:
                return
            ratio_str = ":".join(str(r) for r in ratios)
            pcie_labels = ["x16", "x4", "x4"]
            preview_parts = []
            for gpu_idx, weight in enumerate(ratios):
                if weight > 0:
                    pct = round(100 * weight / total_weight)
                    preview_parts.append(f"GPU{gpu_idx}({pcie_labels[gpu_idx]})≈{pct}%")
            self.moe_preview_var.set(
                f"-ts {ratio_str} 整层分配: " + ", ".join(preview_parts))
            self.remote_override_tensor_var.set("__WEIGHTED_SPLIT__")

        elif mode == "均分到GPU":
            try:
                gpu_count = self.remote_moe_gpu_count_var.get()
            except (tk.TclError, ValueError):
                return
            if gpu_count <= 0:
                return
            equal_str = ":".join(["1"] * gpu_count)
            self.moe_preview_var.set(
                f"-ts {equal_str} 均匀分配到 {gpu_count} 张GPU")
            self.remote_override_tensor_var.set("__AUTO_SPLIT__")

    def _detect_moe_layers(self) -> None:
        """Auto-detect MoE layer count from remote GGUF file."""
        model = self.remote_model_var.get()
        if not model:
            self._remote_log_message("请先选择模型")
            return

        self.moe_detect_btn.config(state=tk.DISABLED)
        self._remote_log_message(f"正在从 {model} 检测层数...")

        def detect_thread():
            success, result = self.remote_process_manager.get_gguf_block_count(
                self.remote_model_dir_var.get(), model)
            self.root.after(0, lambda: self._on_moe_layers_detected(success, result))

        threading.Thread(target=detect_thread, daemon=True).start()

    def _on_moe_layers_detected(self, success: bool, result) -> None:
        """Callback for MoE layer detection."""
        self.moe_detect_btn.config(state=tk.NORMAL)
        if success:
            self.remote_moe_layers_var.set(result)
            self._remote_log_message(f"检测到 {result} 层")
            self._update_moe_preview()
        else:
            self._remote_log_message(f"层数检测失败: {result}")

    def _update_moe_cpu_preview(self) -> None:
        """Update preview for expert-to-CPU mode."""
        try:
            total = self.remote_moe_cpu_total_var.get()
            cpu_layers = self.remote_moe_cpu_layers_var.get()
        except (tk.TclError, ValueError):
            return
        if total <= 0:
            return
        cpu_layers = max(0, min(cpu_layers, total))
        gpu_layers = total - cpu_layers
        ts_str = self.remote_moe_ts_var.get().strip()
        ts_suffix = f"|{ts_str}" if ts_str else ""
        ts_info = f"  -ts {ts_str}" if ts_str else ""
        if cpu_layers <= 0:
            self.moe_preview_var.set(f"expert all on GPU ({total} layers){ts_info}")
            self.remote_override_tensor_var.set("")
        elif cpu_layers >= total:
            self.moe_preview_var.set(
                f"--cpu-moe  all {total} layers expert -> CPU{ts_info}")
            self.remote_override_tensor_var.set(f"__CPU_MOE__{ts_suffix}")
        else:
            self.moe_preview_var.set(
                f"--n-cpu-moe {cpu_layers}  {cpu_layers}L expert -> CPU / "
                f"{gpu_layers}L expert -> GPU{ts_info}")
            self.remote_override_tensor_var.set(f"__NCPU_MOE_{cpu_layers}__{ts_suffix}")

    def _detect_moe_cpu_layers(self) -> None:
        """Auto-detect MoE layer count for cpu-moe mode."""
        model = self.remote_model_var.get()
        if not model:
            self._remote_log_message("请先选择模型")
            return

        self.moe_cpu_detect_btn.config(state=tk.DISABLED)
        self._remote_log_message(f"正在从 {model} 检测层数...")

        def detect_thread():
            success, result = self.remote_process_manager.get_gguf_block_count(
                self.remote_model_dir_var.get(), model)
            self.root.after(0, lambda: self._on_moe_cpu_layers_detected(success, result))

        threading.Thread(target=detect_thread, daemon=True).start()

    def _on_moe_cpu_layers_detected(self, success: bool, result) -> None:
        """Callback for MoE CPU layer detection."""
        self.moe_cpu_detect_btn.config(state=tk.NORMAL)
        if success:
            self.remote_moe_cpu_total_var.set(result)
            self.remote_moe_cpu_layers_var.set(result)
            self._remote_log_message(f"检测到 {result} 层")
            self._update_moe_cpu_preview()
        else:
            self._remote_log_message(f"层数检测失败: {result}")

    def _on_local_moe_mode_changed(self, event=None) -> None:
        """Handle local MoE mode selection change."""
        mode = self.local_moe_mode_var.get()

        # Hide all conditional widgets first
        self.local_moe_layers_label.pack_forget()
        self.local_moe_layers_spin.pack_forget()
        self.local_moe_gpu_label.pack_forget()
        self.local_moe_gpu_combo.pack_forget()
        self.local_moe_ratio_row.pack_forget()
        self.local_moe_cpu_row.pack_forget()
        self.local_moe_custom_row.pack_forget()
        self.local_moe_preview_label.pack_forget()

        if mode == "不分配":
            self.local_override_tensor_var.set("")
            self.local_moe_preview_var.set("")
        elif mode == "专家放CPU":
            self.local_override_tensor_var.set("__CPU_MOE__")
            self.local_moe_cpu_row.pack(fill=tk.X, pady=(3, 0))
            self.local_moe_preview_label.pack(fill=tk.X, pady=(3, 0))
            self._update_local_moe_cpu_preview()
        elif mode == "按比例分配":
            self.local_moe_ratio_row.pack(fill=tk.X, pady=(3, 0))
            self.local_moe_preview_label.pack(fill=tk.X, pady=(3, 0))
            self._update_local_moe_preview()
        elif mode == "均分到GPU":
            self.local_moe_gpu_label.pack(side=tk.LEFT, padx=(10, 0))
            self.local_moe_gpu_combo.pack(side=tk.LEFT, padx=2)
            self.local_moe_preview_label.pack(fill=tk.X, pady=(3, 0))
            self._update_local_moe_preview()
        elif mode == "自定义":
            if self.local_override_tensor_var.get() in ("", "__AUTO_SPLIT__", "__WEIGHTED_SPLIT__"):
                self.local_override_tensor_var.set("")
            self.local_moe_custom_row.pack(fill=tk.X, pady=(3, 0))
            self.local_moe_preview_var.set("")

    def _update_local_moe_preview(self) -> None:
        """Update local MoE preview when ratio/GPU count changes."""
        mode = self.local_moe_mode_var.get()

        if mode == "按比例分配":
            try:
                ratios = [
                    self.local_moe_ratio0_var.get(),
                    self.local_moe_ratio1_var.get(),
                    self.local_moe_ratio2_var.get(),
                ]
            except (tk.TclError, ValueError):
                return
            total_weight = sum(ratios)
            if total_weight <= 0:
                return
            ratio_str = ":".join(str(r) for r in ratios)
            preview_parts = []
            for gpu_idx, weight in enumerate(ratios):
                if weight > 0:
                    pct = round(100 * weight / total_weight)
                    preview_parts.append(f"GPU{gpu_idx}≈{pct}%")
            self.local_moe_preview_var.set(
                f"-ts {ratio_str} 整层分配: " + ", ".join(preview_parts))
            self.local_override_tensor_var.set("__WEIGHTED_SPLIT__")

        elif mode == "均分到GPU":
            try:
                gpu_count = self.local_moe_gpu_count_var.get()
            except (tk.TclError, ValueError):
                return
            if gpu_count <= 0:
                return
            equal_str = ":".join(["1"] * gpu_count)
            self.local_moe_preview_var.set(
                f"-ts {equal_str} 均匀分配到 {gpu_count} 张GPU")
            self.local_override_tensor_var.set("__AUTO_SPLIT__")

    def _update_local_moe_cpu_preview(self) -> None:
        """Update preview for local expert-to-CPU mode."""
        try:
            total = self.local_moe_cpu_total_var.get()
            cpu_layers = self.local_moe_cpu_layers_var.get()
        except (tk.TclError, ValueError):
            return
        if total <= 0:
            return
        cpu_layers = max(0, min(cpu_layers, total))
        gpu_layers = total - cpu_layers
        ts_str = self.local_moe_ts_var.get().strip()
        ts_suffix = f"|{ts_str}" if ts_str else ""
        ts_info = f"  -ts {ts_str}" if ts_str else ""
        if cpu_layers <= 0:
            self.local_moe_preview_var.set(f"expert all on GPU ({total} layers){ts_info}")
            self.local_override_tensor_var.set("")
        elif cpu_layers >= total:
            self.local_moe_preview_var.set(
                f"--cpu-moe  all {total} layers expert -> CPU{ts_info}")
            self.local_override_tensor_var.set(f"__CPU_MOE__{ts_suffix}")
        else:
            self.local_moe_preview_var.set(
                f"--n-cpu-moe {cpu_layers}  {cpu_layers}L expert -> CPU / "
                f"{gpu_layers}L expert -> GPU{ts_info}")
            self.local_override_tensor_var.set(f"__NCPU_MOE_{cpu_layers}__{ts_suffix}")

    def _on_local_lookup_toggle(self) -> None:
        """Toggle local lookup decoding cache path visibility."""
        if self.local_lookup_enabled_var.get():
            self.local_spec_cache_row.pack(fill=tk.X, pady=(3, 0))
        else:
            self.local_spec_cache_row.pack_forget()

    def _auto_local_lookup_cache_name(self) -> None:
        """Generate local lookup cache filename based on current model name."""
        model = self.model_var.get()
        if not model:
            self._log_message("请先选择模型")
            return
        base = model.replace(".gguf", "").replace(" ", "_")
        self.local_lookup_cache_var.set(f"lcd_{base}.bin")

    def _on_local_fit_toggle(self) -> None:
        """Toggle local Fit mode - when enabled, disable conflicting options.

        --fit on automatically calculates GPU layers and tensor split,
        so it conflicts with:
        - Manual GPU layers setting (-ngl)
        - Tensor split (--split-mode row)
        - Custom override tensor rules (-ot)
        """
        fit_enabled = self.fit_var.get()

        state = tk.DISABLED if fit_enabled else tk.NORMAL

        # Reset and disable conflicting options when Fit is enabled
        if fit_enabled:
            self.gpu_layers_var.set(-1)
            self.split_mode_row_var.set(False)
            self.local_override_tensor_var.set("")

        # Disable the widgets
        self.gpu_layers_spinbox.config(state=state)
        self.split_mode_check.config(state=state)
        self.local_override_tensor_entry.config(state=state)

        if fit_enabled:
            self._log_message("Fit(Auto) 已启用 - 将自动计算GPU层数和张量分配")
            self._log_message("提示: GPU层数将设为-1, 张量并行和MoE专家分配将被忽略")
        else:
            self._log_message("Fit(Auto) 已关闭")

    def _on_fit_toggle(self) -> None:
        """Toggle Fit mode - when enabled, disable conflicting options.
        
        --fit on automatically calculates GPU layers and tensor split,
        so it conflicts with:
        - Manual GPU layers setting (-ngl)
        - Tensor split (--split-mode row)
        - Custom override tensor rules (-ot)
        """
        fit_enabled = self.remote_fit_var.get()
        
        state = tk.DISABLED if fit_enabled else tk.NORMAL
        
        # Reset and disable conflicting options when Fit is enabled
        if fit_enabled:
            self.remote_gpu_layers_var.set(-1)
            self.remote_split_mode_row_var.set(False)
            self.remote_override_tensor_var.set("")
        
        # Disable the widgets
        self.remote_gpu_layers_spinbox.config(state=state)
        self.remote_split_mode_check.config(state=state)
        self.remote_override_tensor_entry.config(state=state)
        
        if fit_enabled:
            self._remote_log_message("Fit(Auto) 已启用 - 将自动计算GPU层数和张量分配")
            self._remote_log_message("提示: GPU层数将设为-1, 张量并行和MoE专家分配将被忽略")
        else:
            self._remote_log_message("Fit(Auto) 已关闭")

    def _on_lookup_toggle(self) -> None:
        """Toggle lookup decoding cache path visibility."""
        if self.remote_lookup_enabled_var.get():
            self.spec_cache_row.pack(fill=tk.X, pady=(3, 0))
        else:
            self.spec_cache_row.pack_forget()

    def _auto_lookup_cache_name(self) -> None:
        """Generate lookup cache filename based on current model name."""
        model = self.remote_model_var.get()
        if not model:
            self._remote_log_message("Please select a model first")
            return
        base = model.replace(".gguf", "").replace(" ", "_")
        self.remote_lookup_cache_var.set(f"lcd_{base}.bin")

    def _remote_log_message(self, message: str) -> None:
        """Add message to remote log output."""
        timestamp = datetime.now().strftime("%H:%M:%S")
        formatted = f"[{timestamp}] {message}\n"

        self.remote_log_text.config(state=tk.NORMAL)
        self.remote_log_text.insert(tk.END, formatted)
        self.remote_log_text.see(tk.END)
        self.remote_log_text.config(state=tk.DISABLED)

        self.logger.info(f"[Remote] {message}")

    def _clear_remote_log(self) -> None:
        """Clear the remote log output."""
        self.remote_log_text.config(state=tk.NORMAL)
        self.remote_log_text.delete(1.0, tk.END)
        self.remote_log_text.config(state=tk.DISABLED)

    def _copy_remote_log(self) -> None:
        """Copy remote log contents to clipboard."""
        content = self.remote_log_text.get(1.0, tk.END)
        self.root.clipboard_clear()
        self.root.clipboard_append(content)
        self.statusbar_label.config(text=UIText.LOG_COPIED)

    def _create_status_bar(self, parent: ttk.Frame) -> None:
        """Create status bar at bottom."""
        frame = ttk.Frame(parent)
        frame.pack(fill=tk.X, pady=(5, 0))

        self.statusbar_label = ttk.Label(
            frame,
            text=UIText.READY,
            style="Status.TLabel"
        )
        self.statusbar_label.pack(side=tk.LEFT)

        self.memory_label = ttk.Label(
            frame,
            text="",
            style="Status.TLabel"
        )
        self.memory_label.pack(side=tk.RIGHT)

    def _browse_directory(self) -> None:
        """Browse for llama.cpp directory."""
        directory = filedialog.askdirectory(
            initialdir=self.llama_dir,
            title="Select llama.cpp Directory"
        )
        if directory:
            self.dir_var.set(directory)
            self.llama_dir = directory
            self._scan_models()

    def _scan_models(self) -> None:
        """Scan models directory for GGUF files."""
        self.model_files = []
        self.mmproj_files = []

        models_dir = os.path.join(self.dir_var.get(), "models")

        if not os.path.exists(models_dir):
            self._log_message(f"{UIText.MSG_DIR_NOT_FOUND} {models_dir}")
            return

        try:
            for file in os.listdir(models_dir):
                if file.endswith(".gguf"):
                    if "mmproj" in file.lower():
                        self.mmproj_files.append(file)
                    elif ("-of-" not in file) or ("00001-of-" in file):
                        # Filter out non-model gguf files (imatrix, mtmd, tokenizer, etc.)
                        lower = file.lower()
                        if not any(x in lower for x in ["imatrix", "mtmd", "tokenizer", "vocab", "encoder", "decoder"]):
                            self.model_files.append(file)

            self.model_files.sort(key=lambda x: x.lower())
            self.mmproj_files.sort(key=lambda x: x.lower())

            self.model_combo["values"] = self.model_files
            self.mmproj_combo["values"] = ["(None)"] + self.mmproj_files

            if not self.model_var.get() and self.model_files:
                self.model_var.set(self.model_files[0])

            if not self.mmproj_var.get() and self.mmproj_files:
                self.mmproj_var.set(self.mmproj_files[0])

            self._log_message(
                UIText.MSG_FOUND_MODELS.format(
                    len(self.model_files),
                    len(self.mmproj_files)
                )
            )

        except Exception as e:
            self._log_message(f"{UIText.MSG_SCAN_ERROR} {e}")

    def _open_models_folder(self) -> None:
        """Open models folder in file explorer."""
        models_dir = os.path.join(self.dir_var.get(), "models")
        if os.path.exists(models_dir):
            os.startfile(models_dir)
        else:
            messagebox.showerror(
                UIText.MSG_ERROR,
                f"{UIText.MSG_DIR_NOT_FOUND} {models_dir}"
            )

    def _start_server(self) -> None:
        """Start the llama-server."""
        if not self.model_var.get():
            messagebox.showerror(UIText.MSG_ERROR, UIText.MSG_SELECT_MODEL)
            return

        self._save_ui_to_config()

        self._log_message(UIText.MSG_STARTING)
        self._update_status(UIText.STATUS_LOADING, "Loading.TLabel")
        self.start_btn.config(state=tk.DISABLED)

        def start_thread():
            success = self.process_manager.start(self.config)
            self.root.after(0, lambda: self._on_server_started(success))

        threading.Thread(target=start_thread, daemon=True).start()

    def _on_server_started(self, success: bool) -> None:
        """Callback when server start attempt completes."""
        if success:
            self._log_message(UIText.MSG_STARTED)
            self._update_status(UIText.STATUS_RUNNING, "Running.TLabel")
            self.start_btn.config(state=tk.DISABLED)
            self.stop_btn.config(state=tk.NORMAL)
        else:
            self._log_message(UIText.MSG_START_FAILED)
            self._update_status(UIText.STATUS_STOPPED, "Stopped.TLabel")
            self.start_btn.config(state=tk.NORMAL)
            self.stop_btn.config(state=tk.DISABLED)

    def _stop_server(self) -> None:
        """Stop the llama-server."""
        self._log_message(UIText.MSG_STOPPING)

        def stop_thread():
            success = self.process_manager.stop()
            self.root.after(0, lambda: self._on_server_stopped(success))

        threading.Thread(target=stop_thread, daemon=True).start()

    def _on_server_stopped(self, success: bool) -> None:
        """Callback when server stop attempt completes."""
        if success:
            self._log_message(UIText.MSG_STOPPED)
        else:
            self._log_message(UIText.MSG_STOP_WARNING)

        self._update_status(UIText.STATUS_STOPPED, "Stopped.TLabel")
        self.start_btn.config(state=tk.NORMAL)
        self.stop_btn.config(state=tk.DISABLED)

    def _open_webui(self) -> None:
        """Open the llama-server WebUI in browser."""
        port = self.port_var.get()
        url = f"http://localhost:{port}"
        webbrowser.open(url)
        self._log_message(f"{UIText.MSG_OPENED_WEBUI} {url}")

    def _test_api(self) -> None:
        """Test the server API."""
        host = self.host_var.get()
        if host == "0.0.0.0":
            host = "localhost"
        port = self.port_var.get()

        self._log_message(UIText.MSG_TESTING_API)

        def test_thread():
            result = self.status_monitor.check_health(host, port)
            props = self.status_monitor.get_props(host, port)
            self.root.after(0, lambda: self._on_test_complete(result, props))

        threading.Thread(target=test_thread, daemon=True).start()

    def _on_test_complete(
        self,
        result: Dict[str, Any],
        props: Optional[Dict[str, Any]]
    ) -> None:
        """Callback when API test completes."""
        status = result.get("status", "unknown")

        if status == "ok":
            self._log_message(UIText.MSG_API_OK)
            if props:
                model_alias = props.get("model_alias", "Unknown")
                modalities = props.get("modalities", {})
                vision = modalities.get("vision", False)
                self._log_message(f"  {UIText.MSG_MODEL} {model_alias}")
                vision_status = UIText.MSG_ENABLED if vision else UIText.MSG_DISABLED
                self._log_message(f"  {UIText.MSG_VISION} {vision_status}")
        else:
            error = result.get("error", "Unknown error")
            self._log_message(f"{UIText.MSG_API_FAILED}: {status} - {error}")

    def _save_config(self) -> None:
        """Save current configuration."""
        self._save_ui_to_config()
        self.config.save()
        self._log_message(UIText.CONFIG_SAVED)
        self.statusbar_label.config(text=UIText.CONFIG_SAVED)

    def _reset_defaults(self) -> None:
        """Reset to default configuration."""
        if messagebox.askyesno(UIText.MSG_CONFIRM, UIText.MSG_RESET_CONFIRM):
            self.config.config = ServerConfig.DEFAULT_CONFIG.copy()
            self._load_config_to_ui()
            self._log_message(UIText.MSG_SETTINGS_RESET)

    def _save_ui_to_config(self) -> None:
        """Save UI values to config object."""
        mmproj_value = self.mmproj_var.get()
        if mmproj_value == "(None)":
            mmproj_value = ""

        remote_mmproj_value = self.remote_mmproj_var.get()
        if remote_mmproj_value == "(None)":
            remote_mmproj_value = ""

        self.config.update({
            "llama_dir": self.dir_var.get(),
            "model_file": self.model_var.get(),
            "mmproj_file": mmproj_value,
            "host": self.host_var.get(),
            "port": self.port_var.get(),
            "context_size": self.context_var.get(),
            "gpu_layers": self.gpu_layers_var.get(),
            "main_gpu": self.main_gpu_var.get(),
            "temperature": self.temp_var.get(),
            "top_p": self.top_p_var.get(),
            "top_k": self.top_k_var.get(),
            "repeat_penalty": self.repeat_penalty_var.get(),
            "use_jinja": self.jinja_var.get(),
            "use_flash_attn": self.flash_attn_var.get(),
            "split_mode_row": self.split_mode_row_var.get(),
            "no_mmap": self.no_mmap_var.get(),
            "parallel": self.parallel_var.get(),
            "fit": self.fit_var.get(),
            "fit_target": self.fit_target_var.get(),
            "no_kv_offload": self.no_kv_offload_var.get(),
            "cuda_graph_opt": self.cuda_graph_opt_var.get(),
            "cache_reuse": self.cache_reuse_var.get(),
            "cache_ram": self.cache_ram_var.get(),
            "slot_save": self.slot_save_var.get(),
            "moe_mode": self.local_moe_mode_var.get(),
            "moe_layers": self.local_moe_layers_var.get(),
            "moe_gpu_count": self.local_moe_gpu_count_var.get(),
            "moe_ratio0": self.local_moe_ratio0_var.get(),
            "moe_ratio1": self.local_moe_ratio1_var.get(),
            "moe_ratio2": self.local_moe_ratio2_var.get(),
            "moe_cpu_total": self.local_moe_cpu_total_var.get(),
            "moe_cpu_layers": self.local_moe_cpu_layers_var.get(),
            "moe_ts": self.local_moe_ts_var.get(),
            "override_tensor": self.local_override_tensor_var.get(),
            "lookup_enabled": self.local_lookup_enabled_var.get(),
            "lookup_cache": self.local_lookup_cache_var.get(),
            "lookup_static": self.local_lookup_static_var.get(),
            "draft_max": self.local_draft_max_var.get(),
            "draft_min": self.local_draft_min_var.get(),
            "batch_size": self.batch_size_var.get(),
            "ubatch_size": self.ubatch_size_var.get(),
            "cache_type_k": self.cache_type_k_var.get(),
            "cache_type_v": self.cache_type_v_var.get(),
            "custom_args": self.custom_args_var.get(),
            "remote_host": self.remote_host_var.get(),
            "remote_user": self.remote_user_var.get(),
            "remote_password": self.remote_pass_var.get(),
            "remote_llama_dir": self.remote_llama_dir_var.get(),
            "remote_model_dir": self.remote_model_dir_var.get(),
            "remote_model": self.remote_model_var.get(),
            "remote_mmproj": remote_mmproj_value,
            "remote_port": self.remote_port_var.get(),
            "remote_context": self.remote_context_var.get(),
            "remote_gpu_layers": self.remote_gpu_layers_var.get(),
            "remote_parallel": self.remote_parallel_var.get(),
            "remote_main_gpu": self.remote_main_gpu_var.get(),
            "remote_temp": self.remote_temp_var.get(),
            "remote_top_p": self.remote_top_p_var.get(),
            "remote_top_k": self.remote_top_k_var.get(),
            "remote_repeat_penalty": self.remote_repeat_penalty_var.get(),
            "remote_batch_size": self.remote_batch_size_var.get(),
            "remote_ubatch_size": self.remote_ubatch_size_var.get(),
            "remote_cache_type_k": self.remote_cache_type_k_var.get(),
            "remote_cache_type_v": self.remote_cache_type_v_var.get(),
            "remote_cache_reuse": self.remote_cache_reuse_var.get(),
            "remote_cache_ram": self.remote_cache_ram_var.get(),
            "remote_slot_save": self.remote_slot_save_var.get(),
            "remote_cuda_graph_opt": self.remote_cuda_graph_opt_var.get(),
            "remote_fit": self.remote_fit_var.get(),
            "remote_fit_target": self.remote_fit_target_var.get(),
            "remote_no_kv_offload": self.remote_no_kv_offload_var.get(),
            "remote_custom_args": self.remote_custom_args_var.get(),
            "remote_use_jinja": self.remote_jinja_var.get(),
            "remote_use_flash_attn": self.remote_flash_attn_var.get(),
            "remote_split_mode_row": self.remote_split_mode_row_var.get(),
            "remote_no_mmap": self.remote_no_mmap_var.get(),
            "remote_moe_mode": self.remote_moe_mode_var.get(),
            "remote_moe_layers": self.remote_moe_layers_var.get(),
            "remote_moe_gpu_count": self.remote_moe_gpu_count_var.get(),
            "remote_moe_ratio0": self.remote_moe_ratio0_var.get(),
            "remote_moe_ratio1": self.remote_moe_ratio1_var.get(),
            "remote_moe_ratio2": self.remote_moe_ratio2_var.get(),
            "remote_moe_cpu_total": self.remote_moe_cpu_total_var.get(),
            "remote_moe_cpu_layers": self.remote_moe_cpu_layers_var.get(),
            "remote_moe_ts": self.remote_moe_ts_var.get(),
            "remote_override_tensor": self.remote_override_tensor_var.get(),
            "remote_lookup_enabled": self.remote_lookup_enabled_var.get(),
            "remote_lookup_cache": self.remote_lookup_cache_var.get(),
            "remote_lookup_static": self.remote_lookup_static_var.get(),
            "remote_draft_max": self.remote_draft_max_var.get(),
            "remote_draft_min": self.remote_draft_min_var.get(),
            "multi_instance_enabled": self.multi_enabled_var.get(),
            "multi_instance_count": self.multi_count_var.get(),
            "multi_start_port": self.multi_start_port_var.get(),
            "remote_multi_instance_enabled": self.remote_multi_enabled_var.get(),
            "remote_multi_start_port": self.remote_multi_start_port_var.get(),
            "remote_multi_instance_configs": self.remote_instance_configs,
        })

    def _load_config_to_ui(self) -> None:
        """Load config values to UI widgets."""
        self.dir_var.set(self.config.get("llama_dir", DEFAULT_LLAMA_DIR))
        self.model_var.set(self.config.get("model_file", ""))

        mmproj = self.config.get("mmproj_file", "")
        self.mmproj_var.set(mmproj if mmproj else "(None)")

        self.host_var.set(self.config.get("host", "0.0.0.0"))
        self.port_var.set(self.config.get("port", 8080))
        self.context_var.set(self.config.get("context_size", 32768))
        self.gpu_layers_var.set(self.config.get("gpu_layers", 99))
        local_gpu = self.config.get("main_gpu", "Auto")
        if isinstance(local_gpu, int):
            local_gpu = "Auto" if local_gpu < 0 else str(local_gpu)
        self.main_gpu_var.set(local_gpu)
        self.temp_var.set(self.config.get("temperature", 0.8))
        self.top_p_var.set(self.config.get("top_p", 0.6))
        self.top_k_var.set(self.config.get("top_k", 2))
        self.repeat_penalty_var.set(self.config.get("repeat_penalty", 1.1))
        self.jinja_var.set(self.config.get("use_jinja", True))
        self.flash_attn_var.set(self.config.get("use_flash_attn", False))
        self.split_mode_row_var.set(self.config.get("split_mode_row", False))
        self.no_mmap_var.set(self.config.get("no_mmap", False))
        self.parallel_var.set(self.config.get("parallel", 1))
        self.fit_var.set(self.config.get("fit", False))
        self.fit_target_var.set(self.config.get("fit_target", 1024))
        self.no_kv_offload_var.set(self.config.get("no_kv_offload", False))
        self.cuda_graph_opt_var.set(self.config.get("cuda_graph_opt", False))
        self.cache_reuse_var.set(self.config.get("cache_reuse", 0))
        self.cache_ram_var.set(self.config.get("cache_ram", -1))
        self.slot_save_var.set(self.config.get("slot_save", False))
        self.local_moe_mode_var.set(self.config.get("moe_mode", "不分配"))
        self.local_moe_layers_var.set(self.config.get("moe_layers", 62))
        self.local_moe_gpu_count_var.set(self.config.get("moe_gpu_count", 3))
        self.local_moe_ratio0_var.set(self.config.get("moe_ratio0", 2))
        self.local_moe_ratio1_var.set(self.config.get("moe_ratio1", 1))
        self.local_moe_ratio2_var.set(self.config.get("moe_ratio2", 1))
        self.local_moe_cpu_total_var.set(self.config.get("moe_cpu_total", 0))
        self.local_moe_cpu_layers_var.set(self.config.get("moe_cpu_layers", 0))
        self.local_moe_ts_var.set(self.config.get("moe_ts", ""))
        self.local_override_tensor_var.set(self.config.get("override_tensor", ""))
        self._on_local_moe_mode_changed()
        self.local_lookup_enabled_var.set(self.config.get("lookup_enabled", False))
        self.local_lookup_cache_var.set(self.config.get("lookup_cache", "lookup_cache.bin"))
        self.local_lookup_static_var.set(self.config.get("lookup_static", ""))
        self.local_draft_max_var.set(self.config.get("draft_max", 16))
        self.local_draft_min_var.set(self.config.get("draft_min", 2))
        self._on_local_lookup_toggle()
        self._on_local_fit_toggle()
        self.batch_size_var.set(self.config.get("batch_size", 512))
        self.ubatch_size_var.set(self.config.get("ubatch_size", 512))
        self.cache_type_k_var.set(self.config.get("cache_type_k", "f16"))
        self.cache_type_v_var.set(self.config.get("cache_type_v", "f16"))
        self.custom_args_var.set(self.config.get("custom_args", ""))

        self.remote_host_var.set(self.config.get("remote_host", DEFAULT_REMOTE_HOST))
        self.remote_user_var.set(self.config.get("remote_user", DEFAULT_REMOTE_USER))
        # Password: use default "admin" if empty or not set, and save it
        remote_pass = self.config.get("remote_password", "") or "admin"
        self.remote_pass_var.set(remote_pass)
        if not self.config.get("remote_password"):
            self.config.set("remote_password", "admin")
            self.config.save()
        self.remote_llama_dir_var.set(self.config.get("remote_llama_dir", DEFAULT_REMOTE_LLAMA_DIR))
        self.remote_model_dir_var.set(self.config.get("remote_model_dir", DEFAULT_REMOTE_MODEL_DIR))
        self.remote_model_var.set(self.config.get("remote_model", ""))

        remote_mmproj = self.config.get("remote_mmproj", "")
        self.remote_mmproj_var.set(remote_mmproj if remote_mmproj else "(None)")

        self.remote_port_var.set(self.config.get("remote_port", 8080))
        self.remote_context_var.set(self.config.get("remote_context", 32768))
        self.remote_gpu_layers_var.set(self.config.get("remote_gpu_layers", 99))
        self.remote_parallel_var.set(self.config.get("remote_parallel", 4))
        # Handle backwards compatibility: old config may have int, new config has string
        remote_gpu = self.config.get("remote_main_gpu", "Auto")
        if isinstance(remote_gpu, int):
            remote_gpu = "Auto" if remote_gpu < 0 else str(remote_gpu)
        self.remote_main_gpu_var.set(remote_gpu)
        self.remote_temp_var.set(self.config.get("remote_temp", 0.8))
        self.remote_top_p_var.set(self.config.get("remote_top_p", 0.6))
        self.remote_top_k_var.set(self.config.get("remote_top_k", 2))
        self.remote_repeat_penalty_var.set(self.config.get("remote_repeat_penalty", 1.1))
        self.remote_batch_size_var.set(self.config.get("remote_batch_size", 512))
        self.remote_ubatch_size_var.set(self.config.get("remote_ubatch_size", 512))
        self.remote_cache_type_k_var.set(self.config.get("remote_cache_type_k", "f16"))
        self.remote_cache_type_v_var.set(self.config.get("remote_cache_type_v", "f16"))
        self.remote_cache_reuse_var.set(self.config.get("remote_cache_reuse", 0))
        self.remote_cache_ram_var.set(self.config.get("remote_cache_ram", 8192))
        self.remote_slot_save_var.set(self.config.get("remote_slot_save", False))
        self.remote_cuda_graph_opt_var.set(self.config.get("remote_cuda_graph_opt", True))
        self.remote_fit_var.set(self.config.get("remote_fit", True))
        self.remote_fit_target_var.set(self.config.get("remote_fit_target", 1024))
        self.remote_no_kv_offload_var.set(self.config.get("remote_no_kv_offload", False))
        self._on_fit_toggle()
        self.remote_custom_args_var.set(self.config.get("remote_custom_args", ""))
        self.remote_jinja_var.set(self.config.get("remote_use_jinja", True))
        self.remote_flash_attn_var.set(self.config.get("remote_use_flash_attn", False))
        self.remote_split_mode_row_var.set(self.config.get("remote_split_mode_row", False))
        self.remote_no_mmap_var.set(self.config.get("remote_no_mmap", False))
        self.remote_moe_mode_var.set(self.config.get("remote_moe_mode", "不分配"))
        self.remote_moe_layers_var.set(self.config.get("remote_moe_layers", 62))
        self.remote_moe_gpu_count_var.set(self.config.get("remote_moe_gpu_count", 3))
        self.remote_moe_ratio0_var.set(self.config.get("remote_moe_ratio0", 2))
        self.remote_moe_ratio1_var.set(self.config.get("remote_moe_ratio1", 1))
        self.remote_moe_ratio2_var.set(self.config.get("remote_moe_ratio2", 1))
        self.remote_moe_cpu_total_var.set(self.config.get("remote_moe_cpu_total", 0))
        self.remote_moe_cpu_layers_var.set(self.config.get("remote_moe_cpu_layers", 0))
        self.remote_moe_ts_var.set(self.config.get("remote_moe_ts", ""))
        self.remote_override_tensor_var.set(self.config.get("remote_override_tensor", ""))
        self._on_moe_mode_changed()

        # Lookup decoding config
        self.remote_lookup_enabled_var.set(self.config.get("remote_lookup_enabled", False))
        self.remote_lookup_cache_var.set(self.config.get("remote_lookup_cache", "lookup_cache.bin"))
        self.remote_lookup_static_var.set(self.config.get("remote_lookup_static", ""))
        self.remote_draft_max_var.set(self.config.get("remote_draft_max", 16))
        self.remote_draft_min_var.set(self.config.get("remote_draft_min", 2))
        self._on_lookup_toggle()

        self.multi_enabled_var.set(self.config.get("multi_instance_enabled", False))
        self.multi_count_var.set(self.config.get("multi_instance_count", 3))
        self.multi_start_port_var.set(self.config.get("multi_start_port", 8080))

        if self.multi_enabled_var.get():
            self._on_multi_mode_toggle()

        self.remote_multi_enabled_var.set(self.config.get("remote_multi_instance_enabled", False))
        self.remote_multi_start_port_var.set(self.config.get("remote_multi_start_port", 8080))

        # Load saved instance configurations
        saved_configs = self.config.get("remote_multi_instance_configs", {})
        if saved_configs:
            # Convert string keys back to int (JSON serialization converts them to strings)
            self.remote_instance_configs = {
                int(k): v for k, v in saved_configs.items()
            }
            # Rebuild tree from saved configs
            for port, cfg in sorted(self.remote_instance_configs.items()):
                model = cfg.get("model", "")
                mmproj = cfg.get("mmproj", "") or "(None)"
                main_gpu = cfg.get("main_gpu", -1)
                gpu_display = "Auto" if main_gpu < 0 else f"GPU {main_gpu}"
                self.remote_instance_tree.insert(
                    "", tk.END,
                    values=(port, model, mmproj, gpu_display, UIText.STATUS_STOPPED)
                )

        if self.remote_multi_enabled_var.get():
            self._on_remote_multi_mode_toggle()

    def _main_gpu_str_to_int(self, value: str) -> int:
        """Convert main GPU string to integer value."""
        if value == "Auto":
            return -1
        try:
            return int(value.replace("GPU ", ""))
        except (ValueError, AttributeError):
            return -1

    def _main_gpu_int_to_str(self, value: int) -> str:
        """Convert main GPU integer to string value."""
        if value < 0:
            return "Auto"
        return f"GPU {value}"

    def _update_status(self, text: str, style: str) -> None:
        """Update status label."""
        self.status_label.config(text=text, style=style)

    def _log_message(self, message: str) -> None:
        """Add message to log output."""
        timestamp = datetime.now().strftime("%H:%M:%S")
        formatted = f"[{timestamp}] {message}\n"

        self.log_text.config(state=tk.NORMAL)
        self.log_text.insert(tk.END, formatted)

        if self.auto_scroll_var.get():
            self.log_text.see(tk.END)

        self.log_text.config(state=tk.DISABLED)

        self.logger.info(message)

    def _clear_log(self) -> None:
        """Clear the log output."""
        self.log_text.config(state=tk.NORMAL)
        self.log_text.delete(1.0, tk.END)
        self.log_text.config(state=tk.DISABLED)

    def _copy_log(self) -> None:
        """Copy log contents to clipboard."""
        content = self.log_text.get(1.0, tk.END)
        self.root.clipboard_clear()
        self.root.clipboard_append(content)
        self.statusbar_label.config(text=UIText.LOG_COPIED)

    def _start_periodic_updates(self) -> None:
        """Start periodic status and output updates."""
        self._update_output()
        self._update_server_status()

    def _update_output(self) -> None:
        """Update log with server output."""
        has_output = False

        if self.process_manager.is_running():
            output = self.process_manager.get_output()
            if output:
                self.log_text.config(state=tk.NORMAL)
                self.log_text.insert(tk.END, output)
                has_output = True

        if self.multi_process_manager.get_running_count() > 0:
            multi_outputs = self.multi_process_manager.get_all_output()
            if multi_outputs:
                self.log_text.config(state=tk.NORMAL)
                for port, output in sorted(multi_outputs.items()):
                    self.log_text.insert(tk.END, output)
                has_output = True

        if has_output:
            if self.auto_scroll_var.get():
                self.log_text.see(tk.END)
            self.log_text.config(state=tk.DISABLED)

        self.output_update_job = self.root.after(100, self._update_output)

    def _update_server_status(self) -> None:
        """Update server status periodically."""
        if self.process_manager.is_running():
            if self.status_label.cget("text") != UIText.STATUS_RUNNING:
                self._update_status(UIText.STATUS_RUNNING, "Running.TLabel")
        else:
            if self.process_manager.running:
                self._log_message(UIText.MSG_UNEXPECTED_TERMINATE)
                self.process_manager.running = False

            if self.status_label.cget("text") != UIText.STATUS_STOPPED:
                self._update_status(UIText.STATUS_STOPPED, "Stopped.TLabel")
                self.start_btn.config(state=tk.NORMAL)
                self.stop_btn.config(state=tk.DISABLED)

        self.status_update_job = self.root.after(1000, self._update_server_status)

    def _on_closing(self) -> None:
        """Handle window close event."""
        has_running_single = self.process_manager.is_running()
        has_running_multi = self.multi_process_manager.get_running_count() > 0

        if has_running_single or has_running_multi:
            if messagebox.askyesno(
                UIText.MSG_EXIT_CONFIRM,
                UIText.MSG_SERVER_RUNNING_EXIT
            ):
                if has_running_single:
                    self.process_manager.stop()
                if has_running_multi:
                    self.multi_process_manager.stop_all()
            else:
                return

        if self.status_update_job:
            self.root.after_cancel(self.status_update_job)
        if self.output_update_job:
            self.root.after_cancel(self.output_update_job)

        self._save_ui_to_config()
        self.config.save()

        self.logger.info("Application closing")
        self.root.destroy()

    def run(self) -> None:
        """Run the application main loop."""
        self.root.mainloop()


def main():
    """Main entry point."""
    try:
        app = LauncherGUI()
        app.run()
    except Exception as e:
        import traceback
        error_msg = traceback.format_exc()
        logger.error(f"Application crashed: {error_msg}")
        # Also write to a crash log file
        crash_log_path = os.path.join(os.path.dirname(__file__), "crash.log")
        with open(crash_log_path, "w", encoding="utf-8") as f:
            f.write(f"Crash at {datetime.now()}\n")
            f.write(error_msg)
        print(f"Application crashed. See {crash_log_path} for details.")
        input("Press Enter to exit...")


if __name__ == "__main__":
    main()
