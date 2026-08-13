"""structlog 配置：JSON 结构化日志 + 文件输出 + 屏蔽第三方冗余。

吸收 sage-research/base/logger.py 的思想（文件输出、屏蔽第三方），
但用 structlog 输出 JSON，便于生产环境日志系统收集检索。
"""
import logging
import os
import sys

import structlog


def setup_logging(log_dir: str = "logs", level: int = logging.INFO) -> None:
    # 1. 屏蔽第三方库冗余日志（吸收 logger.py）
    for noisy in ["huggingface_hub", "transformers", "peft", "httpx", "urllib3", "datasets", "httpcore"]:
        logging.getLogger(noisy).setLevel(logging.WARNING)

    # 2. 标准 logging：文件 + 控制台双输出（吸收 logger.py）
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, "medisense.log")

    root = logging.getLogger()
    root.setLevel(level)
    if root.handlers:   # 防重复初始化
        return

    console = logging.StreamHandler(sys.stderr)
    console.setFormatter(logging.Formatter("%(message)s"))
    root.addHandler(console)

    file_handler = logging.FileHandler(log_file, encoding="utf-8")
    file_handler.setFormatter(logging.Formatter("%(message)s"))
    root.addHandler(file_handler)

    # 3. structlog：JSON 输出到标准 logging
    structlog.configure(
        processors=[
            structlog.contextvars.merge_contextvars,
            structlog.processors.add_log_level,
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.processors.format_exc_info,
            structlog.processors.JSONRenderer(),
        ],
        wrapper_class=structlog.make_filtering_bound_logger(level),
        logger_factory=structlog.stdlib.LoggerFactory(),
        cache_logger_on_first_use=True,
    )
