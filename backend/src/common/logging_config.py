"""structlog 配置：JSON 结构化日志 + 按模块分文件输出 + 屏蔽第三方冗余。

专家版 / 患者版日志按 logger 名（模块名）前缀路由到不同文件：
  - logs/expert.log   模块名含 "expert" 的（专家版工作流）
  - logs/patient.log  模块名含 "patient" 的（患者版工作流）
  - logs/app.log      其余（common / database / ingestion / ml 等共享底座）
控制台(stderr)输出全部日志，便于本地开发实时查看。

功能总览：
1. 压制各类第三方库的DEBUG/INFO噪音日志，减少输出干扰
2. 标准logging根日志：控制台(stderr) + 三个本地文件
   handler formatter只用 `%(message)s`：structlog已经生成完整JSON字符串，底层不再二次格式化
3. structlog桥接到 stdlib logging：structlog负责组装JSON，IO输出交给原生logging的handler
4. 每条日志带 `logger` 字段（add_logger_name），用于筛选/排查
5. 防重复初始化：避免FastAPI reload、多次调用setup_logging造成重复挂载handler，日志重复打印

⚠️ 注意调用顺序约束：
1. 必须**先设置FAISS_NO_AVX2_WARNING环境变量**（放在本函数调用之前，因为faiss是C层print，不走logging）
2. 项目入口 main.py 只调用一次 setup_logging()，不要在模块顶层直接执行
3. 全项目业务代码统一使用 structlog.get_logger()，避免混用logging.getLogger产生普通文本日志污染JSON流
"""
import logging
import os
import sys

import structlog

# 屏蔽第三方库冗余日志的 logger 名前缀
_NOISY_LOGGERS = [
    "huggingface_hub", "transformers", "peft", "httpx",
    "urllib3", "datasets", "httpcore",
]


class _LogRouter(logging.Filter):
    """按 logger 名（模块名）子串路由日志到对应文件。

    contains 模式：模块名含该子串才输出到本 handler。
    excludes 模式：模块名含任一排除子串则跳过（兜底文件用）。
    """

    def __init__(self, contains: str | None = None, excludes: tuple[str, ...] = ()):
        super().__init__()
        self.contains = contains
        self.excludes = excludes

    def filter(self, record: logging.LogRecord) -> bool:
        name = record.name
        if self.contains is not None:
            return self.contains in name
        return not any(x in name for x in self.excludes)


def _make_file_handler(log_dir: str, filename: str, router: _LogRouter) -> logging.FileHandler:
    """创建带路由 Filter 的文件 handler"""
    handler = logging.FileHandler(os.path.join(log_dir, filename), encoding="utf-8")
    handler.setFormatter(logging.Formatter("%(message)s"))
    handler.addFilter(router)
    return handler


def setup_logging(log_dir: str = "logs", level: int = logging.INFO) -> None:
    # 1. 屏蔽第三方库冗余日志
    for noisy in _NOISY_LOGGERS:
        logging.getLogger(noisy).setLevel(logging.WARNING)

    # 2. 初始化标准库 logging root logger
    os.makedirs(log_dir, exist_ok=True)

    root = logging.getLogger()
    root.setLevel(level)
    if root.handlers:   # 防重复初始化
        return

    # 控制台输出：全部日志（本地开发实时看）
    console = logging.StreamHandler(sys.stderr)
    console.setFormatter(logging.Formatter("%(message)s"))
    root.addHandler(console)

    # 三个文件，按 logger 名路由
    root.addHandler(_make_file_handler(log_dir, "expert.log", _LogRouter(contains="expert")))
    root.addHandler(_make_file_handler(log_dir, "patient.log", _LogRouter(contains="patient")))
    root.addHandler(_make_file_handler(log_dir, "app.log", _LogRouter(excludes=("expert", "patient"))))

    # 3. structlog 配置：桥接到 stdlib logging
    structlog.configure(
        processors=[
            structlog.contextvars.merge_contextvars,
            structlog.processors.add_log_level,
            structlog.stdlib.add_logger_name,   # 每条日志带 "logger" 字段（模块名）
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.processors.format_exc_info,
            structlog.processors.JSONRenderer(),
        ],
        wrapper_class=structlog.make_filtering_bound_logger(level),
        logger_factory=structlog.stdlib.LoggerFactory(),
        cache_logger_on_first_use=True,
    )
