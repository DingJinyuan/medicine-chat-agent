"""structlog 配置：JSON 结构化日志 + 文件输出 + 屏蔽第三方冗余。

吸收 sage‑research/base/logger.py 的思想（文件输出、屏蔽第三方），
但用 structlog 输出 JSON，便于 Loki / ELK 这类日志系统收集检索。

功能总览：
1. 压制各类第三方库的DEBUG/INFO噪音日志，减少输出干扰
2. 标准logging根日志：控制台(stderr) + 本地文件 双路输出
   handler formatter只用 `%(message)s`：structlog已经生成完整JSON字符串，底层不再二次格式化
3. structlog桥接到 stdlib logging：structlog负责组装JSON，IO输出交给原生logging的handler
4. 防重复初始化：避免FastAPI reload、多次调用setup_logging造成重复挂载handler，日志重复打印

⚠️ 注意调用顺序约束：
1. 必须**先设置FAISS_NO_AVX2_WARNING环境变量**（放在本函数调用之前，因为faiss是C层print，不走logging）
2. 项目入口 main.py 只调用一次 setup_logging()，不要在模块顶层直接执行
3. 全项目业务代码统一使用 structlog.get_logger()，避免混用logging.getLogger产生普通文本日志污染JSON流
"""
import logging
import os
import sys

import structlog # 第三方库，用来方便生成结构化JSON日志

# 设置日志级别：DEBUG / INFO / WARNING / ERROR / CRITICAL
#日志级别，INFO 及以上才输出；DEBUG 会被过滤；可选
def setup_logging(log_dir: str = "logs", level: int = logging.INFO) -> None:
    # 1. 屏蔽第三方库冗余日志（吸收 logger.py）
    for noisy in ["huggingface_hub", "transformers", "peft", "httpx", "urllib3", "datasets", "httpcore"]:
        logging.getLogger(noisy).setLevel(logging.WARNING) #：只允许 WARNING、ERROR、CRITICAL 才打印；INFO、DEBUG 直接丢弃。

    # ========== 2. 初始化标准库 logging root logger ==========
    # 创建日志存储目录，已存在不会报错
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, "medisense.log")

    root = logging.getLogger() # 总老板，最顶层  所有子 logger 默认继承 root 的 level
    root.setLevel(level)
    if root.handlers:   # 防重复初始化  已经添加过控制台 / 文件处理器
        return

    # 控制台输出：输出到stderr（容器环境标准实践，stdout/stderr可被Promtail/Filebeat采集）
    """
    stderr：业务应用日志；日志采集组件 (Promtail,Filebeat) 默认抓取容器 stderr。 程序报错、日志输出
    stdout：程序 print、访问日志。
"""
    # formatter只取message字段：structlog已经拼装完整JSON字符串，底层不要再加时间、级别前缀
    console = logging.StreamHandler(sys.stderr)
    console.setFormatter(logging.Formatter("%(message)s"))
    root.addHandler(console)

    # 文件输出：落盘本地日志文件，utf‑8 避免中文乱码
    file_handler = logging.FileHandler(log_file, encoding="utf-8")
    file_handler.setFormatter(logging.Formatter("%(message)s"))
    root.addHandler(file_handler)

    # ========== 3. structlog 配置：桥接到 stdlib logging ==========
    structlog.configure(
        processors=[
            # 合并contextvars上下文变量，适合透传 request_id，实现链路追踪   自动把请求上下文偷偷塞进来
            structlog.contextvars.merge_contextvars,
            # 自动注入日志级别字段: "level":"info"/"error"  自动加上日志等级
            structlog.processors.add_log_level,
            # ISO8601格式时间戳，Loki/ELK可直接识别解析 timestamp 字段  自动加上标准时间
            structlog.processors.TimeStamper(fmt="iso"),
            # 处理异常堆栈，logger.exception() 时把异常信息放进JSON  自动把报错堆栈信息
            structlog.processors.format_exc_info,
            # 渲染输出JSON字符串，作为message传给stdlib logging handler  拼成一整行 JSON 字符串
            structlog.processors.JSONRenderer(),
        ],
        # 根据日志级别过滤，低于level的日志直接丢弃，不处理  流水线入口的过滤网
        wrapper_class=structlog.make_filtering_bound_logger(level),
        # 核心桥接：structlog 不做IO，委托给Python标准logging系统输出
        logger_factory=structlog.stdlib.LoggerFactory(),
        # 缓存logger实例，减少重复创建开销，提升性能   缓存日志对象，不要反复新建，提升速度
        cache_logger_on_first_use=True,
    )
