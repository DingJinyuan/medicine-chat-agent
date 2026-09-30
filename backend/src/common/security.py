"""API 服务公共安全能力：鉴权 + 限流（两版共用入口 /api/chat）。"""
from __future__ import annotations

import threading
import time
from collections import defaultdict

from fastapi import Header

from src.config import settings
from src.common.errors import AppError, ErrorCode


async def require_api_key(x_api_key: str = Header(default="")) -> None:
    """FastAPI 依赖项：校验请求头 X-API-Key，失败抛 401。

    未配置 APP_API_KEY 时关闭鉴权（开发/演示默认）；配置后严格校验。
    生产环境务必设置 APP_API_KEY。
    """
    if not settings.app_api_key:
        return
    if x_api_key != settings.app_api_key:
        raise AppError(status=401, code=ErrorCode.UNAUTHORIZED, message="invalid or missing X-API-Key")


class RateLimiter:
    """内存固定窗口限流器（单实例演示部署；多实例需换 Redis 共享存储）。"""

    def __init__(self, limit_per_minute: int):
        self.limit = limit_per_minute
        self._hits: dict[str, list[float]] = defaultdict(list)
        self._lock = threading.Lock()  # 保护 _hits 读改写，避免并发竞态

    def check(self, client_id: str) -> bool:
        """检查 client_id 是否仍允许请求，并更新时间窗口。"""
        now = time.time()
        window_start = now - 60
        with self._lock:
            hits = [t for t in self._hits[client_id] if t > window_start]
            hits.append(now)
            self._hits[client_id] = hits
            # 淘汰:client_id 过多时清理已过期的,防止内存单调增长
            if len(self._hits) > 10000:
                self._hits = {k: v for k, v in self._hits.items() if any(t > window_start for t in v)}
            return len(hits) <= self.limit


_rate_limiter: RateLimiter | None = None


def get_rate_limiter() -> RateLimiter:
    """获取全局单例限流器（懒加载）。"""
    global _rate_limiter
    if _rate_limiter is None:
        _rate_limiter = RateLimiter(settings.rate_limit_per_minute)
    return _rate_limiter
