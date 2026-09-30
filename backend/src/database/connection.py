# SQLAlchemy核心组件：数据库引擎、原生SQL文本对象
from sqlalchemy import create_engine, text
# ORM会话工厂、会话类型
from sqlalchemy.orm import sessionmaker, Session
# 上下文管理器装饰器，实现with语法自动管理会话生命周期
from contextlib import contextmanager
# 结构化日志
import structlog
# 项目配置文件，读取数据库连接字符串
from src.config import settings
# 导入ORM模型基类Base，所有数据表模型继承于此
from src.database.models import Base

logger = structlog.get_logger(__name__)


# 创建数据库连接引擎（PostgreSQL + pgvector向量库）
engine = create_engine(
    settings.database_url,
    pool_pre_ping=True,     # 从连接池取出连接前先检测连接是否存活，防止断连报错
    pool_size=10,           # 连接池常驻连接数量
    max_overflow=20,        # 高峰期额外可临时创建的溢出连接数，最大并发=10+20=30
    echo=False,             # True打印所有执行的SQL语句，调试阶段开启，生产关闭
)

# 会话工厂：用来生成数据库会话对象
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)


def init_db():
    """
    数据库初始化函数
    1. 开启PostgreSQL的pgvector向量扩展（向量检索必需）
    2. 根据ORM模型自动创建全部数据表
    幂等安全：重复执行不会报错，不会重复创建扩展和表
    """
    # 获取一条原始数据库连接
    with engine.connect() as conn:
        # IF NOT EXISTS 不存在才创建扩展，防止重复运行报错
        conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
        conn.commit()
        logger.info("pgvector_extension_enabled")

    # 扫描所有继承Base的数据模型，自动生成数据库表
    Base.metadata.create_all(bind=engine)
    logger.info("database_tables_created")

    # 尝试创建 ParadeDB BM25 索引（失败降级，不阻塞）
    init_bm25_index()


def init_bm25_index():
    """创建 ParadeDB BM25 索引（需 ParadeDB pg_search 扩展）。

    给 chunks 表建 BM25 索引，供 bm25_search_chunks 做关键词检索。
    未部署 ParadeDB 时 CREATE INDEX 会失败，捕获后降级为纯向量检索。
    """
    try:
        with engine.connect() as conn:
            conn.execute(text(
                "CREATE INDEX IF NOT EXISTS chunks_bm25_idx "
                "ON chunks USING bm25 (id, content, doc_id)"
            ))
            conn.commit()
        logger.info("bm25_index_ready")
    except Exception as e:
        logger.warning("bm25_index_skipped", error=str(e))


@contextmanager
def get_db() -> Session:
    """
    数据库会话上下文管理器，推荐使用with语句调用
    事务自动管理逻辑：
        ✅ 代码无异常 → 自动commit提交事务
        ❌ 抛出异常 → 自动rollback回滚事务，保证数据一致性
        finally块无论成功失败，都会关闭会话归还连接池
    使用示例：
        with get_db() as db:
            db.query(...)
    """
    # 创建新会话
    db = SessionLocal()
    try:
        # 将会话对象交给with代码块使用
        yield db
        # with内部代码正常跑完之后，提交事务
        db.commit()
    except Exception as e:
        # 出现任何异常，回滚本次所有修改，防止脏数据入库
        db.rollback()
        logger.exception("database_error_rollback")
        # 重新抛出异常，让上层业务代码捕获错误
        raise
    finally:
        # 会话关闭，连接归还连接池，释放资源
        db.close()
