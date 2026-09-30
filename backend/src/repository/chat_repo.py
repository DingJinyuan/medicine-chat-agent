import uuid
from typing import List, Optional, Dict
from datetime import datetime
from sqlalchemy import select
from src.database.connection import get_db
from src.database.models import ChatSession, ChatMessage


def ensure_session(session_id: Optional[str], source: str = "patient") -> str:
    """不存在则新建会话，返回最终session_id。source 区分 'expert' / 'patient'。"""
    sid = session_id or str(uuid.uuid4())
    with get_db() as db:
        stmt = select(ChatSession).where(ChatSession.id == sid)
        exist = db.scalar(stmt)
        if not exist:
            db.add(ChatSession(id=sid, source=source))
    return sid


def add_message(session_id: str, role: str, content: str) -> None:
    """保存一条问答消息"""
    with get_db() as db:
        msg = ChatMessage(
            session_id=session_id,
            role=role,
            content=content,
            created_at=datetime.utcnow()
        )
        db.add(msg)


def get_history(session_id: str) -> List[Dict]:
    """取出完整会话历史，包含时间戳"""
    with get_db() as db:
        stmt = (
            select(ChatMessage)
            .where(ChatMessage.session_id == session_id)
            .order_by(ChatMessage.id.asc())
        )
        rows = db.scalars(stmt).all()
        return [
            {
                "role": m.role,
                "content": m.content,
                "created_at": m.created_at
            }
            for m in rows
        ]


def get_llm_history(session_id: str, limit: int = 20) -> List[Dict]:
    """返回LLM可直接投喂的 {role,content} 格式，取最近limit条"""
    full = get_history(session_id)
    recent_slice = full[-limit:]
    return [{"role": x["role"], "content": x["content"]} for x in recent_slice]
