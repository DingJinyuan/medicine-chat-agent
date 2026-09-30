# 导入 PyMuPDF 库，用于PDF文档解析
import fitz
# Path 用于跨平台、面向对象的文件路径操作
from pathlib import Path
# 类型注解：列表、可选类型
from typing import List, Optional
# Pydantic 数据校验模型，用于结构化输出PDF解析结果
from pydantic import BaseModel
import structlog

logger = structlog.get_logger(__name__)



class ParsedDocument(BaseModel):
    """
    PDF解析完成之后的结构化文档数据模型
    每一份PDF解析完成后都会封装成该对象
    """
    doc_id: str          # 文档唯一标识ID
    title: str           # 文档标题
    content: str         # PDF提取出的全部文本内容
    page_count: int      # PDF总页数
    source: str = "pdf"  # 文档来源类型，默认pdf
    metadata: dict = {}  # 附加元数据，存储文件名、文件路径等额外信息



class PDFParser:
    """
    使用 PyMuPDF (fitz) 解析临床指南PDF文档的解析器
    可批量读取文件夹内全部PDF，提取每页文本，输出结构化文档对象

    免费临床指南下载站点：
      ACC/AHA : https://www.acc.org/guidelines
      NICE    : https://www.nice.org.uk/guidance
      WHO     : https://www.who.int/publications

    使用方式：
    1. 将下载好的PDF文件放到 data/guidelines/ 文件夹
    2. 调用解析流水线开始提取文本
    """

    def __init__(self, pdf_dir: str = "data/guidelines"):
        """
        解析器初始化构造函数
        :param pdf_dir: PDF存放文件夹路径，默认 data/guidelines
        """
        # 将传入字符串路径转为 Path 对象
        self.pdf_dir = Path(pdf_dir)
        # 创建目录，parents=True 自动创建多级父目录；exist_ok=True目录存在不会报错
        self.pdf_dir.mkdir(parents=True, exist_ok=True)
        logger.info("pdf_parser_ready", dir=str(self.pdf_dir))

    def parse_file(self, file_path: str) -> Optional[ParsedDocument]:
        """
        解析单个PDF文件，提取全部页面文本
        :param file_path: 单个PDF文件的完整路径字符串
        :return: 解析成功返回 ParsedDocument 对象；失败/空文件返回 None
        """
        path = Path(file_path)

        # 判断文件是否存在，不存在打印错误日志直接返回
        if not path.exists():
            logger.error("pdf_not_found", path=file_path)
            return None

        try:
            # 使用 fitz 打开PDF文档
            doc = fitz.open(str(path))
            # 存储每一页提取出来的文本
            pages_text = []

            # 遍历PDF所有页面，enumerate拿到页码索引与页面对象
            for page_num, page in enumerate(doc):
                # get_text("text") 获取页面纯文本内容
                text = page.get_text("text")
                # 非空页面，带上页码标记存入列表，方便后续溯源哪一页的文本
                if text.strip():
                    pages_text.append(f"[Page {page_num + 1}]\n{text.strip()}")

            # 关闭前先记录页数（close 后 len(doc) 可能报错）
            page_count = len(doc)
            # 关闭PDF文件句柄，释放文件资源
            doc.close()
            # 使用两个换行符拼接所有页面文本，分隔不同页面
            full_content = "\n\n".join(pages_text)

            # 提取文本为空，大概率是扫描版PDF（图片PDF，没有可复制文字）
            if not full_content.strip():
                logger.warning("pdf_no_text_scanned", name=path.name)
                return None

            # 从文件名生成文档标题：下划线、横杠替换成空格，每个单词首字母大写
            # path.stem → 获取文件名(不带后缀) `.title()` **每个单词第一个字母大写，剩下字母小写**
            title = path.stem.replace("_", " ").replace("-", " ").title()
            logger.info("pdf_parsed", name=path.name, pages=page_count)

            # 封装并返回结构化解析结果对象
            return ParsedDocument(
                doc_id=f"pdf_{path.stem}",
                title=title,
                content=full_content,
                page_count=page_count,
                metadata={"filename": path.name, "filepath": str(path.absolute())}
            )

        except Exception as e:
            # 捕获解析过程所有异常，打印错误日志，返回None
            logger.exception("pdf_parse_failed", path=file_path)
            return None

    def parse_directory(self) -> List[ParsedDocument]:
        """
        批量解析文件夹下所有 .pdf 文件
        :return: 解析成功的 ParsedDocument 对象列表
        """
        # 查找目录下全部pdf文件，按文件名升序排序  `glob` = 通配符扫描文件夹
        pdf_files = sorted(self.pdf_dir.glob("*.pdf"))

        # 文件夹内没有PDF文件，打印警告日志，返回空列表
        if not pdf_files:
            logger.warning("no_pdfs", dir=str(self.pdf_dir))
            return []

        logger.info("pdfs_found", count=len(pdf_files))
        # 循环逐个解析每一个PDF
        docs = [self.parse_file(str(p)) for p in pdf_files]
        # 过滤掉解析失败返回 None 的文件，只保留成功解析文档
        valid = [d for d in docs if d is not None]
        logger.info("pdfs_parsed", parsed=len(valid), total=len(pdf_files))
        return valid

