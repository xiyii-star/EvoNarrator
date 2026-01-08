"""
GROBID PDF解析器
使用GROBID服务进行高精度的学术论文PDF解析

主要功能：
- 识别文档结构（标题、作者、摘要、章节）
- 提取参考文献
- 处理复杂布局（多列、图表、公式）
- 输出结构化章节信息
"""

import logging
import requests
import xml.etree.ElementTree as ET
from typing import List, Optional, Dict
from pathlib import Path
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class PaperSection:
    """论文章节数据结构"""
    title: str
    content: str
    page_num: int
    section_type: str


class GrobidPDFParser:
    """
    GROBID PDF解析器

    使用GROBID服务将PDF转换为结构化的TEI XML，
    然后提取章节信息
    """

    def __init__(self, grobid_url: str = "http://localhost:8070"):
        """
        初始化GROBID解析器

        Args:
            grobid_url: GROBID服务地址
        """
        self.grobid_url = grobid_url.rstrip('/')
        self.tei_ns = {'tei': 'http://www.tei-c.org/ns/1.0'}
        self.timeout = 60  # API超时时间（秒）

        # 检查服务是否可用
        self._check_service()

    def _check_service(self) -> bool:
        """检查GROBID服务是否可用"""
        try:
            response = requests.get(
                f"{self.grobid_url}/api/isalive",
                timeout=5
            )

            if response.status_code == 200 and response.text.strip().lower() == 'true':
                logger.info(f"✅ GROBID服务可用: {self.grobid_url}")
                return True
            else:
                logger.warning(f"⚠️ GROBID服务响应异常: {response.text}")
                return False

        except requests.exceptions.RequestException as e:
            logger.warning(f"⚠️ 无法连接到GROBID服务: {e}")
            logger.info(f"   请确保GROBID服务运行在: {self.grobid_url}")
            logger.info(f"   启动方法: docker run -d -p 8070:8070 lfoppiano/grobid:0.8.0")
            return False

    def extract_sections_from_pdf(self, pdf_path: str) -> List[PaperSection]:
        """
        从PDF中提取章节（使用GROBID）

        Args:
            pdf_path: PDF文件路径

        Returns:
            章节列表
        """
        if not Path(pdf_path).exists():
            logger.error(f"PDF文件不存在: {pdf_path}")
            return []

        try:
            logger.info(f"  📄 使用GROBID解析PDF: {Path(pdf_path).name}")

            # 1. 调用GROBID API
            tei_xml = self._call_grobid_api(pdf_path)

            if not tei_xml:
                return []

            # 2. 解析TEI XML
            sections = self._parse_tei_xml(tei_xml)

            logger.info(f"  ✅ GROBID成功提取 {len(sections)} 个章节")
            return sections

        except Exception as e:
            logger.error(f"  ❌ GROBID解析失败: {e}")
            return []

    def _call_grobid_api(self, pdf_path: str) -> Optional[str]:
        """
        调用GROBID API处理PDF

        Args:
            pdf_path: PDF文件路径

        Returns:
            TEI XML字符串
        """
        try:
            with open(pdf_path, 'rb') as f:
                files = {'input': f}

                # 调用processFulltextDocument API
                response = requests.post(
                    f"{self.grobid_url}/api/processFulltextDocument",
                    files=files,
                    timeout=self.timeout
                )

            if response.status_code != 200:
                logger.error(f"  GROBID API返回错误: {response.status_code}")
                return None

            return response.text

        except requests.exceptions.Timeout:
            logger.error(f"  GROBID API超时（{self.timeout}秒）")
            return None
        except Exception as e:
            logger.error(f"  GROBID API调用失败: {e}")
            return None

    def _parse_tei_xml(self, tei_xml: str) -> List[PaperSection]:
        """
        解析GROBID返回的TEI XML

        Args:
            tei_xml: TEI XML字符串

        Returns:
            章节列表
        """
        try:
            root = ET.fromstring(tei_xml)
            sections = []

            # 1. 提取标题
            title_elem = root.find('.//tei:titleStmt/tei:title[@type="main"]', self.tei_ns)
            if title_elem is not None and title_elem.text:
                sections.append(PaperSection(
                    title='Title',
                    content=self._extract_text(title_elem),
                    page_num=0,
                    section_type='title'
                ))

            # 2. 提取摘要
            abstract = root.find('.//tei:abstract', self.tei_ns)
            if abstract is not None:
                content = self._extract_text(abstract)
                if content.strip():
                    sections.append(PaperSection(
                        title='Abstract',
                        content=content,
                        page_num=0,
                        section_type='abstract'
                    ))

            # 3. 提取正文章节
            body = root.find('.//tei:body', self.tei_ns)
            if body is not None:
                sections.extend(self._parse_body_sections(body))

            # 4. 提取结论（如果在body外）
            back = root.find('.//tei:back', self.tei_ns)
            if back is not None:
                # 有些论文的conclusion在back部分
                for div in back.findall('.//tei:div', self.tei_ns):
                    head = div.find('tei:head', self.tei_ns)
                    if head is not None and head.text:
                        section_title = self._extract_text(head)

                        # 提取段落内容
                        paragraphs = div.findall('.//tei:p', self.tei_ns)
                        content = '\n\n'.join([
                            self._extract_text(p) for p in paragraphs if self._extract_text(p).strip()
                        ])

                        if content.strip():
                            section_type = self._infer_section_type(section_title)
                            sections.append(PaperSection(
                                title=section_title,
                                content=content,
                                page_num=0,
                                section_type=section_type
                            ))

            return sections

        except ET.ParseError as e:
            logger.error(f"  TEI XML解析失败: {e}")
            return []
        except Exception as e:
            logger.error(f"  TEI处理失败: {e}")
            return []

    def _parse_body_sections(self, body: ET.Element) -> List[PaperSection]:
        """
        解析body部分的章节

        Args:
            body: TEI body元素

        Returns:
            章节列表
        """
        sections = []

        # 遍历所有div元素（章节）
        for div in body.findall('.//tei:div', self.tei_ns):
            # 获取章节标题
            head = div.find('tei:head', self.tei_ns)
            section_title = self._extract_text(head) if head is not None else 'Unknown Section'

            # 只提取直接子div的段落，避免嵌套重复
            # 使用XPath的限制：只找当前div下的p，不递归
            paragraphs = []
            for child in div:
                if child.tag == f'{{{self.tei_ns["tei"]}}}p':
                    text = self._extract_text(child)
                    if text.strip():
                        paragraphs.append(text)

            # 如果没有直接段落，可能是有子章节，递归提取
            if not paragraphs:
                sub_divs = div.findall('tei:div', self.tei_ns)
                if sub_divs:
                    # 有子章节，递归处理
                    for sub_div in sub_divs:
                        sub_head = sub_div.find('tei:head', self.tei_ns)
                        sub_title = self._extract_text(sub_head) if sub_head is not None else section_title

                        sub_paragraphs = sub_div.findall('.//tei:p', self.tei_ns)
                        sub_content = '\n\n'.join([
                            self._extract_text(p) for p in sub_paragraphs if self._extract_text(p).strip()
                        ])

                        if sub_content.strip():
                            sub_type = self._infer_section_type(sub_title)
                            sections.append(PaperSection(
                                title=sub_title,
                                content=sub_content,
                                page_num=0,
                                section_type=sub_type
                            ))
                    continue

            # 构建内容
            content = '\n\n'.join(paragraphs)

            if content.strip():
                section_type = self._infer_section_type(section_title)
                sections.append(PaperSection(
                    title=section_title,
                    content=content,
                    page_num=0,
                    section_type=section_type
                ))

        return sections

    def _extract_text(self, element: Optional[ET.Element]) -> str:
        """
        递归提取元素中的所有文本

        Args:
            element: XML元素

        Returns:
            提取的文本
        """
        if element is None:
            return ""

        # 使用itertext()获取所有文本节点
        text_parts = []
        for text in element.itertext():
            text_parts.append(text.strip())

        return ' '.join(text_parts).strip()

    def _infer_section_type(self, title: str) -> str:
        """
        推断章节类型

        Args:
            title: 章节标题

        Returns:
            章节类型
        """
        title_lower = title.lower().strip()

        # 移除编号（如 "1.", "1.1", "I.", "A.", etc.）
        # 修复: 只匹配开头的编号部分,不要匹配单词中的字母
        import re
        # 匹配: 数字编号、罗马数字(后面必须跟.)、字母编号(后面必须跟.)
        title_clean = re.sub(r'^(?:\d+\.)*\d+\s+|^[IVXLCDM]+\.\s+|^[A-Z]\.\s+', '', title_lower, flags=re.IGNORECASE).strip()

        # 匹配章节类型
        if 'abstract' in title_clean:
            return 'abstract'
        elif 'introduction' in title_clean:
            return 'introduction'
        elif any(kw in title_clean for kw in ['related work', 'background', 'literature review', 'prior work']):
            return 'related_work'
        elif any(kw in title_clean for kw in ['method', 'approach', 'model', 'architecture', 'algorithm']):
            return 'method'
        elif any(kw in title_clean for kw in ['experiment', 'evaluation', 'result', 'performance']):
            return 'experiment'
        elif 'discussion' in title_clean:
            return 'discussion'
        elif any(kw in title_clean for kw in ['conclusion', 'summary']):
            return 'conclusion'
        elif any(kw in title_clean for kw in ['limitation', 'weakness']):
            return 'limitation'
        elif any(kw in title_clean for kw in ['future work', 'future direction', 'future research']):
            return 'future_work'
        else:
            return 'other'

    def extract_metadata(self, pdf_path: str) -> Dict:
        """
        提取论文元数据（标题、作者、摘要等）

        Args:
            pdf_path: PDF文件路径

        Returns:
            元数据字典
        """
        try:
            tei_xml = self._call_grobid_api(pdf_path)
            if not tei_xml:
                return {}

            root = ET.fromstring(tei_xml)
            metadata = {}

            # 标题
            title_elem = root.find('.//tei:titleStmt/tei:title[@type="main"]', self.tei_ns)
            if title_elem is not None:
                metadata['title'] = self._extract_text(title_elem)

            # 作者
            authors = []
            for author in root.findall('.//tei:sourceDesc//tei:author', self.tei_ns):
                persName = author.find('.//tei:persName', self.tei_ns)
                if persName is not None:
                    forename = persName.find('tei:forename', self.tei_ns)
                    surname = persName.find('tei:surname', self.tei_ns)

                    name_parts = []
                    if forename is not None and forename.text:
                        name_parts.append(forename.text)
                    if surname is not None and surname.text:
                        name_parts.append(surname.text)

                    if name_parts:
                        authors.append(' '.join(name_parts))

            if authors:
                metadata['authors'] = authors

            # 摘要
            abstract = root.find('.//tei:abstract', self.tei_ns)
            if abstract is not None:
                metadata['abstract'] = self._extract_text(abstract)

            return metadata

        except Exception as e:
            logger.error(f"元数据提取失败: {e}")
            return {}


if __name__ == "__main__":
    # 测试代码
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    parser = GrobidPDFParser()

    # 测试PDF解析
    test_pdf = "./data/papers/sample.pdf"
    if Path(test_pdf).exists():
        sections = parser.extract_sections_from_pdf(test_pdf)

        print(f"\n提取到 {len(sections)} 个章节:\n")
        for i, section in enumerate(sections, 1):
            print(f"{i}. [{section.section_type}] {section.title}")
            print(f"   内容长度: {len(section.content)} 字符")
            print(f"   内容预览: {section.content[:100]}...")
            print()
    else:
        print(f"测试PDF不存在: {test_pdf}")
