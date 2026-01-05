"""
GROBID PDF Parser
Uses GROBID service for high-precision academic paper PDF parsing

Main features:
- Identify document structure (title, authors, abstract, sections)
- Extract references
- Handle complex layouts (multi-column, figures, formulas)
- Output structured section information
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
    """Paper section data structure"""
    title: str
    content: str
    page_num: int
    section_type: str


class GrobidPDFParser:
    """
    GROBID PDF Parser

    Uses GROBID service to convert PDF to structured TEI XML,
    then extracts section information
    """

    def __init__(self, grobid_url: str = "http://localhost:8070"):
        """
        Initialize GROBID parser

        Args:
            grobid_url: GROBID service address
        """
        self.grobid_url = grobid_url.rstrip('/')
        self.tei_ns = {'tei': 'http://www.tei-c.org/ns/1.0'}
        self.timeout = 60  # API timeout (seconds)

        # Check if service is available
        self._check_service()

    def _check_service(self) -> bool:
        """Check if GROBID service is available"""
        try:
            response = requests.get(
                f"{self.grobid_url}/api/isalive",
                timeout=5
            )

            if response.status_code == 200 and response.text.strip().lower() == 'true':
                logger.info(f"✅ GROBID service available: {self.grobid_url}")
                return True
            else:
                logger.warning(f"⚠️ GROBID service response abnormal: {response.text}")
                return False

        except requests.exceptions.RequestException as e:
            logger.warning(f"⚠️ Unable to connect to GROBID service: {e}")
            logger.info(f"   Please ensure GROBID service is running at: {self.grobid_url}")
            logger.info(f"   Start method: docker run -d -p 8070:8070 lfoppiano/grobid:0.8.0")
            return False

    def extract_sections_from_pdf(self, pdf_path: str) -> List[PaperSection]:
        """
        Extract sections from PDF (using GROBID)

        Args:
            pdf_path: PDF file path

        Returns:
            List of sections
        """
        if not Path(pdf_path).exists():
            logger.error(f"PDF file does not exist: {pdf_path}")
            return []

        try:
            logger.info(f"  📄 Parsing PDF with GROBID: {Path(pdf_path).name}")

            # 1. Call GROBID API
            tei_xml = self._call_grobid_api(pdf_path)

            if not tei_xml:
                return []

            # 2. Parse TEI XML
            sections = self._parse_tei_xml(tei_xml)

            logger.info(f"  ✅ GROBID successfully extracted {len(sections)} sections")
            return sections

        except Exception as e:
            logger.error(f"  ❌ GROBID parsing failed: {e}")
            return []

    def _call_grobid_api(self, pdf_path: str) -> Optional[str]:
        """
        Call GROBID API to process PDF

        Args:
            pdf_path: PDF file path

        Returns:
            TEI XML string
        """
        try:
            with open(pdf_path, 'rb') as f:
                files = {'input': f}

                # Call processFulltextDocument API
                response = requests.post(
                    f"{self.grobid_url}/api/processFulltextDocument",
                    files=files,
                    timeout=self.timeout
                )

            if response.status_code != 200:
                logger.error(f"  GROBID API returned error: {response.status_code}")
                return None

            return response.text

        except requests.exceptions.Timeout:
            logger.error(f"  GROBID API timeout ({self.timeout} seconds)")
            return None
        except Exception as e:
            logger.error(f"  GROBID API call failed: {e}")
            return None

    def _parse_tei_xml(self, tei_xml: str) -> List[PaperSection]:
        """
        Parse TEI XML returned by GROBID

        Args:
            tei_xml: TEI XML string

        Returns:
            List of sections
        """
        try:
            root = ET.fromstring(tei_xml)
            sections = []

            # 1. Extract title
            title_elem = root.find('.//tei:titleStmt/tei:title[@type="main"]', self.tei_ns)
            if title_elem is not None and title_elem.text:
                sections.append(PaperSection(
                    title='Title',
                    content=self._extract_text(title_elem),
                    page_num=0,
                    section_type='title'
                ))

            # 2. Extract abstract
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

            # 3. Extract body sections
            body = root.find('.//tei:body', self.tei_ns)
            if body is not None:
                sections.extend(self._parse_body_sections(body))

            # 4. Extract conclusion (if outside body)
            back = root.find('.//tei:back', self.tei_ns)
            if back is not None:
                # Some papers have conclusion in back section
                for div in back.findall('.//tei:div', self.tei_ns):
                    head = div.find('tei:head', self.tei_ns)
                    if head is not None and head.text:
                        section_title = self._extract_text(head)

                        # Extract paragraph content
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
            logger.error(f"  TEI XML parsing failed: {e}")
            return []
        except Exception as e:
            logger.error(f"  TEI processing failed: {e}")
            return []

    def _parse_body_sections(self, body: ET.Element) -> List[PaperSection]:
        """
        Parse sections in body part

        Args:
            body: TEI body element

        Returns:
            List of sections
        """
        sections = []

        # Iterate through all div elements (sections)
        for div in body.findall('.//tei:div', self.tei_ns):
            # Get section title
            head = div.find('tei:head', self.tei_ns)
            section_title = self._extract_text(head) if head is not None else 'Unknown Section'

            # Only extract paragraphs from direct child divs, avoid nested duplication
            # XPath limitation: only find p under current div, no recursion
            paragraphs = []
            for child in div:
                if child.tag == f'{{{self.tei_ns["tei"]}}}p':
                    text = self._extract_text(child)
                    if text.strip():
                        paragraphs.append(text)

            # If no direct paragraphs, may have subsections, extract recursively
            if not paragraphs:
                sub_divs = div.findall('tei:div', self.tei_ns)
                if sub_divs:
                    # Has subsections, process recursively
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

            # Build content
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
        Recursively extract all text from element

        Args:
            element: XML element

        Returns:
            Extracted text
        """
        if element is None:
            return ""

        # Use itertext() to get all text nodes
        text_parts = []
        for text in element.itertext():
            text_parts.append(text.strip())

        return ' '.join(text_parts).strip()

    def _infer_section_type(self, title: str) -> str:
        """
        Infer section type

        Args:
            title: Section title

        Returns:
            Section type
        """
        title_lower = title.lower().strip()

        # Remove numbering (e.g., "1.", "1.1", "I.", "A.", etc.)
        # Fix: only match numbering at the beginning, not letters within words
        import re
        # Match: numeric numbering, Roman numerals (must be followed by .), letter numbering (must be followed by .)
        title_clean = re.sub(r'^(?:\d+\.)*\d+\s+|^[IVXLCDM]+\.\s+|^[A-Z]\.\s+', '', title_lower, flags=re.IGNORECASE).strip()

        # Match section type
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
        Extract paper metadata (title, authors, abstract, etc.)

        Args:
            pdf_path: PDF file path

        Returns:
            Metadata dictionary
        """
        try:
            tei_xml = self._call_grobid_api(pdf_path)
            if not tei_xml:
                return {}

            root = ET.fromstring(tei_xml)
            metadata = {}

            # Title
            title_elem = root.find('.//tei:titleStmt/tei:title[@type="main"]', self.tei_ns)
            if title_elem is not None:
                metadata['title'] = self._extract_text(title_elem)

            # Authors
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

            # Abstract
            abstract = root.find('.//tei:abstract', self.tei_ns)
            if abstract is not None:
                metadata['abstract'] = self._extract_text(abstract)

            return metadata

        except Exception as e:
            logger.error(f"Metadata extraction failed: {e}")
            return {}


if __name__ == "__main__":
    # Test code
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    parser = GrobidPDFParser()

    # Test PDF parsing
    test_pdf = "./data/papers/sample.pdf"
    if Path(test_pdf).exists():
        sections = parser.extract_sections_from_pdf(test_pdf)

        print(f"\nExtracted {len(sections)} sections:\n")
        for i, section in enumerate(sections, 1):
            print(f"{i}. [{section.section_type}] {section.title}")
            print(f"   Content length: {len(section.content)} characters")
            print(f"   Content preview: {section.content[:100]}...")
            print()
    else:
        print(f"Test PDF does not exist: {test_pdf}")
