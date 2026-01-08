"""
PDF下载器模块 - 多源智能PDF下载

支持两种下载方案：

【方案一】直接URL下载（优先级最高）
- 从论文元数据中提取多个可能的PDF URL
- 数据源包括：pdf_url字段、OpenAlex、DOI推断等
- 支持多个URL按顺序尝试，带重试机制
- 支持的PDF源：
  * arXiv (https://arxiv.org/pdf/...)
  * PubMed Central (PMC)
  * PLOS等出版商
  * OpenAlex开放获取链接

【方案二】arXiv标题搜索（降级方案）
- 当方案一失败时，使用论文标题在arXiv搜索
- 智能相似度匹配，选择最佳候选论文
- 依赖python-arxiv库 (可选)

特性：
- 自动重试机制（最多3次）
- PDF格式验证
- 文件大小检查
- Content-Type验证
- 指数退避策略
- 批量下载支持
- 完整的下载统计

依赖：
- requests (必需)
- arxiv (可选，用于方案二)
"""

import requests
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import logging
import re
from urllib.parse import urlparse
import time
from difflib import SequenceMatcher

try:
    import arxiv  # type: ignore
except ImportError:  # pragma: no cover - optional dependency
    arxiv = None

logger = logging.getLogger(__name__)


class PDFDownloader:
    """
    智能PDF下载器

    两种下载方案自动切换：
    1. 方案一（直接URL）：从论文元数据提取PDF链接直接下载
    2. 方案二（arXiv搜索）：当方案一失败时，使用标题在arXiv搜索下载

    核心功能：
    - download_paper(): 下载单篇论文
    - batch_download(): 批量下载多篇论文
    - get_download_stats(): 获取下载统计
    - list_downloaded_papers(): 列出已下载论文

    内部方法：
    - _find_pdf_urls(): 从元数据提取所有可能的PDF链接（方案一）
    - _download_arxiv_by_title(): 通过标题在arXiv搜索下载（方案二）
    - _download_file_with_retry(): 带重试的文件下载
    - _is_valid_pdf_url(): URL有效性检查
    - _generate_filename(): 生成安全的文件名

    使用示例：
        >>> downloader = PDFDownloader(download_dir='./pdfs')
        >>> paper = {'id': 'W123', 'title': 'Some Paper', 'pdf_url': 'https://...'}
        >>> result = downloader.download_paper(paper)
        >>> print(result['status'])  # 'downloaded', 'exists', 或 'failed'
    """

    def __init__(self, download_dir: str = "./data/papers", max_retries: int = 3):
        self.download_dir = Path(download_dir)
        self.download_dir.mkdir(parents=True, exist_ok=True)
        self.max_retries = max_retries
        self.session = requests.Session()

        # 设置请求头，模拟浏览器
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
            'Accept': 'application/pdf,*/*',
            'Accept-Language': 'en-US,en;q=0.9',
            'Accept-Encoding': 'gzip, deflate, br',
            'Connection': 'keep-alive',
            'Upgrade-Insecure-Requests': '1',
        })

        logger.info(f"PDF下载器初始化完成，下载目录: {self.download_dir}")

    def download_paper(self, paper: Dict, overwrite: bool = False) -> Dict:
        """
        下载单篇论文的PDF

        采用两种下载方案（按顺序尝试）：

        【方案一】直接URL下载（优先级最高）
        ├─ 1.1 论文提供的pdf_url字段
        ├─ 1.2 OpenAlex的open_access.oa_url字段
        ├─ 1.3 OpenAlex的oa_locations列表中的url_for_pdf
        └─ 1.4 从DOI推断的多个可能链接：
            ├─ arXiv PDF链接 (https://arxiv.org/pdf/XXXX.XXXXX.pdf)
            ├─ PubMed Central PDF链接
            └─ 出版商特定链接 (如PLOS)

        【方案二】arXiv标题搜索下载（降级方案）
        ├─ 2.1 使用python-arxiv库按标题精确搜索
        ├─ 2.2 如果精确搜索无结果，使用关键词搜索
        └─ 2.3 使用相似度匹配选择最佳候选论文下载

        Args:
            paper: 论文信息字典，需包含id、title等字段
            overwrite: 是否覆盖已存在的文件

        Returns:
            下载结果字典，包含status、filepath、size等信息

        状态码说明：
        - 'exists': 文件已存在，跳过下载
        - 'downloaded': 成功下载
        - 'failed': 所有方案都失败
        """
        paper_id = paper.get('id', 'unknown现只支持OpenAlex的id')
        title = paper.get('title', 'untitled')

        # 生成安全的文件名
        safe_filename = self._generate_filename(paper_id, title)
        filepath = self.download_dir / safe_filename

        # 如果文件已存在且不覆盖
        if filepath.exists() and not overwrite:
            logger.info(f"PDF已存在，跳过: {safe_filename}")
            return {
                'paper_id': paper_id,
                'filename': safe_filename,
                'filepath': str(filepath),
                'status': 'exists',
                'size': filepath.stat().st_size
            }

        # 记录所有尝试的错误信息
        all_errors = []

        # --------------------方案一：利用搜索到的pdf下载url---------------------
        # 优先尝试：如果有arxiv_id，直接使用arxiv API下载（最可靠）
        if paper.get('arxiv_id'):
            logger.info(f"[方案一-arXiv优先] 检测到arXiv ID: {paper['arxiv_id']}")
            arxiv_success, arxiv_url, arxiv_error = self._download_arxiv_by_id(
                paper['arxiv_id'],
                filepath
            )
            if arxiv_success:
                file_size = filepath.stat().st_size
                logger.info(f"[方案一-arXiv优先] 使用arXiv API下载成功: {safe_filename} ({file_size} bytes)")
                return {
                    'paper_id': paper_id,
                    'filename': safe_filename,
                    'filepath': str(filepath),
                    'status': 'downloaded',
                    'size': file_size,
                    'url': arxiv_url or f'arxiv:{paper["arxiv_id"]}',
                    'method': 'arxiv_api_direct'
                }
            else:
                all_errors.append(f"方案一-arXiv API: {arxiv_error}")
                logger.warning(f"[方案一-arXiv优先] arXiv API下载失败: {arxiv_error}，尝试其他方案")

        # 继续尝试其他PDF链接
        pdf_urls = self._find_pdf_urls(paper)  # 返回多个可能的URL

        if pdf_urls:
            # 显示优先级信息
            if paper.get('arxiv_id'):
                logger.info(f"[方案一] 找到 {len(pdf_urls)} 个PDF链接（含arXiv优先链接），开始尝试下载")
            else:
                logger.info(f"[方案一] 找到 {len(pdf_urls)} 个PDF链接，开始尝试下载")

            # 尝试从多个URL下载
            for i, pdf_url in enumerate(pdf_urls):
                try:
                    logger.info(f"[方案一] 尝试下载 {i+1}/{len(pdf_urls)}: {safe_filename} 来源: {pdf_url}")

                    # 带重试机制的下载
                    success, error_msg = self._download_file_with_retry(pdf_url, filepath)

                    if success:
                        file_size = filepath.stat().st_size
                        logger.info(f"[方案一] 下载成功: {safe_filename} ({file_size} bytes)")
                        return {
                            'paper_id': paper_id,
                            'filename': safe_filename,
                            'filepath': str(filepath),
                            'status': 'downloaded',
                            'size': file_size,
                            'url': pdf_url,
                            'method': 'direct_url'
                        }
                    else:
                        all_errors.append(f"方案一-URL{i+1}: {error_msg}")
                        logger.warning(f"[方案一] 来源 {i+1} 下载失败: {error_msg}")
                        time.sleep(1)  # 稍微延迟后尝试下一个源

                except Exception as e:
                    all_errors.append(f"方案一-URL{i+1}: {str(e)}")
                    logger.warning(f"[方案一] 来源 {i+1} 下载异常: {e}")
                    continue

            logger.warning(f"[方案一] 所有 {len(pdf_urls)} 个PDF源都下载失败")
        else:
            logger.warning(f"[方案一] 未找到PDF链接: {paper_id}")
            all_errors.append("方案一: 未找到PDF链接")

        # -------------------方案二：根据论文名称利用arxiv下载--------------------
        if paper.get('title'):
            logger.info(f"[方案二] 尝试通过arXiv标题搜索下载")
            arxiv_success, arxiv_url, arxiv_error = self._download_arxiv_by_title(
                paper.get('title', ''),
                filepath
            )
            if arxiv_success:
                file_size = filepath.stat().st_size
                logger.info(f"[方案二] 通过arXiv标题搜索下载成功: {safe_filename} ({file_size} bytes)")
                return {
                    'paper_id': paper_id,
                    'filename': safe_filename,
                    'filepath': str(filepath),
                    'status': 'downloaded',
                    'size': file_size,
                    'url': arxiv_url or 'arxiv_search',
                    'method': 'arxiv_title_search'
                }
            else:
                all_errors.append(f"方案二: {arxiv_error}")
                logger.warning(f"[方案二] 通过arXiv标题下载失败: {arxiv_error}")
        else:
            logger.info(f"[方案二] 跳过（论文无标题信息）")
            all_errors.append("方案二: 论文无标题信息")

        # 所有方案都失败
        logger.error(f"所有下载方案都失败: {paper_id}")
        return {
            'paper_id': paper_id,
            'filename': safe_filename,
            'status': 'failed',
            'error': ' | '.join(all_errors)
        }

    def batch_download(self, papers: List[Dict], max_downloads: int = 5) -> Dict:
        """
        批量下载PDF

        Args:
            papers: 论文列表
            max_downloads: 最大下载数量

        Returns:
            下载统计结果
        """
        logger.info(f"开始批量下载 {len(papers)} 篇论文的PDF (最多{max_downloads}篇)")

        results = []
        downloaded = 0

        for i, paper in enumerate(papers):
            if downloaded >= max_downloads:
                logger.info(f"已达到最大下载数量 {max_downloads}，停止下载")
                break

            result = self.download_paper(paper)
            results.append(result)

            if result['status'] == 'downloaded':
                downloaded += 1

            # 添加延迟避免被限制
            time.sleep(1)

            logger.info(f"进度: {i+1}/{len(papers)}, 已下载: {downloaded}")

        # 统计结果
        stats = {
            'total_papers': len(papers),
            'attempted': len(results),
            'downloaded': sum(1 for r in results if r['status'] == 'downloaded'),
            'exists': sum(1 for r in results if r['status'] == 'exists'),
            'failed': sum(1 for r in results if r['status'] in ['failed', 'error', 'no_pdf_url']),
            'results': results
        }

        logger.info(f"批量下载完成: {stats['downloaded']} 成功, {stats['failed']} 失败")
        return stats


    def _download_arxiv_by_id(
        self,
        arxiv_id: str,
        filepath: Path
    ) -> Tuple[bool, Optional[str], Optional[str]]:
        """
        直接通过arXiv ID使用arxiv API下载PDF

        这是最可靠的arXiv论文下载方式，利用python-arxiv库直接获取PDF。

        Args:
            arxiv_id: arXiv ID（如 "2301.12345" 或 "2301.12345v2"）
            filepath: 目标文件路径

        Returns:
            (success, url, error_message) 元组
        """
        if not arxiv_id:
            return False, None, "Empty arXiv ID"

        if arxiv is None:
            return False, None, "python-arxiv库未安装"

        # 清理arXiv ID（移除可能的版本号）
        clean_id = arxiv_id.split('v')[0] if 'v' in arxiv_id else arxiv_id

        logger.info(f"使用arXiv API直接下载: {arxiv_id}")

        try:
            # 使用arxiv库的Client搜索论文
            client = arxiv.Client()
            search = arxiv.Search(id_list=[clean_id])

            # 获取论文对象
            paper = next(client.results(search), None)

            if not paper:
                return False, None, f"arXiv未找到ID: {arxiv_id}"

            # 使用arxiv库的download_pdf方法下载
            # 注意：这个方法会自动处理重试和错误
            try:
                paper.download_pdf(dirpath=str(filepath.parent), filename=filepath.name)

                # 验证下载的文件
                if not filepath.exists():
                    return False, None, "下载完成但文件不存在"

                if filepath.stat().st_size < 1024:
                    filepath.unlink(missing_ok=True)
                    return False, None, f"下载的文件太小 ({filepath.stat().st_size} bytes)"

                # 验证PDF格式
                with open(filepath, 'rb') as f:
                    header = f.read(5)
                    if not header.startswith(b'%PDF-'):
                        filepath.unlink(missing_ok=True)
                        return False, None, "下载的文件不是有效的PDF"

                return True, paper.pdf_url, None

            except Exception as download_error:
                return False, None, f"arXiv API下载失败: {str(download_error)}"

        except Exception as e:
            logger.error(f"arXiv API下载过程中发生错误: {e}")
            return False, None, str(e)

    def _download_arxiv_by_title(self, title_query: str, filepath: Path, max_results: int = 5) -> Tuple[bool, Optional[str], Optional[str]]:
        """
        根据论文标题在arXiv搜索并尝试下载PDF（方案二的核心方法）

        智能搜索策略：
        1. 首先尝试精确标题搜索 (ti:"exact title")
        2. 如果无结果，降级为关键词搜索 (all:"title keywords")
        3. 使用SequenceMatcher计算相似度，选择最匹配的候选论文
        4. 按相似度从高到低依次尝试下载

        依赖：需要安装 python-arxiv 库 (pip install arxiv)

        Args:
            title_query: 论文标题或关键词
            filepath: 目标文件路径
            max_results: 搜索返回的最大结果数（默认5）

        Returns:
            (success, url, error_message) 元组
            - success: 是否下载成功
            - url: 成功下载的PDF URL（失败时为None）
            - error_message: 失败时的错误信息（成功时为None）
        """
        if not title_query:
            return False, None, "Empty title query"

        if arxiv is None:
            return False, None, "python-arxiv库未安装，无法执行方案二"

        title_query = title_query.strip()
        logger.info(f"尝试通过arXiv标题搜索下载: {title_query}")

        try:
            search = arxiv.Search(
                query=f'ti:"{title_query}"',
                max_results=max_results,
                sort_by=arxiv.SortCriterion.Relevance
            )
            results_list = list(search.results())

            if not results_list:
                logger.info("精确标题搜索未找到匹配，尝试关键词搜索")
                search_broad = arxiv.Search(
                    query=f'all:"{title_query}"',
                    max_results=max_results,
                    sort_by=arxiv.SortCriterion.Relevance
                )
                results_list = list(search_broad.results())

            if not results_list:
                return False, None, f"未在arXiv中搜索到与 '{title_query}' 匹配的论文"

            def result_score(result):
                return SequenceMatcher(None, result.title.lower(), title_query.lower()).ratio()

            ranked_results = sorted(results_list, key=result_score, reverse=True)

            for candidate in ranked_results:
                pdf_url = getattr(candidate, 'pdf_url', None)
                if not pdf_url:
                    entry_id = getattr(candidate, 'entry_id', '')
                    if entry_id and '/abs/' in entry_id:
                        pdf_url = entry_id.replace('/abs/', '/pdf/') + '.pdf'

                if not pdf_url:
                    logger.debug(f"候选结果缺少PDF链接，跳过: {candidate.title}")
                    continue

                success, error_msg = self._download_file_with_retry(pdf_url, filepath)
                if success:
                    return True, pdf_url, None

                logger.warning(f"通过arXiv候选 {candidate.entry_id} 下载失败: {error_msg}")

            return False, None, "所有arXiv候选下载失败"
        except Exception as e:
            logger.error(f"arXiv搜索或下载过程中发生错误: {e}")
            return False, None, str(e)

    def _find_pdf_urls(self, paper: Dict) -> List[str]:
        """
        查找论文的所有可能PDF链接（方案一的核心方法）

        按优先级从多个数据源提取PDF链接：

        0. 【优先】arXiv ID直接构建（来自跨库映射）
        1. 论文自带的pdf_url字段
        2. OpenAlex的open_access.oa_url
        3. OpenAlex的open_access.oa_locations列表
        4. 从DOI推断的链接：
           - arXiv PDF (从DOI中提取arxiv ID)
           - PubMed Central PDF (从DOI中提取PMC ID)
           - 出版商特定链接 (如PLOS的printable版本)

        Args:
            paper: 论文信息字典

        Returns:
            去重后的有效PDF URL列表
        """
        pdf_urls = []

        # 0. 【最高优先级】如果论文有arxiv_id（来自跨库映射），直接构建arXiv PDF链接
        # 这是跨库映射优化的核心：利用验证过的arXiv ID直接下载
        if paper.get('arxiv_id'):
            arxiv_id = paper['arxiv_id']
            # 清理arxiv_id（移除可能的版本号）
            clean_arxiv_id = arxiv_id.split('v')[0] if 'v' in arxiv_id else arxiv_id
            arxiv_pdf = f"https://arxiv.org/pdf/{clean_arxiv_id}.pdf"
            pdf_urls.append(arxiv_pdf)
            logger.debug(f"优先使用跨库映射的arXiv ID: {arxiv_id} -> {arxiv_pdf}")

        # 1. 论文提供的PDF链接（第二优先级）
        if paper.get('pdf_url'):
            # 避免重复添加（如果pdf_url已经是arXiv链接）
            if paper['pdf_url'] not in pdf_urls:
                pdf_urls.append(paper['pdf_url'])

        # 2. 尝试从OpenAlex获取更多PDF链接
        open_access = paper.get('open_access', {})
        if isinstance(open_access, dict):
            # OpenAlex的oa_url
            if open_access.get('oa_url'):
                pdf_urls.append(open_access['oa_url'])

            # 如果有oa_locations，提取所有PDF链接
            oa_locations = open_access.get('oa_locations', [])
            for location in oa_locations:
                if isinstance(location, dict) and location.get('url_for_pdf'):
                    pdf_urls.append(location['url_for_pdf'])

        # 3. 尝试从DOI构建PDF链接
        doi = paper.get('doi', '')
        if doi:
            # 尝试arXiv链接
            if 'arxiv' in doi.lower():
                arxiv_id = re.search(r'(\d{4}\.\d{4,5})', doi)
                if arxiv_id:
                    arxiv_pdf = f"https://arxiv.org/pdf/{arxiv_id.group(1)}.pdf"
                    if arxiv_pdf not in pdf_urls:
                        pdf_urls.append(arxiv_pdf)

            # 尝试PubMed Central链接
            if 'pmc' in doi.lower() or 'pubmed' in doi.lower():
                pmc_match = re.search(r'pmc(\d+)', doi.lower())
                if pmc_match:
                    pmc_pdf = f"https://www.ncbi.nlm.nih.gov/pmc/articles/PMC{pmc_match.group(1)}/pdf/"
                    if pmc_pdf not in pdf_urls:
                        pdf_urls.append(pmc_pdf)

            # 尝试DOI解析服务
            if doi.startswith('10.'):
                # 某些发布商提供直接PDF访问
                if '10.1371/journal' in doi:  # PLOS
                    plos_pdf = f"https://journals.plos.org/plosone/article/file?id={doi}&type=printable"
                    if plos_pdf not in pdf_urls:
                        pdf_urls.append(plos_pdf)

        # 5. 去重和过滤
        unique_urls = []
        for url in pdf_urls:
            if url and url not in unique_urls and self._is_valid_pdf_url(url):
                unique_urls.append(url)

        logger.debug(f"为论文 {paper.get('id')} 找到 {len(unique_urls)} 个PDF源")
        return unique_urls

    def _is_valid_pdf_url(self, url: str) -> bool:
        """检查URL是否可能是有效的PDF链接"""
        if not url:
            return False

        # 基本URL格式检查
        parsed = urlparse(url)
        if not parsed.scheme or not parsed.netloc:
            return False

        # 排除明显不是PDF的URL
        exclude_patterns = [
            'javascript:', 'mailto:', '#', 'facebook.com', 'twitter.com',
            'linkedin.com', 'youtube.com', 'instagram.com'
        ]

        url_lower = url.lower()
        for pattern in exclude_patterns:
            if pattern in url_lower:
                return False

        return True

    def _download_file_with_retry(self, url: str, filepath: Path, timeout: int = 60) -> tuple[bool, str]:
        """带重试机制的文件下载"""
        for attempt in range(self.max_retries):
            try:
                # 为每个请求添加额外的请求头
                headers = self.session.headers.copy()

                # 添加 Referer (假装从文章页面点击过来)
                parsed_url = urlparse(url)
                if 'journal-of-hepatology' in url:
                    # 针对这个期刊，添加 Referer
                    referer = url.replace('/pdf', '').replace('/pdfExtended', '')
                    headers['Referer'] = referer
                elif parsed_url.netloc:
                    # 通用策略：设置 Referer 为同域名首页
                    headers['Referer'] = f"{parsed_url.scheme}://{parsed_url.netloc}/"

                response = self.session.get(url, headers=headers, stream=True, timeout=timeout)

                # 检查状态码
                if response.status_code == 403:
                    return False, f"Access forbidden (403) - {url}"
                elif response.status_code == 404:
                    return False, f"File not found (404) - {url}"
                elif response.status_code != 200:
                    if attempt < self.max_retries - 1:
                        logger.warning(f"状态码 {response.status_code}，重试 {attempt + 1}/{self.max_retries}")
                        time.sleep(2 ** attempt)  # 指数退避
                        continue
                    else:
                        return False, f"HTTP {response.status_code} - {url}"

                response.raise_for_status()

                # 检查Content-Type
                content_type = response.headers.get('content-type', '').lower()
                content_length = response.headers.get('content-length')

                # 如果明确不是PDF，跳过
                if 'text/html' in content_type and 'application/pdf' not in content_type:
                    return False, f"Content-Type is HTML, not PDF - {url}"

                # 如果文件太小，可能是错误页面
                if content_length and int(content_length) < 1024:
                    return False, f"File too small ({content_length} bytes) - {url}"

                # 开始下载
                downloaded_size = 0
                with open(filepath, 'wb') as f:
                    for chunk in response.iter_content(chunk_size=8192):
                        if chunk:
                            f.write(chunk)
                            downloaded_size += len(chunk)

                # 验证下载的文件
                if filepath.stat().st_size < 1024:
                    filepath.unlink(missing_ok=True)
                    return False, f"Downloaded file too small ({filepath.stat().st_size} bytes)"

                # 简单的PDF格式验证
                with open(filepath, 'rb') as f:
                    header = f.read(5)
                    if not header.startswith(b'%PDF-'):
                        filepath.unlink(missing_ok=True)
                        return False, "Downloaded file is not a valid PDF"

                return True, "Success"

            except requests.exceptions.Timeout:
                if attempt < self.max_retries - 1:
                    logger.warning(f"下载超时，重试 {attempt + 1}/{self.max_retries}")
                    time.sleep(2 ** attempt)
                    continue
                else:
                    return False, f"Download timeout after {self.max_retries} attempts"

            except requests.exceptions.RequestException as e:
                if attempt < self.max_retries - 1:
                    logger.warning(f"请求异常，重试 {attempt + 1}/{self.max_retries}: {e}")
                    time.sleep(2 ** attempt)
                    continue
                else:
                    return False, f"Request failed after {self.max_retries} attempts: {str(e)}"

            except Exception as e:
                logger.error(f"下载文件时发生未知错误: {e}")
                filepath.unlink(missing_ok=True)
                return False, f"Unknown error: {str(e)}"

        return False, f"Max retries ({self.max_retries}) exceeded"

    def _generate_filename(self, paper_id: str, title: str) -> str:
        """生成安全的文件名"""
        # 清理标题，移除特殊字符
        safe_title = re.sub(r'[^\w\s-]', '', title).strip()
        safe_title = re.sub(r'[\s]+', '_', safe_title)

        # 截断过长的标题
        if len(safe_title) > 50:
            safe_title = safe_title[:50]

        # 组合文件名
        filename = f"{paper_id}_{safe_title}.pdf"
        return filename

    def get_download_stats(self) -> Dict:
        """获取下载统计"""
        if not self.download_dir.exists():
            return {'total_files': 0, 'total_size': 0}

        pdf_files = list(self.download_dir.glob('*.pdf'))
        total_size = sum(f.stat().st_size for f in pdf_files)

        return {
            'total_files': len(pdf_files),
            'total_size': total_size,
            'total_size_mb': round(total_size / (1024 * 1024), 2),
            'download_dir': str(self.download_dir)
        }

    def list_downloaded_papers(self) -> List[Dict]:
        """列出已下载的论文"""
        if not self.download_dir.exists():
            return []

        pdf_files = list(self.download_dir.glob('*.pdf'))

        papers = []
        for pdf_file in pdf_files:
            # 从文件名解析paper_id
            filename = pdf_file.stem
            parts = filename.split('_', 1)
            paper_id = parts[0] if parts else filename

            papers.append({
                'paper_id': paper_id,
                'filename': pdf_file.name,
                'filepath': str(pdf_file),
                'size': pdf_file.stat().st_size,
                'modified_time': pdf_file.stat().st_mtime
            })

        return sorted(papers, key=lambda x: x['modified_time'], reverse=True)


if __name__ == "__main__":
    # 测试代码
    downloader = PDFDownloader()

    # 创建测试论文数据
    test_paper = {
        'id': 'W2741809807',
        'title': 'Attention Is All You Need',
        'pdf_url': 'https://arxiv.org/pdf/1706.03762.pdf'
    }

    # 测试下载
    result = downloader.download_paper(test_paper)
    print(f"下载结果: {result}")

    # 查看统计
    stats = downloader.get_download_stats()
    print(f"下载统计: {stats}")