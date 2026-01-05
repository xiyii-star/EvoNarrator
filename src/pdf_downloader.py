"""
PDF Downloader Module - Multi-source Intelligent PDF Download

Supports two download strategies:

[Strategy 1] Direct URL Download (Highest Priority)
- Extract multiple possible PDF URLs from paper metadata
- Data sources include: pdf_url field, OpenAlex, DOI inference, etc.
- Support multiple URLs with sequential attempts and retry mechanism
- Supported PDF sources:
  * arXiv (https://arxiv.org/pdf/...)
  * PubMed Central (PMC)
  * PLOS and other publishers
  * OpenAlex open access links

[Strategy 2] arXiv Title Search (Fallback Strategy)
- When Strategy 1 fails, search arXiv using paper title
- Intelligent similarity matching to select best candidate paper
- Depends on python-arxiv library (optional)

Features:
- Automatic retry mechanism (up to 3 attempts)
- PDF format validation
- File size checking
- Content-Type validation
- Exponential backoff strategy
- Batch download support
- Complete download statistics

Dependencies:
- requests (required)
- arxiv (optional, for Strategy 2)
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
    Intelligent PDF Downloader

    Two download strategies with automatic switching:
    1. Strategy 1 (Direct URL): Extract PDF links from paper metadata and download directly
    2. Strategy 2 (arXiv Search): When Strategy 1 fails, search arXiv using title

    Core Functions:
    - download_paper(): Download a single paper
    - batch_download(): Batch download multiple papers
    - get_download_stats(): Get download statistics
    - list_downloaded_papers(): List downloaded papers

    Internal Methods:
    - _find_pdf_urls(): Extract all possible PDF links from metadata (Strategy 1)
    - _download_arxiv_by_title(): Search and download from arXiv by title (Strategy 2)
    - _download_file_with_retry(): File download with retry
    - _is_valid_pdf_url(): URL validity check
    - _generate_filename(): Generate safe filename

    Usage Example:
        >>> downloader = PDFDownloader(download_dir='./pdfs')
        >>> paper = {'id': 'W123', 'title': 'Some Paper', 'pdf_url': 'https://...'}
        >>> result = downloader.download_paper(paper)
        >>> print(result['status'])  # 'downloaded', 'exists', or 'failed'
    """

    def __init__(self, download_dir: str = "./data/papers", max_retries: int = 3):
        self.download_dir = Path(download_dir)
        self.download_dir.mkdir(parents=True, exist_ok=True)
        self.max_retries = max_retries
        self.session = requests.Session()

        # Set request headers to simulate browser
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
            'Accept': 'application/pdf,*/*',
            'Accept-Language': 'en-US,en;q=0.9',
            'Accept-Encoding': 'gzip, deflate, br',
            'Connection': 'keep-alive',
            'Upgrade-Insecure-Requests': '1',
        })

        logger.info(f"PDF downloader initialized, download directory: {self.download_dir}")

    def download_paper(self, paper: Dict, overwrite: bool = False) -> Dict:
        """
        Download PDF for a single paper

        Uses two download strategies (tried in order):

        [Strategy 1] Direct URL Download (Highest Priority)
        ├─ 1.1 pdf_url field provided by paper
        ├─ 1.2 OpenAlex open_access.oa_url field
        ├─ 1.3 url_for_pdf in OpenAlex oa_locations list
        └─ 1.4 Multiple possible links inferred from DOI:
            ├─ arXiv PDF link (https://arxiv.org/pdf/XXXX.XXXXX.pdf)
            ├─ PubMed Central PDF link
            └─ Publisher-specific links (e.g., PLOS)

        [Strategy 2] arXiv Title Search Download (Fallback Strategy)
        ├─ 2.1 Exact search by title using python-arxiv library
        ├─ 2.2 If exact search fails, use keyword search
        └─ 2.3 Use similarity matching to select best candidate paper

        Args:
            paper: Paper information dictionary, must contain id, title, etc.
            overwrite: Whether to overwrite existing files

        Returns:
            Download result dictionary containing status, filepath, size, etc.

        Status codes:
        - 'exists': File already exists, skipped download
        - 'downloaded': Successfully downloaded
        - 'failed': All strategies failed
        """
        paper_id = paper.get('id', 'unknown_only_supports_OpenAlex_id')
        title = paper.get('title', 'untitled')

        # Generate safe filename
        safe_filename = self._generate_filename(paper_id, title)
        filepath = self.download_dir / safe_filename

        # If file exists and not overwriting
        if filepath.exists() and not overwrite:
            logger.info(f"PDF already exists, skipping: {safe_filename}")
            return {
                'paper_id': paper_id,
                'filename': safe_filename,
                'filepath': str(filepath),
                'status': 'exists',
                'size': filepath.stat().st_size
            }

        # Record all attempted error messages
        all_errors = []

        # --------------------Strategy 1: Use discovered PDF download URLs---------------------
        # Priority attempt: If arxiv_id exists, use arXiv API directly (most reliable)
        if paper.get('arxiv_id'):
            logger.info(f"[Strategy 1-arXiv Priority] Detected arXiv ID: {paper['arxiv_id']}")
            arxiv_success, arxiv_url, arxiv_error = self._download_arxiv_by_id(
                paper['arxiv_id'],
                filepath
            )
            if arxiv_success:
                file_size = filepath.stat().st_size
                logger.info(f"[Strategy 1-arXiv Priority] Download successful using arXiv API: {safe_filename} ({file_size} bytes)")
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
                all_errors.append(f"Strategy 1-arXiv API: {arxiv_error}")
                logger.warning(f"[Strategy 1-arXiv Priority] arXiv API download failed: {arxiv_error}, trying other strategies")

        # Continue trying other PDF links
        pdf_urls = self._find_pdf_urls(paper)  # Returns multiple possible URLs

        if pdf_urls:
            # Display priority information
            if paper.get('arxiv_id'):
                logger.info(f"[Strategy 1] Found {len(pdf_urls)} PDF links (including arXiv priority link), starting download attempts")
            else:
                logger.info(f"[Strategy 1] Found {len(pdf_urls)} PDF links, starting download attempts")

            # Try downloading from multiple URLs
            for i, pdf_url in enumerate(pdf_urls):
                try:
                    logger.info(f"[Strategy 1] Attempting download {i+1}/{len(pdf_urls)}: {safe_filename} from: {pdf_url}")

                    # Download with retry mechanism
                    success, error_msg = self._download_file_with_retry(pdf_url, filepath)

                    if success:
                        file_size = filepath.stat().st_size
                        logger.info(f"[Strategy 1] Download successful: {safe_filename} ({file_size} bytes)")
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
                        all_errors.append(f"Strategy 1-URL{i+1}: {error_msg}")
                        logger.warning(f"[Strategy 1] Source {i+1} download failed: {error_msg}")
                        time.sleep(1)  # Brief delay before trying next source

                except Exception as e:
                    all_errors.append(f"Strategy 1-URL{i+1}: {str(e)}")
                    logger.warning(f"[Strategy 1] Source {i+1} download exception: {e}")
                    continue

            logger.warning(f"[Strategy 1] All {len(pdf_urls)} PDF sources failed to download")
        else:
            logger.warning(f"[Strategy 1] No PDF links found: {paper_id}")
            all_errors.append("Strategy 1: No PDF links found")

        # -------------------Strategy 2: Download using arXiv based on paper title--------------------
        if paper.get('title'):
            logger.info(f"[Strategy 2] Attempting download via arXiv title search")
            arxiv_success, arxiv_url, arxiv_error = self._download_arxiv_by_title(
                paper.get('title', ''),
                filepath
            )
            if arxiv_success:
                file_size = filepath.stat().st_size
                logger.info(f"[Strategy 2] Download successful via arXiv title search: {safe_filename} ({file_size} bytes)")
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
                all_errors.append(f"Strategy 2: {arxiv_error}")
                logger.warning(f"[Strategy 2] Download via arXiv title failed: {arxiv_error}")
        else:
            logger.info(f"[Strategy 2] Skipped (paper has no title information)")
            all_errors.append("Strategy 2: Paper has no title information")

        # All strategies failed
        logger.error(f"All download strategies failed: {paper_id}")
        return {
            'paper_id': paper_id,
            'filename': safe_filename,
            'status': 'failed',
            'error': ' | '.join(all_errors)
        }

    def batch_download(self, papers: List[Dict], max_downloads: int = 5) -> Dict:
        """
        Batch download PDFs

        Args:
            papers: List of papers
            max_downloads: Maximum number of downloads

        Returns:
            Download statistics result
        """
        logger.info(f"Starting batch download of {len(papers)} papers' PDFs (max {max_downloads})")

        results = []
        downloaded = 0

        for i, paper in enumerate(papers):
            if downloaded >= max_downloads:
                logger.info(f"Reached maximum download count {max_downloads}, stopping download")
                break

            result = self.download_paper(paper)
            results.append(result)

            if result['status'] == 'downloaded':
                downloaded += 1

            # Add delay to avoid rate limiting
            time.sleep(1)

            logger.info(f"Progress: {i+1}/{len(papers)}, Downloaded: {downloaded}")

        # Statistics result
        stats = {
            'total_papers': len(papers),
            'attempted': len(results),
            'downloaded': sum(1 for r in results if r['status'] == 'downloaded'),
            'exists': sum(1 for r in results if r['status'] == 'exists'),
            'failed': sum(1 for r in results if r['status'] in ['failed', 'error', 'no_pdf_url']),
            'results': results
        }

        logger.info(f"Batch download completed: {stats['downloaded']} successful, {stats['failed']} failed")
        return stats


    def _download_arxiv_by_id(
        self,
        arxiv_id: str,
        filepath: Path
    ) -> Tuple[bool, Optional[str], Optional[str]]:
        """
        Download PDF directly via arXiv ID using arxiv API

        This is the most reliable way to download arXiv papers, using the python-arxiv library to fetch PDFs directly.

        Args:
            arxiv_id: arXiv ID (e.g., "2301.12345" or "2301.12345v2")
            filepath: Target file path

        Returns:
            (success, url, error_message) tuple
        """
        if not arxiv_id:
            return False, None, "Empty arXiv ID"

        if arxiv is None:
            return False, None, "python-arxiv library not installed"

        # Clean arXiv ID (remove possible version number)
        clean_id = arxiv_id.split('v')[0] if 'v' in arxiv_id else arxiv_id

        logger.info(f"Downloading directly using arXiv API: {arxiv_id}")

        try:
            # Use arxiv library's Client to search for paper
            client = arxiv.Client()
            search = arxiv.Search(id_list=[clean_id])

            # Get paper object
            paper = next(client.results(search), None)

            if not paper:
                return False, None, f"arXiv ID not found: {arxiv_id}"

            # Use arxiv library's download_pdf method to download
            # Note: This method automatically handles retries and errors
            try:
                paper.download_pdf(dirpath=str(filepath.parent), filename=filepath.name)

                # Verify downloaded file
                if not filepath.exists():
                    return False, None, "Download completed but file does not exist"

                if filepath.stat().st_size < 1024:
                    filepath.unlink(missing_ok=True)
                    return False, None, f"Downloaded file too small ({filepath.stat().st_size} bytes)"

                # Verify PDF format
                with open(filepath, 'rb') as f:
                    header = f.read(5)
                    if not header.startswith(b'%PDF-'):
                        filepath.unlink(missing_ok=True)
                        return False, None, "Downloaded file is not a valid PDF"

                return True, paper.pdf_url, None

            except Exception as download_error:
                return False, None, f"arXiv API download failed: {str(download_error)}"

        except Exception as e:
            logger.error(f"Error occurred during arXiv API download: {e}")
            return False, None, str(e)

    def _download_arxiv_by_title(self, title_query: str, filepath: Path, max_results: int = 5) -> Tuple[bool, Optional[str], Optional[str]]:
        """
        Search arXiv by paper title and attempt to download PDF (core method for Strategy 2)

        Intelligent search strategy:
        1. First attempt exact title search (ti:"exact title")
        2. If no results, downgrade to keyword search (all:"title keywords")
        3. Use SequenceMatcher to calculate similarity and select best matching candidate paper
        4. Try downloading in order from highest to lowest similarity

        Dependency: Requires python-arxiv library (pip install arxiv)

        Args:
            title_query: Paper title or keywords
            filepath: Target file path
            max_results: Maximum number of search results (default 5)

        Returns:
            (success, url, error_message) tuple
            - success: Whether download succeeded
            - url: PDF URL of successful download (None if failed)
            - error_message: Error message if failed (None if successful)
        """
        if not title_query:
            return False, None, "Empty title query"

        if arxiv is None:
            return False, None, "python-arxiv library not installed, cannot execute Strategy 2"

        title_query = title_query.strip()
        logger.info(f"Attempting download via arXiv title search: {title_query}")

        try:
            search = arxiv.Search(
                query=f'ti:"{title_query}"',
                max_results=max_results,
                sort_by=arxiv.SortCriterion.Relevance
            )
            results_list = list(search.results())

            if not results_list:
                logger.info("Exact title search found no matches, trying keyword search")
                search_broad = arxiv.Search(
                    query=f'all:"{title_query}"',
                    max_results=max_results,
                    sort_by=arxiv.SortCriterion.Relevance
                )
                results_list = list(search_broad.results())

            if not results_list:
                return False, None, f"No papers matching '{title_query}' found in arXiv"

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
                    logger.debug(f"Candidate result missing PDF link, skipping: {candidate.title}")
                    continue

                success, error_msg = self._download_file_with_retry(pdf_url, filepath)
                if success:
                    return True, pdf_url, None

                logger.warning(f"Download via arXiv candidate {candidate.entry_id} failed: {error_msg}")

            return False, None, "All arXiv candidates failed to download"
        except Exception as e:
            logger.error(f"Error occurred during arXiv search or download: {e}")
            return False, None, str(e)

    def _find_pdf_urls(self, paper: Dict) -> List[str]:
        """
        Find all possible PDF links for a paper (core method for Strategy 1)

        Extract PDF links from multiple data sources by priority:

        0. [Priority] Direct construction from arXiv ID (from cross-database mapping)
        1. Paper's own pdf_url field
        2. OpenAlex's open_access.oa_url
        3. OpenAlex's open_access.oa_locations list
        4. Links inferred from DOI:
           - arXiv PDF (extract arxiv ID from DOI)
           - PubMed Central PDF (extract PMC ID from DOI)
           - Publisher-specific links (e.g., PLOS printable version)

        Args:
            paper: Paper information dictionary

        Returns:
            Deduplicated list of valid PDF URLs
        """
        pdf_urls = []

        # 0. [Highest Priority] If paper has arxiv_id (from cross-database mapping), directly construct arXiv PDF link
        # This is the core of cross-database mapping optimization: use verified arXiv ID to download directly
        if paper.get('arxiv_id'):
            arxiv_id = paper['arxiv_id']
            # Clean arxiv_id (remove possible version number)
            clean_arxiv_id = arxiv_id.split('v')[0] if 'v' in arxiv_id else arxiv_id
            arxiv_pdf = f"https://arxiv.org/pdf/{clean_arxiv_id}.pdf"
            pdf_urls.append(arxiv_pdf)
            logger.debug(f"Priority using cross-database mapped arXiv ID: {arxiv_id} -> {arxiv_pdf}")

        # 1. PDF link provided by paper (second priority)
        if paper.get('pdf_url'):
            # Avoid duplicate addition (if pdf_url is already an arXiv link)
            if paper['pdf_url'] not in pdf_urls:
                pdf_urls.append(paper['pdf_url'])

        # 2. Try to get more PDF links from OpenAlex
        open_access = paper.get('open_access', {})
        if isinstance(open_access, dict):
            # OpenAlex's oa_url
            if open_access.get('oa_url'):
                pdf_urls.append(open_access['oa_url'])

            # If has oa_locations, extract all PDF links
            oa_locations = open_access.get('oa_locations', [])
            for location in oa_locations:
                if isinstance(location, dict) and location.get('url_for_pdf'):
                    pdf_urls.append(location['url_for_pdf'])

        # 3. Try to construct PDF links from DOI
        doi = paper.get('doi', '')
        if doi:
            # Try arXiv link
            if 'arxiv' in doi.lower():
                arxiv_id = re.search(r'(\d{4}\.\d{4,5})', doi)
                if arxiv_id:
                    arxiv_pdf = f"https://arxiv.org/pdf/{arxiv_id.group(1)}.pdf"
                    if arxiv_pdf not in pdf_urls:
                        pdf_urls.append(arxiv_pdf)

            # Try PubMed Central link
            if 'pmc' in doi.lower() or 'pubmed' in doi.lower():
                pmc_match = re.search(r'pmc(\d+)', doi.lower())
                if pmc_match:
                    pmc_pdf = f"https://www.ncbi.nlm.nih.gov/pmc/articles/PMC{pmc_match.group(1)}/pdf/"
                    if pmc_pdf not in pdf_urls:
                        pdf_urls.append(pmc_pdf)

            # Try DOI resolution service
            if doi.startswith('10.'):
                # Some publishers provide direct PDF access
                if '10.1371/journal' in doi:  # PLOS
                    plos_pdf = f"https://journals.plos.org/plosone/article/file?id={doi}&type=printable"
                    if plos_pdf not in pdf_urls:
                        pdf_urls.append(plos_pdf)

        # 5. Deduplication and filtering
        unique_urls = []
        for url in pdf_urls:
            if url and url not in unique_urls and self._is_valid_pdf_url(url):
                unique_urls.append(url)

        logger.debug(f"Found {len(unique_urls)} PDF sources for paper {paper.get('id')}")
        return unique_urls

    def _is_valid_pdf_url(self, url: str) -> bool:
        """Check if URL is likely a valid PDF link"""
        if not url:
            return False

        # Basic URL format check
        parsed = urlparse(url)
        if not parsed.scheme or not parsed.netloc:
            return False

        # Exclude URLs that are obviously not PDFs
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
        """File download with retry mechanism"""
        for attempt in range(self.max_retries):
            try:
                # Add extra request headers for each request
                headers = self.session.headers.copy()

                # Add Referer (pretend to click from article page)
                parsed_url = urlparse(url)
                if 'journal-of-hepatology' in url:
                    # For this journal, add Referer
                    referer = url.replace('/pdf', '').replace('/pdfExtended', '')
                    headers['Referer'] = referer
                elif parsed_url.netloc:
                    # General strategy: set Referer to same domain homepage
                    headers['Referer'] = f"{parsed_url.scheme}://{parsed_url.netloc}/"

                response = self.session.get(url, headers=headers, stream=True, timeout=timeout)

                # Check status code
                if response.status_code == 403:
                    return False, f"Access forbidden (403) - {url}"
                elif response.status_code == 404:
                    return False, f"File not found (404) - {url}"
                elif response.status_code != 200:
                    if attempt < self.max_retries - 1:
                        logger.warning(f"Status code {response.status_code}, retry {attempt + 1}/{self.max_retries}")
                        time.sleep(2 ** attempt)  # Exponential backoff
                        continue
                    else:
                        return False, f"HTTP {response.status_code} - {url}"

                response.raise_for_status()

                # Check Content-Type
                content_type = response.headers.get('content-type', '').lower()
                content_length = response.headers.get('content-length')

                # If explicitly not PDF, skip
                if 'text/html' in content_type and 'application/pdf' not in content_type:
                    return False, f"Content-Type is HTML, not PDF - {url}"

                # If file too small, might be error page
                if content_length and int(content_length) < 1024:
                    return False, f"File too small ({content_length} bytes) - {url}"

                # Start download
                downloaded_size = 0
                with open(filepath, 'wb') as f:
                    for chunk in response.iter_content(chunk_size=8192):
                        if chunk:
                            f.write(chunk)
                            downloaded_size += len(chunk)

                # Verify downloaded file
                if filepath.stat().st_size < 1024:
                    filepath.unlink(missing_ok=True)
                    return False, f"Downloaded file too small ({filepath.stat().st_size} bytes)"

                # Simple PDF format validation
                with open(filepath, 'rb') as f:
                    header = f.read(5)
                    if not header.startswith(b'%PDF-'):
                        filepath.unlink(missing_ok=True)
                        return False, "Downloaded file is not a valid PDF"

                return True, "Success"

            except requests.exceptions.Timeout:
                if attempt < self.max_retries - 1:
                    logger.warning(f"Download timeout, retry {attempt + 1}/{self.max_retries}")
                    time.sleep(2 ** attempt)
                    continue
                else:
                    return False, f"Download timeout after {self.max_retries} attempts"

            except requests.exceptions.RequestException as e:
                if attempt < self.max_retries - 1:
                    logger.warning(f"Request exception, retry {attempt + 1}/{self.max_retries}: {e}")
                    time.sleep(2 ** attempt)
                    continue
                else:
                    return False, f"Request failed after {self.max_retries} attempts: {str(e)}"

            except Exception as e:
                logger.error(f"Unknown error occurred while downloading file: {e}")
                filepath.unlink(missing_ok=True)
                return False, f"Unknown error: {str(e)}"

        return False, f"Max retries ({self.max_retries}) exceeded"

    def _generate_filename(self, paper_id: str, title: str) -> str:
        """Generate safe filename"""
        # Clean title, remove special characters
        safe_title = re.sub(r'[^\w\s-]', '', title).strip()
        safe_title = re.sub(r'[\s]+', '_', safe_title)

        # Truncate overly long title
        if len(safe_title) > 50:
            safe_title = safe_title[:50]

        # Combine filename
        filename = f"{paper_id}_{safe_title}.pdf"
        return filename

    def get_download_stats(self) -> Dict:
        """Get download statistics"""
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
        """List downloaded papers"""
        if not self.download_dir.exists():
            return []

        pdf_files = list(self.download_dir.glob('*.pdf'))

        papers = []
        for pdf_file in pdf_files:
            # Parse paper_id from filename
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
    # Test code
    downloader = PDFDownloader()

    # Create test paper data
    test_paper = {
        'id': 'W2741809807',
        'title': 'Attention Is All You Need',
        'pdf_url': 'https://arxiv.org/pdf/1706.03762.pdf'
    }

    # Test download
    result = downloader.download_paper(test_paper)
    print(f"Download result: {result}")

    # View statistics
    stats = downloader.get_download_stats()
    print(f"Download statistics: {stats}")