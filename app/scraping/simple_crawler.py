"""
Windows-Compatible Web Scraper
Enhanced with anti-bot evasion and retry logic
"""

import asyncio
import requests
import time
from typing import Optional, Callable, Dict
from bs4 import BeautifulSoup
from markdownify import markdownify as md
from urllib.parse import urlparse
import random


class SimpleNewsArticleCrawler:
    """
    Enhanced HTTP-based web crawler with anti-bot evasion.
    
    Features:
    - Rotating User-Agent headers
    - Retry logic with exponential backoff
    - Cookie handling
    - Better content extraction heuristics
    
    Attributes:
        progress_callback: Optional function to call with progress updates
        timeout: Scraping timeout in seconds
        max_retries: Maximum retry attempts for failed requests
        scrape_results: Detailed results for debugging
    """
    
    def __init__(
        self, 
        progress_callback: Optional[Callable[[str], None]] = None,
        timeout: int = 30,
        max_retries: int = 3
    ):
        """
        Initialize the news article crawler.
        
        Args:
            progress_callback: Optional function for progress updates
            timeout: Timeout in seconds for scraping operations (default: 30)
            max_retries: Maximum retry attempts (default: 3)
        """
        self.progress_callback = progress_callback
        self.timeout = timeout
        self.max_retries = max_retries
        self.scrape_results = {}
        
        # Multiple User-Agent strings to rotate
        self.user_agents = [
            'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/121.0.0.0 Safari/537.36',
            'Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:122.0) Gecko/20100101 Firefox/122.0',
            'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/121.0.0.0 Safari/537.36',
            'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Edge/121.0.0.0',
        ]
    
    def _get_headers(self) -> Dict[str, str]:
        """
        Generate request headers with random User-Agent.
        
        Returns:
            Dict: HTTP headers mimicking a real browser
        """
        return {
            'User-Agent': random.choice(self.user_agents),
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,*/*;q=0.8',
            'Accept-Language': 'en-US,en;q=0.9',
            'Accept-Encoding': 'gzip, deflate, br',
            'DNT': '1',
            'Connection': 'keep-alive',
            'Upgrade-Insecure-Requests': '1',
            'Sec-Fetch-Dest': 'document',
            'Sec-Fetch-Mode': 'navigate',
            'Sec-Fetch-Site': 'none',
            'Sec-Fetch-User': '?1',
            'Cache-Control': 'max-age=0',
        }
    
    def _update_progress(self, message: str) -> None:
        """
        Internal method to update progress via callback.
        
        Args:
            message: Progress message to send
        """
        if self.progress_callback:
            try:
                self.progress_callback(message)
            except Exception as e:
                print(f"Progress callback error: {str(e)}")
    
    def _extract_article_content(self, soup: BeautifulSoup, url: str) -> str:
        """
        Extract main article content from HTML with improved heuristics.
        
        Args:
            soup: BeautifulSoup parsed HTML
            url: Original URL (for domain-specific handling)
        
        Returns:
            str: Extracted article content as Markdown
        """
        # Remove unwanted elements
        for element in soup(['script', 'style', 'nav', 'header', 'footer', 
                            'aside', 'iframe', 'noscript', 'form']):
            element.decompose()
        
        # Remove common ad/social/newsletter classes
        unwanted_patterns = [
            'advertisement', 'ad-', 'social-share', 'newsletter', 
            'related-posts', 'comments', 'sidebar', 'promo',
            'widget', 'subscribe', 'newsletter', 'popup',
            'cookie', 'gdpr', 'consent'
        ]
        
        for pattern in unwanted_patterns:
            for element in soup.find_all(class_=lambda x: x and pattern in x.lower()):
                element.decompose()
            for element in soup.find_all(id=lambda x: x and pattern in x.lower()):
                element.decompose()
        
        # Try multiple strategies to find article content
        article_content = None
        
        # Strategy 1: <article> tag
        article_tag = soup.find('article')
        if article_tag:
            article_content = article_tag
        
        # Strategy 2: Common article container patterns
        if not article_content:
            selectors = [
                {'class': lambda x: x and 'article-body' in x.lower()},
                {'class': lambda x: x and 'post-content' in x.lower()},
                {'class': lambda x: x and 'entry-content' in x.lower()},
                {'class': lambda x: x and 'article-content' in x.lower()},
                {'class': lambda x: x and 'story-body' in x.lower()},
                {'class': lambda x: x and 'main-content' in x.lower()},
                {'id': lambda x: x and 'article' in x.lower()},
                {'id': lambda x: x and 'content' in x.lower()},
                {'role': 'main'}
            ]
            
            for selector in selectors:
                article_content = soup.find('div', selector)
                if article_content:
                    break
        
        # Strategy 3: <main> tag
        if not article_content:
            article_content = soup.find('main')
        
        # Strategy 4: Find largest text block (heuristic)
        if not article_content:
            all_divs = soup.find_all('div')
            if all_divs:
                # Find div with most text content
                max_text_length = 0
                best_div = None
                for div in all_divs:
                    text_length = len(div.get_text(strip=True))
                    if text_length > max_text_length:
                        max_text_length = text_length
                        best_div = div
                
                if best_div and max_text_length > 500:  # Minimum threshold
                    article_content = best_div
        
        # Fallback: Use body
        if not article_content:
            article_content = soup.find('body')
        
        if not article_content:
            return ""
        
        # Convert to Markdown
        markdown_content = md(str(article_content), heading_style="ATX")
        
        # Clean up excessive whitespace
        lines = [line.strip() for line in markdown_content.split('\n')]
        cleaned_lines = [line for line in lines if line]
        
        return '\n\n'.join(cleaned_lines)
    
    async def scrape_url(self, url: str) -> Optional[str]:
        """
        Scrape a single news article URL with retry logic.
        
        Args:
            url: News article URL to scrape
        
        Returns:
            Optional[str]: Cleaned Markdown content or None if scraping failed
        """
        last_error = None
        
        for attempt in range(self.max_retries):
            try:
                if attempt > 0:
                    # Exponential backoff: 2s, 4s, 8s
                    wait_time = 2 ** attempt
                    self._update_progress(f"⏳ Retry {attempt + 1}/{self.max_retries} for {url} (waiting {wait_time}s)...")
                    await asyncio.sleep(wait_time)
                
                self._update_progress(f"🌐 Scraping news article: {url}...")
                
                # Create session with cookies enabled
                session = requests.Session()
                
                # Run blocking requests in executor
                loop = asyncio.get_event_loop()
                response = await loop.run_in_executor(
                    None,
                    lambda: session.get(
                        url, 
                        headers=self._get_headers(), 
                        timeout=self.timeout, 
                        allow_redirects=True
                    )
                )
                
                # Store response details
                self.scrape_results[url] = {
                    'status_code': response.status_code,
                    'content_type': response.headers.get('content-type', 'unknown'),
                    'content_length': len(response.content),
                    'final_url': response.url,
                    'attempt': attempt + 1,
                    'success': False,
                    'error': None
                }
                
                # Check for anti-bot challenges
                if response.status_code == 403:
                    # Check if it's a Cloudflare or similar challenge
                    if 'cloudflare' in response.text.lower() or 'just a moment' in response.text.lower():
                        last_error = "Cloudflare anti-bot protection detected"
                        self.scrape_results[url]['error'] = last_error
                        self._update_progress(f"⚠️ {url}: Cloudflare protection detected")
                        continue
                    else:
                        last_error = f"HTTP 403: Access Forbidden"
                        self.scrape_results[url]['error'] = last_error
                        self._update_progress(f"⚠️ {url}: Access forbidden (403)")
                        # Don't retry 403s - they won't succeed
                        break
                
                # Check response status
                response.raise_for_status()
                
                # Parse HTML
                soup = BeautifulSoup(response.content, 'html.parser')
                
                # Extract title
                title_tag = soup.find('title')
                page_title = title_tag.get_text().strip() if title_tag else "Unknown"
                self.scrape_results[url]['page_title'] = page_title
                
                # Extract article content
                content = self._extract_article_content(soup, url)
                
                if not content or len(content.strip()) < 100:
                    last_error = f"Minimal content ({len(content)} chars)"
                    self.scrape_results[url]['error'] = last_error
                    self._update_progress(f"⚠️ {url} returned minimal content")
                    self._update_progress(f"   Page title: {page_title}")
                    continue
                
                # Success!
                self.scrape_results[url]['success'] = True
                self.scrape_results[url]['extracted_length'] = len(content)
                
                self._update_progress(f"✅ Scraped article from {url} ({len(content):,} characters)")
                self._update_progress(f"   Title: {page_title}")
                
                return content
        
            except requests.Timeout:
                last_error = f"Timeout (>{self.timeout}s)"
                self.scrape_results[url] = {'success': False, 'error': last_error, 'attempt': attempt + 1}
                self._update_progress(f"❌ Timeout scraping {url}")
                
            except requests.HTTPError as e:
                last_error = f"HTTP {e.response.status_code}: {e.response.reason}"
                self.scrape_results[url] = {
                    'success': False, 
                    'error': last_error,
                    'status_code': e.response.status_code,
                    'attempt': attempt + 1
                }
                self._update_progress(f"❌ HTTP error scraping {url}: {last_error}")
                # Don't retry 4xx errors
                if 400 <= e.response.status_code < 500:
                    break
                
            except requests.RequestException as e:
                last_error = str(e)
                self.scrape_results[url] = {'success': False, 'error': last_error, 'attempt': attempt + 1}
                self._update_progress(f"❌ Request error scraping {url}: {last_error}")
                
            except Exception as e:
                last_error = str(e)
                self.scrape_results[url] = {'success': False, 'error': last_error, 'attempt': attempt + 1}
                self._update_progress(f"❌ Error scraping {url}: {last_error}")
        
        # All retries failed
        self._update_progress(f"❌ Failed to scrape {url} after {self.max_retries} attempts")
        return None
    
    async def scrape_multiple(
        self, 
        urls: list[str]
    ) -> Dict[str, Optional[str]]:
        """
        Scrape multiple news article URLs concurrently.
        
        Args:
            urls: List of news article URLs to scrape
        
        Returns:
            Dict[str, Optional[str]]: Mapping of URL -> content
        """
        if not urls:
            self._update_progress("⚠️ No URLs provided to scrape")
            return {}
        
        self._update_progress(f"🚀 Starting concurrent scrape of {len(urls)} article(s)...")
        
        # Create async tasks for all URLs
        tasks = [self.scrape_url(url) for url in urls]
        
        # Gather results
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Build result dictionary
        scraped_data = {}
        success_count = 0
        
        for url, result in zip(urls, results):
            if isinstance(result, Exception):
                scraped_data[url] = None
                self.scrape_results[url] = {'success': False, 'error': str(result)}
                self._update_progress(f"❌ {url} failed with exception: {str(result)}")
            elif result is not None:
                scraped_data[url] = result
                success_count += 1
            else:
                scraped_data[url] = None
        
        # Final summary
        self._update_progress(f"📊 Scraping complete: {success_count}/{len(urls)} article(s) successful")
        
        # Show failed scrapes
        for url, details in self.scrape_results.items():
            if not details.get('success', False):
                error = details.get('error', 'Unknown error')
                self._update_progress(f"   ⚠️ {url}: {error}")
        
        return scraped_data
    
    def get_scrape_diagnostics(self) -> Dict[str, Dict]:
        """Get detailed diagnostics about scraping results."""
        return self.scrape_results
    
    def validate_url(self, url: str) -> bool:
        """Validate URL format before scraping."""
        if not url or not isinstance(url, str):
            return False
        
        url = url.strip()
        
        if not url.startswith(('http://', 'https://')):
            return False
        
        if len(url) < 10:
            return False
        
        return True
    
    def get_domain(self, url: str) -> Optional[str]:
        """Extract domain name from URL."""
        try:
            parsed = urlparse(url)
            domain = parsed.netloc
            
            if domain.startswith('www.'):
                domain = domain[4:]
            
            return domain if domain else None
        except Exception:
            return None