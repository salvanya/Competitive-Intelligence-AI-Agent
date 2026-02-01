"""
Windows-Compatible Web Scraper
Simple HTTP-based scraper using requests + BeautifulSoup
Replaces crawl4ai for Windows compatibility
"""

import asyncio
import requests
from typing import Optional, Callable, Dict
from bs4 import BeautifulSoup
from markdownify import markdownify as md
from urllib.parse import urlparse


class SimpleNewsArticleCrawler:
    """
    Simple HTTP-based web crawler for AI news articles.
    
    Windows-compatible alternative to crawl4ai that uses:
    - requests for HTTP fetching
    - BeautifulSoup for HTML parsing
    - markdownify for content conversion
    
    Attributes:
        progress_callback: Optional function to call with progress updates
        timeout: Scraping timeout in seconds
        headers: HTTP headers to mimic browser requests
    
    Example:
        >>> crawler = SimpleNewsArticleCrawler(progress_callback=st.status.update)
        >>> results = await crawler.scrape_multiple(["https://techcrunch.com/..."])
        >>> content = results["https://techcrunch.com/..."]
    """
    
    def __init__(
        self, 
        progress_callback: Optional[Callable[[str], None]] = None,
        timeout: int = 30
    ):
        """
        Initialize the news article crawler.
        
        Args:
            progress_callback: Optional function for progress updates
            timeout: Timeout in seconds for scraping operations (default: 30)
        """
        self.progress_callback = progress_callback
        self.timeout = timeout
        
        # Mimic browser to avoid 403 errors
        self.headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
            'Accept-Language': 'en-US,en;q=0.5',
            'Accept-Encoding': 'gzip, deflate, br',
            'DNT': '1',
            'Connection': 'keep-alive',
            'Upgrade-Insecure-Requests': '1'
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
                # Silently fail if callback errors (don't break scraping)
                print(f"Progress callback error: {str(e)}")
    
    def _extract_article_content(self, soup: BeautifulSoup, url: str) -> str:
        """
        Extract main article content from HTML.
        
        Uses heuristics to identify article content:
        1. Look for <article> tags
        2. Look for common article class names
        3. Remove navigation, ads, scripts, styles
        
        Args:
            soup: BeautifulSoup parsed HTML
            url: Original URL (for domain-specific handling)
        
        Returns:
            str: Extracted article content as Markdown
        """
        # Remove unwanted elements
        for element in soup(['script', 'style', 'nav', 'header', 'footer', 
                            'aside', 'iframe', 'noscript']):
            element.decompose()
        
        # Remove common ad/social classes
        for classname in ['advertisement', 'social-share', 'newsletter', 
                         'related-posts', 'comments', 'sidebar']:
            for element in soup.find_all(class_=lambda x: x and classname in x.lower()):
                element.decompose()
        
        # Try to find article content
        article_content = None
        
        # Strategy 1: Look for <article> tag
        article_tag = soup.find('article')
        if article_tag:
            article_content = article_tag
        
        # Strategy 2: Look for common article containers
        if not article_content:
            for selector in [
                {'class': lambda x: x and 'article' in x.lower()},
                {'class': lambda x: x and 'post-content' in x.lower()},
                {'class': lambda x: x and 'entry-content' in x.lower()},
                {'id': lambda x: x and 'article' in x.lower()},
                {'role': 'main'}
            ]:
                article_content = soup.find('div', selector)
                if article_content:
                    break
        
        # Strategy 3: Use the main tag
        if not article_content:
            article_content = soup.find('main')
        
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
        Scrape a single news article URL and return cleaned content.
        
        Uses requests + BeautifulSoup for scraping and converts to Markdown.
        
        Args:
            url: News article URL to scrape
        
        Returns:
            Optional[str]: Cleaned Markdown content or None if scraping failed
        
        Example:
            >>> content = await crawler.scrape_url("https://techcrunch.com/ai-news")
            >>> if content:
            ...     print(f"Scraped {len(content)} characters")
        """
        try:
            self._update_progress(f"🌐 Scraping news article: {url}...")
            
            # Run blocking requests in executor to avoid blocking event loop
            loop = asyncio.get_event_loop()
            response = await loop.run_in_executor(
                None,
                lambda: requests.get(url, headers=self.headers, timeout=self.timeout)
            )
            
            # Check response status
            response.raise_for_status()
            
            # Parse HTML
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # Extract article content
            content = self._extract_article_content(soup, url)
            
            if not content or len(content.strip()) < 100:
                self._update_progress(
                    f"⚠️ {url} returned minimal content ({len(content) if content else 0} chars)"
                )
                return None
            
            self._update_progress(
                f"✅ Scraped article from {url} ({len(content):,} characters)"
            )
            
            return content
        
        except requests.Timeout:
            self._update_progress(
                f"❌ Timeout scraping {url} (>{self.timeout}s)"
            )
            return None
        
        except requests.RequestException as e:
            self._update_progress(
                f"❌ HTTP error scraping {url}: {str(e)}"
            )
            return None
        
        except Exception as e:
            self._update_progress(
                f"❌ Error scraping {url}: {str(e)}"
            )
            return None
    
    async def scrape_multiple(
        self, 
        urls: list[str]
    ) -> Dict[str, Optional[str]]:
        """
        Scrape multiple news article URLs concurrently.
        
        Launches async tasks for all URLs simultaneously and gathers results.
        Failed scrapes return None but don't block other URLs from processing.
        
        Args:
            urls: List of news article URLs to scrape
        
        Returns:
            Dict[str, Optional[str]]: Mapping of URL -> content
                                     (None if scraping failed)
        
        Example:
            >>> urls = ["https://techcrunch.com/ai1", "https://venturebeat.com/ai2"]
            >>> results = await crawler.scrape_multiple(urls)
            >>> successful = {url: content for url, content in results.items() if content}
            >>> print(f"Successfully scraped {len(successful)}/{len(urls)} articles")
        """
        if not urls:
            self._update_progress("⚠️ No URLs provided to scrape")
            return {}
        
        self._update_progress(
            f"🚀 Starting concurrent scrape of {len(urls)} article(s)..."
        )
        
        # Create async tasks for all URLs
        tasks = [self.scrape_url(url) for url in urls]
        
        # Gather results (return_exceptions=True prevents one failure from stopping others)
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Build result dictionary
        scraped_data = {}
        success_count = 0
        
        for url, result in zip(urls, results):
            if isinstance(result, Exception):
                # Task raised an exception
                scraped_data[url] = None
                self._update_progress(
                    f"❌ {url} failed with exception: {str(result)}"
                )
            elif result is not None:
                # Successful scrape
                scraped_data[url] = result
                success_count += 1
            else:
                # Returned None (handled error in scrape_url)
                scraped_data[url] = None
        
        # Final summary
        self._update_progress(
            f"📊 Scraping complete: {success_count}/{len(urls)} article(s) successful"
        )
        
        return scraped_data
    
    def validate_url(self, url: str) -> bool:
        """
        Validate URL format before scraping.
        
        Args:
            url: URL string to validate
        
        Returns:
            bool: True if URL appears valid, False otherwise
        """
        if not url or not isinstance(url, str):
            return False
        
        url = url.strip()
        
        # Must start with http:// or https://
        if not url.startswith(('http://', 'https://')):
            return False
        
        # Must have a domain
        if len(url) < 10:  # Minimum: https://a.b
            return False
        
        return True
    
    def get_domain(self, url: str) -> Optional[str]:
        """
        Extract domain name from URL.
        
        Useful for identifying news sources.
        
        Args:
            url: Article URL
        
        Returns:
            Optional[str]: Domain name or None if invalid URL
        
        Example:
            >>> domain = crawler.get_domain("https://techcrunch.com/2026/01/ai-news")
            >>> print(domain)
            techcrunch.com
        """
        try:
            parsed = urlparse(url)
            domain = parsed.netloc
            
            # Remove 'www.' prefix if present
            if domain.startswith('www.'):
                domain = domain[4:]
            
            return domain if domain else None
        except Exception:
            return None