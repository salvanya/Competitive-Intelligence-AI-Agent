"""
Web Scraping using crawl4ai
Async crawler for AI news article content extraction
"""

import asyncio
from typing import Optional, Callable, Dict
from crawl4ai import AsyncWebCrawler


class NewsArticleCrawler:
    """
    Async web crawler for AI news articles using crawl4ai.
    
    Provides concurrent scraping capabilities with progress callbacks
    for real-time UI updates. Handles errors gracefully and returns
    cleaned Markdown content for downstream processing.
    
    Attributes:
        progress_callback: Optional function to call with progress updates
        timeout: Scraping timeout in seconds
    
    Example:
        >>> crawler = NewsArticleCrawler(progress_callback=st.status.update)
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
    
    async def scrape_url(self, url: str) -> Optional[str]:
        """
        Scrape a single news article URL and return cleaned content.
        
        Uses crawl4ai's AsyncWebCrawler with optimized settings for
        news article extraction. Returns Markdown format which is
        cleaner than raw HTML for LLM processing.
        
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
            
            # Initialize async crawler (crawl4ai 0.8.0 syntax)
            async with AsyncWebCrawler(verbose=False) as crawler:
                # Execute crawl
                result = await crawler.arun(
                    url=url,
                    word_count_threshold=10,
                    exclude_external_links=True,
                    page_timeout=self.timeout * 1000,
                )
                
                if result.success:
                    # Extract markdown content (0.8.0 uses .markdown attribute)
                    content = result.markdown
                    
                    if not content or len(content.strip()) < 100:
                        self._update_progress(
                            f"⚠️ {url} returned minimal content ({len(content) if content else 0} chars)"
                        )
                        return None
                    
                    self._update_progress(
                        f"✅ Scraped article from {url} ({len(content):,} characters)"
                    )
                    
                    return content
                else:
                    error_msg = getattr(result, 'error_message', 'Unknown error')
                    self._update_progress(
                        f"❌ Failed to scrape {url}: {error_msg}"
                    )
                    return None
        
        except asyncio.TimeoutError:
            self._update_progress(
                f"❌ Timeout scraping {url} (>{self.timeout}s)"
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
            from urllib.parse import urlparse
            parsed = urlparse(url)
            domain = parsed.netloc
            
            # Remove 'www.' prefix if present
            if domain.startswith('www.'):
                domain = domain[4:]
            
            return domain if domain else None
        except Exception:
            return None