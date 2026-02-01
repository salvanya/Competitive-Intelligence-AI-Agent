"""
Web Scraping using crawl4ai
Async crawler for competitor website content extraction
"""

import asyncio
from typing import Optional, Callable, Dict
from crawl4ai import AsyncWebCrawler


class CompetitorCrawler:
    """
    Async web crawler for competitor websites using crawl4ai.
    
    Provides concurrent scraping capabilities with progress callbacks
    for real-time UI updates. Handles errors gracefully and returns
    cleaned Markdown content for downstream processing.
    
    Attributes:
        progress_callback: Optional function to call with progress updates
        timeout: Scraping timeout in seconds
    
    Example:
        >>> crawler = CompetitorCrawler(progress_callback=st.status.update)
        >>> results = await crawler.scrape_multiple(["https://example.com"])
        >>> content = results["https://example.com"]
    """
    
    def __init__(
        self, 
        progress_callback: Optional[Callable[[str], None]] = None,
        timeout: int = 30
    ):
        """
        Initialize the competitor crawler.
        
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
        Scrape a single URL and return cleaned content.
        
        Uses crawl4ai's AsyncWebCrawler with optimized settings for
        competitive intelligence extraction. Returns Markdown format
        which is cleaner than raw HTML for LLM processing.
        
        Args:
            url: Website URL to scrape
        
        Returns:
            Optional[str]: Cleaned Markdown content or None if scraping failed
        
        Example:
            >>> content = await crawler.scrape_url("https://competitor.com")
            >>> if content:
            ...     print(f"Scraped {len(content)} characters")
        """
        try:
            self._update_progress(f"🌐 Scraping {url}...")
            
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
                        f"✅ Scraped {url} ({len(content):,} characters)"
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
        Scrape multiple URLs concurrently.
        
        Launches async tasks for all URLs simultaneously and gathers results.
        Failed scrapes return None but don't block other URLs from processing.
        
        Args:
            urls: List of website URLs to scrape
        
        Returns:
            Dict[str, Optional[str]]: Mapping of URL -> content
                                     (None if scraping failed)
        
        Example:
            >>> urls = ["https://comp1.com", "https://comp2.com"]
            >>> results = await crawler.scrape_multiple(urls)
            >>> successful = {url: content for url, content in results.items() if content}
            >>> print(f"Successfully scraped {len(successful)}/{len(urls)} URLs")
        """
        if not urls:
            self._update_progress("⚠️ No URLs provided to scrape")
            return {}
        
        self._update_progress(
            f"🚀 Starting concurrent scrape of {len(urls)} URL(s)..."
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
            f"📊 Scraping complete: {success_count}/{len(urls)} successful"
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