"""
Streamlit UI for AI News Summarizer & Analyzer
Main entry point for the application
"""

import streamlit as st
import asyncio
import sys
from pathlib import Path
from typing import List, Optional

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from app.config import AppConfig
from app.scraping.simple_crawler import SimpleNewsArticleCrawler
from app.extraction.chain import ExtractionChain
from app.extraction.schemas import NewsArticleProfile
from app.vectorstore.store import NewsVectorStore
from app.synthesis.report_generator import SynthesisAgent
from app.utils.rate_limiter import RateLimiter
from app.utils.logger import ProgressLogger


# Page configuration
st.set_page_config(
    page_title="AI News Summarizer & Analyzer",
    page_icon="📰",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: 700;
        color: #1f77b4;
        margin-bottom: 0.5rem;
    }
    .sub-header {
        font-size: 1.2rem;
        color: #666;
        margin-bottom: 2rem;
    }
    .stProgress > div > div > div > div {
        background-color: #1f77b4;
    }
    .success-box {
        padding: 1rem;
        border-radius: 0.5rem;
        background-color: #d4edda;
        border: 1px solid #c3e6cb;
        color: #155724;
    }
    .error-box {
        padding: 1rem;
        border-radius: 0.5rem;
        background-color: #f8d7da;
        border: 1px solid #f5c6cb;
        color: #721c24;
    }
    .info-box {
        padding: 1rem;
        border-radius: 0.5rem;
        background-color: #d1ecf1;
        border: 1px solid #bee5eb;
        color: #0c5460;
    }
</style>
""", unsafe_allow_html=True)


def initialize_session_state():
    """Initialize session state variables."""
    if 'api_key' not in st.session_state:
        st.session_state.api_key = None
    
    if 'api_key_input' not in st.session_state:
        st.session_state.api_key_input = ""
    
    if 'report_generated' not in st.session_state:
        st.session_state.report_generated = False
    
    if 'current_report' not in st.session_state:
        st.session_state.current_report = ""
    
    if 'analyzed_articles' not in st.session_state:
        st.session_state.analyzed_articles = []


def set_api_key():
    """Callback function to set API key when button is clicked."""
    if st.session_state.api_key_input.strip():
        st.session_state.api_key = st.session_state.api_key_input.strip()


def render_sidebar():
    """Render sidebar with configuration options."""
    with st.sidebar:
        st.header("⚙️ Configuration")
        
        # API Key Input with submit button
        st.text_input(
            "Google AI Studio API Key",
            type="password",
            help="Get your key at https://aistudio.google.com/app/apikey",
            placeholder="Enter your API key here...",
            key="api_key_input",
            on_change=None  # Don't auto-submit
        )
        
        # Submit button for API key
        if st.button("Submit API Key", use_container_width=True, type="primary"):
            set_api_key()
        
        # Show API key status
        if st.session_state.api_key:
            st.success("✅ API Key loaded")
        else:
            st.warning("⚠️ API key required")
        
        st.markdown("---")
        
        # Rate Limits Info
        st.markdown("### 📊 Rate Limits")
        st.info("""
        **Free Tier Limits:**
        - 15 requests/minute
        - 1M tokens/day
        
        The app automatically manages rate limiting.
        """)
        
        st.markdown("---")
        
        # Model Info
        st.markdown("### 🤖 Model Configuration")
        st.code("""
Model: gemini-2.0-flash
Extraction Temp: 0.0
Synthesis Temp: 0.7
Scraper: requests + BeautifulSoup
        """)
        
        st.markdown("---")
        
        # About
        st.markdown("### ℹ️ About")
        st.markdown("""
        **AI News Summarizer & Analyzer**
        
        Analyze AI news with:
        - Google Gemini 2.0 Flash
        - LangChain LCEL
        - Qdrant Vector Store
        - Simple HTTP Scraper (Windows-compatible)
        
        [View Documentation](#) | [GitHub](#)
        """)


def render_header():
    """Render main header."""
    st.markdown('<h1 style="color: #1f77b4;">AI News Summarizer & Analyzer</h1>', 
                unsafe_allow_html=True)
    st.markdown(
        '<p class="sub-header">AI-powered news intelligence using Google Gemini, '
        'LangChain, and semantic search</p>',
        unsafe_allow_html=True
    )
    
    st.markdown("""
    **How it works:**
    1. Enter 1-5 AI news article URLs
    2. Specify your analysis objective
    3. Get instant AI-powered news intelligence
    
    The system scrapes articles, extracts structured insights, and generates comprehensive 
    analysis reports with technology trends, industry impact, use cases, and priority rankings.
    """)
    
    st.markdown("---")


def validate_urls(urls: List[str]) -> tuple[List[str], List[str]]:
    """
    Validate URL list.
    
    Args:
        urls: List of URL strings
    
    Returns:
        tuple: (valid_urls, invalid_urls)
    """
    valid = []
    invalid = []
    
    for url in urls:
        url = url.strip()
        if not url:
            continue
        
        if url.startswith(('http://', 'https://')):
            valid.append(url)
        else:
            invalid.append(url)
    
    return valid, invalid


async def run_analysis(
    urls: List[str],
    objective: str,
    progress_container,
    report_container
):
    """
    Main analysis pipeline.
    
    Args:
        urls: List of news article URLs
        objective: Analysis objective
        progress_container: Streamlit container for progress updates
        report_container: Streamlit container for report display
    """
    with progress_container:
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        # Initialize configuration
        config = AppConfig(google_api_key=st.session_state.api_key)
        rate_limiter = RateLimiter(config.MAX_RPM)
        
        # Step 1: Scraping (20% progress)
        status_text.markdown("### 🌐 Phase 1: Article Scraping")
        st.info(f"Scraping {len(urls)} news article(s)...")
        progress_bar.progress(0.05)
        
        def update_progress(msg: str):
            status_text.markdown(f"**Current:** {msg}")
        
        crawler = SimpleNewsArticleCrawler(
            progress_callback=update_progress,
            timeout=config.SCRAPE_TIMEOUT
        )
        
        scraped_data = await crawler.scrape_multiple(urls)
        progress_bar.progress(0.20)
        
        # Show scraping results
        successful_scrapes = sum(1 for content in scraped_data.values() if content)
        st.success(f"✅ Scraped {successful_scrapes}/{len(urls)} article(s) successfully")
        
        # Show detailed diagnostics
        diagnostics = crawler.get_scrape_diagnostics()
        with st.expander("📋 Scraping Details", expanded=(successful_scrapes < len(urls))):
            for url, details in diagnostics.items():
                if details.get('success'):
                    st.success(f"✅ **{url}**")
                    st.write(f"- Page Title: {details.get('page_title', 'N/A')}")
                    st.write(f"- Content Length: {details.get('extracted_length', 0):,} chars")
                    st.write(f"- Status Code: {details.get('status_code', 'N/A')}")
                else:
                    st.error(f"❌ **{url}**")
                    st.write(f"- Error: {details.get('error', 'Unknown')}")
                    if 'status_code' in details:
                        st.write(f"- Status Code: {details.get('status_code')}")
                    if 'page_title' in details:
                        st.write(f"- Page Title: {details.get('page_title')}")
                st.markdown("---")
        
        # Step 2: Extraction (20% -> 50% progress)
        status_text.markdown("### 🔍 Phase 2: News Intelligence Extraction")
        st.info("Extracting structured news insights...")
        progress_bar.progress(0.25)
        
        extractor = ExtractionChain(st.session_state.api_key, config)
        profiles: List[NewsArticleProfile] = []
        
        for idx, (url, content) in enumerate(scraped_data.items(), 1):
            update_progress(f"Extracting insights from article {idx}/{len(scraped_data)}...")
            
            if content:
                # Respect rate limit
                await rate_limiter.acquire()
                
                try:
                    profile = await extractor.extract(url, content)
                    profiles.append(profile)
                except Exception as e:
                    # Create error profile
                    error_profile = NewsArticleProfile(
                        headline=url,
                        article_url=url,
                        news_source="Unknown",
                        article_summary=f"Extraction error: {str(e)}",
                        scrape_success=False,
                        error_message=f"Extraction error: {str(e)}"
                    )
                    profiles.append(error_profile)
            else:
                # Create error profile for failed scrape
                error_profile = NewsArticleProfile(
                    headline=url,
                    article_url=url,
                    news_source="Unknown",
                    article_summary="Article scraping failed",
                    scrape_success=False,
                    error_message="Website scraping failed"
                )
                profiles.append(error_profile)
            
            # Update progress incrementally
            progress_bar.progress(0.25 + (0.25 * idx / len(scraped_data)))
        
        progress_bar.progress(0.50)
        
        # Show extraction results
        valid_profiles = [p for p in profiles if p.scrape_success]
        st.success(f"✅ Extracted insights from {len(valid_profiles)}/{len(profiles)} article(s)")
        
        # Store profiles in session state
        st.session_state.analyzed_articles = profiles
        
        # Step 3: Vector Store (50% -> 60% progress)
        status_text.markdown("### 📊 Phase 3: Semantic Indexing")
        st.info("Building semantic search index...")
        progress_bar.progress(0.55)
        
        vectorstore = NewsVectorStore(st.session_state.api_key, config)
        
        for profile in valid_profiles:
            await vectorstore.ingest_article(profile)
        
        progress_bar.progress(0.60)
        st.success(f"✅ Indexed {len(valid_profiles)} news article(s)")
        
        # Step 4: Synthesis (60% -> 100% progress)
        status_text.markdown("### ✍️ Phase 4: Report Generation")
        st.info("Generating comprehensive news analysis...")
        progress_bar.progress(0.65)
        
        # Prepare synthesis agent
        synthesizer = SynthesisAgent(st.session_state.api_key, config)
        
        # Respect rate limit before synthesis
        await rate_limiter.acquire()
        
        progress_bar.progress(0.70)
    
    # Display report in separate container
    with report_container:
        st.markdown("---")
        st.markdown("## AI News Analysis Report")
        
        # Create placeholder for streaming report
        report_placeholder = st.empty()
        full_report = ""
        
        # Stream report generation
        async for chunk in synthesizer.generate_report_stream(profiles, objective):
            full_report += chunk
            report_placeholder.markdown(full_report)
        
        # Store report in session state
        st.session_state.current_report = full_report
        st.session_state.report_generated = True
        
        # Update progress to complete
        with progress_container:
            progress_bar.progress(1.0)
            status_text.markdown("### ✅ Analysis Complete!")


def main():
    """Main application entry point."""
    # Initialize session state
    initialize_session_state()
    
    # Render UI components
    render_sidebar()
    render_header()
    
    # Check for API key
    if not st.session_state.api_key:
        st.warning("👈 Please enter your Google AI Studio API key in the sidebar to begin")
        
        with st.expander("ℹ️ How to get an API key"):
            st.markdown("""
            1. Go to [Google AI Studio](https://aistudio.google.com/app/apikey)
            2. Sign in with your Google account
            3. Click "Create API Key"
            4. Copy the key and paste it in the sidebar
            
            **Note:** The API key is stored only in your browser session and is never saved.
            """)
        
        st.stop()
    
    # Main input section
    st.header("1️⃣ Enter AI News Article URLs")
    st.markdown("Enter 1-5 AI news article URLs (one per field)")
    
    # URL inputs
    col1, col2 = st.columns(2)
    
    with col1:
        url1 = st.text_input(
            "Article 1 URL",
            key="url_1",
            placeholder="https://techcrunch.com/ai/..."
        )
        
        url2 = st.text_input(
            "Article 2 URL",
            key="url_2",
            placeholder="https://venturebeat.com/ai/..."
        )
        
        url3 = st.text_input(
            "Article 3 URL",
            key="url_3",
            placeholder="https://theverge.com/ai/..."
        )
    
    with col2:
        url4 = st.text_input(
            "Article 4 URL",
            key="url_4",
            placeholder="https://arstechnica.com/ai/..."
        )
        
        url5 = st.text_input(
            "Article 5 URL",
            key="url_5",
            placeholder="https://wired.com/ai/..."
        )
    
    # Collect and validate URLs
    urls = [url for url in [url1, url2, url3, url4, url5] if url.strip()]
    
    if not urls:
        st.info("👆 Add at least one news article URL to begin analysis")
        st.stop()
    
    # Validate URLs
    valid_urls, invalid_urls = validate_urls(urls)
    
    if invalid_urls:
        st.error(f"❌ Invalid URLs detected: {', '.join(invalid_urls)}")
        st.info("URLs must start with http:// or https://")
        st.stop()
    
    # Analysis objective
    st.header("2️⃣ Analysis Objective")
    st.markdown("Specify what you'd like to analyze about these AI news articles")
    
    objective = st.text_area(
        "What would you like to analyze?",
        value="Analyze the AI news developments focusing on technology trends, industry impact, practical use cases, and provide a prioritized investigation roadmap.",
        height=100,
        help="Be specific about what aspects you want to focus on (e.g., technology trends, industry impact, use cases, priority ranking)"
    )
    
    if not objective.strip():
        st.warning("Please specify an analysis objective")
        st.stop()
    
    # Generate button
    st.markdown("---")
    
    col_btn1, col_btn2, col_btn3 = st.columns([2, 1, 2])
    
    with col_btn2:
        generate_button = st.button(
            "🚀 Generate Analysis",
            type="primary",
            use_container_width=True
        )
    
    # Show summary before generation
    with st.expander("📋 Analysis Summary", expanded=True):
        url_list = "\n".join([f"- {url}" for url in valid_urls])
        st.markdown(f"""
**Articles to analyze:** {len(valid_urls)}

{url_list}

**Analysis objective:** {objective[:200]}{"..." if len(objective) > 200 else ""}

**Estimated time:** ~{len(valid_urls) * 30} seconds
        """)
    
    # Analysis execution
    if generate_button:
        # Create containers for progress and report
        progress_container = st.container()
        report_container = st.container()
        
        # Run async analysis
        try:
            asyncio.run(
                run_analysis(
                    valid_urls,
                    objective,
                    progress_container,
                    report_container
                )
            )
        except Exception as e:
            st.error(f"❌ An error occurred during analysis: {str(e)}")
            
            with st.expander("🔍 Error Details"):
                st.code(str(e))
                st.markdown("""
                **Troubleshooting:**
                1. Check your internet connection
                2. Verify API key is valid
                3. Ensure URLs are accessible
                4. Try with fewer articles
                5. Some websites may block automated scraping
                """)
    
    # Download report button (if report exists)
    if st.session_state.report_generated and st.session_state.current_report:
        st.markdown("---")
        
        col_dl1, col_dl2, col_dl3 = st.columns([2, 1, 2])
        
        with col_dl2:
            st.download_button(
                label="📥 Download Report",
                data=st.session_state.current_report,
                file_name="ai_news_analysis_report.md",
                mime="text/markdown",
                use_container_width=True
            )
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; color: #666; padding: 2rem 0;'>
        <p><strong>AI News Summarizer & Analyzer</strong></p>
        <p>Built with Google Gemini 2.0 Flash | LangChain | Qdrant | Simple HTTP Scraper</p>
        <p style='font-size: 0.9rem;'>
            <a href='#' style='color: #1f77b4; text-decoration: none;'>Documentation</a> • 
            <a href='#' style='color: #1f77b4; text-decoration: none;'>GitHub</a> • 
            <a href='#' style='color: #1f77b4; text-decoration: none;'>Report Issues</a>
        </p>
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()