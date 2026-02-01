"""
Streamlit UI for Competitive Intelligence Agent
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
from app.scraping.crawler import CompetitorCrawler
from app.extraction.chain import ExtractionChain
from app.extraction.schemas import CompetitorProfile
from app.vectorstore.store import CompetitorVectorStore
from app.synthesis.report_generator import SynthesisAgent
from app.utils.rate_limiter import RateLimiter
from app.utils.logger import ProgressLogger


# Page configuration
st.set_page_config(
    page_title="Competitive Intelligence Agent",
    page_icon="🔍",
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
    
    if 'report_generated' not in st.session_state:
        st.session_state.report_generated = False
    
    if 'current_report' not in st.session_state:
        st.session_state.current_report = ""
    
    if 'analyzed_profiles' not in st.session_state:
        st.session_state.analyzed_profiles = []


def render_sidebar():
    """Render sidebar with configuration options."""
    with st.sidebar:
        st.header("⚙️ Configuration")
        
        # API Key Input
        api_key = st.text_input(
            "Google AI Studio API Key",
            type="password",
            help="Get your key at https://aistudio.google.com/app/apikey",
            placeholder="Enter your API key here..."
        )
        
        if api_key:
            st.session_state.api_key = api_key
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
        """)
        
        st.markdown("---")
        
        # About
        st.markdown("### ℹ️ About")
        st.markdown("""
        **Competitive Intelligence Agent**
        
        AI-powered competitive analysis using:
        - Google Gemini 2.0 Flash
        - LangChain LCEL
        - Qdrant Vector Store
        - crawl4ai Web Scraper
        
        [View Documentation](#) | [GitHub](#)
        """)


def render_header():
    """Render main header."""
    st.markdown('<p class="main-header">🔍 Competitive Intelligence Synthesis Agent</p>', 
                unsafe_allow_html=True)
    st.markdown(
        '<p class="sub-header">AI-powered competitive analysis using Google Gemini, '
        'LangChain, and semantic search</p>',
        unsafe_allow_html=True
    )
    
    st.markdown("""
    **How it works:**
    1. Enter 1-3 competitor URLs
    2. Specify your analysis objective
    3. Get instant AI-powered competitive intelligence
    
    The system scrapes websites, extracts structured data, and generates comprehensive 
    analysis reports with pricing comparisons, feature gaps, and strategic recommendations.
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
        urls: List of competitor URLs
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
        status_text.markdown("### 🌐 Phase 1: Web Scraping")
        st.info(f"Scraping {len(urls)} competitor website(s)...")
        progress_bar.progress(0.05)
        
        def update_progress(msg: str):
            status_text.markdown(f"**Current:** {msg}")
        
        crawler = CompetitorCrawler(
            progress_callback=update_progress,
            timeout=config.SCRAPE_TIMEOUT
        )
        
        scraped_data = await crawler.scrape_multiple(urls)
        progress_bar.progress(0.20)
        
        # Show scraping results
        successful_scrapes = sum(1 for content in scraped_data.values() if content)
        st.success(f"✅ Scraped {successful_scrapes}/{len(urls)} websites successfully")
        
        # Step 2: Extraction (20% -> 50% progress)
        status_text.markdown("### 🔍 Phase 2: Intelligence Extraction")
        st.info("Extracting structured competitive intelligence...")
        progress_bar.progress(0.25)
        
        extractor = ExtractionChain(st.session_state.api_key, config)
        profiles: List[CompetitorProfile] = []
        
        for idx, (url, content) in enumerate(scraped_data.items(), 1):
            update_progress(f"Extracting data from {url} ({idx}/{len(scraped_data)})...")
            
            if content:
                # Respect rate limit
                await rate_limiter.acquire()
                
                try:
                    profile = await extractor.extract(url, content)
                    profiles.append(profile)
                except Exception as e:
                    # Create error profile
                    error_profile = CompetitorProfile(
                        company_name=url,
                        website_url=url,
                        scrape_success=False,
                        error_message=f"Extraction error: {str(e)}"
                    )
                    profiles.append(error_profile)
            else:
                # Create error profile for failed scrape
                error_profile = CompetitorProfile(
                    company_name=url,
                    website_url=url,
                    scrape_success=False,
                    error_message="Website scraping failed"
                )
                profiles.append(error_profile)
            
            # Update progress incrementally
            progress_bar.progress(0.25 + (0.25 * idx / len(scraped_data)))
        
        progress_bar.progress(0.50)
        
        # Show extraction results
        valid_profiles = [p for p in profiles if p.scrape_success]
        st.success(f"✅ Extracted data from {len(valid_profiles)}/{len(profiles)} competitors")
        
        # Store profiles in session state
        st.session_state.analyzed_profiles = profiles
        
        # Step 3: Vector Store (50% -> 60% progress)
        status_text.markdown("### 📊 Phase 3: Semantic Indexing")
        st.info("Building semantic search index...")
        progress_bar.progress(0.55)
        
        vectorstore = CompetitorVectorStore(st.session_state.api_key, config)
        
        for profile in valid_profiles:
            await vectorstore.ingest_profile(profile)
        
        progress_bar.progress(0.60)
        st.success(f"✅ Indexed {len(valid_profiles)} competitor profile(s)")
        
        # Step 4: Synthesis (60% -> 100% progress)
        status_text.markdown("### ✍️ Phase 4: Report Generation")
        st.info("Generating comprehensive competitive analysis...")
        progress_bar.progress(0.65)
        
        # Prepare synthesis agent
        synthesizer = SynthesisAgent(st.session_state.api_key, config)
        
        # Respect rate limit before synthesis
        await rate_limiter.acquire()
        
        progress_bar.progress(0.70)
    
    # Display report in separate container
    with report_container:
        st.markdown("---")
        st.markdown("## 📄 Competitive Analysis Report")
        
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
            st.balloons()


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
    st.header("1️⃣ Enter Competitor URLs")
    st.markdown("Enter 1-3 competitor websites to analyze (one per field)")
    
    # URL inputs
    col1, col2, col3 = st.columns(3)
    
    with col1:
        url1 = st.text_input(
            "Competitor 1 URL",
            key="url_1",
            placeholder="https://competitor1.com"
        )
    
    with col2:
        url2 = st.text_input(
            "Competitor 2 URL",
            key="url_2",
            placeholder="https://competitor2.com"
        )
    
    with col3:
        url3 = st.text_input(
            "Competitor 3 URL",
            key="url_3",
            placeholder="https://competitor3.com"
        )
    
    # Collect and validate URLs
    urls = [url for url in [url1, url2, url3] if url.strip()]
    
    if not urls:
        st.info("👆 Add at least one competitor URL to begin analysis")
        st.stop()
    
    # Validate URLs
    valid_urls, invalid_urls = validate_urls(urls)
    
    if invalid_urls:
        st.error(f"❌ Invalid URLs detected: {', '.join(invalid_urls)}")
        st.info("URLs must start with http:// or https://")
        st.stop()
    
    # Analysis objective
    st.header("2️⃣ Analysis Objective")
    st.markdown("Specify what you'd like to analyze about these competitors")
    
    objective = st.text_area(
        "What would you like to analyze?",
        value="Analyze the competitive landscape focusing on pricing strategies, key features, market positioning, and identify opportunities for differentiation.",
        height=100,
        help="Be specific about what aspects you want to focus on (e.g., pricing, features, target markets, technology)"
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
        st.markdown(f"""
        **Competitors to analyze:** {len(valid_urls)}
        {chr(10).join([f"- {url}" for url in valid_urls])}
        
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
                4. Try with fewer competitors
                """)
    
    # Download report button (if report exists)
    if st.session_state.report_generated and st.session_state.current_report:
        st.markdown("---")
        
        col_dl1, col_dl2, col_dl3 = st.columns([2, 1, 2])
        
        with col_dl2:
            st.download_button(
                label="📥 Download Report",
                data=st.session_state.current_report,
                file_name="competitive_analysis_report.md",
                mime="text/markdown",
                use_container_width=True
            )
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; color: #666; padding: 2rem 0;'>
        <p><strong>Competitive Intelligence Synthesis Agent</strong></p>
        <p>Built with Google Gemini 2.0 Flash | LangChain | Qdrant | crawl4ai</p>
        <p style='font-size: 0.9rem;'>
            <a href='#' style='color: #1f77b4; text-decoration: none;'>Documentation</a> • 
            <a href='#' style='color: #1f77b4; text-decoration: none;'>GitHub</a> • 
            <a href='#' style='color: #1f77b4; text-decoration: none;'>Report Issues</a>
        </p>
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()