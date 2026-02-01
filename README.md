# 📰 AI News Summarizer & Analyzer

AI-powered news intelligence using Google Gemini, LangChain, and semantic search.

![Python](https://img.shields.io/badge/python-3.12-blue.svg)
![LangChain](https://img.shields.io/badge/LangChain-0.3.15-green.svg)
![Streamlit](https://img.shields.io/badge/Streamlit-1.41.1-red.svg)
![License](https://img.shields.io/badge/license-MIT-blue.svg)

---

## 🎯 Features

- **🌐 Article Scraping:** Automated AI news article extraction using crawl4ai
- **🔍 Structured Extraction:** Temperature=0 deterministic data extraction with Pydantic validation
- **📊 Semantic Search:** Vector-based news article similarity search using Qdrant
- **🤖 ReAct Reasoning:** Multi-step agent decision-making with chain-of-thought
- **✍️ Intelligence Reports:** Creative analysis with few-shot prompting (Temperature=0.7)
- **⚡ Real-time Streaming:** Live report generation in the UI

---

## 🏗️ Architecture
```
[Streamlit UI] → [Rate Limiter] → [Scraper (crawl4ai)]
                                        ↓
                                  [Extraction Chain (temp=0)]
                                        ↓
                                  [Vector Store (Qdrant)]
                                        ↓
                                  [ReAct Agent (temp=0.7)]
                                        ↓
                                  [Synthesis Agent + Streaming]
```

### **System Components**

- **Scraping Module:** Async concurrent article scraping with error handling
- **Extraction Chain:** LCEL chain with JSON mode enforcement
- **Vector Store:** In-memory Qdrant for semantic search
- **ReAct Agent:** Reasoning + Acting pattern with tool calling
- **Synthesis Agent:** Few-shot prompting with streaming output

---

## 🚀 Quick Start

### **Prerequisites**

- Python 3.12+
- Google AI Studio API Key ([Get one here](https://aistudio.google.com/app/apikey))

### **Installation**

1. **Clone the repository**
```bash
git clone https://github.com/yourusername/ai-news-summarizer.git
cd ai-news-summarizer
```

2. **Create virtual environment**
```bash
python -m venv venv

# Windows
venv\Scripts\activate

# macOS/Linux
source venv/bin/activate
```

3. **Install dependencies**
```bash
pip install -r requirements.txt
playwright install chromium
```

4. **Set up environment variables (optional)**
```bash
# Copy the example file
cp .env.example .env

# Edit .env and add your Google API key
# Note: The app also accepts API keys via the UI
```

5. **Run the application**
```bash
streamlit run app/main.py
```

6. **Open your browser**

Navigate to `http://localhost:8501`

---

## 📖 Usage

### **Basic Workflow**

1. **Enter API Key:** Paste your Google AI Studio API key in the sidebar
2. **Add URLs:** Enter 1-5 AI news article URLs (e.g., from TechCrunch, VentureBeat, The Verge)
3. **Specify Objective:** Describe what you want to analyze (technology trends, industry impact, use cases)
4. **Generate Analysis:** Click the button and watch real-time report generation
5. **Download Report:** Save the analysis as a Markdown file

### **Example Analysis Objectives**

- "Analyze AI news developments focusing on technology trends and industry impact"
- "Identify the most critical AI developments and provide a prioritized investigation roadmap"
- "Evaluate emerging AI technologies, practical use cases, and affected industries"
- "Assess the relevance and potential impact of recent AI announcements"

### **Recommended News Sources**

- **TechCrunch AI:** https://techcrunch.com/category/artificial-intelligence/
- **VentureBeat AI:** https://venturebeat.com/category/ai/
- **The Verge AI:** https://www.theverge.com/ai-artificial-intelligence
- **Ars Technica AI:** https://arstechnica.com/tag/artificial-intelligence/
- **Wired AI:** https://www.wired.com/tag/artificial-intelligence/

---

## 🧪 Technical Deep Dive

### **LLM Configuration**

| Component | Model | Temperature | Purpose |
|-----------|-------|-------------|---------|
| Extraction | gemini-2.0-flash | 0.0 | Deterministic data extraction |
| Synthesis | gemini-2.0-flash | 0.7 | Creative reasoning & insights |
| Embeddings | text-embedding-004 | N/A | Semantic search (768 dims) |

### **Rate Limiting**

Enforces 15 requests/minute (Google AI Studio free tier):
```python
await rate_limiter.acquire()  # Waits if limit reached
```

### **Pydantic Schemas**
```python
class NewsArticleProfile(BaseModel):
    headline: str
    article_url: str
    news_source: str
    publication_date: Optional[str]
    article_summary: str
    key_technologies: List[str]
    use_cases: List[str]
    affected_industries: List[str]
    potential_impact: Optional[ImpactLevel]
    relevance_score: Optional[float]
    recommended_priority: Optional[int]
    key_insights: List[str]
    # ... full schema validation
```

### **LCEL Chain Pattern**
```python
chain = (
    prompt.partial(format_instructions=parser.get_format_instructions())
    | llm
    | parser
)
```

---

## 📂 Project Structure
```
ai-news-summarizer/
├── app/
│   ├── main.py                      # Streamlit UI entry point
│   ├── config.py                    # AppConfig (Pydantic model)
│   │
│   ├── scraping/
│   │   └── crawler.py               # NewsArticleCrawler (crawl4ai)
│   │
│   ├── extraction/
│   │   ├── schemas.py               # NewsArticleProfile, ImpactLevel
│   │   └── chain.py                 # ExtractionChain (LCEL, temp=0)
│   │
│   ├── vectorstore/
│   │   └── store.py                 # NewsVectorStore (Qdrant)
│   │
│   ├── agents/
│   │   ├── tools.py                 # NewsAnalysisTools (ReAct tools)
│   │   └── react_agent.py           # ReActAgent (reasoning)
│   │
│   ├── synthesis/
│   │   └── report_generator.py      # SynthesisAgent (streaming)
│   │
│   └── utils/
│       ├── rate_limiter.py          # RateLimiter (15 RPM)
│       └── logger.py                # ProgressLogger
│
├── .streamlit/
│   └── config.toml                  # Streamlit configuration
│
├── .env.example                     # Environment variables template
├── .gitignore                       # Git ignore rules
├── requirements.txt                 # Python dependencies
├── packages.txt                     # System packages (Streamlit Cloud)
└── README.md                        # This file
```

---

## 🚢 Deployment

### **Option 1: Streamlit Cloud (Recommended)**

1. Push your code to GitHub
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Connect your repository
4. Select `app/main.py` as the main file
5. Click "Deploy"

**Result:** Live URL at `https://your-app.streamlit.app`

### **Option 2: Local Docker**
```bash
# Build image
docker build -t ai-news-summarizer .

# Run container
docker run -p 8501:8501 ai-news-summarizer
```

### **Option 3: Production Server**
```bash
# Install system dependencies
sudo apt-get update
sudo apt-get install -y chromium chromium-driver

# Install Python dependencies
pip install -r requirements.txt
playwright install chromium

# Run with production settings
streamlit run app/main.py --server.port 8501 --server.headless true
```

---

## 🔐 Security & Privacy

- **API Keys:** Session-based only, never persisted to disk
- **Data Storage:** In-memory vector store, no database
- **Logs:** No logging of sensitive data
- **Dependencies:** All open-source, vetted packages

---

## 🛠️ Tech Stack

| Layer | Technology | Version | Purpose |
|-------|------------|---------|---------|
| **LLM** | Google Gemini 2.0 Flash | Latest | Fast, 1M context window |
| **Framework** | LangChain | 0.3.15 | Production orchestration |
| **UI** | Streamlit | 1.41.1 | Rapid web app development |
| **Scraping** | crawl4ai | 0.8.0 | Free, async web scraping |
| **Vector DB** | Qdrant | 1.12.1 | In-memory semantic search |
| **Validation** | Pydantic | 2.10.5 | Type safety & validation |
| **Async** | asyncio | Built-in | Concurrent operations |

---

## 📊 Example Output

### **Generated Report Sections**

- **Executive Summary:** Key findings and critical developments
- **Key AI Developments:** Detailed analysis of significant news items
- **Technology Trend Analysis:** Emerging technologies and convergence patterns
- **Industry Impact Assessment:** Sector-specific insights and transformation areas
- **Use Case Opportunities:** Practical applications and business opportunities
- **Recommended Investigation Priority:** Actionable reading order with rationale
- **Strategic Insights:** Forward-looking analysis and recommendations

---

## 🔍 Advanced Features

### **Custom Analysis Tools**

The ReAct agent includes specialized tools:

- `search_articles()` - Semantic search for relevant articles
- `analyze_relevance()` - Statistical analysis of relevance scores
- `identify_technology_trends()` - Technology mention frequency analysis
- `analyze_industry_impact()` - Industry-specific impact assessment
- `prioritize_articles()` - Priority-based ranking system
- `identify_use_cases()` - Use case categorization
- `get_comprehensive_summary()` - Overall news intelligence summary

### **Custom Synthesis Prompts**

Modify the few-shot examples in `app/synthesis/report_generator.py` to customize output style.

### **Extended Rate Limits**

Upgrade to paid tier and update `config.py`:
```python
MAX_RPM: int = 1000  # Paid tier limit
```

---

## 🐛 Troubleshooting

### **Common Issues**

**1. Playwright Installation Failed**
```bash
# Manually install Playwright
playwright install chromium --with-deps
```

**2. Dependency Conflicts**
```bash
# Clean reinstall
pip uninstall -y langchain langchain-core langchain-google-genai
pip install -r requirements.txt
```

**3. Rate Limit Errors**

- Wait 60 seconds between analyses
- Reduce number of articles
- Upgrade to paid API tier

**4. Scraping Failures**

- Verify URLs are accessible
- Check for anti-bot protection
- Try fewer URLs
- Some news sites may block automated access

---

## 📝 Development

### **Running Tests**
```bash
# Install test dependencies
pip install pytest pytest-asyncio

# Run tests
pytest tests/
```

### **Code Quality**
```bash
# Format code
black app/

# Type checking
mypy app/

# Linting
ruff check app/
```

---

## 🤝 Contributing

Contributions are welcome! Please:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 🙏 Acknowledgments

- Built with [LangChain](https://github.com/langchain-ai/langchain)
- Powered by [Google Gemini](https://ai.google.dev/)
- Web scraping by [crawl4ai](https://github.com/unclecode/crawl4ai)
- Vector search by [Qdrant](https://qdrant.tech/)

---

## 📧 Contact

**Project Maintainer:** Your Name

- GitHub: [@yourusername](https://github.com/yourusername)
- LinkedIn: [Your Profile](https://linkedin.com/in/yourprofile)
- Email: your.email@example.com

---

## 🗺️ Roadmap

- [ ] Multi-language support for international news
- [ ] PDF report export with charts/visualizations
- [ ] Batch processing for >5 articles
- [ ] Historical trend tracking and comparison
- [ ] Integration with RSS feeds for automated monitoring
- [ ] Custom report templates
- [ ] API endpoint for programmatic access
- [ ] Sentiment analysis integration
- [ ] Alert system for critical developments

---

**⭐ If you find this project useful, please give it a star!**