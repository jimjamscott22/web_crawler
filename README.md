# Web Crawler

![Python Version](https://img.shields.io/badge/python-3.8%2B-blue.svg)
![License](https://img.shields.io/badge/license-MIT-green.svg)

A powerful BFS-based web crawler that respects robots.txt, searches pages for user-provided queries, and exports results in multiple formats. Features both CLI and GUI interfaces with advanced crawling options.

## 🚀 Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Run in interactive mode
python crawler.py

# Run in GUI mode
python crawler.py --gui

# Run with CLI arguments
python crawler.py --url https://example.com --query "privacy" --max-urls 25
```

## ✨ Features

- **🔍 BFS Traversal**: Breadth-first search algorithm for systematic web crawling
- **🤖 Robots.txt Compliance**: Respects website crawling policies automatically
- **🔎 Content Search**: Find pages containing specific search terms
- **📊 Export Results**: Export to JSON and CSV formats with full metadata
- **📝 Comprehensive Logging**: Python logging module with file and console output
- **🎯 Depth-Limited Crawling**: Control how deep the crawler explores from the seed URL
- **🌐 Domain-Restricted Mode**: Optionally limit crawling to the same domain only
- **🖥️ Dual Interface**: Both command-line and graphical user interfaces
- **⚡ Smart Prioritization**: Same-domain URLs are prioritized over external links
- **📈 Real-time Progress**: Live updates during crawling operations
- **📉 Performance Metrics**: Track response times and crawl statistics

## 📋 Requirements

- **Python**: 3.8 or higher
- **Libraries**:
  - `requests` - HTTP library for fetching web pages
  - `beautifulsoup4` - HTML parsing and link extraction

Install all dependencies:

```bash
pip install -r requirements.txt
```

## 🎮 Usage

### Interactive CLI Mode

Simply run the script and follow the prompts:

```bash
python crawler.py
```

You'll be asked to enter:
1. Starting URL (e.g., `https://example.com`)
2. Search query (e.g., `privacy`)

### Command-Line Arguments

For automation and scripting, use command-line arguments:

```bash
# Basic usage
python crawler.py --url https://example.com --query "privacy"

# With depth limiting
python crawler.py --url https://example.com --query "contact" --max-depth 2

# Same-domain only crawling
python crawler.py --url https://example.com --query "about" --same-domain-only

# Export results to JSON
python crawler.py --url https://example.com --query "search" --export-json results.json

# Export results to CSV
python crawler.py --url https://example.com --query "search" --export-csv results.csv

# Combine multiple options
python crawler.py \
  --url https://example.com \
  --query "privacy" \
  --max-urls 50 \
  --max-depth 3 \
  --same-domain-only \
  --export-json \
  --log-level DEBUG
```

### GUI Mode

Launch the graphical interface:

```bash
python crawler.py --gui
```

The GUI provides:
- Input fields for URL, search query, and max URLs
- Max depth control (optional)
- "Same Domain Only" checkbox
- Export format selection (JSON, CSV, Both, or None)
- Log level dropdown
- Real-time output display
- Stop button to cancel crawling

### Command-Line Options

| Option | Short | Description | Default |
|--------|-------|-------------|---------|
| `--url` | `-u` | Starting URL to crawl | Interactive prompt |
| `--query` | `-q` | Search query to look for | Interactive prompt |
| `--max-urls` | `-m` | Maximum URLs to visit | 25 |
| `--max-depth` | `-d` | Maximum crawl depth | Unlimited |
| `--same-domain-only` | `-s` | Only crawl same domain | False |
| `--export-json` | | Export to JSON file | None |
| `--export-csv` | | Export to CSV file | None |
| `--log-level` | | Logging level | INFO |
| `--gui` | | Run in GUI mode | False |

## 📊 Export Formats

### JSON Export

JSON files include complete crawl metadata and detailed URL information:

```json
{
  "metadata": {
    "start_url": "https://example.com",
    "search_query": "privacy",
    "max_urls": 25,
    "max_depth": 2,
    "same_domain_only": false,
    "urls_visited": 25,
    "urls_matched": 5,
    "total_time": 45.3,
    "timestamp": "2026-01-01T14:30:45.123456"
  },
  "visited_urls": [
    {
      "url": "https://example.com",
      "depth": 0,
      "timestamp": "2026-01-01T14:30:15.123456",
      "response_time": 0.234,
      "status": "visited",
      "error": null,
      "matched": true
    }
  ],
  "matching_urls": [
    "https://example.com/privacy"
  ]
}
```

### CSV Export

CSV files are optimized for spreadsheet applications like Excel:

| URL | Depth | Status | Matched | Response Time (s) | Error | Timestamp |
|-----|-------|--------|---------|-------------------|-------|-----------|
| https://example.com | 0 | visited | Yes | 0.234 | | 2026-01-01T14:30:15 |
| https://example.com/about | 1 | visited | No | 0.156 | | 2026-01-01T14:30:16 |

Auto-generated filenames include timestamps:
- `crawl_results_2026-01-01_14-30-45.json`
- `crawl_results_2026-01-01_14-30-45.csv`

## 📝 Logging

The crawler uses Python's `logging` module with multiple levels:

- **DEBUG**: Detailed information for diagnosing problems
- **INFO**: General informational messages (default)
- **WARNING**: Warning messages for non-critical issues
- **ERROR**: Error messages for serious problems

Logs are saved to `crawler.log` and displayed in the console (or GUI).

Example log format:
```
2026-01-01 14:30:45 - WebCrawler - INFO - STARTING WEB CRAWLER
2026-01-01 14:30:45 - WebCrawler - INFO - Start URL: https://example.com
2026-01-01 14:30:46 - WebCrawler - INFO - [1/25] Depth 0: https://example.com
2026-01-01 14:30:46 - WebCrawler - INFO -   [Match!] Query 'privacy' found on this page
```

## 🎯 Depth-Limited Crawling

Control how deep the crawler explores from the seed URL:

```bash
# Crawl only the seed page (depth 0)
python crawler.py --url https://example.com --query "test" --max-depth 0

# Crawl seed page and its direct links (depth 0-1)
python crawler.py --url https://example.com --query "test" --max-depth 1

# Crawl up to 3 levels deep
python crawler.py --url https://example.com --query "test" --max-depth 3
```

**Depth levels:**
- **Depth 0**: Seed URL only
- **Depth 1**: Direct links from seed URL
- **Depth 2**: Links found on depth 1 pages
- And so on...

URLs beyond the maximum depth are automatically filtered out.

## 🌐 Domain-Restricted Mode

Limit crawling to the same domain as the seed URL:

```bash
# Only crawl pages on example.com
python crawler.py --url https://example.com --query "test" --same-domain-only

# Combine with depth limiting
python crawler.py --url https://example.com --query "test" --same-domain-only --max-depth 2
```

**When enabled:**
- Only URLs from the same domain are added to the frontier
- External links are completely ignored
- Subdomain handling: `blog.example.com` ≠ `example.com`

**When disabled (default):**
- Same-domain URLs are prioritized (added to front of queue)
- External URLs are still crawled (added to back of queue)

## 🔧 How It Works

### Architecture

1. **URL Frontier (BFS Queue)**
   - Manages URLs to visit using a double-ended queue (deque)
   - Prevents duplicate URLs with a "seen" set
   - Prioritizes same-domain URLs over external links
   - Tracks depth of each URL from the seed

2. **Robots.txt Compliance**
   - Fetches and parses robots.txt for each domain
   - Caches parsers to avoid repeated fetches
   - Respects crawl permissions for the User-Agent

3. **Page Processing Pipeline**
   - **Fetch**: Downloads HTML with timeout and error handling
   - **Parse**: Extracts links using BeautifulSoup
   - **Normalize**: Converts relative URLs to absolute URLs
   - **Search**: Extracts text and searches for query (case-insensitive)

4. **Error Resilience**
   - All network operations wrapped in try/except
   - Errors logged but don't stop crawling
   - Handles timeouts, connection errors, HTTP errors

### URL Prioritization

The crawler uses a smart prioritization system:

1. Same-domain URLs added to front of queue (high priority)
2. External URLs added to back of queue (low priority)
3. URLs shuffled within priority groups for variety
4. Depth constraints applied before adding to queue

## 📸 Sample Output

### CLI Output

```
============================================================
STARTING WEB CRAWLER
============================================================
Start URL: https://example.com
Search Query: 'privacy'
Target: 25 unique URLs
Max Depth: 2
Same Domain Only: False
============================================================

[1/25] Depth 0: https://example.com
  [Match!] Query 'privacy' found on this page
  [Links] Found 15 links on this page
    → https://example.com/about
    → https://example.com/contact
    → https://example.com/privacy
    ... and 12 more

[2/25] Depth 1: https://example.com/about
  [Links] Found 8 links on this page
    → https://example.com/team
    → https://example.com/careers
    ... and 6 more

...

============================================================
CRAWL RESULTS
============================================================

📋 ALL VISITED URLs (25 of target 25):
----------------------------------------
   1. [Depth 0] https://example.com (visited)
   2. [Depth 1] https://example.com/about (visited)
   3. [Depth 1] https://example.com/contact (visited)
   ...

🔍 SEARCH RESULTS for 'privacy':
----------------------------------------
  Pages containing the query: 5

  URLs with matches:
    1. https://example.com
    2. https://example.com/privacy
    3. https://example.com/legal
    ...

============================================================
CRAWL COMPLETE
============================================================

✅ Results exported to JSON: crawl_results_2026-01-01_14-30-45.json
✅ Results exported to CSV: crawl_results_2026-01-01_14-30-45.csv
```

### GUI Interface

The GUI provides an intuitive interface with:
- **Input Section**: Fields for URL, query, max URLs, and max depth
- **Options**: Checkboxes and dropdowns for crawl settings
- **Control Buttons**: Run and Stop buttons
- **Output Display**: Scrollable text area with real-time progress
- **Export Options**: Radio buttons for JSON, CSV, Both, or None

## 🛡️ Error Handling

The crawler handles various error conditions gracefully:

- **Network Errors**: Timeout, connection refused, DNS failures
- **HTTP Errors**: 404, 403, 500, etc.
- **Parsing Errors**: Malformed HTML, invalid URLs
- **Robots.txt Issues**: Missing or inaccessible robots.txt files

All errors are logged with appropriate detail levels and don't crash the crawler.

## 🏗️ Code Structure

```
crawler.py
├── Configuration Constants
├── Logging Setup
├── Robots.txt Compliance
│   ├── get_robots_parser()
│   └── can_crawl()
├── URL Frontier (BFS Queue)
│   └── URLFrontier class
├── Page Fetching and Parsing
│   ├── fetch_page()
│   ├── extract_links()
│   ├── extract_text()
│   └── search_page()
├── Main Crawler Logic
│   ├── crawl()
│   └── print_results()
├── Export Functionality
│   ├── export_to_json()
│   └── export_to_csv()
├── User Input and CLI
│   ├── get_user_input()
│   ├── parse_arguments()
│   └── main()
└── GUI Wrapper
    ├── TextRedirector class
    └── run_gui()
```

## 🧪 Testing Examples

Test depth limiting:
```bash
python crawler.py --url https://example.com --query "test" --max-depth 0
# Should only visit the seed URL

python crawler.py --url https://example.com --query "test" --max-depth 1
# Should visit seed + direct links only
```

Test domain restriction:
```bash
python crawler.py --url https://example.com --query "test" --same-domain-only
# Should only visit example.com pages
```

Test export formats:
```bash
python crawler.py --url https://example.com --query "test" --export-json --export-csv
# Should create both JSON and CSV files with timestamps
```

Test logging levels:
```bash
python crawler.py --url https://example.com --query "test" --log-level DEBUG
# Should show detailed debug information
```

## 📄 Files

- **crawler.py**: Main application file
- **requirements.txt**: Python dependencies
- **README.md**: This documentation file
- **crawler.log**: Log file (created automatically)
- **crawl_results_*.json**: Exported JSON results (if requested)
- **crawl_results_*.csv**: Exported CSV results (if requested)

## 🔒 Best Practices

- The crawler identifies itself as `StudentWebCrawler/1.0`
- Request timeout set to 10 seconds
- Respects robots.txt policies
- Only processes HTML content (skips PDFs, images, etc.)
- Uses case-insensitive search
- Removes URL fragments to avoid duplicate pages
- Press `Ctrl+C` to stop crawling at any time

## 🐛 Troubleshooting

**Issue**: "Module not found" error  
**Solution**: Install dependencies with `pip install -r requirements.txt`

**Issue**: GUI won't start  
**Solution**: Ensure tkinter is installed (usually included with Python)

**Issue**: No pages match search query  
**Solution**: Try a more common search term or check spelling

**Issue**: Crawler seems slow  
**Solution**: Some websites have rate limiting or slow response times. This is normal.

**Issue**: Getting blocked by websites  
**Solution**: This is expected behavior. The crawler respects robots.txt. Try different sites.

## 📚 Technical Details

### Libraries Used

1. **requests**: HTTP library for fetching web pages
   - Handles timeouts, errors, and HTTP headers
   - Provides simple API for GET requests

2. **beautifulsoup4 (bs4)**: HTML parser
   - Extracts links from anchor tags
   - Removes script/style elements for clean text
   - Handles malformed HTML gracefully

3. **urllib.parse** (built-in): URL manipulation
   - `urljoin()`: Converts relative to absolute URLs
   - `urlparse()`: Parses URL components

4. **urllib.robotparser** (built-in): Robots.txt compliance
   - Fetches and parses robots.txt
   - Checks crawl permissions per URL

5. **logging** (built-in): Logging framework
   - Multiple log levels and handlers
   - File and console output

6. **tkinter** (built-in): GUI framework
   - Cross-platform graphical interface
   - Thread-safe text redirection

## 📝 License

This project is available under the MIT License.

## 🙏 Acknowledgments

Built as an educational web crawler demonstrating:
- BFS algorithm implementation
- HTTP/HTML processing
- Robots.txt compliance
- Data export formats
- Python logging best practices
- GUI development with tkinter

---

**Note**: This crawler is designed for educational purposes. Always respect website terms of service and crawling policies when using web crawlers.
