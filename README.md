# Web Crawler

A high-performance Python web crawler with async support, authentication, and dual interfaces. Uses Breadth-First Search (BFS) to crawl websites, respects robots.txt, and can search pages for specific content.

## Features

- **🚀 Async Crawling**: Concurrent fetching for 5-10x faster performance
- **🔐 Authentication Support**: Login to websites and crawl authenticated pages
- **📝 Form Detection**: Automatically detect and categorize forms
- **🍪 Session Persistence**: Maintain cookies across crawls
- **⏱️ Politeness Controls**: Per-domain rate limiting to avoid getting blocked
- **🔄 Retry Logic**: Automatic retries with exponential backoff
- **🖥️ Dual Interface**: Easy-to-use GUI or command-line interface
- **🔍 Content Search**: Search crawled pages for specific keywords
- **🤖 Robots.txt Compliance**: Respects website crawling rules
- **📁 Smart URL Filtering**: Automatically skips non-HTML files (PDFs, images, etc.)
- **💾 Export Options**: Save results as JSON or CSV
- **⚙️ Configurable**: Adjust concurrency, delays, depth, domain restrictions, and more

## Requirements

- Python 3.8 or higher
- tkinter (usually included with Python)

## Installation

### 1. Set Up Virtual Environment (Recommended)

Using a virtual environment is **strongly recommended** to keep dependencies isolated:

```bash
# Navigate to the project directory
cd /path/to/web_crawler

# Create virtual environment
python -m venv venv

# Activate virtual environment
# On Linux/Mac:
source venv/bin/activate
# On Windows:
venv\Scripts\activate
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

This will install:

- **requests**: HTTP requests and web page fetching
- **beautifulsoup4**: HTML parsing and link extraction
- **aiohttp**: Async HTTP client for high-performance crawling
- **mechanicalsoup**: Form handling and authentication
- **beautifulsoup4**: HTML parsing and link extraction
- **aiohttp**: Async HTTP client for high-performance crawling

## Usage

### GUI Mode (Recommended for Beginners)

Start the graphical interface:

```bash
python crawler.py --gui
```

The GUI provides:

- Input fields for start URL and search query
- **Performance Options**: Toggle async mode, set concurrency and delay
- **Authentication Panel**: Enable login for authenticated crawling
- Options for max URLs, depth, and domain restrictions
- Real-time crawl progress display
- Export buttons for saving results

### Command-Line Mode

Basic usage (async mode by default):

```bash
python crawler.py
```

You'll be prompted to enter:

1. Starting URL (e.g., `https://example.com`)
2. Search query (e.g., "contact" or "privacy")

### Authentication Examples

Crawl authenticated pages by providing login credentials:

```bash
# Basic authentication
python crawler.py --url https://example.com/dashboard \
  --query "reports" \
  --login \
  --login-url https://example.com/login \
  --username your_email@example.com \
  --password your_password

# With custom field names
python crawler.py --url https://site.com/members \
  --login \
  --login-url https://site.com/signin \
  --username user@example.com \
  --password pass123 \
  --username-field "email" \
  --password-field "pass"

# Sessions are saved - reuse on next run
python crawler.py --url https://example.com/dashboard --query "data"
```

**Note:** Sessions are automatically saved to `crawler_session.pkl` and reused in future runs.

### Advanced Options

```bash
# Fast concurrent crawling (10 workers, 0.5s delay between requests)
python crawler.py --url https://example.com --query "privacy" --concurrency 10 --delay 0.5

# Crawl 100 URLs with depth limit
python crawler.py --url https://example.com --query "contact" --max-urls 100 --max-depth 3

# Same-domain only mode
python crawler.py --url https://example.com --query "about" --same-domain-only

# Use synchronous mode (original behavior, slower but simpler)
python crawler.py --url https://example.com --query "privacy" --sync

# Export results automatically
python crawler.py --url https://example.com --query "privacy" --export-json --export-csv
```

### All Command-Line Options

| Option | Short | Description | Default |
|--------|-------|-------------|---------|
| `--url` | `-u` | Starting URL to crawl | (interactive) |
| `--query` | `-q` | Search query to look for | (interactive) |
| `--max-urls` | `-m` | Maximum URLs to visit | 25 |
| `--max-depth` | `-d` | Maximum crawl depth | unlimited |
| `--same-domain-only` | `-s` | Only crawl same domain | false |
| `--concurrency` | `-c` | Concurrent requests | 5 |
| `--delay` | | Politeness delay (seconds) | 1.0 |
| `--sync` | | Use sync mode instead of async | false |
| `--export-json` | | Export to JSON file | - |
| `--export-csv` | | Export to CSV file | - |
| `--log-level` | | DEBUG, INFO, WARNING, ERROR | INFO |
| `--gui` | | Run in GUI mode | false |

## Technical Details

### Async Architecture

The crawler uses `aiohttp` for high-performance async HTTP requests:

- **Concurrent Fetching**: Multiple URLs fetched simultaneously (configurable workers)
- **Domain Rate Limiting**: Per-domain politeness delays prevent server overload
- **Connection Pooling**: Reuses connections for efficiency
- **Retry with Backoff**: Failed requests retry with exponential backoff

### URL Frontier System (BFS Queue)

- Uses Python's deque (double-ended queue) for efficient BFS traversal
- URLFrontier class features:
  - Duplicate prevention using a "seen" set
  - Same-domain URL prioritization (front of queue)
  - External URLs added to back (lower priority)
  - URL shuffling for varied crawl patterns

### Smart URL Filtering

Automatically skips non-HTML files to avoid wasting time:
- Documents: .pdf, .doc, .docx, .xls, .xlsx, .ppt, .pptx
- Images: .jpg, .jpeg, .png, .gif, .svg, .webp
- Media: .mp3, .mp4, .avi, .mov
- Archives: .zip, .rar, .7z, .tar, .gz

### Robots.txt Compliance

- Checks robots.txt before visiting any URL
- Uses urllib.robotparser.RobotFileParser
- Per-domain caching to avoid repeated fetches
- Assumes allowed if robots.txt is unavailable

### Libraries Used

- **aiohttp**: Async HTTP client for concurrent requests
- **requests**: Sync HTTP requests (fallback and robots.txt)
- **beautifulsoup4**: HTML parsing and link extraction
- **urllib.parse** (built-in): URL normalization and joining
- **urllib.robotparser** (built-in): Robots.txt compliance checking
- **tkinter** (built-in): GUI interface

## Troubleshooting

### ImportError: No module named 'tkinter'

On Linux, tkinter may need to be installed separately:

```bash
# Ubuntu/Debian
sudo apt-get install python3-tk

# Fedora
sudo dnf install python3-tkinter

# Arch Linux
sudo pacman -S tk
```

### Virtual Environment Issues**

If you have trouble activating the virtual environment, ensure you're using the correct command for your OS (see Installation section).

**Connection Errors**

If you encounter frequent connection errors:
- Check your internet connection
- Some websites may block crawlers
- Try increasing the timeout in the code if needed

## Output Files

The crawler can generate:

- **crawler.log**: Detailed log of all crawler activity
- **results.json**: Crawl results in JSON format
- **results.csv**: Crawl results in CSV format

## Deactivating Virtual Environment

When you're done using the crawler:

```bash
deactivate
```

## License

This project is for personal use. Please respect robots.txt and website terms of service when crawling.