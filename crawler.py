"""
Web Crawler
===========
A high-performance async web crawler with BFS traversal, concurrent fetching,
per-domain rate limiting, and search functionality.

Features:
- Async HTTP requests with aiohttp for 5-10x faster crawling
- Per-domain politeness delays to avoid getting blocked
- Configurable concurrency (number of parallel workers)
- Retry logic with exponential backoff
- Respects robots.txt
- GUI and CLI interfaces

Libraries used:
- aiohttp: For async HTTP requests (primary)
- requests: For sync operations and robots.txt (fallback)
- BeautifulSoup (bs4): For parsing HTML and extracting links
- urllib.parse: For URL normalization and joining
- urllib.robotparser: For robots.txt compliance
"""

import argparse
import asyncio
import csv
import json
import logging
import mechanicalsoup
import random
import sys
import threading
import time
from contextlib import redirect_stdout
from collections import deque, defaultdict
from datetime import datetime
from urllib.parse import urlparse, urljoin
from urllib.robotparser import RobotFileParser

import requests
from bs4 import BeautifulSoup

# Try to import aiohttp, fall back to sync-only mode if not available
try:
    import aiohttp
    ASYNC_AVAILABLE = True
except ImportError:
    ASYNC_AVAILABLE = False
    print("Warning: aiohttp not installed. Using synchronous mode only.")
    print("Install with: pip install aiohttp")


# ============================================================================
# CONFIGURATION
# ============================================================================

MAX_URLS_TO_VISIT = 25  # Total unique URLs to crawl
REQUEST_TIMEOUT = 10     # Seconds to wait for a response
USER_AGENT = "PersonalWebCrawler/2.0"  # Identifies our crawler to websites

# Async crawling configuration
DEFAULT_CONCURRENCY = 5  # Number of concurrent requests
DEFAULT_POLITENESS_DELAY = 1.0  # Seconds between requests to same domain
MAX_RETRIES = 3  # Maximum retry attempts for failed requests
RETRY_BACKOFF = 2.0  # Exponential backoff multiplier

# URL filtering - skip these file extensions
SKIP_EXTENSIONS = {
    '.pdf', '.jpg', '.jpeg', '.png', '.gif', '.bmp', '.svg', '.webp',
    '.mp3', '.mp4', '.avi', '.mov', '.wmv', '.flv', '.webm',
    '.zip', '.rar', '.7z', '.tar', '.gz',
    '.doc', '.docx', '.xls', '.xlsx', '.ppt', '.pptx',
    '.exe', '.dmg', '.iso', '.bin'
}

# New configuration options
DEFAULT_MAX_DEPTH = None  # None = unlimited
DEFAULT_SAME_DOMAIN_ONLY = False
DEFAULT_EXPORT_FORMAT = "json"  # json, csv, or both
LOG_FILE = "crawler.log"
LOG_LEVEL = logging.INFO


# ============================================================================
# LOGGING SETUP
# ============================================================================

def setup_logging(log_level=LOG_LEVEL, log_file=LOG_FILE):
    """
    Configure logging for the crawler.
    
    Args:
        log_level: The logging level (DEBUG, INFO, WARNING, ERROR)
        log_file: Path to the log file
    """
    # Create logger
    logger = logging.getLogger("WebCrawler")
    logger.setLevel(log_level)
    
    # Remove any existing handlers
    logger.handlers = []
    
    # Create formatters
    file_formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    console_formatter = logging.Formatter('%(message)s')
    
    # File handler
    file_handler = logging.FileHandler(log_file, encoding='utf-8')
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(file_formatter)
    logger.addHandler(file_handler)
    
    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(log_level)
    console_handler.setFormatter(console_formatter)
    logger.addHandler(console_handler)
    
    return logger

# Initialize default logger
logger = setup_logging()


# ============================================================================
# ROBOTS.TXT COMPLIANCE
# ============================================================================

def get_robots_parser(base_url):
    """
    Create a RobotFileParser for a given website's robots.txt.
    
    Args:
        base_url: The base URL of the website (e.g., "https://example.com")
    
    Returns:
        RobotFileParser object configured for the site, or None if unavailable
    """
    try:
        # Construct the robots.txt URL
        parsed = urlparse(base_url)
        robots_url = f"{parsed.scheme}://{parsed.netloc}/robots.txt"
        
        # Create and configure the parser
        parser = RobotFileParser()
        parser.set_url(robots_url)
        parser.read()
        return parser
    except Exception as e:
        logger.warning(f"Could not fetch robots.txt for {base_url}: {e}")
        return None


def can_crawl(url, robots_cache):
    """
    Check if we're allowed to crawl a URL based on robots.txt rules.
    
    Args:
        url: The URL to check
        robots_cache: Dictionary caching RobotFileParser objects per domain
    
    Returns:
        True if crawling is allowed, False otherwise
    """
    try:
        parsed = urlparse(url)
        base_url = f"{parsed.scheme}://{parsed.netloc}"
        
        # Check cache first, fetch robots.txt if not cached
        if base_url not in robots_cache:
            robots_cache[base_url] = get_robots_parser(base_url)
        
        parser = robots_cache[base_url]
        
        # If we couldn't get robots.txt, assume allowed (common practice)
        if parser is None:
            return True
        
        # Check if our user agent is allowed to access this URL
        return parser.can_fetch(USER_AGENT, url)
    except Exception:
        # If anything goes wrong, allow crawling (fail open)
        return True


# ============================================================================
# URL FRONTIER (BFS QUEUE) WITH PRIORITIZATION
# ============================================================================

class URLFrontier:
    """
    Manages the queue of URLs to visit (the "frontier").
    
    Features:
    - Prevents duplicate URLs
    - Prioritizes same-domain URLs over external URLs
    - Uses BFS (Breadth-First Search) traversal order
    - Tracks depth of each URL from the seed
    - Supports domain-restricted crawling mode
    """
    
    def __init__(self, start_url, max_depth=None, same_domain_only=False):
        """
        Initialize the frontier with a starting URL.
        
        Args:
            start_url: The seed URL to begin crawling from
            max_depth: Maximum depth to crawl (None for unlimited)
            same_domain_only: If True, only crawl URLs from the same domain
        """
        self.start_domain = urlparse(start_url).netloc
        self.queue = deque([(start_url, 0)])  # BFS queue with (url, depth) tuples
        self.seen = {start_url}  # Set of all URLs we've seen (to prevent duplicates)
        self.max_depth = max_depth
        self.same_domain_only = same_domain_only
        self.url_depths = {start_url: 0}  # Track depth of each URL
    
    def add_urls(self, urls, parent_depth):
        """
        Add new URLs to the frontier with prioritization.
        
        Same-domain URLs are added to the front (higher priority).
        External URLs are added to the back (lower priority).
        URLs are shuffled within their priority group for variety.
        
        Args:
            urls: List of URLs to add
            parent_depth: Depth of the parent URL
        """
        # Calculate depth for new URLs
        new_depth = parent_depth + 1
        
        # Check depth limit
        if self.max_depth is not None and new_depth > self.max_depth:
            logger.debug(f"Skipping {len(urls)} URLs - would exceed max depth {self.max_depth}")
            return
        
        # Filter out duplicates
        new_urls = [url for url in urls if url not in self.seen]
        
        if not new_urls:
            return
        
        # Separate into same-domain and external URLs
        same_domain = []
        external = []
        
        for url in new_urls:
            if urlparse(url).netloc == self.start_domain:
                same_domain.append(url)
            else:
                # Skip external URLs if same_domain_only mode is active
                if not self.same_domain_only:
                    external.append(url)
        
        # Shuffle for variety (prevents always visiting links in page order)
        random.shuffle(same_domain)
        random.shuffle(external)
        
        # Mark all as seen and track depths
        for url in same_domain + external:
            self.seen.add(url)
            self.url_depths[url] = new_depth
        
        # Add same-domain URLs to front (higher priority)
        # We use extendleft with reversed to maintain shuffle order
        for url in reversed(same_domain):
            self.queue.appendleft((url, new_depth))
        
        # Add external URLs to back (lower priority)
        for url in external:
            self.queue.append((url, new_depth))
    
    def get_next(self):
        """
        Get the next URL to visit.
        
        Returns:
            Tuple of (url, depth), or (None, None) if the queue is empty
        """
        if self.queue:
            return self.queue.popleft()
        return None, None
    
    def is_empty(self):
        """Check if there are no more URLs to visit."""
        return len(self.queue) == 0
    
    def get_depth(self, url):
        """Get the depth of a URL."""
        return self.url_depths.get(url, 0)


# ============================================================================
# DOMAIN RATE LIMITER (POLITENESS)
# ============================================================================

class DomainRateLimiter:
    """
    Enforces politeness delays between requests to the same domain.
    This prevents overwhelming servers and getting blocked.
    """
    
    def __init__(self, delay=DEFAULT_POLITENESS_DELAY):
        """
        Initialize the rate limiter.
        
        Args:
            delay: Minimum seconds between requests to the same domain
        """
        self.delay = delay
        self.last_request_time = defaultdict(float)
        self._lock = asyncio.Lock() if ASYNC_AVAILABLE else threading.Lock()
    
    def get_domain(self, url):
        """Extract domain from URL."""
        return urlparse(url).netloc
    
    async def wait_async(self, url):
        """Async wait for politeness delay if needed."""
        if not ASYNC_AVAILABLE:
            return
            
        domain = self.get_domain(url)
        async with self._lock:
            now = time.time()
            elapsed = now - self.last_request_time[domain]
            if elapsed < self.delay:
                wait_time = self.delay - elapsed
                await asyncio.sleep(wait_time)
            self.last_request_time[domain] = time.time()
    
    def wait_sync(self, url):
        """Sync wait for politeness delay if needed."""
        domain = self.get_domain(url)
        with self._lock:
            now = time.time()
            elapsed = now - self.last_request_time[domain]
            if elapsed < self.delay:
                wait_time = self.delay - elapsed
                time.sleep(wait_time)
            self.last_request_time[domain] = time.time()


# ============================================================================
# URL FILTERING
# ============================================================================

def should_skip_url(url):
    """
    Check if a URL should be skipped based on file extension.
    
    Args:
        url: The URL to check
    
    Returns:
        True if the URL should be skipped, False otherwise
    """
    parsed = urlparse(url)
    path = parsed.path.lower()
    
    for ext in SKIP_EXTENSIONS:
        if path.endswith(ext):
            return True
    return False


# ============================================================================
# ASYNC PAGE FETCHING
# ============================================================================

async def fetch_page_async(session, url, rate_limiter, retries=MAX_RETRIES):
    """
    Async fetch a web page's HTML content with retry logic.
    
    Args:
        session: aiohttp ClientSession
        url: The URL to fetch
        rate_limiter: DomainRateLimiter instance
        retries: Number of retry attempts remaining
    
    Returns:
        Tuple of (html_content, error_message, response_time)
    """
    if not ASYNC_AVAILABLE:
        html, error = fetch_page(url)
        return html, error, 0.0
    
    # Wait for politeness delay
    await rate_limiter.wait_async(url)
    
    headers = {"User-Agent": USER_AGENT}
    start_time = time.time()
    
    for attempt in range(retries):
        try:
            async with session.get(url, headers=headers, timeout=aiohttp.ClientTimeout(total=REQUEST_TIMEOUT)) as response:
                response_time = time.time() - start_time
                
                # Check for HTTP errors
                if response.status >= 400:
                    return None, f"HTTP error: {response.status}", response_time
                
                # Only process HTML content
                content_type = response.headers.get("Content-Type", "")
                if "text/html" not in content_type.lower():
                    return None, f"Not HTML (Content-Type: {content_type})", response_time
                
                html = await response.text()
                return html, None, response_time
                
        except asyncio.TimeoutError:
            if attempt < retries - 1:
                wait_time = RETRY_BACKOFF ** attempt
                logger.debug(f"Timeout fetching {url}, retrying in {wait_time:.1f}s...")
                await asyncio.sleep(wait_time)
            else:
                return None, "Request timed out", time.time() - start_time
        except aiohttp.ClientError as e:
            if attempt < retries - 1:
                wait_time = RETRY_BACKOFF ** attempt
                logger.debug(f"Error fetching {url}: {e}, retrying in {wait_time:.1f}s...")
                await asyncio.sleep(wait_time)
            else:
                return None, f"Connection error: {str(e)}", time.time() - start_time
        except Exception as e:
            return None, f"Request failed: {str(e)}", time.time() - start_time
    
    return None, "Max retries exceeded", time.time() - start_time


# ============================================================================
# PAGE FETCHING AND PARSING (SYNC)
# ============================================================================

def create_browser():
    """
    Create a MechanicalSoup browser with our crawler user agent.
    """
    return mechanicalsoup.StatefulBrowser(user_agent=USER_AGENT)


def fetch_page(url, browser=None):
    """
    Fetch a web page's HTML content.
    
    Args:
        url: The URL to fetch
        browser: Optional MechanicalSoup browser for session reuse
    
    Returns:
        Tuple of (html_content, error_message)
        - On success: (html_string, None)
        - On failure: (None, error_description)
    """
    try:
        browser = browser or create_browser()
        response = browser.open(url, timeout=REQUEST_TIMEOUT)
        
        # Check for HTTP errors (4xx, 5xx)
        response.raise_for_status()
        
        # Only process HTML content
        content_type = response.headers.get("Content-Type", "")
        if "text/html" not in content_type.lower():
            return None, f"Not HTML (Content-Type: {content_type})"
        
        return response.text, None
        
    except requests.exceptions.Timeout:
        return None, "Request timed out"
    except requests.exceptions.ConnectionError:
        return None, "Connection error"
    except requests.exceptions.HTTPError as e:
        return None, f"HTTP error: {e.response.status_code}"
    except requests.exceptions.RequestException as e:
        return None, f"Request failed: {str(e)}"


def extract_links(html, base_url):
    """
    Extract and normalize all links from an HTML page.
    
    Args:
        html: The HTML content as a string
        base_url: The URL of the page (used for resolving relative links)
    
    Returns:
        List of normalized, absolute URLs found on the page
    """
    links = []
    
    try:
        soup = BeautifulSoup(html, "html.parser")
        
        # Find all anchor tags with href attributes
        for anchor in soup.find_all("a", href=True):
            href = anchor["href"]
            
            # Skip empty hrefs, javascript links, and anchors
            if not href or href.startswith(("#", "javascript:", "mailto:", "tel:")):
                continue
            
            # Convert relative URLs to absolute URLs
            absolute_url = urljoin(base_url, href)
            
            # Parse and validate the URL
            parsed = urlparse(absolute_url)
            
            # Only keep HTTP/HTTPS URLs
            if parsed.scheme in ("http", "https"):
                # Remove fragment (the part after #) for cleaner URLs
                clean_url = f"{parsed.scheme}://{parsed.netloc}{parsed.path}"
                if parsed.query:
                    clean_url += f"?{parsed.query}"
                links.append(clean_url)
        
    except Exception as e:
        logger.warning(f"Error parsing links: {e}")
    
    return links


def extract_text(html):
    """
    Extract readable text from an HTML page.
    
    Args:
        html: The HTML content as a string
    
    Returns:
        Plain text content of the page (lowercased for search)
    """
    try:
        soup = BeautifulSoup(html, "html.parser")
        
        # Remove script and style elements (not readable content)
        for element in soup(["script", "style", "meta", "link"]):
            element.decompose()
        
        # Get text and normalize whitespace
        text = soup.get_text(separator=" ", strip=True)
        return text.lower()
        
    except Exception:
        return ""


def search_page(html, query):
    """
    Check if a page contains the search query.
    
    Args:
        html: The HTML content as a string
        query: The search term to look for
    
    Returns:
        True if the query is found, False otherwise
    """
    text = extract_text(html)
    return query.lower() in text


# ============================================================================
# MAIN CRAWLER LOGIC
# ============================================================================

def crawl(start_url, search_query, max_urls=MAX_URLS_TO_VISIT, max_depth=None,
          same_domain_only=False, stop_event=None, progress_callback=None):
    """
    Main crawling function using BFS traversal.
    
    Args:
        start_url: The seed URL to start crawling from
        search_query: The term to search for on each page
        max_urls: Maximum number of unique URLs to visit
        max_depth: Maximum depth to crawl (None for unlimited)
        same_domain_only: If True, only crawl URLs from the same domain
        stop_event: Optional threading.Event to request a stop
        progress_callback: Optional callback with (visited, max_urls)
    
    Returns:
        Tuple of (visited_urls_data, matching_urls, crawl_metadata)
    """
    logger.info("\n" + "=" * 60)
    logger.info("STARTING WEB CRAWLER")
    logger.info("=" * 60)
    logger.info(f"Start URL: {start_url}")
    logger.info(f"Search Query: '{search_query}'")
    logger.info(f"Target: {max_urls} unique URLs")
    if max_depth is not None:
        logger.info(f"Max Depth: {max_depth}")
    else:
        logger.info(f"Max Depth: Unlimited")
    logger.info(f"Same Domain Only: {same_domain_only}")
    logger.info("=" * 60 + "\n")
    
    # Record start time
    start_time = time.time()
    
    # Initialize data structures
    frontier = URLFrontier(start_url, max_depth=max_depth, same_domain_only=same_domain_only)
    robots_cache = {}  # Cache robots.txt parsers per domain
    browser = create_browser()
    visited_urls_data = []  # List of dicts with URL info
    matching_urls = []  # URLs where the query was found
    
    # Main crawling loop
    while len(visited_urls_data) < max_urls and not frontier.is_empty():
        if stop_event and stop_event.is_set():
            logger.info("  [Stopped] Crawl cancelled by user.")
            break
        
        # Get next URL from the frontier
        current_url, depth = frontier.get_next()
        
        if current_url is None:
            break
        
        url_number = len(visited_urls_data) + 1
        logger.info(f"\n[{url_number}/{max_urls}] Depth {depth}: {current_url}")
        
        # Check robots.txt compliance
        if not can_crawl(current_url, robots_cache):
            logger.info("  [Skipped] Blocked by robots.txt")
            continue
        
        # Record fetch start time
        fetch_start = time.time()
        
        # Fetch the page
        html, error = fetch_page(current_url, browser)
        
        # Calculate response time
        response_time = time.time() - fetch_start
        
        # Create URL data record
        url_data = {
            'url': current_url,
            'depth': depth,
            'timestamp': datetime.now().isoformat(),
            'response_time': response_time,
            'status': 'visited',
            'error': None,
            'matched': False
        }
        
        if error:
            logger.error(f"  [Error] {error}")
            url_data['status'] = 'error'
            url_data['error'] = error
            # Still count as visited (we attempted it)
            visited_urls_data.append(url_data)
            if progress_callback:
                progress_callback(len(visited_urls_data), max_urls)
            continue
        
        # Successfully fetched - add to visited list
        visited_urls_data.append(url_data)
        if progress_callback:
            progress_callback(len(visited_urls_data), max_urls)
        
        # Search for the query
        if search_page(html, search_query):
            matching_urls.append(current_url)
            url_data['matched'] = True
            logger.info(f"  [Match!] Query '{search_query}' found on this page")
        
        # Extract links from the page
        links = extract_links(html, current_url)
        logger.info(f"  [Links] Found {len(links)} links on this page")
        
        # Display discovered links (first 5 for brevity)
        if links:
            display_count = min(5, len(links))
            for link in links[:display_count]:
                logger.info(f"    → {link}")
            if len(links) > display_count:
                logger.info(f"    ... and {len(links) - display_count} more")
        
        # Add new links to the frontier
        frontier.add_urls(links, depth)
    
    # Calculate total time
    total_time = time.time() - start_time
    
    # Create metadata
    crawl_metadata = {
        'start_url': start_url,
        'search_query': search_query,
        'max_urls': max_urls,
        'max_depth': max_depth,
        'same_domain_only': same_domain_only,
        'urls_visited': len(visited_urls_data),
        'urls_matched': len(matching_urls),
        'total_time': total_time,
        'timestamp': datetime.now().isoformat()
    }
    
    return visited_urls_data, matching_urls, crawl_metadata


# ============================================================================
# ASYNC CRAWLER LOGIC
# ============================================================================

async def crawl_async(start_url, search_query, max_urls=MAX_URLS_TO_VISIT, max_depth=None,
                      same_domain_only=False, concurrency=DEFAULT_CONCURRENCY,
                      politeness_delay=DEFAULT_POLITENESS_DELAY, stop_event=None):
    """
    High-performance async crawling function using BFS traversal with concurrent fetching.
    
    Args:
        start_url: The seed URL to start crawling from
        search_query: The term to search for on each page
        max_urls: Maximum number of unique URLs to visit
        max_depth: Maximum depth to crawl (None for unlimited)
        same_domain_only: If True, only crawl URLs from the same domain
        concurrency: Number of concurrent requests
        politeness_delay: Seconds between requests to same domain
        stop_event: Optional threading.Event to request a stop
    
    Returns:
        Tuple of (visited_urls_data, matching_urls, crawl_metadata)
    """
    if not ASYNC_AVAILABLE:
        logger.warning("Async mode not available, falling back to sync mode")
        return crawl(start_url, search_query, max_urls, max_depth, same_domain_only, stop_event)
    
    logger.info("\n" + "=" * 60)
    logger.info("STARTING ASYNC WEB CRAWLER")
    logger.info("=" * 60)
    logger.info(f"Start URL: {start_url}")
    logger.info(f"Search Query: '{search_query}'")
    logger.info(f"Target: {max_urls} unique URLs")
    logger.info(f"Concurrency: {concurrency} workers")
    logger.info(f"Politeness Delay: {politeness_delay}s per domain")
    if max_depth is not None:
        logger.info(f"Max Depth: {max_depth}")
    else:
        logger.info(f"Max Depth: Unlimited")
    logger.info(f"Same Domain Only: {same_domain_only}")
    logger.info("=" * 60 + "\n")
    
    # Record start time
    start_time = time.time()
    
    # Initialize data structures
    frontier = URLFrontier(start_url, max_depth=max_depth, same_domain_only=same_domain_only)
    robots_cache = {}  # Cache robots.txt parsers per domain
    rate_limiter = DomainRateLimiter(delay=politeness_delay)
    visited_urls_data = []
    matching_urls = []
    urls_in_progress = set()
    results_lock = asyncio.Lock()
    
    # Semaphore for concurrency control
    semaphore = asyncio.Semaphore(concurrency)
    
    async def process_url(session, url, depth):
        """Process a single URL asynchronously."""
        nonlocal visited_urls_data, matching_urls
        
        async with semaphore:
            if stop_event and stop_event.is_set():
                return
            
            # Check if we've already hit the limit
            async with results_lock:
                if len(visited_urls_data) >= max_urls:
                    return
                url_number = len(visited_urls_data) + 1
            
            logger.info(f"\n[{url_number}/{max_urls}] Depth {depth}: {url}")
            
            # Check robots.txt compliance (sync operation, but fast due to caching)
            if not can_crawl(url, robots_cache):
                logger.info("  [Skipped] Blocked by robots.txt")
                return
            
            # Skip non-HTML file extensions
            if should_skip_url(url):
                logger.info("  [Skipped] Non-HTML file extension")
                return
            
            # Fetch the page
            html, error, response_time = await fetch_page_async(session, url, rate_limiter)
            
            # Create URL data record
            url_data = {
                'url': url,
                'depth': depth,
                'timestamp': datetime.now().isoformat(),
                'response_time': response_time,
                'status': 'visited',
                'error': None,
                'matched': False
            }
            
            if error:
                logger.error(f"  [Error] {error}")
                url_data['status'] = 'error'
                url_data['error'] = error
                async with results_lock:
                    visited_urls_data.append(url_data)
                return
            
            # Search for the query
            if search_page(html, search_query):
                url_data['matched'] = True
                logger.info(f"  [Match!] Query '{search_query}' found on this page")
                async with results_lock:
                    matching_urls.append(url)
            
            # Add to visited list
            async with results_lock:
                visited_urls_data.append(url_data)
            
            # Extract links from the page
            links = extract_links(html, url)
            
            # Filter out non-HTML links
            links = [link for link in links if not should_skip_url(link)]
            
            logger.info(f"  [Links] Found {len(links)} valid links on this page")
            
            # Display discovered links (first 5 for brevity)
            if links:
                display_count = min(5, len(links))
                for link in links[:display_count]:
                    logger.info(f"    → {link}")
                if len(links) > display_count:
                    logger.info(f"    ... and {len(links) - display_count} more")
            
            # Add new links to the frontier
            frontier.add_urls(links, depth)
    
    # Create aiohttp session with connection pooling
    connector = aiohttp.TCPConnector(limit=concurrency * 2, limit_per_host=2)
    timeout = aiohttp.ClientTimeout(total=REQUEST_TIMEOUT)
    
    async with aiohttp.ClientSession(connector=connector, timeout=timeout) as session:
        while len(visited_urls_data) < max_urls and not frontier.is_empty():
            if stop_event and stop_event.is_set():
                logger.info("  [Stopped] Crawl cancelled by user.")
                break
            
            # Get batch of URLs to process concurrently
            batch = []
            while len(batch) < concurrency and not frontier.is_empty():
                url, depth = frontier.get_next()
                if url and url not in urls_in_progress:
                    urls_in_progress.add(url)
                    batch.append((url, depth))
            
            if not batch:
                break
            
            # Process batch concurrently
            tasks = [process_url(session, url, depth) for url, depth in batch]
            await asyncio.gather(*tasks, return_exceptions=True)
            
            # Remove from in-progress set
            for url, _ in batch:
                urls_in_progress.discard(url)
    
    # Calculate total time
    total_time = time.time() - start_time
    
    # Create metadata
    crawl_metadata = {
        'start_url': start_url,
        'search_query': search_query,
        'max_urls': max_urls,
        'max_depth': max_depth,
        'same_domain_only': same_domain_only,
        'concurrency': concurrency,
        'politeness_delay': politeness_delay,
        'urls_visited': len(visited_urls_data),
        'urls_matched': len(matching_urls),
        'total_time': total_time,
        'timestamp': datetime.now().isoformat(),
        'mode': 'async'
    }
    
    return visited_urls_data, matching_urls, crawl_metadata


def run_async_crawl(start_url, search_query, max_urls=MAX_URLS_TO_VISIT, max_depth=None,
                    same_domain_only=False, concurrency=DEFAULT_CONCURRENCY,
                    politeness_delay=DEFAULT_POLITENESS_DELAY, stop_event=None):
    """
    Wrapper to run async crawl from sync code.
    
    This is useful for running from the GUI or other sync contexts.
    """
    return asyncio.run(crawl_async(
        start_url, search_query, max_urls, max_depth,
        same_domain_only, concurrency, politeness_delay, stop_event
    ))


def print_results(visited_urls_data, matching_urls, search_query, max_urls=MAX_URLS_TO_VISIT):
    """
    Print the final crawling results.
    
    Args:
        visited_urls_data: List of dictionaries with URL information
        matching_urls: List of URLs where the query was found
        search_query: The search term used
        max_urls: Target number of URLs for this crawl
    """
    logger.info("\n" + "=" * 60)
    logger.info("CRAWL RESULTS")
    logger.info("=" * 60)
    
    # Print all visited URLs
    logger.info(f"\n📋 ALL VISITED URLs ({len(visited_urls_data)} of target {max_urls}):")
    logger.info("-" * 40)
    for i, url_data in enumerate(visited_urls_data, 1):
        url = url_data['url']
        depth = url_data.get('depth', 0)
        status = url_data.get('status', 'visited')
        logger.info(f"  {i:2}. [Depth {depth}] {url} ({status})")
    
    # Print search results
    logger.info(f"\n🔍 SEARCH RESULTS for '{search_query}':")
    logger.info("-" * 40)
    logger.info(f"  Pages containing the query: {len(matching_urls)}")
    
    if matching_urls:
        logger.info(f"\n  URLs with matches:")
        for i, url in enumerate(matching_urls, 1):
            logger.info(f"    {i}. {url}")
    else:
        logger.info(f"\n  No pages contained the query '{search_query}'")
    
    logger.info("\n" + "=" * 60)
    logger.info("CRAWL COMPLETE")
    logger.info("=" * 60)


# ============================================================================
# EXPORT FUNCTIONALITY
# ============================================================================

def export_to_json(visited_urls_data, matching_urls, crawl_metadata, filename=None):
    """
    Export crawl results to JSON format.
    
    Args:
        visited_urls_data: List of dictionaries with URL information
        matching_urls: List of URLs where the query was found
        crawl_metadata: Dictionary with crawl metadata
        filename: Output filename (auto-generated if None)
    
    Returns:
        The filename used
    """
    if filename is None:
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        filename = f"crawl_results_{timestamp}.json"
    
    data = {
        'metadata': crawl_metadata,
        'visited_urls': visited_urls_data,
        'matching_urls': matching_urls
    }
    
    try:
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        logger.info(f"\n✅ Results exported to JSON: {filename}")
        return filename
    except Exception as e:
        logger.error(f"Failed to export to JSON: {e}")
        return None


def export_to_csv(visited_urls_data, matching_urls, crawl_metadata, filename=None):
    """
    Export crawl results to CSV format.
    
    Args:
        visited_urls_data: List of dictionaries with URL information
        matching_urls: List of URLs where the query was found
        crawl_metadata: Dictionary with crawl metadata
        filename: Output filename (auto-generated if None)
    
    Returns:
        The filename used
    """
    if filename is None:
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        filename = f"crawl_results_{timestamp}.csv"
    
    try:
        with open(filename, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            
            # Write metadata header
            writer.writerow(['# Crawl Metadata'])
            writer.writerow(['Start URL', crawl_metadata['start_url']])
            writer.writerow(['Search Query', crawl_metadata['search_query']])
            writer.writerow(['Max URLs', crawl_metadata['max_urls']])
            writer.writerow(['Max Depth', crawl_metadata.get('max_depth', 'Unlimited')])
            writer.writerow(['Same Domain Only', crawl_metadata.get('same_domain_only', False)])
            writer.writerow(['URLs Visited', crawl_metadata['urls_visited']])
            writer.writerow(['URLs Matched', crawl_metadata['urls_matched']])
            writer.writerow(['Total Time (seconds)', f"{crawl_metadata['total_time']:.2f}"])
            writer.writerow(['Timestamp', crawl_metadata['timestamp']])
            writer.writerow([])  # Empty row
            
            # Write URL data
            writer.writerow(['URL', 'Depth', 'Status', 'Matched', 'Response Time (s)', 'Error', 'Timestamp'])
            for url_data in visited_urls_data:
                writer.writerow([
                    url_data['url'],
                    url_data.get('depth', 0),
                    url_data.get('status', 'visited'),
                    'Yes' if url_data.get('matched', False) else 'No',
                    f"{url_data.get('response_time', 0):.3f}",
                    url_data.get('error', ''),
                    url_data.get('timestamp', '')
                ])
        
        logger.info(f"✅ Results exported to CSV: {filename}")
        return filename
    except Exception as e:
        logger.error(f"Failed to export to CSV: {e}")
        return None


# ============================================================================
# USER INPUT AND PROGRAM ENTRY
# ============================================================================

def get_user_input():
    """
    Get the starting URL and search query from the user.
    
    Returns:
        Tuple of (start_url, search_query)
    """
    logger.info("\n" + "=" * 60)
    logger.info("WEB CRAWLER Application")
    logger.info("=" * 60)
    
    # Get starting URL
    while True:
        start_url = input("\nEnter the starting URL (e.g., https://example.com): ").strip()
        
        if not start_url:
            logger.info("  Please enter a valid URL.")
            continue
        
        # Add https:// if no scheme provided
        if not start_url.startswith(("http://", "https://")):
            start_url = "https://" + start_url
        
        # Basic URL validation
        parsed = urlparse(start_url)
        if parsed.scheme and parsed.netloc:
            break
        else:
            logger.info("  Invalid URL format. Please try again.")
    
    # Get search query
    while True:
        search_query = input("Enter the search query: ").strip()
        
        if search_query:
            break
        else:
            logger.info("  Please enter a search term.")
    
    return start_url, search_query


def parse_arguments():
    """
    Parse command-line arguments.
    
    Returns:
        argparse.Namespace with parsed arguments
    """
    parser = argparse.ArgumentParser(
        description='Web Crawler - High-performance async crawler with search functionality',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Interactive mode (async by default)
  python crawler.py
  
  # CLI mode with all options
  python crawler.py --url https://example.com --query "privacy" --max-urls 50 --max-depth 3
  
  # Fast concurrent crawling
  python crawler.py --url https://example.com --query "contact" --concurrency 10 --delay 0.5
  
  # GUI mode
  python crawler.py --gui
  
  # Sync mode (original behavior)
  python crawler.py --url https://example.com --query "privacy" --sync
  
  # Same-domain only mode with export
  python crawler.py --url https://example.com --query "contact" --same-domain-only --export-json
        """
    )
    
    parser.add_argument('--url', '-u', type=str, help='Starting URL to crawl')
    parser.add_argument('--query', '-q', type=str, help='Search query to look for')
    parser.add_argument('--max-urls', '-m', type=int, default=MAX_URLS_TO_VISIT,
                        help=f'Maximum number of URLs to visit (default: {MAX_URLS_TO_VISIT})')
    parser.add_argument('--max-depth', '-d', type=int, default=None,
                        help='Maximum crawl depth (default: unlimited)')
    parser.add_argument('--same-domain-only', '-s', action='store_true',
                        help='Only crawl URLs from the same domain as the seed URL')
    
    # Async options
    parser.add_argument('--sync', action='store_true',
                        help='Use synchronous crawling instead of async (slower but simpler)')
    parser.add_argument('--concurrency', '-c', type=int, default=DEFAULT_CONCURRENCY,
                        help=f'Number of concurrent requests (default: {DEFAULT_CONCURRENCY})')
    parser.add_argument('--delay', type=float, default=DEFAULT_POLITENESS_DELAY,
                        help=f'Politeness delay in seconds between requests to same domain (default: {DEFAULT_POLITENESS_DELAY})')
    
    # Export options
    parser.add_argument('--export-json', type=str, nargs='?', const='', metavar='FILE',
                        help='Export results to JSON file (auto-generated filename if not specified)')
    parser.add_argument('--export-csv', type=str, nargs='?', const='', metavar='FILE',
                        help='Export results to CSV file (auto-generated filename if not specified)')
    parser.add_argument('--log-level', type=str, choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
                        default='INFO', help='Set logging level (default: INFO)')
    parser.add_argument('--gui', action='store_true', help='Run in GUI mode')
    
    return parser.parse_args()


def main():
    """
    Main entry point for the web crawler program.
    """
    args = parse_arguments()
    
    # Check if GUI mode
    if args.gui:
        run_gui()
        return
    
    # Setup logging with specified level
    global logger
    log_level = getattr(logging, args.log_level)
    logger = setup_logging(log_level=log_level)
    
    try:
        # Get user input (interactive or from args)
        if args.url and args.query:
            start_url = args.url
            search_query = args.query
            
            # Add https:// if no scheme provided
            if not start_url.startswith(("http://", "https://")):
                start_url = "https://" + start_url
        else:
            start_url, search_query = get_user_input()
        
        # Run the crawler (async by default if available)
        if args.sync or not ASYNC_AVAILABLE:
            if not args.sync:
                logger.info("Async mode not available, using sync mode")
            visited_urls_data, matching_urls, crawl_metadata = crawl(
                start_url, 
                search_query,
                max_urls=args.max_urls,
                max_depth=args.max_depth,
                same_domain_only=args.same_domain_only
            )
        else:
            visited_urls_data, matching_urls, crawl_metadata = run_async_crawl(
                start_url, 
                search_query,
                max_urls=args.max_urls,
                max_depth=args.max_depth,
                same_domain_only=args.same_domain_only,
                concurrency=args.concurrency,
                politeness_delay=args.delay
            )
        
        # Print results
        print_results(visited_urls_data, matching_urls, search_query, max_urls=args.max_urls)
        
        # Export results if requested
        if args.export_json is not None:
            filename = args.export_json if args.export_json else None
            export_to_json(visited_urls_data, matching_urls, crawl_metadata, filename)
        
        if args.export_csv is not None:
            filename = args.export_csv if args.export_csv else None
            export_to_csv(visited_urls_data, matching_urls, crawl_metadata, filename)
        
    except KeyboardInterrupt:
        logger.info("\n\n[Interrupted] Crawler stopped by user.")
    except Exception as e:
        logger.error(f"\n[Critical Error] An unexpected error occurred: {e}")
        logger.error("The crawler has stopped.")
        logger.debug("Exception details:", exc_info=True)


# ============================================================================ 
# OPTIONAL: TKINTER GUI WRAPPER 
# ============================================================================ 


class TextRedirector:
    """
    Redirects stdout to a Tkinter Text/ScrolledText widget in a thread-safe way.
    Also logs to the file logger.
    """

    def __init__(self, widget, text_logger):
        self.widget = widget
        self.text_logger = text_logger

    def write(self, message):
        if not message:
            return
        # Ensure UI updates happen on the main thread
        self.widget.after(0, lambda: self._append(message))
        # Also write to text logger if it exists
        if self.text_logger and message.strip():
            # Remove ANSI codes and special formatting for log file
            clean_message = message.strip()
            if clean_message:
                self.text_logger.info(clean_message)

    def flush(self):
        # No-op needed for file-like API
        pass

    def _append(self, message):
        was_disabled = str(self.widget.cget("state")) == "disabled"
        if was_disabled:
            self.widget.configure(state="normal")
        self.widget.insert("end", message)
        self.widget.see("end")
        if was_disabled:
            self.widget.configure(state="disabled")


def format_results_for_gui(visited_urls_data, matching_urls, crawl_metadata):
    lines = []
    lines.append("CRAWL SUMMARY")
    lines.append("=" * 60)
    lines.append(f"Start URL: {crawl_metadata.get('start_url', '')}")
    lines.append(f"Search Query: {crawl_metadata.get('search_query', '')}")
    lines.append(f"Max URLs: {crawl_metadata.get('max_urls', '')}")
    max_depth = crawl_metadata.get("max_depth")
    max_depth_text = max_depth if max_depth is not None else "Unlimited"
    lines.append(f"Max Depth: {max_depth_text}")
    lines.append(f"Same Domain Only: {crawl_metadata.get('same_domain_only', False)}")
    lines.append(f"URLs Visited: {crawl_metadata.get('urls_visited', len(visited_urls_data))}")
    lines.append(f"URLs Matched: {crawl_metadata.get('urls_matched', len(matching_urls))}")
    total_time = crawl_metadata.get("total_time", 0.0)
    lines.append(f"Total Time (s): {total_time:.2f}")
    timestamp = crawl_metadata.get("timestamp")
    if timestamp:
        lines.append(f"Timestamp: {timestamp}")

    lines.append("")
    lines.append("MATCHES")
    lines.append("-" * 60)
    if matching_urls:
        for i, url in enumerate(matching_urls, 1):
            lines.append(f"{i:2}. {url}")
    else:
        lines.append("No matches found.")

    lines.append("")
    lines.append("VISITED URLS")
    lines.append("-" * 60)
    lines.append(f"{'#':>3}  {'Depth':>5}  {'Status':<7}  {'Match':<5}  URL")
    lines.append("-" * 60)
    for i, url_data in enumerate(visited_urls_data, 1):
        depth = url_data.get("depth", 0)
        status = url_data.get("status", "visited")
        matched = "Yes" if url_data.get("matched") else "No"
        url = url_data.get("url", "")
        lines.append(f"{i:3}  {depth:5}  {status:<7}  {matched:<5}  {url}")
        error = url_data.get("error")
        if error:
            lines.append(f"      error: {error}")

    return "\n".join(lines)


def run_gui():
    """
    Tkinter GUI wrapper around the existing crawler with all new features.
    """
    import os
    import tkinter as tk
    from tkinter import messagebox, scrolledtext, filedialog, ttk
    
    # Setup GUI logger with file output only
    gui_logger = logging.getLogger("WebCrawler.GUI")
    gui_logger.setLevel(logging.INFO)
    gui_logger.handlers = []
    file_handler = logging.FileHandler(LOG_FILE, encoding='utf-8')
    file_handler.setFormatter(logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    ))
    gui_logger.addHandler(file_handler)

    root = tk.Tk()
<<<<<<< HEAD
    root.title("Web Crawler - Async Edition" if ASYNC_AVAILABLE else "Web Crawler")
    root.geometry("900x750")
=======
    root.title("Web Crawler - Enhanced Edition")
    root.geometry("900x700")
    root.configure(background="#f6f3ee")
    style = ttk.Style(root)
    if "clam" in style.theme_names():
        style.theme_use("clam")
    style.configure("TFrame", background="#f6f3ee")
    style.configure("TLabelframe", background="#f6f3ee")
    style.configure("TLabelframe.Label", background="#f6f3ee", font=("Segoe UI", 10, "bold"))
    style.configure("TLabel", background="#f6f3ee")
>>>>>>> 2c70ca6369b91adfb6c25eb7a9ce0298ecafbc3c

    url_var = tk.StringVar()
    query_var = tk.StringVar()
    max_urls_var = tk.StringVar(value=str(MAX_URLS_TO_VISIT))
    max_depth_var = tk.StringVar(value="")
    same_domain_var = tk.BooleanVar(value=False)
    export_format_var = tk.StringVar(value="json")
    log_level_var = tk.StringVar(value="INFO")
<<<<<<< HEAD
    
    # Async options
    use_async_var = tk.BooleanVar(value=ASYNC_AVAILABLE)
    concurrency_var = tk.StringVar(value=str(DEFAULT_CONCURRENCY))
    delay_var = tk.StringVar(value=str(DEFAULT_POLITENESS_DELAY))
    
=======
    export_dir_var = tk.StringVar(value=os.getcwd())
    status_var = tk.StringVar(value="Stopped")
    progress_var = tk.StringVar(value=f"0 / {MAX_URLS_TO_VISIT}")
>>>>>>> 2c70ca6369b91adfb6c25eb7a9ce0298ecafbc3c
    stop_event_holder = {"event": None}

    inputs_frame = ttk.LabelFrame(root, text="Inputs", padding=10)
    inputs_frame.grid(row=0, column=0, sticky="ew", padx=10, pady=6)
    inputs_frame.grid_columnconfigure(1, weight=1)

    ttk.Label(inputs_frame, text="Starting URL:").grid(row=0, column=0, sticky="w", padx=6, pady=4)
    url_entry = ttk.Entry(inputs_frame, textvariable=url_var, width=70)
    url_entry.grid(row=0, column=1, sticky="ew", padx=6, pady=4)

    ttk.Label(inputs_frame, text="Search Query:").grid(row=1, column=0, sticky="w", padx=6, pady=4)
    query_entry = ttk.Entry(inputs_frame, textvariable=query_var, width=40)
    query_entry.grid(row=1, column=1, sticky="ew", padx=6, pady=4)

<<<<<<< HEAD
    # Row 3: Async options
    async_frame = tk.LabelFrame(root, text="Performance Options", padx=8, pady=4)
    async_frame.grid(row=3, column=0, columnspan=3, sticky="ew", padx=8, pady=4)
    
    async_check = tk.Checkbutton(
        async_frame, text="Async mode (faster)", 
        variable=use_async_var,
        state="normal" if ASYNC_AVAILABLE else "disabled"
    )
    async_check.pack(side=tk.LEFT, padx=5)
    
    tk.Label(async_frame, text="Concurrency:").pack(side=tk.LEFT, padx=(20, 5))
    concurrency_entry = tk.Entry(async_frame, textvariable=concurrency_var, width=5)
    concurrency_entry.pack(side=tk.LEFT)
    
    tk.Label(async_frame, text="Delay (sec):").pack(side=tk.LEFT, padx=(20, 5))
    delay_entry = tk.Entry(async_frame, textvariable=delay_var, width=5)
    delay_entry.pack(side=tk.LEFT)
    
    # Row 4: Same Domain Only checkbox
    same_domain_check = tk.Checkbutton(
        root, text="Crawl same domain only", 
        variable=same_domain_var
    )
    same_domain_check.grid(row=4, column=0, columnspan=2, sticky="w", padx=8, pady=4)

    # Row 5: Export format and Log level
    tk.Label(root, text="Export Format:").grid(row=5, column=0, sticky="w", padx=8, pady=4)
    export_frame = tk.Frame(root)
    export_frame.grid(row=5, column=1, sticky="w", padx=8, pady=4)
    tk.Radiobutton(export_frame, text="JSON", variable=export_format_var, value="json").pack(side=tk.LEFT)
    tk.Radiobutton(export_frame, text="CSV", variable=export_format_var, value="csv").pack(side=tk.LEFT)
    tk.Radiobutton(export_frame, text="Both", variable=export_format_var, value="both").pack(side=tk.LEFT)
    tk.Radiobutton(export_frame, text="None", variable=export_format_var, value="none").pack(side=tk.LEFT)
    
    tk.Label(root, text="Log Level:").grid(row=5, column=1, sticky="e", padx=8, pady=4)
    log_level_menu = tk.OptionMenu(root, log_level_var, "DEBUG", "INFO", "WARNING", "ERROR")
    log_level_menu.grid(row=5, column=2, sticky="w", padx=8, pady=4)

    # Row 7: Log box (moved down to make room for async options)
    log_box = scrolledtext.ScrolledText(root, width=100, height=28, state="normal")
    log_box.grid(row=7, column=0, columnspan=3, padx=8, pady=8, sticky="nsew")

    # Grid stretch
    root.grid_columnconfigure(1, weight=1)
    root.grid_rowconfigure(7, weight=1)
=======
    options_frame = ttk.LabelFrame(root, text="Options", padding=10)
    options_frame.grid(row=1, column=0, sticky="ew", padx=10, pady=6)
    options_frame.grid_columnconfigure(1, weight=1)

    ttk.Label(options_frame, text="Max URLs (1-200):").grid(row=0, column=0, sticky="w", padx=6, pady=4)
    max_urls_entry = ttk.Entry(options_frame, textvariable=max_urls_var, width=10)
    max_urls_entry.grid(row=0, column=1, sticky="w", padx=6, pady=4)

    ttk.Label(options_frame, text="Max Depth (optional):").grid(row=0, column=2, sticky="w", padx=6, pady=4)
    max_depth_entry = ttk.Entry(options_frame, textvariable=max_depth_var, width=10)
    max_depth_entry.grid(row=0, column=3, sticky="w", padx=6, pady=4)

    same_domain_check = ttk.Checkbutton(
        options_frame, text="Crawl same domain only",
        variable=same_domain_var
    )
    same_domain_check.grid(row=1, column=0, columnspan=2, sticky="w", padx=6, pady=4)

    ttk.Label(options_frame, text="Log Level:").grid(row=1, column=2, sticky="e", padx=6, pady=4)
    log_level_menu = ttk.Combobox(
        options_frame,
        textvariable=log_level_var,
        values=("DEBUG", "INFO", "WARNING", "ERROR"),
        width=12,
        state="readonly",
    )
    log_level_menu.grid(row=1, column=3, sticky="w", padx=6, pady=4)

    output_frame = ttk.LabelFrame(root, text="Output", padding=10)
    output_frame.grid(row=2, column=0, sticky="ew", padx=10, pady=6)
    output_frame.grid_columnconfigure(1, weight=1)

    ttk.Label(output_frame, text="Export Format:").grid(row=0, column=0, sticky="w", padx=6, pady=4)
    export_frame = ttk.Frame(output_frame)
    export_frame.grid(row=0, column=1, columnspan=3, sticky="w", padx=6, pady=4)
    export_json_radio = ttk.Radiobutton(export_frame, text="JSON", variable=export_format_var, value="json")
    export_csv_radio = ttk.Radiobutton(export_frame, text="CSV", variable=export_format_var, value="csv")
    export_both_radio = ttk.Radiobutton(export_frame, text="Both", variable=export_format_var, value="both")
    export_none_radio = ttk.Radiobutton(export_frame, text="None", variable=export_format_var, value="none")
    export_json_radio.pack(side=tk.LEFT, padx=(0, 6))
    export_csv_radio.pack(side=tk.LEFT, padx=(0, 6))
    export_both_radio.pack(side=tk.LEFT, padx=(0, 6))
    export_none_radio.pack(side=tk.LEFT)

    ttk.Label(output_frame, text="Export Directory:").grid(row=1, column=0, sticky="w", padx=6, pady=4)
    export_dir_entry = ttk.Entry(output_frame, textvariable=export_dir_var, state="readonly")
    export_dir_entry.grid(row=1, column=1, sticky="ew", padx=6, pady=4)
    export_dir_button = ttk.Button(output_frame, text="Browse", width=10)
    export_dir_button.grid(row=1, column=2, sticky="w", padx=6, pady=4)

    ttk.Label(output_frame, text="Status:").grid(row=2, column=0, sticky="w", padx=6, pady=4)
    status_label = ttk.Label(output_frame, textvariable=status_var)
    status_label.grid(row=2, column=1, sticky="w", padx=6, pady=4)

    ttk.Label(output_frame, text="Progress:").grid(row=3, column=0, sticky="w", padx=6, pady=4)
    progress_bar = ttk.Progressbar(output_frame, maximum=MAX_URLS_TO_VISIT, value=0)
    progress_bar.grid(row=3, column=1, sticky="ew", padx=6, pady=4)
    progress_label = ttk.Label(output_frame, textvariable=progress_var)
    progress_label.grid(row=3, column=2, sticky="w", padx=6, pady=4)

    # Buttons
    button_frame = ttk.Frame(root)
    button_frame.grid(row=3, column=0, pady=6, padx=10, sticky="ew")
    start_button = ttk.Button(button_frame, text="Run Crawl", width=15)
    start_button.pack(side=tk.LEFT, padx=5)
    stop_button = ttk.Button(button_frame, text="Stop", state="disabled", width=15)
    stop_button.pack(side=tk.LEFT, padx=5)

    notebook = ttk.Notebook(root)
    notebook.grid(row=4, column=0, padx=10, pady=8, sticky="nsew")

    log_tab = ttk.Frame(notebook)
    results_tab = ttk.Frame(notebook)
    notebook.add(log_tab, text="Log")
    notebook.add(results_tab, text="Results")

    log_box = scrolledtext.ScrolledText(log_tab, width=100, height=30, state="disabled")
    log_box.pack(fill="both", expand=True)

    results_box = scrolledtext.ScrolledText(results_tab, width=100, height=30, state="disabled")
    results_box.pack(fill="both", expand=True)

    # Grid stretch
    root.grid_columnconfigure(0, weight=1)
    root.grid_rowconfigure(4, weight=1)

    def set_inputs_state(enabled):
        normal_state = "normal" if enabled else "disabled"
        readonly_state = "readonly" if enabled else "disabled"
        for widget in (
            url_entry,
            query_entry,
            max_urls_entry,
            max_depth_entry,
            same_domain_check,
            export_json_radio,
            export_csv_radio,
            export_both_radio,
            export_none_radio,
            export_dir_button,
        ):
            try:
                widget.configure(state=normal_state)
            except tk.TclError:
                pass
        log_level_menu.configure(state=readonly_state)
        export_dir_entry.configure(state=readonly_state)

    def set_results_text(text):
        results_box.configure(state="normal")
        results_box.delete("1.0", "end")
        results_box.insert("end", text)
        results_box.configure(state="disabled")

    def update_progress(visited, max_urls):
        progress_bar.configure(maximum=max_urls)
        progress_bar.configure(value=visited)
        progress_var.set(f"{visited} / {max_urls}")

    def choose_export_dir():
        selected = filedialog.askdirectory(initialdir=export_dir_var.get() or os.getcwd())
        if selected:
            export_dir_var.set(selected)

    export_dir_button.configure(command=choose_export_dir)
>>>>>>> 2c70ca6369b91adfb6c25eb7a9ce0298ecafbc3c

    def start_crawl():
        start_url = url_var.get().strip()
        search_query = query_var.get().strip()

        if not start_url or not search_query:
            messagebox.showwarning("Missing Input", "Please provide both URL and query.")
            return

        if not start_url.startswith(("http://", "https://")):
            start_url = "https://" + start_url

        try:
            max_urls_val = int(max_urls_var.get())
        except ValueError:
            messagebox.showwarning("Invalid Input", "Max URLs must be a number.")
            return

        if max_urls_val < 1 or max_urls_val > 200:
            messagebox.showwarning("Invalid Input", "Max URLs must be between 1 and 200.")
            return
        
        # Parse max depth
        max_depth_val = None
        depth_str = max_depth_var.get().strip()
        if depth_str:
            try:
                max_depth_val = int(depth_str)
                if max_depth_val < 0:
                    messagebox.showwarning("Invalid Input", "Max depth must be non-negative.")
                    return
            except ValueError:
                messagebox.showwarning("Invalid Input", "Max depth must be a number or empty.")
                return
        
        # Parse async options
        use_async = use_async_var.get() and ASYNC_AVAILABLE
        try:
            concurrency_val = int(concurrency_var.get())
            if concurrency_val < 1 or concurrency_val > 20:
                messagebox.showwarning("Invalid Input", "Concurrency must be between 1 and 20.")
                return
        except ValueError:
            messagebox.showwarning("Invalid Input", "Concurrency must be a number.")
            return
        
        try:
            delay_val = float(delay_var.get())
            if delay_val < 0:
                messagebox.showwarning("Invalid Input", "Delay must be non-negative.")
                return
        except ValueError:
            messagebox.showwarning("Invalid Input", "Delay must be a number.")
            return
        
        same_domain_val = same_domain_var.get()
        export_format = export_format_var.get()
        log_level = log_level_var.get()

        # Clear previous logs
        log_box.configure(state="normal")
        log_box.delete("1.0", "end")
        log_box.configure(state="disabled")
        set_results_text("")

        update_progress(0, max_urls_val)
        status_var.set("Running...")
        set_inputs_state(False)
        start_button.config(state="disabled")
        stop_button.config(state="normal")
        stop_event_holder["event"] = threading.Event()

        def worker():
            # Setup logger for this thread
            thread_logger = logging.getLogger("WebCrawler")
            thread_logger.setLevel(getattr(logging, log_level))
            
            redirector = TextRedirector(log_box, gui_logger)
            try:
                with redirect_stdout(redirector):
<<<<<<< HEAD
                    if use_async:
                        # Use async crawl in a new event loop
                        visited_urls_data, matching_urls, crawl_metadata = run_async_crawl(
                            start_url,
                            search_query,
                            max_urls=max_urls_val,
                            max_depth=max_depth_val,
                            same_domain_only=same_domain_val,
                            concurrency=concurrency_val,
                            politeness_delay=delay_val,
                            stop_event=stop_event_holder["event"],
                        )
                    else:
                        visited_urls_data, matching_urls, crawl_metadata = crawl(
                            start_url,
                            search_query,
                            max_urls=max_urls_val,
                            max_depth=max_depth_val,
                            same_domain_only=same_domain_val,
                            stop_event=stop_event_holder["event"],
                        )
=======
                    visited_urls_data, matching_urls, crawl_metadata = crawl(
                        start_url,
                        search_query,
                        max_urls=max_urls_val,
                        max_depth=max_depth_val,
                        same_domain_only=same_domain_val,
                        stop_event=stop_event_holder["event"],
                        progress_callback=lambda visited, total: root.after(
                            0, lambda: update_progress(visited, total)
                        ),
                    )
>>>>>>> 2c70ca6369b91adfb6c25eb7a9ce0298ecafbc3c
                    print_results(visited_urls_data, matching_urls, search_query, max_urls=max_urls_val)
                    formatted = format_results_for_gui(
                        visited_urls_data,
                        matching_urls,
                        crawl_metadata,
                    )
                    root.after(0, lambda: set_results_text(formatted))
                    
                    # Export results
                    if export_format in ("json", "both"):
                        filename = None
                        export_dir = export_dir_var.get().strip()
                        if export_dir:
                            timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
                            filename = os.path.join(export_dir, f"crawl_results_{timestamp}.json")
                        export_to_json(visited_urls_data, matching_urls, crawl_metadata, filename)
                    if export_format in ("csv", "both"):
                        filename = None
                        export_dir = export_dir_var.get().strip()
                        if export_dir:
                            timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
                            filename = os.path.join(export_dir, f"crawl_results_{timestamp}.csv")
                        export_to_csv(visited_urls_data, matching_urls, crawl_metadata, filename)
                    
            except Exception as e:
                with redirect_stdout(redirector):
                    print(f"[GUI Error] {e}")
                gui_logger.error(f"GUI Error: {e}", exc_info=True)
            finally:
                def finish_ui():
                    start_button.config(state="normal")
                    stop_button.config(state="disabled")
                    set_inputs_state(True)
                    if stop_event_holder["event"] and stop_event_holder["event"].is_set():
                        status_var.set("Stopped")
                    else:
                        status_var.set("Done")
                root.after(0, finish_ui)

        threading.Thread(target=worker, daemon=True).start()

    def stop_crawl():
        event = stop_event_holder.get("event")
        if event:
            event.set()
        status_var.set("Stopped")
        stop_button.config(state="disabled")

<<<<<<< HEAD
    # Row 6: Buttons
    button_frame = tk.Frame(root)
    button_frame.grid(row=6, column=0, columnspan=3, pady=6, padx=8, sticky="ew")
    
    start_button = tk.Button(button_frame, text="Run Crawl", command=start_crawl, width=15)
    start_button.pack(side=tk.LEFT, padx=5)

    stop_button = tk.Button(button_frame, text="Stop", command=stop_crawl, state="disabled", width=15)
    stop_button.pack(side=tk.LEFT, padx=5)
    
    # Add async status label
    if ASYNC_AVAILABLE:
        status_label = tk.Label(button_frame, text="✓ Async mode available", fg="green")
    else:
        status_label = tk.Label(button_frame, text="⚠ Async not available (install aiohttp)", fg="orange")
    status_label.pack(side=tk.RIGHT, padx=5)
=======
    start_button.configure(command=start_crawl)
    stop_button.configure(command=stop_crawl)
>>>>>>> 2c70ca6369b91adfb6c25eb7a9ce0298ecafbc3c

    url_entry.focus_set()
    root.mainloop()


if __name__ == "__main__":
    main()

