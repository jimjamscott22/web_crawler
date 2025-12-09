"""
Web Crawler Assignment
======================
A BFS-based web crawler that visits 25 unique URLs, respects robots.txt,
and searches pages for a user-provided query.

Libraries used:
- requests: For fetching web pages
- BeautifulSoup (bs4): For parsing HTML and extracting links
- urllib.parse: For URL normalization and joining
- urllib.robotparser: For robots.txt compliance
"""

import random
import sys
import threading
from contextlib import redirect_stdout
from collections import deque
from urllib.parse import urlparse, urljoin
from urllib.robotparser import RobotFileParser

import requests
from bs4 import BeautifulSoup


# ============================================================================
# CONFIGURATION
# ============================================================================

MAX_URLS_TO_VISIT = 25  # Total unique URLs to crawl
REQUEST_TIMEOUT = 10     # Seconds to wait for a response
USER_AGENT = "StudentWebCrawler/1.0"  # Identifies our crawler to websites


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
        print(f"  [Warning] Could not fetch robots.txt for {base_url}: {e}")
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
    """
    
    def __init__(self, start_url):
        """
        Initialize the frontier with a starting URL.
        
        Args:
            start_url: The seed URL to begin crawling from
        """
        self.start_domain = urlparse(start_url).netloc
        self.queue = deque([start_url])  # BFS queue
        self.seen = {start_url}  # Set of all URLs we've seen (to prevent duplicates)
    
    def add_urls(self, urls):
        """
        Add new URLs to the frontier with prioritization.
        
        Same-domain URLs are added to the front (higher priority).
        External URLs are added to the back (lower priority).
        URLs are shuffled within their priority group for variety.
        
        Args:
            urls: List of URLs to add
        """
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
                external.append(url)
        
        # Shuffle for variety (prevents always visiting links in page order)
        random.shuffle(same_domain)
        random.shuffle(external)
        
        # Mark all as seen
        for url in new_urls:
            self.seen.add(url)
        
        # Add same-domain URLs to front (higher priority)
        # We use extendleft with reversed to maintain shuffle order
        for url in reversed(same_domain):
            self.queue.appendleft(url)
        
        # Add external URLs to back (lower priority)
        self.queue.extend(external)
    
    def get_next(self):
        """
        Get the next URL to visit.
        
        Returns:
            The next URL, or None if the queue is empty
        """
        if self.queue:
            return self.queue.popleft()
        return None
    
    def is_empty(self):
        """Check if there are no more URLs to visit."""
        return len(self.queue) == 0


# ============================================================================
# PAGE FETCHING AND PARSING
# ============================================================================

def fetch_page(url):
    """
    Fetch a web page's HTML content.
    
    Args:
        url: The URL to fetch
    
    Returns:
        Tuple of (html_content, error_message)
        - On success: (html_string, None)
        - On failure: (None, error_description)
    """
    try:
        headers = {"User-Agent": USER_AGENT}
        response = requests.get(url, headers=headers, timeout=REQUEST_TIMEOUT)
        
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
        print(f"  [Warning] Error parsing links: {e}")
    
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

def crawl(start_url, search_query, max_urls=MAX_URLS_TO_VISIT, stop_event=None):
    """
    Main crawling function using BFS traversal.
    
    Args:
        start_url: The seed URL to start crawling from
        search_query: The term to search for on each page
        max_urls: Maximum number of unique URLs to visit
        stop_event: Optional threading.Event to request a stop
    
    Returns:
        Tuple of (visited_urls, matching_urls)
    """
    print("\n" + "=" * 60)
    print("STARTING WEB CRAWLER")
    print("=" * 60)
    print(f"Start URL: {start_url}")
    print(f"Search Query: '{search_query}'")
    print(f"Target: {max_urls} unique URLs")
    print("=" * 60 + "\n")
    
    # Initialize data structures
    frontier = URLFrontier(start_url)
    robots_cache = {}  # Cache robots.txt parsers per domain
    visited_urls = []  # URLs we've successfully visited
    matching_urls = []  # URLs where the query was found
    
    # Main crawling loop
    while len(visited_urls) < max_urls and not frontier.is_empty():
        if stop_event and stop_event.is_set():
            print("  [Stopped] Crawl cancelled by user.")
            break
        # Get next URL from the frontier
        current_url = frontier.get_next()
        
        if current_url is None:
            break
        
        url_number = len(visited_urls) + 1
        print(f"\n[{url_number}/{max_urls}] Visiting: {current_url}")
        
        # Check robots.txt compliance
        if not can_crawl(current_url, robots_cache):
            print("  [Skipped] Blocked by robots.txt")
            continue
        
        # Fetch the page
        html, error = fetch_page(current_url)
        
        if error:
            print(f"  [Error] {error}")
            # Still count as visited (we attempted it)
            visited_urls.append(current_url)
            continue
        
        # Successfully fetched - add to visited list
        visited_urls.append(current_url)
        
        # Search for the query
        if search_page(html, search_query):
            matching_urls.append(current_url)
            print(f"  [Match!] Query '{search_query}' found on this page")
        
        # Extract links from the page
        links = extract_links(html, current_url)
        print(f"  [Links] Found {len(links)} links on this page")
        
        # Display discovered links (first 5 for brevity)
        if links:
            display_count = min(5, len(links))
            for link in links[:display_count]:
                print(f"    → {link}")
            if len(links) > display_count:
                print(f"    ... and {len(links) - display_count} more")
        
        # Add new links to the frontier
        frontier.add_urls(links)
    
    return visited_urls, matching_urls


def print_results(visited_urls, matching_urls, search_query, max_urls=MAX_URLS_TO_VISIT):
    """
    Print the final crawling results.
    
    Args:
        visited_urls: List of all URLs that were visited
        matching_urls: List of URLs where the query was found
        search_query: The search term used
        max_urls: Target number of URLs for this crawl
    """
    print("\n" + "=" * 60)
    print("CRAWL RESULTS")
    print("=" * 60)
    
    # Print all visited URLs
    print(f"\n📋 ALL VISITED URLs ({len(visited_urls)} of target {max_urls}):")
    print("-" * 40)
    for i, url in enumerate(visited_urls, 1):
        print(f"  {i:2}. {url}")
    
    # Print search results
    print(f"\n🔍 SEARCH RESULTS for '{search_query}':")
    print("-" * 40)
    print(f"  Pages containing the query: {len(matching_urls)}")
    
    if matching_urls:
        print(f"\n  URLs with matches:")
        for i, url in enumerate(matching_urls, 1):
            print(f"    {i}. {url}")
    else:
        print(f"\n  No pages contained the query '{search_query}'")
    
    print("\n" + "=" * 60)
    print("CRAWL COMPLETE")
    print("=" * 60)


# ============================================================================
# USER INPUT AND PROGRAM ENTRY
# ============================================================================

def get_user_input():
    """
    Get the starting URL and search query from the user.
    
    Returns:
        Tuple of (start_url, search_query)
    """
    print("\n" + "=" * 60)
    print("WEB CRAWLER Application")
    print("=" * 60)
    
    # Get starting URL
    while True:
        start_url = input("\nEnter the starting URL (e.g., https://example.com): ").strip()
        
        if not start_url:
            print("  Please enter a valid URL.")
            continue
        
        # Add https:// if no scheme provided
        if not start_url.startswith(("http://", "https://")):
            start_url = "https://" + start_url
        
        # Basic URL validation
        parsed = urlparse(start_url)
        if parsed.scheme and parsed.netloc:
            break
        else:
            print("  Invalid URL format. Please try again.")
    
    # Get search query
    while True:
        search_query = input("Enter the search query: ").strip()
        
        if search_query:
            break
        else:
            print("  Please enter a search term.")
    
    return start_url, search_query


def main():
    """
    Main entry point for the web crawler program.
    """
    try:
        # Get user input
        start_url, search_query = get_user_input()
        
        # Run the crawler
        visited_urls, matching_urls = crawl(start_url, search_query)
        
        # Print results
        print_results(visited_urls, matching_urls, search_query)
        
    except KeyboardInterrupt:
        print("\n\n[Interrupted] Crawler stopped by user.")
    except Exception as e:
        print(f"\n[Critical Error] An unexpected error occurred: {e}")
        print("The crawler has stopped.")


# ============================================================================ 
# OPTIONAL: TKINTER GUI WRAPPER 
# ============================================================================ 


class TextRedirector:
    """
    Redirects stdout to a Tkinter Text/ScrolledText widget in a thread-safe way.
    """

    def __init__(self, widget):
        self.widget = widget

    def write(self, message):
        if not message:
            return
        # Ensure UI updates happen on the main thread
        self.widget.after(0, lambda: self._append(message))

    def flush(self):
        # No-op needed for file-like API
        pass

    def _append(self, message):
        self.widget.insert("end", message)
        self.widget.see("end")


def run_gui():
    """
    Minimal Tkinter GUI wrapper around the existing crawler.
    """
    import tkinter as tk
    from tkinter import messagebox, scrolledtext

    root = tk.Tk()
    root.title("Web Crawler (25 URLs, BFS)")

    url_var = tk.StringVar()
    query_var = tk.StringVar()
    max_urls_var = tk.StringVar(value=str(MAX_URLS_TO_VISIT))
    stop_event_holder = {"event": None}

    tk.Label(root, text="Starting URL:").grid(row=0, column=0, sticky="w", padx=8, pady=4)
    url_entry = tk.Entry(root, textvariable=url_var, width=70)
    url_entry.grid(row=0, column=1, sticky="we", padx=8, pady=4)

    tk.Label(root, text="Search Query:").grid(row=1, column=0, sticky="w", padx=8, pady=4)
    query_entry = tk.Entry(root, textvariable=query_var, width=40)
    query_entry.grid(row=1, column=1, sticky="we", padx=8, pady=4)

    tk.Label(root, text="Max URLs (1-200):").grid(row=2, column=0, sticky="w", padx=8, pady=4)
    max_urls_entry = tk.Entry(root, textvariable=max_urls_var, width=10)
    max_urls_entry.grid(row=2, column=1, sticky="w", padx=8, pady=4)

    log_box = scrolledtext.ScrolledText(root, width=100, height=30, state="normal")
    log_box.grid(row=4, column=0, columnspan=2, padx=8, pady=8, sticky="nsew")

    # Grid stretch
    root.grid_columnconfigure(1, weight=1)
    root.grid_rowconfigure(4, weight=1)

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

        # Clear previous logs
        log_box.delete("1.0", "end")

        # Disable button during crawl
        start_button.config(state="disabled")
        stop_button.config(state="normal")
        stop_event_holder["event"] = threading.Event()

        def worker():
            logger = TextRedirector(log_box)
            try:
                with redirect_stdout(logger):
                    visited_urls, matching_urls = crawl(
                        start_url,
                        search_query,
                        max_urls=max_urls_val,
                        stop_event=stop_event_holder["event"],
                    )
                    print_results(visited_urls, matching_urls, search_query, max_urls=max_urls_val)
            except Exception as e:
                with redirect_stdout(logger):
                    print(f"[GUI Error] {e}")
            finally:
                # Re-enable button on UI thread
                root.after(0, lambda: (start_button.config(state="normal"), stop_button.config(state="disabled")))

        threading.Thread(target=worker, daemon=True).start()

    def stop_crawl():
        event = stop_event_holder.get("event")
        if event:
            event.set()
        stop_button.config(state="disabled")

    start_button = tk.Button(root, text="Run Crawl", command=start_crawl)
    start_button.grid(row=3, column=0, pady=6, padx=8, sticky="w")

    stop_button = tk.Button(root, text="Stop", command=stop_crawl, state="disabled")
    stop_button.grid(row=3, column=1, pady=6, padx=8, sticky="e")

    url_entry.focus_set()
    root.mainloop()


if __name__ == "__main__":
    # Run GUI if "--gui" is passed, otherwise default to CLI.
    if "--gui" in sys.argv:
        run_gui()
    else:
        main()

