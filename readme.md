================================================================================
WEB CRAWLER - Application
================================================================================

PROGRAM DESCRIPTION
-------------------
This is a Python web crawler that uses Breadth-First Search (BFS) to visit
25 unique URLs starting from a user-provided seed URL. It searches each page
for a user-specified query and reports which pages contain the search term.


HOW TO RUN THE PROGRAM
----------------------
1. Open a terminal/command prompt
2. Navigate to the folder containing crawler.py
3. Run the command:
   
   python crawler.py

4. When prompted:
   - Enter a starting URL (e.g., https://example.com)
   - Enter a search query (e.g., "privacy" or "contact")

5. The crawler will:
   - Visit up to 25 unique URLs
   - Display progress for each page
   - Show links discovered on each page
   - At the end, display all results


ENVIRONMENT USED
----------------
- Operating System: Windows 10/11
- Python Version: Python 3.8 or higher (tested on Python 3.14)
- IDE: PyCharm


REQUIRED LIBRARIES
------------------
This crawler uses the libraries specified in the assignment:

1. requests
   - Purpose: Fetching web page content via HTTP
   - Install: pip install requests

2. beautifulsoup4 (bs4)
   - Purpose: Parsing HTML and extracting links/text
   - Install: pip install beautifulsoup4

3. urllib.parse (built-in)
   - Purpose: URL parsing and normalization with urljoin
   - No installation needed (part of Python standard library)

4. urllib.robotparser (built-in)
   - Purpose: Parsing robots.txt for crawl permission checking
   - No installation needed (part of Python standard library)

To install all external dependencies at once:
   pip install requests beautifulsoup4


MY APPROACH
-----------

1. URL Frontier System (BFS Queue)
   - Used Python's deque (double-ended queue) for efficient BFS traversal
   - Implemented a URLFrontier class that:
     * Prevents duplicate URLs using a "seen" set
     * Prioritizes same-domain URLs (added to front of queue)
     * Adds external URLs to back of queue (lower priority)
     * Shuffles URLs for variety in crawl pattern

2. Robots.txt Compliance
   - Before visiting any URL, the crawler checks robots.txt
   - Uses urllib.robotparser.RobotFileParser
   - Caches robots.txt per domain to avoid repeated fetches
   - If robots.txt is unavailable, assumes crawling is allowed

3. Page Processing Pipeline
   - Fetch: Uses requests library with timeout and error handling
   - Parse: BeautifulSoup extracts all <a href="..."> links
   - Normalize: urljoin converts relative URLs to absolute URLs
   - Search: Extracts text content and searches for query (case-insensitive)

4. Error Resilience
   - Every network operation is wrapped in try/except
   - Errors are logged but don't stop the crawler
   - Handles: timeouts, connection errors, HTTP errors, parse errors


CHALLENGES AND SOLUTIONS
------------------------

Challenge 1: Duplicate URLs
   Problem: The same URL could appear multiple times on different pages
   Solution: Used a Python set called "seen" to track all discovered URLs.
             New URLs are only added to the queue if not already in the set.

Challenge 2: Relative URLs
   Problem: Links like "/about" or "../contact" are relative, not absolute
   Solution: Used urllib.parse.urljoin(base_url, relative_url) to convert
             all relative URLs to absolute URLs before adding to the queue.

Challenge 3: Robots.txt Performance
   Problem: Fetching robots.txt for every URL would be slow
   Solution: Created a cache (dictionary) that stores RobotFileParser objects
             per domain. Each domain's robots.txt is fetched only once.

Challenge 4: Non-HTML Content
   Problem: Crawler might try to parse PDFs, images, or other non-HTML files
   Solution: Check the Content-Type header before processing. Only process
             pages with "text/html" in the Content-Type.

Challenge 5: URL Fragments and Query Strings
   Problem: URLs like "page.html#section" and "page.html" are the same page
   Solution: Strip URL fragments (the # part) when normalizing URLs to
             prevent visiting the same page multiple times.

Challenge 6: Graceful Error Handling
   Problem: Network errors, timeout, or malformed HTML could crash the program
   Solution: Wrapped all external operations in try/except blocks. Errors are
             logged and the crawler continues to the next URL.


PROGRAM STRUCTURE
-----------------

crawler.py is organized into these sections:

1. CONFIGURATION
   - Constants like MAX_URLS_TO_VISIT and REQUEST_TIMEOUT

2. ROBOTS.TXT COMPLIANCE
   - get_robots_parser(): Fetches and parses robots.txt
   - can_crawl(): Checks if a URL can be crawled

3. URL FRONTIER
   - URLFrontier class: Manages the BFS queue with prioritization

4. PAGE FETCHING AND PARSING
   - fetch_page(): Downloads HTML content
   - extract_links(): Finds all links on a page
   - extract_text(): Gets readable text from HTML
   - search_page(): Checks if query exists on page

5. MAIN CRAWLER LOGIC
   - crawl(): The main BFS crawling loop
   - print_results(): Displays final results

6. USER INPUT
   - get_user_input(): Prompts for URL and query
   - main(): Program entry point


SAMPLE OUTPUT
-------------
When running the crawler, you'll see output like:

[1/25] Visiting: https://example.com
  [Links] Found 12 links on this page
    → https://example.com/about
    → https://example.com/contact
    ... and 10 more

[2/25] Visiting: https://example.com/about
  [Match!] Query 'privacy' found on this page
  [Links] Found 8 links on this page
  ...

At the end:
- List of all 25 visited URLs
- Count of pages containing the query
- List of URLs where the query was found


NOTES
-----

1. From the project root, run python crawler.py.
2. When prompted, provide:
   - A starting URL (for example, `https://example.com`)
   - A search query (for example, privacy)
3. The crawler will visit up to 25 pages, print progress for each visit, and finish with a list of URLs that contained the query text.

GUI Usage
---------

1. From the project root, run python gui.py.
2. Enter the starting URL and search query in the form.
3. Click Start Crawl to begin; progress appears in the output area as pages are visited.
4. Use Stop to end the crawl early and keep the results collected so far.

Tips
----

- Press Ctrl+C to stop the crawl early; partial results collected so far will remain on screen.
- Use small sites while testing to avoid long crawl times or rate limits.

License
-------
- MIT License