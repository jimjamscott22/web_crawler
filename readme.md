Web Crawler
===========

This project is a console-based Python crawler that performs a breadth-first crawl starting from a user-supplied seed URL. It visits up to 25 unique pages, obeys each site's robots.txt rules, searches the body text of every page for a user-provided query, and prints both the crawl progress and the matches it finds.

Features
--------

- Breadth-first traversal that prioritizes in-domain links to keep the crawl relevant
- robots.txt compliance with caching so domains are not requested repeatedly
- Link extraction, normalization, and deduplication to avoid revisiting pages
- Graceful error handling for timeouts, HTTP errors, and invalid pages
- Summary report showing which URLs contained the search term

Requirements
------------

- Python 3.8+ (tested on Windows 11 with Python 3.11; compatibility with Python 3.14 has not yet been verified)
- requests and beautifulsoup4 (see requirements.txt for the authoritative list)
- Tkinter is part of the standard library on Windows; on Linux, you may need to install your distro's Tk package (often named python3-tk) to use the GUI.

Setup
-----

1. Clone or download this repository.
2. (Optional but recommended) Create and activate a virtual environment:
   - python -m venv .venv
   - Windows PowerShell: .venv\Scripts\Activate.ps1
3. Install dependencies:
   - pip install -r requirements.txt

Usage
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