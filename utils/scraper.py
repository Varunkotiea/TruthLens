import logging
import re
import requests
from bs4 import BeautifulSoup

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0.0.0 Safari/537.36"
    ),
    "Accept-Language": "en-US,en;q=0.9",
}

REQUEST_TIMEOUT = 15
CONTENT_CLASS_HINTS = [
    "article", "article-body", "article-content", "article__body",
    "story-body", "entry-content", "post-content", "content-body",
    "main-content", "body-copy", "td-post-content",
]


def _extract_with_bs4(soup: BeautifulSoup) -> str:
    for tag in soup(["script", "style", "nav", "footer", "header",
                     "aside", "form", "iframe", "noscript"]):
        tag.decompose()

    for cls in CONTENT_CLASS_HINTS:
        container = soup.find(class_=re.compile(cls, re.I))
        if container:
            text = container.get_text(separator=" ", strip=True)
            if len(text) > 200:
                return text

    article = soup.find("article")
    if article:
        return article.get_text(separator=" ", strip=True)

    paragraphs = soup.find_all("p")
    return " ".join(p.get_text(strip=True) for p in paragraphs)


def _extract_with_newspaper(url: str) -> str:
    try:
        from newspaper import Article
        art = Article(url)
        art.download()
        art.parse()
        return art.text
    except Exception as e:
        logging.warning(f"newspaper3k fallback failed: {e}")
        return ""


def get_article_text(url: str) -> dict:
    result = {"title": "", "text": "", "url": url, "error": None}

    if not url.startswith(("http://", "https://")):
        result["error"] = "Invalid URL. Please include http:// or https://"
        return result

    try:
        response = requests.get(url, headers=HEADERS, timeout=REQUEST_TIMEOUT)
        response.raise_for_status()
    except requests.exceptions.Timeout:
        result["error"] = "Request timed out. The website took too long to respond."
        return result
    except requests.exceptions.ConnectionError:
        result["error"] = "Could not connect to the URL. Check if the site is reachable."
        return result
    except requests.exceptions.HTTPError as e:
        result["error"] = f"HTTP error {e.response.status_code}: {e}"
        return result
    except Exception as e:
        result["error"] = f"Unexpected error fetching URL: {e}"
        return result

    soup = BeautifulSoup(response.text, "lxml")
    
    title_tag = soup.find("title")
    og_title = soup.find("meta", property="og:title")
    result["title"] = (
        og_title["content"] if og_title and og_title.get("content")
        else title_tag.get_text(strip=True) if title_tag
        else "Untitled"
    )

    text = _extract_with_bs4(soup)

    if len(text.split()) < 50:
        text = _extract_with_newspaper(url)

    if len(text.split()) < 30:
        result["error"] = (
            "Could not extract enough article text from this URL. "
            "The page may be paywalled, JavaScript-rendered, or block scrapers."
        )
        return result

    result["text"] = text
    return result