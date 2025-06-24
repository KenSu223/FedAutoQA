# scraper.py
"""
scraper.py
----------
Download the HCUP FAQ page and extract (question, answer, url_anchor) tuples.
Writes `faq_pairs.jsonl`, one JSON object per line:
{ "id": "...", "question": "...", "answer": "...", "url": "..." }
"""

import json, uuid, requests
from bs4 import BeautifulSoup
from pathlib import Path
from urllib.parse import urljoin
import re

def slugify(text: str) -> str:
    # convert to lowercase, replace non-alphanumerics with underscores
    slug = re.sub(r'[^a-z0-9]+', '_', text.lower())
    return slug.strip('_')

FAQ_URL = "https://hcup-us.ahrq.gov/tech_assist/faq.jsp"
OUT_FILE = Path("faq_pairs.jsonl")

def fetch_html(url: str) -> str:
    print(f"Fetching {url} …")
    resp = requests.get(url, timeout=30)
    resp.raise_for_status()
    html = resp.text
    # Write raw HTML to file for debugging
    debug_file = Path("debug_fetched.html")
    debug_file.write_text(html, encoding="utf-8")
    print(f"Wrote fetched HTML to {debug_file}")
    return html

def parse_pairs(html: str, base_url: str):
    """
    Extract FAQ (question, answer) pairs.

    Heuristic:
    * A question lives in a <li> whose text ends with '?' (often inside an <h3>).
    * The answer is every subsequent sibling element—ignoring whitespace strings—
      until the next <li> is reached.
    """
    soup = BeautifulSoup(html, "html.parser")

    # Locate the main FAQ section
    header = soup.find(lambda tag: tag.name in ["h1", "h2", "h3"] and "Frequently Asked Questions" in tag.get_text())
    faq_list = header.find_next("ul") if header else soup

    for li in faq_list.find_all("li"):
        # Only consider list items with a <strong> question element
        strong = li.find("strong")
        if not strong:
            continue
        question = strong.get_text(strip=True)

        # Everything after the <strong> in this same <li> is part of the answer
        answer_parts = []
        for sib in strong.next_siblings:
            if isinstance(sib, str):
                text = sib.strip()
                if text:
                    answer_parts.append(text)
            else:
                answer_parts.append(sib.get_text(" ", strip=True))
        answer = " ".join(answer_parts).strip()
        if not answer:
            continue

        # Generate a stable fragment or fallback to a UUID
        frag = li.get("id", uuid.uuid4().hex)
        url = f"{base_url}#{frag}"

        yield {
            "id": frag,
            "question": question,
            "answer": answer,
            "url": url,
        }

def parse_text_pairs(text: str):
    """
    Parse plain-text FAQ into (question, answer) tuples.
    Assumes questions end with '?' on their own line.
    """
    lines = text.splitlines()
    question = None
    answer_lines = []
    for line in lines:
        stripped = line.strip()
        if not stripped:
            continue
        # start of a question
        if stripped.endswith('?'):
            if question:
                yield question, ' '.join(answer_lines).strip()
            question = stripped
            answer_lines = []
        elif question:
            answer_lines.append(stripped)
    if question:
        yield question, ' '.join(answer_lines).strip()

def parse_text_file(path: Path):
    text = path.read_text(encoding="utf-8")
    for question, answer in parse_text_pairs(text):
        frag = slugify(question)
        yield {
            "id": frag or uuid.uuid4().hex,
            "question": question,
            "answer": answer,
            "url": ""
        }

def main():
    text_path = Path("QA.txt")
    with OUT_FILE.open("w", encoding="utf-8") as fp:
        if text_path.exists():
            for obj in parse_text_file(text_path):
                json.dump(obj, fp, ensure_ascii=False)
                fp.write("\n")
        else:
            html = fetch_html(FAQ_URL)
            for obj in parse_pairs(html, FAQ_URL):
                json.dump(obj, fp, ensure_ascii=False)
                fp.write("\n")
    print(f"Wrote {OUT_FILE}")

if __name__ == "__main__":
    main()
