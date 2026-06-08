import requests
from bs4 import BeautifulSoup
import sqlite3
import os
import time
from tqdm import tqdm
from config import RAW_PDF_DIR, DB_PATH

SEBI_BASE = "https://www.sebi.gov.in"
SEBI_FILINGS_URL = "https://www.sebi.gov.in/sebiweb/home/HomeAction.do?doListing=yes&sid=3&ssid=15&smid=10"

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
}

def init_db():
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("""
        CREATE TABLE IF NOT EXISTS filings (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            company_name TEXT,
            filing_type TEXT,
            filing_date TEXT,
            sebi_url TEXT UNIQUE,
            pdf_url TEXT,
            local_path TEXT,
            downloaded INTEGER DEFAULT 0,
            parsed INTEGER DEFAULT 0
        )
    """)
    conn.commit()
    conn.close()
    print("Database initialized.")

def scrape_filings_page(url):
    try:
        resp = requests.get(url, headers=HEADERS, timeout=15)
        soup = BeautifulSoup(resp.text, "lxml")
        filings = []
        for row in soup.select("table tr"):
            cols = row.find_all("td")
            if len(cols) < 3:
                continue
            link_tag = row.find("a", href=True)
            if not link_tag:
                continue
            href = link_tag["href"]
            if "/filings/public-issues/" not in href:
                continue
            full_url = SEBI_BASE + href if href.startswith("/") else href
            company_name = link_tag.text.strip()
            date_text = cols[-1].text.strip() if cols else ""
            filing_type = "DRHP"
            if "udrhp" in href.lower():
                filing_type = "UDRHP"
            elif "rhp" in href.lower():
                filing_type = "RHP"
            filings.append({
                "company_name": company_name,
                "filing_type": filing_type,
                "filing_date": date_text,
                "sebi_url": full_url
            })
        return filings
    except Exception as e:
        print(f"Error scraping {url}: {e}")
        return []

def get_pdf_url(filing_page_url):
    try:
        resp = requests.get(filing_page_url, headers=HEADERS, timeout=15)
        soup = BeautifulSoup(resp.text, "lxml")
        for a in soup.find_all("a", href=True):
            href = a["href"]
            if href.endswith(".pdf") or "attachdocs" in href:
                if href.startswith("/"):
                    return SEBI_BASE + href
                return href
        return None
    except Exception as e:
        print(f"Error getting PDF URL from {filing_page_url}: {e}")
        return None

def download_pdf(pdf_url, company_name):
    try:
        os.makedirs(RAW_PDF_DIR, exist_ok=True)
        safe_name = "".join(c for c in company_name if c.isalnum() or c in " _-")
        safe_name = safe_name.strip().replace(" ", "_")[:50]
        filename = f"{safe_name}.pdf"
        local_path = os.path.join(RAW_PDF_DIR, filename)
        if os.path.exists(local_path):
            return local_path
        resp = requests.get(pdf_url, headers=HEADERS, timeout=60, stream=True)
        if resp.status_code == 200:
            with open(local_path, "wb") as f:
                for chunk in resp.iter_content(chunk_size=8192):
                    f.write(chunk)
            return local_path
        return None
    except Exception as e:
        print(f"Error downloading {pdf_url}: {e}")
        return None

def save_filing(filing):
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    try:
        c.execute("""
            INSERT OR IGNORE INTO filings
            (company_name, filing_type, filing_date, sebi_url, pdf_url, local_path, downloaded)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        """, (
            filing["company_name"],
            filing["filing_type"],
            filing["filing_date"],
            filing["sebi_url"],
            filing.get("pdf_url"),
            filing.get("local_path"),
            1 if filing.get("local_path") else 0
        ))
        conn.commit()
    except Exception as e:
        print(f"DB error: {e}")
    finally:
        conn.close()

def run_scraper(max_filings=10):
    print(f"Starting SEBI scraper (max {max_filings} filings)...")
    init_db()
    filings = scrape_filings_page(SEBI_FILINGS_URL)
    print(f"Found {len(filings)} filings on page.")
    filings = filings[:max_filings]
    for filing in tqdm(filings, desc="Processing filings"):
        pdf_url = get_pdf_url(filing["sebi_url"])
        filing["pdf_url"] = pdf_url
        if pdf_url:
            local_path = download_pdf(pdf_url, filing["company_name"])
            filing["local_path"] = local_path
        save_filing(filing)
        time.sleep(1)
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("SELECT COUNT(*) FROM filings WHERE downloaded=1")
    count = c.fetchone()[0]
    conn.close()
    print(f"\nDone. {count} PDFs downloaded to {RAW_PDF_DIR}/")

if __name__ == "__main__":
    run_scraper(max_filings=10)