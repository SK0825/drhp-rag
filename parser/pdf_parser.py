import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pdfplumber
import fitz
import json
import re
from tqdm import tqdm
from config import RAW_PDF_DIR, PARSED_DIR, CHUNKS_DIR, CHUNK_SIZE, CHUNK_OVERLAP

SEBI_SECTIONS = [
    "risk factors",
    "introduction",
    "summary",
    "objects of the issue",
    "basis for issue price",
    "statement of tax benefits",
    "industry overview",
    "business overview",
    "regulations and policies",
    "history and corporate structure",
    "management",
    "promoters",
    "related party transactions",
    "dividend policy",
    "financial statements",
    "financial information",
    "management discussion and analysis",
    "outstanding litigation",
    "government approvals",
    "other regulatory disclosures",
    "issue structure",
    "terms of the issue",
    "restrictions on foreign ownership",
]

def detect_section(text):
    text_lower = text.lower().strip()
    for section in SEBI_SECTIONS:
        if text_lower.startswith(section) or section in text_lower[:80]:
            return section.title()
    return None

def extract_with_pdfplumber(pdf_path):
    pages = []
    try:
        with pdfplumber.open(pdf_path) as pdf:
            for i, page in enumerate(pdf.pages):
                page_data = {"page_num": i + 1, "text": "", "tables": []}
                text = page.extract_text()
                if text:
                    page_data["text"] = text.strip()
                tables = page.extract_tables()
                for table in tables:
                    if table:
                        md_table = table_to_markdown(table)
                        page_data["tables"].append(md_table)
                pages.append(page_data)
    except Exception as e:
        print(f"pdfplumber error on {pdf_path}: {e}")
    return pages

def table_to_markdown(table):
    if not table or not table[0]:
        return ""
    rows = []
    header = [str(cell or "").strip() for cell in table[0]]
    rows.append("| " + " | ".join(header) + " |")
    rows.append("| " + " | ".join(["---"] * len(header)) + " |")
    for row in table[1:]:
        cells = [str(cell or "").strip().replace("\n", " ") for cell in row]
        rows.append("| " + " | ".join(cells) + " |")
    return "\n".join(rows)

def chunk_text(text, chunk_size=CHUNK_SIZE, overlap=CHUNK_OVERLAP):
    words = text.split()
    chunks = []
    start = 0
    while start < len(words):
        end = start + chunk_size
        chunk = " ".join(words[start:end])
        chunks.append(chunk)
        start += chunk_size - overlap
    return chunks

def parse_pdf(pdf_path, company_name):
    print(f"\nParsing: {company_name}")
    pages = extract_with_pdfplumber(pdf_path)
    print(f"  Extracted {len(pages)} pages")

    current_section = "General"
    all_chunks = []
    full_text_by_section = {}

    for page in pages:
        text = page["text"]
        if not text:
            continue
        detected = detect_section(text)
        if detected:
            current_section = detected

        if current_section not in full_text_by_section:
            full_text_by_section[current_section] = []
        full_text_by_section[current_section].append(text)

        for table_md in page["tables"]:
            if table_md:
                full_text_by_section[current_section].append(table_md)

    chunk_id = 0
    for section, texts in full_text_by_section.items():
        combined = " ".join(texts)
        chunks = chunk_text(combined)
        for chunk in chunks:
            if len(chunk.strip()) < 50:
                continue
            all_chunks.append({
                "chunk_id": f"{company_name}_{chunk_id}",
                "company": company_name,
                "section": section,
                "text": chunk,
                "word_count": len(chunk.split())
            })
            chunk_id += 1

    print(f"  Created {len(all_chunks)} chunks across {len(full_text_by_section)} sections")
    print(f"  Sections found: {list(full_text_by_section.keys())[:8]}")
    return all_chunks

def save_chunks(chunks, company_name):
    os.makedirs(CHUNKS_DIR, exist_ok=True)
    out_path = os.path.join(CHUNKS_DIR, f"{company_name}_chunks.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(chunks, f, indent=2, ensure_ascii=False)
    print(f"  Saved to {out_path}")
    return out_path

def parse_all_pdfs():
    os.makedirs(PARSED_DIR, exist_ok=True)
    pdf_files = [f for f in os.listdir(RAW_PDF_DIR) if f.lower().endswith(".pdf")]
    print(f"Found {len(pdf_files)} PDFs to parse")

    all_stats = []
    for pdf_file in tqdm(pdf_files, desc="Parsing PDFs"):
        company_name = os.path.splitext(pdf_file)[0].lower().replace(" ", "_")
        pdf_path = os.path.join(RAW_PDF_DIR, pdf_file)
        chunks = parse_pdf(pdf_path, company_name)
        save_chunks(chunks, company_name)
        all_stats.append({
            "company": company_name,
            "chunks": len(chunks)
        })

    print("\n=== Parsing Summary ===")
    total = 0
    for s in all_stats:
        print(f"  {s['company']}: {s['chunks']} chunks")
        total += s['chunks']
    print(f"  Total chunks: {total}")

if __name__ == "__main__":
    parse_all_pdfs()