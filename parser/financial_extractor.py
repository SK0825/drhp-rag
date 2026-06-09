import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pdfplumber
import json
from config import RAW_PDF_DIR, CHUNKS_DIR

def extract_tables_from_pdf(pdf_path):
    all_tables = []
    try:
        with pdfplumber.open(pdf_path) as pdf:
            for i, page in enumerate(pdf.pages):
                tables = page.extract_tables()
                for table in tables:
                    if table and len(table) > 1:
                        # convert to markdown
                        rows = []
                        for row in table:
                            cells = [str(cell or "").strip().replace("\n", " ") for cell in row]
                            rows.append(" | ".join(cells))
                        table_text = "\n".join(rows)
                        if any(kw in table_text.lower() for kw in
                               ["revenue", "income", "profit", "loss", "ebitda",
                                "assets", "crore", "million", "lakh", "₹", "rs."]):
                            all_tables.append({
                                "page": i + 1,
                                "text": table_text
                            })
    except Exception as e:
        print(f"Error extracting tables from {pdf_path}: {e}")
    return all_tables

def extract_financials_from_pdf(company_name):
    # find the pdf
    pdf_file = None
    for f in os.listdir(RAW_PDF_DIR):
        if f.lower().replace(".pdf", "").replace(" ", "_") == company_name.lower():
            pdf_file = f
            break
    
    if not pdf_file:
        print(f"PDF not found for {company_name}")
        return {}
    
    pdf_path = os.path.join(RAW_PDF_DIR, pdf_file)
    print(f"Extracting financial tables from {pdf_file}...")
    
    tables = extract_tables_from_pdf(pdf_path)
    print(f"  Found {len(tables)} financial tables")
    
    if not tables:
        return {}
    
    # take the most relevant tables (first 5 financial tables found)
    context = "\n\n---\n\n".join([f"Page {t['page']}:\n{t['text']}" for t in tables[:5]])
    return context

def save_financial_tables(company_name, context):
    path = os.path.join(CHUNKS_DIR, f"{company_name}_financials.txt")
    with open(path, "w", encoding="utf-8") as f:
        f.write(context)
    print(f"  Saved to {path}")

def extract_all():
    pdf_files = [f for f in os.listdir(RAW_PDF_DIR) if f.lower().endswith(".pdf")]
    for pdf_file in pdf_files:
        company_name = os.path.splitext(pdf_file)[0].lower().replace(" ", "_")
        context = extract_financials_from_pdf(company_name)
        if context:
            save_financial_tables(company_name, context)
    print("\nDone extracting financial tables.")

if __name__ == "__main__":
    extract_all()