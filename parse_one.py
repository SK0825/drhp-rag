import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from parser.pdf_parser import parse_pdf, save_chunks

pdf_path = "data/raw_pdfs/union_bank.pdf"
company_name = "union_bank"

chunks = parse_pdf(pdf_path, company_name)
save_chunks(chunks, company_name)
print(f"Done! Created {len(chunks)} chunks")