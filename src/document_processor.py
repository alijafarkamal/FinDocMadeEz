import fitz
import pandas as pd
import json
from typing import List, Dict, Any
import os
from pathlib import Path

class DocumentProcessor:
    def __init__(self, pdfs_dir: str = "pdfs"):
        self.pdfs_dir = Path(pdfs_dir)
        self.documents_cache = {}
        
    def extract_text_from_pdf(self, pdf_path: str) -> str:
        if pdf_path in self.documents_cache:
            return self.documents_cache[pdf_path]
            
        try:
            doc = fitz.open(pdf_path)
            text = ""
            for page_num in range(len(doc)):
                page = doc[page_num]
                page_text = page.get_text()
                if page_text.strip():  # Only add non-empty pages
                    text += f"\n--- Page {page_num + 1} ---\n"
                    text += page_text
                    text += "\n"
            doc.close()
            
            # Clean up the text
            text = self._clean_text(text)
            
            self.documents_cache[pdf_path] = text
            return text
        except Exception as e:
            print(f"Error processing {pdf_path}: {e}")
            return ""
    
    def _clean_text(self, text: str) -> str:
        """Clean and normalize extracted text"""
        import re
        
        # Remove excessive whitespace
        text = re.sub(r'\s+', ' ', text)
        
        # Remove page headers/footers
        text = re.sub(r'Page \d+ of \d+', '', text)
        
        # Normalize line breaks
        text = text.replace('\r\n', '\n').replace('\r', '\n')
        
        # Remove empty lines
        lines = [line.strip() for line in text.split('\n') if line.strip()]
        text = '\n'.join(lines)
        
        return text
    
    def load_financebench_data(self) -> Dict[str, Any]:
        questions_df = pd.read_json("data/financebench_open_source.jsonl", lines=True)
        meta_df = pd.read_json("data/financebench_document_information.jsonl", lines=True)
        
        full_df = pd.merge(questions_df, meta_df, on="doc_name")
        
        return {
            "questions": questions_df.to_dict('records'),
            "metadata": meta_df.to_dict('records'),
            "full_data": full_df.to_dict('records')
        }
    
    def get_document_text(self, doc_name: str) -> str:
        pdf_path = self.pdfs_dir / f"{doc_name}.pdf"
        if pdf_path.exists():
            return self.extract_text_from_pdf(str(pdf_path))
        return ""
    
    def extract_financial_metrics(self, text: str) -> Dict[str, Any]:
        metrics = {
            "revenue": None,
            "net_income": None,
            "total_assets": None,
            "total_liabilities": None,
            "cash_flow": None,
            "debt": None,
            "equity": None
        }
        
        lines = text.split('\n')
        for line in lines:
            line = line.strip()
            if not line:
                continue
                
            if any(keyword in line.lower() for keyword in ['revenue', 'sales', 'net sales']):
                metrics["revenue"] = self._extract_amount(line)
            elif any(keyword in line.lower() for keyword in ['net income', 'net earnings']):
                metrics["net_income"] = self._extract_amount(line)
            elif any(keyword in line.lower() for keyword in ['total assets']):
                metrics["total_assets"] = self._extract_amount(line)
            elif any(keyword in line.lower() for keyword in ['total liabilities']):
                metrics["total_liabilities"] = self._extract_amount(line)
            elif any(keyword in line.lower() for keyword in ['cash flow', 'cash flows']):
                metrics["cash_flow"] = self._extract_amount(line)
            elif any(keyword in line.lower() for keyword in ['debt', 'long-term debt']):
                metrics["debt"] = self._extract_amount(line)
            elif any(keyword in line.lower() for keyword in ['equity', 'shareholders equity']):
                metrics["equity"] = self._extract_amount(line)
        
        return metrics
    
    def _extract_amount(self, text: str) -> float:
        import re
        amount_pattern = r'\$?([\d,]+\.?\d*)'
        matches = re.findall(amount_pattern, text)
        if matches:
            try:
                return float(matches[0].replace(',', ''))
            except:
                pass
        return None
    
    def get_available_documents(self) -> List[str]:
        return [f.stem for f in self.pdfs_dir.glob("*.pdf")] 