#!/usr/bin/env python3

import os
import sys
import pandas as pd
import numpy as np
from tqdm import tqdm

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from langchain.schema import Document
from langchain_community.vectorstores import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings

def initialize_rag_system():
    print("🔧 Initializing RAG System with Real FinanceBench Data...")
    
    # Paths
    current_dir = os.getcwd()
    questions_file = os.path.join(current_dir, "data", "financebench_open_source.jsonl")
    docs_file = os.path.join(current_dir, "data", "financebench_document_information.jsonl")
    db_dir = os.path.join(current_dir, "db")
    persistent_directory = os.path.join(db_dir, "hackathon3")
    questions_db_dir = os.path.join(db_dir, "question_match_db")
    
    # Create db directory
    os.makedirs(db_dir, exist_ok=True)
    
    print("📊 Loading FinanceBench data...")
    
    # Load data
    questions = pd.read_json(questions_file, lines=True)
    docs_meta = pd.read_json(docs_file, lines=True)
    
    # Merge to get doc metadata in questions
    questions = questions.merge(docs_meta, on="doc_name", how="left")
    
    print(f"✅ Loaded {len(questions)} questions and {len(docs_meta)} documents")
    
    # Build documents from evidence snippets for retrieval store
    print("🔨 Building document chunks...")
    documents = []
    for _, row in tqdm(questions.iterrows(), total=len(questions), desc="Processing questions"):
        company = row.get("company_x")
        doc_name = row.get("doc_name")
        doc_type = row.get("doc_type", "")
        doc_period = row.get("doc_period", None)
        question_id = row.get("financebench_id")
        question_type = row.get("question_type", "")
        evidence_list = row.get("evidence", [])
        
        for evidence in evidence_list:
            snippet = evidence.get("evidence_text", "").strip()
            if not snippet:
                continue
            metadata = {
                "financebench_id": question_id,
                "company": company,
                "doc_name": doc_name,
                "doc_type": doc_type,
                "doc_period": doc_period,
                "question_type": question_type,
                "evidence_doc_name": evidence.get("evidence_doc_name", ""),
                "evidence_page_num": evidence.get("evidence_page_num", -1),
            }
            documents.append(Document(page_content=snippet, metadata=metadata))
    
    print(f"📝 Created {len(documents)} document snippets")
    
    # Split long snippets into chunks
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    docs_chunks = text_splitter.split_documents(documents)
    print(f"✂️ Split into {len(docs_chunks)} document chunks")
    
    # Initialize embedding model
    print("🧠 Loading embedding model...")
    embedding_model = HuggingFaceEmbeddings(model_name="BAAI/bge-small-en-v1.5")
    
    # Create or load retrieval vector store
    if not os.path.exists(persistent_directory):
        print("🏗️ Creating retrieval vector store...")
        retrieval_db = Chroma.from_documents(docs_chunks, embedding_model, persist_directory=persistent_directory)
        retrieval_db.persist()
        print("✅ Retrieval vector store created and persisted")
    else:
        print("📂 Loading existing retrieval vector store...")
        retrieval_db = Chroma(persist_directory=persistent_directory, embedding_function=embedding_model)
        print("✅ Retrieval vector store loaded")
    
    # Create or load question match vector store
    if not os.path.exists(questions_db_dir):
        print("🏗️ Creating question match vector store...")
        question_docs = []
        for _, row in tqdm(questions.iterrows(), total=len(questions), desc="Preparing questions for embedding"):
            question_text = row["question"]
            gold_answer = row["answer"]
            metadata = {
                "financebench_id": row["financebench_id"],
                "gold_answer": gold_answer,
                "company": row.get("company_x", ""),
            }
            question_docs.append(Document(page_content=question_text, metadata=metadata))
        
        questions_db = Chroma.from_documents(question_docs, embedding_model, persist_directory=questions_db_dir)
        questions_db.persist()
        print("✅ Question match vector store created and persisted")
    else:
        print("📂 Loading existing question match vector store...")
        questions_db = Chroma(persist_directory=questions_db_dir, embedding_function=embedding_model)
        print("✅ Question match vector store loaded")
    
    print("\n🎉 RAG System Initialization Complete!")
    print(f"📊 Statistics:")
    print(f"   - Questions loaded: {len(questions)}")
    print(f"   - Documents loaded: {len(docs_meta)}")
    print(f"   - Document chunks: {len(docs_chunks)}")
    print(f"   - Vector stores: 2 (retrieval + question matching)")
    print(f"   - Embedding model: BAAI/bge-small-en-v1.5")
    
    return True

if __name__ == "__main__":
    try:
        initialize_rag_system()
        print("\n✅ RAG system is ready for real data!")
    except Exception as e:
        print(f"❌ Error initializing RAG system: {e}")
        import traceback
        traceback.print_exc() 