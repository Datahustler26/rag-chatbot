#!/usr/bin/env python3
"""Generate test PDFs for RAG chatbot testing."""

import os
from pathlib import Path
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, PageBreak, Spacer
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch

# Sample content from various technical domains
DOMAINS = {
    "machine_learning": """
    Machine Learning is a subset of artificial intelligence that enables systems to learn and improve from experience without being explicitly programmed. 
    It focuses on developing algorithms and statistical models that allow computers to analyze large amounts of data and make predictions or decisions based on patterns.
    
    Deep Learning, a specialized subset of machine learning, uses artificial neural networks with multiple layers (hence "deep") to process data and extract features.
    Convolutional Neural Networks (CNNs) are particularly effective for image processing tasks, while Recurrent Neural Networks (RNNs) excel at sequential data analysis.
    
    Transfer Learning allows models trained on large datasets to be adapted for specific tasks with smaller datasets, significantly reducing training time and data requirements.
    Fine-tuning pre-trained models has become a standard practice in modern machine learning applications.
    """,
    
    "nlp": """
    Natural Language Processing (NLP) is a branch of artificial intelligence that deals with the interaction between computers and human language.
    It involves techniques for text processing, language understanding, and generation.
    
    Transformer models revolutionized NLP by introducing the attention mechanism, which allows models to focus on relevant parts of the input sequence.
    The BERT model (Bidirectional Encoder Representations from Transformers) introduced bidirectional training of transformers, significantly improving performance on various NLP tasks.
    
    Large Language Models (LLMs) like GPT series have demonstrated remarkable capabilities in understanding and generating human language.
    These models are trained on vast amounts of text data and can perform a wide variety of language tasks without task-specific training.
    """,
    
    "databases": """
    Database systems are critical components of modern applications, providing persistent storage and efficient querying of data.
    Relational databases, based on the ACID properties, have been the standard for decades.
    
    Vector databases represent a newer paradigm, optimized for storing and querying high-dimensional vectors.
    They are particularly useful for similarity search and nearest neighbor queries, which are fundamental to machine learning applications.
    
    Qdrant is a vector database that provides fast and accurate similarity search with production-ready infrastructure.
    It uses Hierarchical Navigable Small World (HNSW) graphs for efficient nearest neighbor search.
    """,
    
    "distributed_systems": """
    Distributed systems consist of multiple computers or nodes that work together to achieve a common goal.
    Key challenges include managing consistency, availability, and partition tolerance (CAP theorem).
    
    Consensus algorithms like Raft and Paxos ensure that all nodes agree on a consistent state despite failures.
    These algorithms are fundamental to building reliable distributed systems.
    
    Microservices architecture decomposes applications into smaller, independent services that communicate through APIs.
    This approach provides scalability, flexibility, and allows teams to deploy services independently.
    """,
    
    "web_development": """
    Web development encompasses both frontend and backend technologies for building web applications.
    Frontend technologies like HTML, CSS, and JavaScript provide user interfaces, while backend technologies handle business logic and data management.
    
    FastAPI is a modern web framework for building APIs with Python, featuring automatic validation and documentation.
    It leverages Python type hints and provides high performance comparable to Node.js and Go frameworks.
    
    REST (Representational State Transfer) and GraphQL are popular architectural styles for building web APIs.
    Each has its advantages: REST is simple and widely understood, while GraphQL provides more flexible data querying.
    """,
}

LOREM = """
The advancement of artificial intelligence has opened new possibilities in numerous fields. 
Organizations across industries are leveraging machine learning and deep learning to solve complex problems.
Data-driven decision making has become essential in competitive markets.
The integration of AI into existing systems requires careful consideration of architectural patterns and scalability requirements.
Cloud computing platforms provide the infrastructure necessary for training and deploying large-scale ML models.
"""

def generate_test_pdf(filename: str, domain: str, num_pages: int = 250):
    """Generate a test PDF with substantial content."""
    doc = SimpleDocTemplate(
        filename,
        pagesize=letter,
        rightMargin=0.5 * inch,
        leftMargin=0.5 * inch,
        topMargin=0.5 * inch,
        bottomMargin=0.5 * inch,
    )
    
    styles = getSampleStyleSheet()
    title_style = ParagraphStyle(
        'CustomTitle',
        parent=styles['Heading1'],
        fontSize=24,
        textColor='#1f2937',
        spaceAfter=12,
    )
    
    content_style = ParagraphStyle(
        'CustomBody',
        parent=styles['BodyText'],
        fontSize=11,
        leading=14,
        spaceAfter=10,
    )
    
    # Build document content
    elements = []
    
    # Title
    elements.append(Paragraph(f"Comprehensive Guide to {domain.replace('_', ' ').title()}", title_style))
    elements.append(Spacer(1, 0.3 * inch))
    
    # Add domain-specific content
    if domain in DOMAINS:
        for paragraph_text in DOMAINS[domain].strip().split('\n\n'):
            if paragraph_text.strip():
                elements.append(Paragraph(paragraph_text.strip(), content_style))
                elements.append(Spacer(1, 0.1 * inch))
    
    # Fill remaining pages with Lorem ipsum
    current_page = 1
    while current_page < num_pages:
        # Add Lorem content in paragraphs
        for _ in range(5):
            if current_page >= num_pages:
                break
            elements.append(Paragraph(LOREM * 3, content_style))
            elements.append(Spacer(1, 0.1 * inch))
            current_page += 0.3  # Rough estimate
        
        if current_page < num_pages:
            elements.append(PageBreak())
            current_page += 1
    
    # Build PDF
    doc.build(elements)
    print(f"✓ Generated {filename} ({num_pages} pages)")

def main():
    """Generate test PDFs in the project root."""
    # Create test_pdfs directory
    pdf_dir = Path(__file__).parent.parent / "test_pdfs"
    pdf_dir.mkdir(exist_ok=True)
    
    domains = list(DOMAINS.keys())
    
    # Generate 10+ test PDFs with 200+ pages each
    print("Generating test PDFs for RAG chatbot ingestion testing...\n")
    
    for i, domain in enumerate(domains, 1):
        filename = pdf_dir / f"{i:02d}_{domain}.pdf"
        generate_test_pdf(str(filename), domain, num_pages=250)
    
    # Generate additional PDFs for more comprehensive testing
    extra_domains = ["machine_learning", "nlp", "databases"]
    for i, domain in enumerate(extra_domains, len(domains) + 1):
        filename = pdf_dir / f"{i:02d}_{domain}_advanced.pdf"
        generate_test_pdf(str(filename), domain, num_pages=220)
    
    print(f"\n✓ Generated {len(domains) + len(extra_domains)} test PDFs in {pdf_dir}")
    print(f"  Total PDFs: {len(domains) + len(extra_domains)}")
    print(f"  Each PDF: 220-250 pages")

if __name__ == "__main__":
    main()
