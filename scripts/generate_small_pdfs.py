#!/usr/bin/env python3
"""Generate smaller test PDFs to avoid memory issues."""

from pathlib import Path
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, PageBreak, Spacer
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch

DOMAINS = {
    "02_nlp": "Natural Language Processing",
    "03_databases": "Database Systems",
    "04_distributed_systems": "Distributed Systems",
    "05_web_development": "Web Development",
    "06_machine_learning_advanced": "Advanced Machine Learning",
    "07_nlp_advanced": "Advanced NLP",
    "08_databases_advanced": "Advanced Database Topics",
    "09_computer_vision": "Computer Vision",
    "10_cloud_computing": "Cloud Computing",
}

SAMPLE_TEXT = """
This is a comprehensive technical document covering important concepts and best practices. 
The document provides detailed information about modern software development, architecture patterns, and implementation strategies.
Each section builds upon previous knowledge to create a complete understanding of the subject matter.
Key takeaways include scalability, performance optimization, and proper system design principles.
These concepts are fundamental to building robust and maintainable applications.
"""

def generate_small_pdf(filename: str, title: str, num_pages: int = 70):
    """Generate a smaller test PDF."""
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
        'Title',
        parent=styles['Heading1'],
        fontSize=20,
        spaceAfter=12,
    )
    
    body_style = ParagraphStyle(
        'Body',
        parent=styles['BodyText'],
        fontSize=10,
        leading=12,
        spaceAfter=8,
    )
    
    elements = []
    elements.append(Paragraph(title, title_style))
    elements.append(Spacer(1, 0.2 * inch))
    
    # Add repeated content to fill pages
    for i in range(num_pages):
        elements.append(Paragraph(SAMPLE_TEXT, body_style))
        elements.append(Spacer(1, 0.05 * inch))
        if i % 5 == 4:  # Page break every few iterations
            elements.append(PageBreak())
    
    doc.build(elements)
    print(f"✓ Created {filename}")

def main():
    pdf_dir = Path(__file__).parent.parent / "test_pdfs"
    
    print("Creating smaller test PDFs...\n")
    for filename, title in DOMAINS.items():
        pdf_path = pdf_dir / f"{filename}.pdf"
        generate_small_pdf(str(pdf_path), title, num_pages=70)
    
    print(f"\n✓ Created {len(DOMAINS)} smaller PDFs (70 pages each)")

if __name__ == "__main__":
    main()
