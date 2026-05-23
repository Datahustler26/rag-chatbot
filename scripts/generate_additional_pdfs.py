#!/usr/bin/env python3
"""Generate additional test PDFs."""

import os
from pathlib import Path
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, PageBreak, Spacer
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch

ADDITIONAL_CONTENT = {
    "computer_vision": """
    Computer Vision is a field of artificial intelligence that trains computers to interpret and understand the visual world.
    Using digital images from cameras and videos and deep learning models, machines can accurately identify and classify objects.
    
    Object detection, semantic segmentation, and instance segmentation are fundamental computer vision tasks.
    State-of-the-art models like YOLO, Faster R-CNN, and Mask R-CNN have revolutionized the field.
    
    Image classification networks like ResNet, VGG, and EfficientNet have demonstrated remarkable performance.
    Transfer learning from these pre-trained models has become standard practice.
    """,
    
    "cloud_computing": """
    Cloud computing provides on-demand access to computing resources over the internet.
    Major providers like AWS, Azure, and Google Cloud offer comprehensive services for compute, storage, and databases.
    
    Containerization technologies like Docker and orchestration platforms like Kubernetes have transformed cloud deployments.
    Microservices and serverless architectures enable scalable and cost-effective applications.
    
    Infrastructure as Code (IaC) tools like Terraform and CloudFormation automate resource provisioning.
    This enables reproducible, version-controlled infrastructure management.
    """,
}

LOREM = """
The rapid evolution of technology continues to create new opportunities and challenges.
Organizations must balance innovation with stability and security.
Data privacy and regulatory compliance have become critical considerations in modern software development.
The integration of AI and machine learning into existing systems requires careful planning and execution.
Cloud-native architectures are becoming the standard for new application development.
"""

def generate_test_pdf(filename: str, domain: str, content: str, num_pages: int = 240):
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
    for paragraph_text in content.strip().split('\n\n'):
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
    """Generate additional test PDFs."""
    pdf_dir = Path(__file__).parent.parent / "test_pdfs"
    
    print("Generating additional test PDFs...\n")
    
    for i, (domain, content) in enumerate(ADDITIONAL_CONTENT.items(), 9):
        filename = pdf_dir / f"{i:02d}_{domain}.pdf"
        generate_test_pdf(str(filename), domain, content, num_pages=230)
    
    print(f"\n✓ Total test PDFs now: 10+")

if __name__ == "__main__":
    main()
