#!/usr/bin/env python3
"""Upload all test PDFs to the RAG chatbot via API."""

import asyncio
import aiohttp
from pathlib import Path
import sys

async def upload_pdf(session, pdf_path: str):
    """Upload a single PDF file."""
    pdf_file = Path(pdf_path)
    if not pdf_file.exists():
        print(f"✗ File not found: {pdf_path}")
        return False
    
    try:
        with open(pdf_file, 'rb') as f:
            form = aiohttp.FormData()
            form.add_field('file', f, filename=pdf_file.name)
            
            async with session.post('http://localhost:8000/api/ingest', data=form) as resp:
                if resp.status == 200:
                    print(f"✓ Uploaded: {pdf_file.name}")
                    return True
                else:
                    print(f"✗ Failed: {pdf_file.name} (status: {resp.status})")
                    text = await resp.text()
                    print(f"  Response: {text[:200]}")
                    return False
    except Exception as e:
        print(f"✗ Error uploading {pdf_file.name}: {e}")
        return False

async def main():
    """Upload all PDFs sequentially."""
    pdf_dir = Path(__file__).parent.parent / "test_pdfs"
    
    # Get all PDFs except the first one (already uploaded)
    pdf_files = sorted(pdf_dir.glob("*.pdf"))[1:]  # Skip first PDF (already uploaded)
    
    if not pdf_files:
        print("No PDFs to upload.")
        return
    
    print(f"Uploading {len(pdf_files)} PDFs...\n")
    
    # Use persistent session for better performance
    connector = aiohttp.TCPConnector(limit=1)  # Limit concurrent connections
    async with aiohttp.ClientSession(connector=connector) as session:
        success_count = 0
        for pdf_file in pdf_files:
            if await upload_pdf(session, str(pdf_file)):
                success_count += 1
            await asyncio.sleep(1)  # Wait between uploads to avoid overwhelming server
    
    print(f"\n✓ Successfully uploaded: {success_count}/{len(pdf_files)} PDFs")

if __name__ == "__main__":
    asyncio.run(main())
