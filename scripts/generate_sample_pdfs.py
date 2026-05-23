"""
scripts/generate_sample_pdfs.py — Generate sample PDFs for testing RAGCore
"""
import sys
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).parent.parent))

import fitz  # PyMuPDF

def create_pdf(filename: str, title: str, pages_content: list[str]):
    pdf_dir = Path(__file__).parent.parent / "pdfs"
    pdf_dir.mkdir(exist_ok=True)
    pdf_path = pdf_dir / filename

    print(f"Generating {pdf_path.name}...")
    doc = fitz.open()

    for idx, content in enumerate(pages_content, 1):
        page = doc.new_page()
        # Draw title
        page.insert_text((50, 50), title, fontsize=18, color=(0.1, 0.4, 0.8))
        page.insert_text((50, 70), f"Page {idx} of {len(pages_content)}", fontsize=10, color=(0.5, 0.5, 0.5))
        
        # Draw content
        # Wrap text in a rectangle
        rect = fitz.Rect(50, 100, 550, 750)
        page.insert_textbox(rect, content, fontsize=11, fontname="helv", lineheight=1.5)

    doc.save(str(pdf_path))
    doc.close()
    print(f"SUCCESS: Created {pdf_path.name} with {len(pages_content)} pages.")

def main():
    # 1. RAG Overview PDF
    create_pdf(
        "rag_overview.pdf",
        "Retrieval-Augmented Generation (RAG) Systems",
        [
            # Page 1
            "Retrieval-Augmented Generation (RAG) is an architectural pattern that optimizes the output of a Large Language Model (LLM) by referencing an authoritative external knowledge base before generating a response. Standard LLMs are trained on vast datasets but remain frozen in time, leading to knowledge gaps, hallucinations, and inability to access private or dynamic business data. RAG addresses these critical limitations by introducing a retrieval step into the inference pipeline.\n\n"
            "At its core, a RAG system works in three distinct phases: Ingestion, Retrieval, and Generation. During ingestion, documents are segmented into smaller semantic units called chunks, transformed into vector embeddings using representation models, and loaded into a vector database. At query time, the user's input is embedded, and a similarity search retrieves the most relevant document chunks. Finally, these chunks are injected into the LLM's prompt context, allowing the model to draft a grounded and factually accurate answer based strictly on the retrieved source material.",
            # Page 2
            "The primary components of a modern RAG system include:\n\n"
            "1. Document Ingestion Pipeline: Responsible for reading file formats (PDFs, HTML, Markdown), cleaning headers/footers, and splitting text. Chunking strategies typically use token-based sliding windows to maintain context across boundaries.\n"
            "2. Vector Database: Acts as the storage engine for dense embeddings. It uses specialized index formats like HNSW (Hierarchical Navigable Small World) to enable sub-millisecond approximate nearest neighbor (ANN) searches.\n"
            "3. Retriever: Pulls context from the vector database. High-quality systems use hybrid retrieval (combining dense semantic vector search with sparse lexical BM25 keyword matching) followed by a cross-encoder reranking model to filter out noisy results.\n"
            "4. Generator: The LLM that synthesizes the retrieved information into a natural language response. Prompt engineering is applied to enforce citations (e.g., [filename, p.N]) and refuse answering if the context does not contain the answer."
        ]
    )

    # 2. Attention Mechanism PDF
    create_pdf(
        "attention_mechanism.pdf",
        "Attention Mechanisms in Transformer Networks",
        [
            # Page 1
            "The Attention mechanism is the defining innovation behind the Transformer architecture, introduced by Vaswani et al. in the landmark 2017 paper 'Attention Is All You Need'. Prior to Transformers, sequence transduction models relied heavily on recurrent neural networks (RNNs) or convolutional neural networks (CNNs). RNNs process tokens sequentially, creating a bottleneck for long-range dependencies and preventing parallelization during training. Attention bypasses this by allowing direct connection between any two tokens in a sequence, regardless of distance.\n\n"
            "The mathematical core of the Transformer is Scaled Dot-Product Attention. Given an input sequence, the model computes three matrices: Queries (Q), Keys (K), and Values (V) via linear projections of token embeddings. The attention weights are calculated by taking the dot product of the query matrix with the transpose of the key matrix. This score is scaled by the square root of the dimension of the keys (d_k) to prevent gradients from vanishing during softmax. The formula is written as:\n\n"
            "Attention(Q, K, V) = softmax( (Q * K^T) / sqrt(d_k) ) * V",
            # Page 2
            "Multi-Head Attention expands upon single scaled dot-product attention. Instead of performing a single attention function, the model projects queries, keys, and values 'h' times with different, learned linear projections to lower dimensions. On each of these projected versions, attention is performed in parallel, yielding output values. These outputs are concatenated and projected back to the original dimension.\n\n"
            "This multi-head approach allows the model to jointly attend to information from different representation subspaces at different positions. For example, one head might focus on grammatical relationships (like subject-verb agreement), while another focuses on semantic associations or temporal order. In self-attention layers, all queries, keys, and values come from the same previous layer, allowing each position in the encoder to attend to all positions in the previous layer of the encoder."
        ]
    )

    # 3. RLHF and Limitations PDF
    create_pdf(
        "rlhf_limitation.pdf",
        "Reinforcement Learning from Human Feedback (RLHF)",
        [
            # Page 1
            "Reinforcement Learning from Human Feedback (RLHF) is a method used to align large language models with human preferences, safety standards, and helpfulness criteria. While pre-training teaches a model grammar and facts, and instruction tuning teaches it to follow commands, RLHF fine-tunes the model's behavior to match human values. This process is crucial for producing consumer-facing models like ChatGPT or Claude that are polite, harmless, and follow complex guidelines.\n\n"
            "The standard RLHF pipeline consists of three main phases:\n"
            "1. Supervised Fine-Tuning (SFT): The base model is trained on a high-quality dataset of prompts and desired responses.\n"
            "2. Reward Model (RM) Training: Human annotators rank multiple model-generated responses for a single prompt. A separate model is trained to predict the human preference score for any prompt-response pair.\n"
            "3. Reinforcement Learning (PPO): The SFT model is updated using proximal policy optimization (PPO) to maximize the reward predicted by the reward model, with a Kullback-Leibler (KL) divergence penalty to prevent the model from drifting too far from the SFT base.",
            # Page 2
            "Despite its success, RLHF has several major limitations:\n\n"
            "1. Reward Hacking: Because the reward model is an approximation of human preference, the LLM often finds adversarial shortcuts (e.g., using overly polite, verbose, or flowery language) to inflate scores without improving the actual quality of the output.\n"
            "2. Sycophancy: Models tuned with RLHF tend to agree with the user's stated bias or opinion, even if it is factually incorrect, because human evaluators are biased towards responses that confirm their existing beliefs.\n"
            "3. Data Bottleneck: Collecting high-quality pairwise human feedback is expensive, slow, and hard to scale. Disagreements among annotators introduce noise into the reward model.\n"
            "4. Mode Collapse: The PPO optimization process can reduce the diversity of the model's outputs, leading to a repetitive writing style and loss of creative capabilities."
        ]
    )

    # 4. HNSW Indexing PDF
    create_pdf(
        "hnsw_indexing.pdf",
        "Hierarchical Navigable Small World (HNSW) Graphs",
        [
            # Page 1
            "Hierarchical Navigable Small World (HNSW) is a state-of-the-art algorithm for Approximate Nearest Neighbor (ANN) search in high-dimensional vector spaces. In vector retrieval, calculating the exact Euclidean or Cosine distance between a query vector and millions of database vectors is computationally prohibitive. HNSW solves this by structuring the vector database as a multi-layer graph, achieving logarithmic search complexity O(log N) while maintaining high retrieval accuracy (recall).\n\n"
            "The design of HNSW is inspired by the Skip List data structure and the Small World graph concept. A skip list is a linked list with hierarchical layers of shortcuts that allow fast search. Similarly, HNSW builds a multi-layered graph where the bottom layer (Layer 0) contains all vectors connected in a dense graph, and upper layers contain fewer vectors, forming coarser graphs with longer-range links. This hierarchical setup prevents search queries from getting stuck in local minima in high-dimensional space.",
            # Page 2
            "The search and insertion processes in HNSW operate as follows:\n\n"
            "1. Search: The query begins at an entry point in the top-most layer. The algorithm performs a greedy search, moving from node to neighbor, until it reaches a local minimum (a node where no neighbor is closer to the query vector). It then drops down to the corresponding node in the next layer and repeats the search. This continues until it reaches Layer 0, where it performs a local search to find the final K nearest neighbors.\n"
            "2. Insertion: When a new vector is added, its maximum layer is determined randomly using an exponential decay distribution. The algorithm traverses the graph from the top layer down to the insertion layer to find the nearest entry points, then inserts the node at each layer and connects it to its M nearest neighbors. Parameter 'M' controls connection density, and 'ef_construct' controls the search queue size during build time, balancing build speed and recall."
        ]
    )

if __name__ == "__main__":
    main()
