#!/usr/bin/env python3
"""
CLI-based Markdown Document Chatbot using RAG (Retrieval-Augmented Generation)
This application only accepts markdown (.md) files and provides a command-line interface.
"""

import os
import re
import sys
import argparse
import glob
from pathlib import Path
import uuid
from typing import List, Dict, Tuple

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from langchain_groq import ChatGroq

class MarkdownChatbot:
    def __init__(self, api_key: str):
        """Initialize the chatbot with Groq API key."""
        self.api_key = api_key
        os.environ["GROQ_API_KEY"] = api_key
        self.embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        self.vector_store = None
        self.all_metadata = []
        
    def extract_text_from_markdown(self, file_path: str) -> Tuple[str, List[Dict]]:
        """Extract text from a markdown file."""
        try:
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as file:
                content = file.read()
                # Remove any surrogate characters that might cause issues
                content = re.sub(r'[\ud800-\udfff]', '', content)
                
                metadata = [{
                    "doc_id": str(uuid.uuid4()),
                    "page": 1,
                    "text": content,
                    "source": os.path.basename(file_path)
                }]
                
                return content, metadata
        except Exception as e:
            print(f"Error processing {file_path}: {e}")
            return "", []
    
    def get_text_chunks(self, text: str, metadata: List[Dict]) -> Tuple[List[str], List[Dict]]:
        """Split text into chunks and associate metadata."""
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        chunks = text_splitter.split_text(text)
        chunk_metadata = []
        
        for i, chunk in enumerate(chunks):
            chunk_meta = metadata[0].copy()  # Use the first metadata entry as base
            chunk_meta["paragraph"] = i + 1
            chunk_metadata.append(chunk_meta)
        
        return chunks, chunk_metadata
    
    def load_markdown_files(self, file_paths: List[str]) -> bool:
        """Load multiple markdown files and create vector store."""
        if not file_paths:
            print("No markdown files provided.")
            return False
        
        print(f"Loading {len(file_paths)} markdown files...")
        
        all_chunks = []
        all_metadata = []
        
        for file_path in file_paths:
            if not file_path.endswith('.md'):
                print(f"Skipping non-markdown file: {file_path}")
                continue
            
            if not os.path.exists(file_path):
                print(f"File not found: {file_path}")
                continue
            
            print(f"Processing: {file_path}")
            text, metadata = self.extract_text_from_markdown(file_path)
            
            if text:
                chunks, chunk_metadata = self.get_text_chunks(text, metadata)
                all_chunks.extend(chunks)
                all_metadata.extend(chunk_metadata)
            else:
                print(f"No text extracted from: {file_path}")
        
        if not all_chunks:
            print("No text chunks were extracted from the provided files.")
            return False
        
        print(f"Creating vector store with {len(all_chunks)} chunks...")
        
        # Create FAISS vector store
        self.vector_store = FAISS.from_texts(all_chunks, embedding=self.embeddings, metadatas=all_metadata)
        self.all_metadata = all_metadata
        
        # Save the vector store
        self.vector_store.save_local("faiss_index")
        print("Vector store created and saved successfully!")
        
        return True
    
    def get_conversational_chain(self):
        """Create QA chain with custom prompt using Groq LLaMA model."""
        prompt_template = """
        YOU ARE A HIGHLY DISCIPLINED, EVIDENCE-DRIVEN QUESTION-ANSWERING AGENT TRAINED TO EXTRACT, ANALYZE, AND REPORT INFORMATION *EXCLUSIVELY* FROM THE PROVIDED CONTEXT. YOUR RESPONSES MUST BE THOROUGH, FACTUALLY PRECISE, AND METICULOUSLY SOURCED.

        ###INSTRUCTIONS###

        - YOU MUST PROVIDE A COMPREHENSIVE AND ACCURATE ANSWER TO THE USER'S QUESTION *USING ONLY THE INFORMATION FOUND IN THE GIVEN CONTEXT*
        - PRESENT THE ANSWER IN BULLET POINTS FOR CLARITY AND READABILITY
        - ALL CLAIMS OR FACTS MUST BE SUPPORTED BY CLEAR CITATIONS IN THE FORMAT: `(Document ID: [ID], Page: [X], Paragraph: [Y], Source: [Source])`
        - IF THE CONTEXT DOES *NOT* CONTAIN SUFFICIENT INFORMATION TO ANSWER, RESPOND EXACTLY: `"Answer is not available in the context."`
        - DO NOT FABRICATE, ASSUME, OR HALLUCINATE ANY INFORMATION OUTSIDE THE PROVIDED MATERIAL

        ###CHAIN OF THOUGHTS TO FOLLOW###

        1. **UNDERSTAND** the user's question carefully and precisely
        2. **IDENTIFY** all key elements and concepts within the question
        3. **SCAN** the provided context for all *relevant sections* tied to the query
        4. **EXTRACT** the factual content that addresses the question directly
        5. **VERIFY** the relevance and accuracy of the extracted information
        6. **CITE** all supporting evidence in the required format: `(Document ID: [ID], Page: [X], Paragraph: [Y], Source: [Source])`
        7. **RESPOND** with a clear, complete, and well-structured answer in bullet points using only the validated context
        8. **RETURN** "Answer is not available in the context" if *no* relevant information exists

        ###WHAT NOT TO DO###

        - DO NOT INVENT INFORMATION NOT PRESENT IN THE CONTEXT
        - NEVER OMIT REQUIRED CITATIONS — THEY ARE MANDATORY FOR EVERY CLAIM
        - DO NOT PROVIDE VAGUE OR INCOMPLETE ANSWERS
        - NEVER PARAPHRASE IF IT RISKS ALTERING THE FACTUAL MEANING OF SOURCE TEXT
        - AVOID GENERALIZATIONS NOT GROUNDED IN THE SPECIFIC CITED MATERIAL
        - DO NOT MENTION OR SPECULATE ABOUT DATA OUTSIDE THE GIVEN CONTEXT

        ###EXAMPLE RESPONSE FORMAT###

        **Answer**:
        - The primary benefit of hydrogen fuel is its zero-emission nature, which contributes significantly to reducing environmental pollution (Document ID: H-2023-01, Page: 4, Paragraph: 2, Source: hydrogen_report.md).
        
        Context:\n{context}\n
        Question:\n{question}\n
        
        Answer:
        """
        
        model = ChatGroq(model_name="llama-3.3-70b-versatile", temperature=0.3, api_key=self.api_key)
        prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
        return load_qa_chain(model, chain_type="stuff", prompt=prompt)
    
    def get_theme_summary(self, docs, question: str) -> str:
        """Generate a summary of recurring themes using Groq LLaMA model."""
        prompt_template = """
        YOU ARE A WORLD-CLASS DOCUMENT SYNTHESIS EXPERT SPECIALIZED IN ANALYZING MULTI-SOURCE TEXTUAL DATA TO EXTRACT THEMES AND KEY INSIGHTS. YOUR TASK IS TO EXAMINE THE PROVIDED DOCUMENT EXCERPTS AND SUMMARIZE RECURRING THEMES RELATED TO THE USER'S QUESTION IN A CONVERSATIONAL, EASY-TO-UNDERSTAND MANNER — WHILE MAINTAINING ACADEMIC RIGOR THROUGH ACCURATE CITATIONS.

        ###INSTRUCTIONS###

        - CAREFULLY READ all document excerpts provided
        - IDENTIFY recurring themes, patterns, or key insights SPECIFICALLY RELEVANT to the question: **"{question}"**
        - SUMMARIZE the themes in a NATURAL, CONVERSATIONAL TONE using BULLET POINTS for clarity
        - INCLUDE ACCURATE CITATIONS in the format: `(Document ID: [ID], Page: [X], Paragraph: [Y], Source: [Source])` IMMEDIATELY after each claim or thematic insight
        - DO NOT INCLUDE INFORMATION that cannot be directly traced to the context with a proper citation
        - MAINTAIN A BALANCE between ACCESSIBLE LANGUAGE and ANALYTICAL DEPTH

        ###CHAIN OF THOUGHTS TO FOLLOW###

        1. **UNDERSTAND** the question: parse what specific themes or insights should be sought
        2. **SCAN** all document excerpts to locate content related to the question
        3. **GROUP** related excerpts together by common themes or insights
        4. **ANALYZE** each group to extract the core idea represented
        5. **PHRASE** your findings in bullet points in a smooth, conversational tone
        6. **CITE** every theme with specific references using the format: `(Document ID: [ID], Page: [X], Paragraph: [Y], Source: [Source])`
        7. **VERIFY** that no part of the answer includes unstated assumptions or uncited claims

        ###WHAT NOT TO DO###

        - NEVER INVENT THEMES OR INSIGHTS THAT ARE NOT GROUNDED IN THE PROVIDED CONTEXT
        - DO NOT OMIT CITATIONS FOR ANY CLAIM OR GENERALIZATION
        - DO NOT REPEAT VERBATIM TEXT FROM THE DOCUMENTS WITHOUT INTERPRETING THE THEMES
        - NEVER WRITE IN A FORMAL, STILTED, OR OVERLY TECHNICAL TONE — THE SUMMARY SHOULD BE CONVERSATIONAL
        - AVOID LISTING QUOTES WITHOUT SYNTHESIZING MEANING OR GROUPING THEM INTO COHERENT THEMES

        Excerpts:\n{context}\n

        Summary:
        """
        
        model = ChatGroq(model_name="llama-3.3-70b-versatile", temperature=0.5, api_key=self.api_key)
        prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
        context = "\n".join([f"Doc ID: {doc.metadata['doc_id']}, Page: {doc.metadata['page']}, Para: {doc.metadata['paragraph']}, Source: {doc.metadata['source']}\n{doc.page_content}" for doc in docs])
        chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)
        response = chain({"input_documents": docs, "question": question}, return_only_outputs=True)
        return response["output_text"]
    
    def answer_question(self, question: str, show_citations: bool = True, show_themes: bool = True) -> Dict:
        """Answer a question based on the loaded documents."""
        if not self.vector_store:
            return {"error": "No documents loaded. Please load markdown files first."}
        
        print(f"\nProcessing question: {question}")
        print("Searching for relevant information...")
        
        # Perform similarity search
        docs = self.vector_store.similarity_search(question, k=5)
        
        if not docs:
            return {"error": "No relevant documents found for your question."}
        
        # Generate answer
        chain = self.get_conversational_chain()
        response = chain({"input_documents": docs, "question": question}, return_only_outputs=True)
        
        result = {
            "answer": response["output_text"],
            "citations": [],
            "themes": ""
        }
        
        # Prepare citations
        if show_citations:
            for doc in docs:
                citation = {
                    "doc_id": doc.metadata["doc_id"],
                    "source": doc.metadata["source"],
                    "page": doc.metadata["page"],
                    "paragraph": doc.metadata["paragraph"],
                    "content": doc.page_content[:200] + "..." if len(doc.page_content) > 200 else doc.page_content
                }
                result["citations"].append(citation)
        
        # Generate themes
        if show_themes:
            print("Generating theme summary...")
            result["themes"] = self.get_theme_summary(docs, question)
        
        return result

def find_markdown_files(directory: str) -> List[str]:
    """Find all markdown files in a directory and its subdirectories."""
    pattern = os.path.join(directory, "**", "*.md")
    return glob.glob(pattern, recursive=True)

def print_result(result: Dict):
    """Print the result in a formatted way."""
    if "error" in result:
        print(f"\nError: {result['error']}")
        return
    
    print("\n" + "="*80)
    print("ANSWER")
    print("="*80)
    print(result["answer"])
    
    if result["citations"]:
        print("\n" + "="*80)
        print("CITATIONS")
        print("="*80)
        for i, citation in enumerate(result["citations"], 1):
            print(f"\n[{i}] Document ID: {citation['doc_id']}")
            print(f"    Source: {citation['source']}")
            print(f"    Page: {citation['page']}, Paragraph: {citation['paragraph']}")
            print(f"    Content: {citation['content']}")
    
    if result["themes"]:
        print("\n" + "="*80)
        print("RECURRING THEMES")
        print("="*80)
        print(result["themes"])
    
    print("\n" + "="*80)

def interactive_mode(chatbot: MarkdownChatbot):
    """Run the chatbot in interactive mode."""
    print("\n🤖 Markdown Chatbot - Interactive Mode")
    print("Type 'quit', 'exit', or 'q' to exit")
    print("Type 'help' for available commands")
    print("-" * 50)
    
    while True:
        try:
            question = input("\n💬 Your question: ").strip()
            
            if question.lower() in ['quit', 'exit', 'q']:
                print("Goodbye!")
                break
            elif question.lower() == 'help':
                print("\nAvailable commands:")
                print("- Type any question about your markdown documents")
                print("- 'quit', 'exit', 'q': Exit the chatbot")
                print("- 'help': Show this help message")
                continue
            elif not question:
                print("Please enter a question.")
                continue
            
            result = chatbot.answer_question(question)
            print_result(result)
            
        except KeyboardInterrupt:
            print("\n\nGoodbye!")
            break
        except Exception as e:
            print(f"Error: {e}")

def main():
    parser = argparse.ArgumentParser(description="CLI-based Markdown Document Chatbot using RAG")
    parser.add_argument("--api-key", required=True, help="Groq API key")
    parser.add_argument("--files", nargs="+", help="Markdown files to process")
    parser.add_argument("--directory", help="Directory containing markdown files (searches recursively)")
    parser.add_argument("--question", help="Single question to ask (non-interactive mode)")
    parser.add_argument("--no-citations", action="store_true", help="Don't show citations")
    parser.add_argument("--no-themes", action="store_true", help="Don't show themes")
    
    args = parser.parse_args()
    
    # Initialize chatbot
    print("🚀 Initializing Markdown Chatbot...")
    chatbot = MarkdownChatbot(args.api_key)
    
    # Collect markdown files
    files_to_process = []
    
    if args.files:
        files_to_process.extend(args.files)
    
    if args.directory:
        if os.path.isdir(args.directory):
            dir_files = find_markdown_files(args.directory)
            files_to_process.extend(dir_files)
            print(f"Found {len(dir_files)} markdown files in directory: {args.directory}")
        else:
            print(f"Directory not found: {args.directory}")
            return
    
    if not files_to_process:
        print("No markdown files specified. Use --files or --directory to specify files.")
        return
    
    # Remove duplicates and filter for .md files
    unique_files = list(set(f for f in files_to_process if f.endswith('.md')))
    print(f"Processing {len(unique_files)} unique markdown files...")
    
    # Load documents
    if not chatbot.load_markdown_files(unique_files):
        print("Failed to load documents. Exiting.")
        return
    
    # Handle single question or interactive mode
    if args.question:
        result = chatbot.answer_question(
            args.question,
            show_citations=not args.no_citations,
            show_themes=not args.no_themes
        )
        print_result(result)
    else:
        interactive_mode(chatbot)

if __name__ == "__main__":
    main()
