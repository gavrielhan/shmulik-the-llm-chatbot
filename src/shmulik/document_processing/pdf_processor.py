"""
PDF Document Processing Module using LangChain
"""
import os
import re
import unicodedata
from typing import List, Optional, Dict, Any
try:
    from langchain_community.document_loaders import PyPDFLoader
except ImportError:
    from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
from loguru import logger

# Try to import PyMuPDF for better multilingual support
try:
    import fitz  # PyMuPDF
    HAS_PYMUPDF = True
except ImportError:
    HAS_PYMUPDF = False
    logger.warning("PyMuPDF not available. Install with: pip install PyMuPDF")


class PDFProcessor:
    """Class for processing PDF documents and splitting them into chunks"""
    
    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 200, use_pymupdf: bool = True):
        """
        Initialize PDF processor with best practices
        
        Args:
            chunk_size: Size of text chunks for splitting
            chunk_overlap: Overlap between consecutive chunks
            use_pymupdf: Whether to use PyMuPDF for better multilingual extraction
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.use_pymupdf = use_pymupdf and HAS_PYMUPDF
        
        # Smart chunking with better separators for multilingual content
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
            separators=["\n\n", "\n", ". ", "! ", "? ", "; ", ": ", " ", ""]
        )
        
        # Common header/footer patterns (case insensitive)
        self.header_footer_patterns = [
            r'^.*page \d+.*$',  # Page numbers
            r'^\d+$',           # Simple page numbers
            r'^.*copyright.*$', # Copyright notices
            r'^.*confidential.*$', # Confidential notices
            r'^.*\d+\s*/\s*\d+.*$', # Page x/y format
        ]
        
        logger.info(f"PDFProcessor initialized: chunk_size={chunk_size}, chunk_overlap={chunk_overlap}, pymupdf={self.use_pymupdf}")
    
    def clean_text(self, text: str) -> str:
        """
        Clean and normalize text following best practices
        
        Args:
            text: Raw text to clean
            
        Returns:
            Cleaned and normalized text
        """
        if not text or not text.strip():
            return ""
        
        # Unicode normalization (NFKC - compatibility decomposition + canonical composition)
        text = unicodedata.normalize('NFKC', text)
        
        # Remove common header/footer patterns
        lines = text.split('\n')
        cleaned_lines = []
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
                
            # Skip lines matching header/footer patterns
            is_header_footer = False
            for pattern in self.header_footer_patterns:
                if re.match(pattern, line.lower()):
                    is_header_footer = True
                    break
            
            if not is_header_footer:
                cleaned_lines.append(line)
        
        # Rejoin with proper spacing
        cleaned_text = '\n'.join(cleaned_lines)
        
        # Remove excessive whitespace
        cleaned_text = re.sub(r'\n\s*\n\s*\n', '\n\n', cleaned_text)  # Multiple newlines -> double newline
        cleaned_text = re.sub(r'[ \t]+', ' ', cleaned_text)  # Multiple spaces -> single space
        
        return cleaned_text.strip()
    
    def validate_extraction(self, original_path: str, extracted_docs: List[Document]) -> Dict[str, Any]:
        """
        Validate PDF extraction quality
        
        Args:
            original_path: Path to original PDF
            extracted_docs: List of extracted documents
            
        Returns:
            Validation metrics
        """
        total_chars = sum(len(doc.page_content) for doc in extracted_docs)
        total_words = sum(len(doc.page_content.split()) for doc in extracted_docs)
        
        validation_stats = {
            "total_pages": len(set(doc.metadata.get('page', 0) for doc in extracted_docs)),
            "total_chunks": len(extracted_docs),
            "total_characters": total_chars,
            "total_words": total_words,
            "avg_chunk_size": total_chars / len(extracted_docs) if extracted_docs else 0,
            "avg_words_per_chunk": total_words / len(extracted_docs) if extracted_docs else 0
        }
        
        # Basic quality checks
        if total_words < 100:
            logger.warning(f"Very few words extracted ({total_words}). Check PDF quality.")
        
        if validation_stats["avg_chunk_size"] < self.chunk_size * 0.3:
            logger.warning(f"Chunks seem too small (avg: {validation_stats['avg_chunk_size']}). Check text extraction.")
        
        logger.info(f"Extraction validation: {validation_stats}")
        return validation_stats
        
    def load_pdf_with_pymupdf(self, pdf_path: str) -> List[Document]:
        """
        Load PDF using PyMuPDF for better multilingual extraction
        
        Args:
            pdf_path: Path to the PDF file
            
        Returns:
            List of Document objects
        """
        if not HAS_PYMUPDF:
            raise ImportError("PyMuPDF not available. Install with: pip install PyMuPDF")
        
        logger.info(f"Loading PDF with PyMuPDF from: {pdf_path}")
        documents = []
        
        try:
            pdf_doc = fitz.open(pdf_path)
            
            for page_num in range(pdf_doc.page_count):
                page = pdf_doc[page_num]
                
                # Extract text with better Unicode handling
                text = page.get_text("text")
                
                # Clean the extracted text
                cleaned_text = self.clean_text(text)
                
                if cleaned_text.strip():  # Only add non-empty pages
                    doc = Document(
                        page_content=cleaned_text,
                        metadata={
                            "page": page_num + 1,
                            "source": pdf_path,
                            "extraction_method": "PyMuPDF"
                        }
                    )
                    documents.append(doc)
            
            pdf_doc.close()
            logger.info(f"Successfully loaded {len(documents)} pages from PDF using PyMuPDF")
            return documents
            
        except Exception as e:
            logger.error(f"Error loading PDF with PyMuPDF: {str(e)}")
            raise
    
    def load_pdf(self, pdf_path: str) -> List[Document]:
        """
        Load PDF document using best available method
        
        Args:
            pdf_path: Path to the PDF file
            
        Returns:
            List of Document objects
        """
        if not os.path.exists(pdf_path):
            raise FileNotFoundError(f"PDF file not found: {pdf_path}")
            
        logger.info(f"Loading PDF from: {pdf_path}")
        
        # Try PyMuPDF first if available and enabled
        if self.use_pymupdf:
            try:
                return self.load_pdf_with_pymupdf(pdf_path)
            except Exception as e:
                logger.warning(f"PyMuPDF loading failed: {e}. Falling back to PyPDFLoader.")
        
        # Fallback to PyPDFLoader
        try:
            loader = PyPDFLoader(pdf_path)
            documents = loader.load()
            
            # Clean the text from PyPDFLoader as well
            cleaned_documents = []
            for doc in documents:
                cleaned_text = self.clean_text(doc.page_content)
                if cleaned_text.strip():
                    cleaned_doc = Document(
                        page_content=cleaned_text,
                        metadata={**doc.metadata, "extraction_method": "PyPDFLoader"}
                    )
                    cleaned_documents.append(cleaned_doc)
            
            logger.info(f"Successfully loaded {len(cleaned_documents)} pages from PDF using PyPDFLoader")
            return cleaned_documents
        except Exception as e:
            logger.error(f"Error loading PDF: {str(e)}")
            raise
    
    def split_documents(self, documents: List[Document]) -> List[Document]:
        """
        Split documents into smaller chunks
        
        Args:
            documents: List of Document objects to split
            
        Returns:
            List of chunked Document objects
        """
        logger.info(f"Splitting {len(documents)} documents into chunks")
        
        try:
            chunks = self.text_splitter.split_documents(documents)
            logger.info(f"Created {len(chunks)} text chunks")
            return chunks
        except Exception as e:
            logger.error(f"Error splitting documents: {str(e)}")
            raise
    
    def process_pdf(self, pdf_path: str) -> List[Document]:
        """
        Complete enhanced PDF processing pipeline: load -> clean -> split -> validate
        
        Args:
            pdf_path: Path to the PDF file
            
        Returns:
            List of processed Document chunks with validation
        """
        logger.info(f"Starting enhanced PDF processing pipeline for: {pdf_path}")
        
        try:
            # Load the PDF using best available method
            documents = self.load_pdf(pdf_path)
            
            if not documents:
                raise ValueError("No documents were loaded from the PDF")
            
            # Split into chunks with smart chunking
            chunks = self.split_documents(documents)
            
            if not chunks:
                raise ValueError("No chunks were created from the documents")
            
            # Add enhanced metadata
            for i, chunk in enumerate(chunks):
                chunk.metadata.update({
                    'chunk_id': i,
                    'source_file': os.path.basename(pdf_path),
                    'chunk_size': len(chunk.page_content),
                    'word_count': len(chunk.page_content.split()),
                    'extraction_method': chunk.metadata.get('extraction_method', 'unknown')
                })
            
            # Validate extraction quality
            validation_stats = self.validate_extraction(pdf_path, chunks)
            
            logger.info(f"PDF processing complete. Generated {len(chunks)} chunks with {validation_stats['total_words']} words")
            return chunks
            
        except Exception as e:
            logger.error(f"Error in PDF processing pipeline: {str(e)}")
            raise
    
    def get_document_stats(self, documents: List[Document]) -> dict:
        """
        Get statistics about the processed documents
        
        Args:
            documents: List of Document objects
            
        Returns:
            Dictionary with document statistics
        """
        total_chars = sum(len(doc.page_content) for doc in documents)
        total_words = sum(len(doc.page_content.split()) for doc in documents)
        
        return {
            'total_documents': len(documents),
            'total_characters': total_chars,
            'total_words': total_words,
            'average_chars_per_doc': total_chars / len(documents) if documents else 0,
            'average_words_per_doc': total_words / len(documents) if documents else 0
        }


def create_pdf_processor(chunk_size: int = 1000, chunk_overlap: int = 200) -> PDFProcessor:
    """
    Factory function to create a PDFProcessor instance
    
    Args:
        chunk_size: Size of text chunks
        chunk_overlap: Overlap between chunks
        
    Returns:
        PDFProcessor instance
    """
    return PDFProcessor(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
