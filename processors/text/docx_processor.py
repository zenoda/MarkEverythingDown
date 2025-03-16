from processors.base import BaseDocumentProcessor, StructuredDocument, DocumentSection, DocumentElement, DocumentType
from pathlib import Path
import mammoth
import re

class DocxProcessor(BaseDocumentProcessor):
    """Processor for DOCX documents"""
    
    def process(self, file_path: str) -> StructuredDocument:
        """Process DOCX document"""
        document = StructuredDocument(
            title=Path(file_path).stem,
            source_file=file_path,
            doc_type=DocumentType.WORD
        )
        
        try:
            # Convert DOCX to Markdown using mammoth
            with open(file_path, "rb") as docx_file:
                result = mammoth.convert_to_markdown(docx_file)
            markdown_content = result.value
            
            # Create a single section with the markdown content
            section = DocumentSection(title="Document Content")
            section.add_element(DocumentElement(
                content=markdown_content,
                element_type="markdown"
            ))
            document.add_section(section)
            
            # Store raw markdown in metadata
            document.metadata["markdown"] = markdown_content
            
            # Extract any messages/warnings
            if result.messages:
                document.metadata["conversion_messages"] = [msg.message for msg in result.messages]
                
        except Exception as e:
            print(f"Error extracting content from DOCX: {e}")
            # Create error section
            error_section = DocumentSection(title="Error")
            error_section.add_element(DocumentElement(
                content=f"Failed to process document: {str(e)}",
                element_type="paragraph"
            ))
            document.add_section(error_section)
        
        return document