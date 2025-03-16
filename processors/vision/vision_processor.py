import os
import base64
from typing import Dict, List, Optional, Union, Any, Tuple
import requests
from pathlib import Path
from PIL import Image
from io import BytesIO

from processors.base import BaseDocumentProcessor, StructuredDocument, DocumentSection, DocumentElement, DocumentType

class VisionDocumentProcessor(BaseDocumentProcessor):
    """Document processor using Qwen2.5-VL"""
    
    # Class variables to store API configuration
    DEFAULT_BASE_URL = "https://dashscope-intl.aliyuncs.com/compatible-mode/v1"
    DEFAULT_MODEL = "qwen2.5-vl-72b-instruct"
    
    # Class method to set API configuration
    @classmethod
    def configure_api(cls, api_key: str, base_url: Optional[str] = None, model: Optional[str] = None):
        """
        Configure API settings for all instances
        
        Args:
            api_key: API key for the vision model
            base_url: Base URL for the API (defaults to Dashscope endpoint)
            model: Model name to use (defaults to qwen2.5-vl-72b-instruct)
        """
        cls.api_key = api_key
        cls.base_url = base_url or cls.DEFAULT_BASE_URL
        cls.model = model or cls.DEFAULT_MODEL
    
    def __init__(
        self, 
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        model: Optional[str] = None
    ):
        """
        Initialize vision document processor
        
        Args:
            api_key: API key for the vision model
            base_url: Base URL for the API
            model: Model name to use
        """
        super().__init__()
        
        # Try to get API key from different sources in order of priority
        self.api_key = api_key or getattr(self.__class__, 'api_key', None) or os.environ.get("QWEN_API_KEY")
        
        # Same for base_url and model
        self.base_url = base_url or getattr(self.__class__, 'base_url', None) or self.DEFAULT_BASE_URL
        self.model = model or getattr(self.__class__, 'model', None) or self.DEFAULT_MODEL
        
        # Validate API key
        if not self.api_key:
            raise ValueError(
                "API key not provided. Either pass it to the constructor, "
                "use VisionDocumentProcessor.configure_api(), or set the QWEN_API_KEY environment variable."
            )
            
    def process(self, file_path: str, max_concurrent: int = 2) -> StructuredDocument:
        """
        Process document using Qwen2.5-VL
        
        Args:
            file_path: Path to document file
            max_concurrent: Maximum number of concurrent API calls for PDFs
            
        Returns:
            StructuredDocument: Processed document
        """
        # Get document type
        doc_type = DocumentType.from_file_extension(file_path)
        
        # Create document with basic metadata
        document = StructuredDocument(
            title=Path(file_path).stem,
            source_file=file_path,
            doc_type=doc_type
        )
        
        # If file is PDF, process each page separately
        if doc_type == DocumentType.PDF:
            self._process_pdf(file_path, document, max_concurrent)
        else:
            # For images and other document types
            image_content = self._prepare_image(file_path)
            
            # Get markdown response directly
            markdown_response = self._call_api(image_content)
            
            # Create a section for the entire document
            section = DocumentSection(title=Path(file_path).stem, level=1)
            section.add_element(DocumentElement(
                content=markdown_response,
                element_type="markdown"
            ))
            document.add_section(section)
            
            # Store raw markdown in metadata
            document.metadata["markdown"] = markdown_response
        
        return document
    
    def _prepare_image(self, file_path: str) -> str:
        """Read image file and encode as base64"""
        with open(file_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode("utf-8")
    
    def _process_pdf(self, file_path: str, document: StructuredDocument, max_concurrent: int = 2) -> None:
        """
        Process PDF document with parallel page processing
        
        Args:
            file_path: Path to PDF file
            document: Document to add content to
            max_concurrent: Maximum number of concurrent API calls
        """
        from pdf2image import convert_from_path
        import tempfile
        import concurrent.futures
        
        # Storage for combined markdown from all pages
        all_markdown = [None] * 0  # Will resize based on page count
        all_sections = [None] * 0  # Will resize based on page count
        
        try:
            # Convert PDF to images
            with tempfile.TemporaryDirectory() as path:
                # Check if we can convert the PDF
                try:
                    print("Converting PDF to images...")
                    images = convert_from_path(file_path)
                    page_count = len(images)
                    print(f"Converted {page_count} pages")
                    
                    # Resize result arrays
                    all_markdown = [None] * page_count
                    all_sections = [None] * page_count
                    
                except Exception as e:
                    print(f"Error converting PDF to images: {str(e)}")
                    error_section = DocumentSection(title="Error")
                    error_section.add_element(DocumentElement(
                        content=f"Failed to convert PDF to images: {str(e)}",
                        element_type="paragraph"
                    ))
                    document.add_section(error_section)
                    return
                
                # Save all images first
                image_paths = []
                for i, image in enumerate(images):
                    temp_image_path = os.path.join(path, f"page_{i+1}.jpg")
                    image.save(temp_image_path, "JPEG")
                    image_paths.append((i, temp_image_path))
                
                # Process pages in parallel
                with concurrent.futures.ThreadPoolExecutor(max_workers=max_concurrent) as executor:
                    # Submit all tasks and store the futures
                    future_to_page = {
                        executor.submit(self._process_single_page, path, index, temp_path): (index, temp_path) 
                        for index, temp_path in image_paths
                    }
                    
                    # Process completed tasks as they finish
                    completed = 0
                    for future in concurrent.futures.as_completed(future_to_page):
                        index, temp_path = future_to_page[future]
                        try:
                            section, markdown = future.result()
                            all_sections[index] = section
                            all_markdown[index] = markdown
                            completed += 1
                            print(f"Completed page {index+1}/{page_count} ({completed}/{page_count} done)")
                        except Exception as e:
                            print(f"Error processing page {index+1}: {str(e)}")
                            error_section = DocumentSection(title=f"Error on Page {index+1}")
                            error_section.add_element(DocumentElement(
                                content=f"Failed to process page: {str(e)}",
                                element_type="paragraph"
                            ))
                            all_sections[index] = error_section
                            all_markdown[index] = f"## Page {index+1}\n\nError processing page: {str(e)}"
                
                # Add all sections to document in correct order
                for section in all_sections:
                    if section:
                        document.add_section(section)
                        
                # Combine all markdown
                valid_markdown = [md for md in all_markdown if md]
                if valid_markdown:
                    document.metadata["markdown"] = "\n\n".join(valid_markdown)
                    
        except Exception as e:
            print(f"Error in PDF processing: {str(e)}")
            error_section = DocumentSection(title="Error")
            error_section.add_element(DocumentElement(
                content=f"Failed to process PDF: {str(e)}",
                element_type="paragraph"
            ))
            document.add_section(error_section)

    def _process_single_page(self, temp_dir: str, page_index: int, image_path: str) -> tuple:
        """
        Process a single page in the PDF
        
        Args:
            temp_dir: Temporary directory 
            page_index: Index of the page (0-based)
            image_path: Path to the image file
            
        Returns:
            tuple: (section, markdown) for the page
        """
        print(f"Started processing page {page_index+1}")
        page_num = page_index + 1  # Convert to 1-based page numbers for display
        
        # Process image
        image_content = self._prepare_image(image_path)
        markdown_response = self._call_api(image_content)
        
        # Create section for this page
        section = DocumentSection(title=f"Page {page_num}", level=1)
        section.metadata["page"] = page_num
        section.add_element(DocumentElement(
            content=markdown_response,
            element_type="markdown"
        ))
        
        # Format markdown with page indicator
        page_markdown = f"## Page {page_num}\n\n{markdown_response}"
        
        return section, page_markdown
        
    def _call_api(self, image_content: str) -> str:
        """
        Call vision model API to get markdown representation of document
        
        Args:
            image_content: Base64 encoded image
            
        Returns:
            str: Markdown representation of the document
        """
        from openai import OpenAI
        
        print(f"Using API: {self.base_url}")
        print(f"Using model: {self.model}")
        
        # Configure the client
        client = OpenAI(
            api_key=self.api_key,
            base_url=self.base_url,
        )
        
        # Format message for LM Studio compatibility
        markdown_messages = [
            {
                "role": "system",
                "content": '''You are an AI specialized in recognizing and extracting text. 
                Your mission is to analyze the image document and generate the result in markdown format, use markdown syntax to preserve the title level of the original document.
                When encountering chat history, use the format of chat message, clearly document the sender of each message.
                '''
            },
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Convert this document to well-formatted markdown with proper headers, lists, code blocks, and tables:"},
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/jpeg;base64,{image_content}"}
                    }
                ]
            }
        ]
        
        try:
            print("Requesting markdown conversion...")
            # Use streaming for more reliable results
            markdown_parts = []
            markdown_completion = client.chat.completions.create(
                model=self.model,
                messages=markdown_messages,
                stream=True
            )
            
            print("Receiving markdown stream...", end="", flush=True)
            for chunk in markdown_completion:
                if chunk.choices and chunk.choices[0].delta.content:
                    content = chunk.choices[0].delta.content
                    markdown_parts.append(content)
                    # Print progress indicator
                    print(".", end="", flush=True)
            
            markdown_response = "".join(markdown_parts)
            print("\nMarkdown response received successfully")
            return markdown_response
            
        except Exception as e:
            print(f"Error during API call: {str(e)}")
            return f"# API Error\n\nFailed to process document: {str(e)}"