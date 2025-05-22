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
    MAX_IMAGE_SIZE = 1920 * 1080

    # Class method to set API configuration
    @classmethod
    def configure_api(cls, api_key: str, base_url: Optional[str] = None, model: Optional[str] = None,
                      max_image_size: Optional[int] = None):
        """
        Configure API settings for all instances
        
        Args:
            api_key: API key for the vision model
            base_url: Base URL for the API (defaults to Dashscope endpoint)
            model: Model name to use (defaults to qwen2.5-vl-72b-instruct)
            max_image_size: Max image size to use (defaults to MAX_IMAGE_SIZE)
        """
        cls.api_key = api_key
        cls.base_url = base_url or cls.DEFAULT_BASE_URL
        cls.model = model or cls.DEFAULT_MODEL
        cls.max_image_size = max_image_size or cls.MAX_IMAGE_SIZE

    def __init__(
            self,
            api_key: Optional[str] = None,
            base_url: Optional[str] = None,
            model: Optional[str] = None,
            temperature: float = 0.0,
            max_tokens: Optional[int] = None
    ):
        """
        Initialize vision document processor
        
        Args:
            api_key: API key for the vision model
            base_url: Base URL for the API
            model: Model name to use
            temperature: Temperature for generation (0.0-1.0). Lower values make output more deterministic.
            max_tokens: Maximum tokens to generate. None means using model's default limit.
        """
        super().__init__()

        # Try to get API key from different sources in order of priority
        self.api_key = api_key or getattr(self.__class__, 'api_key', None) or os.environ.get("QWEN_API_KEY")

        # Same for base_url and model
        self.base_url = base_url or getattr(self.__class__, 'base_url', None) or self.DEFAULT_BASE_URL
        self.model = model or getattr(self.__class__, 'model', None) or self.DEFAULT_MODEL

        # Store generation parameters
        self.temperature = float(temperature) if temperature is not None else 0.0
        self.max_tokens = int(max_tokens) if max_tokens is not None else None

        # Validate API key
        if not self.api_key:
            raise ValueError(
                "API key not provided. Either pass it to the constructor, "
                "use VisionDocumentProcessor.configure_api(), or set the QWEN_API_KEY environment variable."
            )

    def process(self, file_path: str, max_concurrent: int = 2, images_per_batch: int = 1,
                dynamic_batching: bool = True, max_tokens_per_batch: int = 4000) -> StructuredDocument:
        """
        Process document using Qwen2.5-VL
        
        Args:
            file_path: Path to document file
            max_concurrent: Maximum number of concurrent API calls for PDFs
            images_per_batch: Number of consecutive pages to process in a single API call
                (1 = traditional single page processing, 2+ = multi-image processing)
            dynamic_batching: Whether to dynamically determine batch sizes based on image complexity
            max_tokens_per_batch: Maximum tokens per batch when using dynamic batching
            
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
        if (doc_type == DocumentType.PDF):
            if images_per_batch > 1:
                # Use multi-image processing if specified
                self._process_pdf_multi(file_path, document, max_concurrent, images_per_batch,
                                        dynamic_batching, max_tokens_per_batch)
            else:
                # Use original single-page processing
                self._process_pdf(file_path, document, max_concurrent)

            # Ensure page markers are removed from final document output
            if "markdown" in document.metadata:
                document.metadata["markdown"] = self._remove_page_markers(document.metadata["markdown"])

                # Also update section content to remove page markers
                for section in document.sections:
                    for element in section.elements:
                        if element.element_type == "markdown":
                            element.content = self._remove_page_markers(element.content)

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
                    images = convert_from_path(file_path, dpi=300)
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
                    temp_image_path = os.path.join(path, f"page_{i + 1}.jpg")
                    actual_size = image.width * image.height
                    if actual_size > self.max_image_size:
                        zoom_factor = pow(self.max_image_size / actual_size, 0.5)
                        width, height = image.width * zoom_factor, image.height * zoom_factor
                        image = image.resize((int(width), int(height)), Image.Resampling.BILINEAR)
                    image.save(temp_image_path, "JPEG", quality=90)
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
                            print(f"Completed page {index + 1}/{page_count} ({completed}/{page_count} done)")
                        except Exception as e:
                            print(f"Error processing page {index + 1}: {str(e)}")
                            error_section = DocumentSection(title=f"Error on Page {index + 1}")
                            error_section.add_element(DocumentElement(
                                content=f"Failed to process page: {str(e)}",
                                element_type="paragraph"
                            ))
                            all_sections[index] = error_section
                            all_markdown[index] = f"## Page {index + 1}\n\nError processing page: {str(e)}"

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

    def _process_pdf_multi(self, file_path: str, document: StructuredDocument, max_concurrent: int = 2,
                           images_per_batch: int = 2,
                           dynamic_batching: bool = True, max_tokens_per_batch: int = 4000) -> None:
        """
        Process PDF document with multi-image batches for improved context
        
        Args:
            file_path: Path to PDF file
            document: Document to add content to
            max_concurrent: Maximum number of concurrent API calls
            images_per_batch: Number of consecutive pages to process in a single API call
            dynamic_batching: Whether to dynamically determine batch sizes based on image complexity
            max_tokens_per_batch: Maximum tokens per batch when using dynamic batching
        """
        from pdf2image import convert_from_path
        import tempfile
        import concurrent.futures

        # Storage for combined markdown from all pages
        all_markdown = []
        all_sections = []

        try:
            # Convert PDF to images
            with tempfile.TemporaryDirectory() as path:
                # Check if we can convert the PDF
                try:
                    print("Converting PDF to images...")
                    images = convert_from_path(file_path, dpi=300)
                    page_count = len(images)
                    print(f"Converted {page_count} pages")

                    # Initialize result arrays
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
                    temp_image_path = os.path.join(path, f"page_{i + 1}.jpg")
                    actual_size = image.width * image.height
                    if actual_size > self.max_image_size:
                        zoom_factor = self.max_image_size / actual_size
                        width, height = image.width * zoom_factor, image.height * zoom_factor
                        image = image.resize((int(width), int(height)), Image.Resampling.BILINEAR)
                    image.save(temp_image_path, "JPEG", quality=90)
                    image_paths.append((i, temp_image_path))

                # Create batch groups
                batch_tasks = []

                for i in range(0, len(image_paths), images_per_batch):
                    batch = image_paths[i:i + images_per_batch]
                    if batch:
                        # For each batch, all indices and paths
                        indices = [idx for idx, _ in batch]
                        paths = [p for _, p in batch]
                        batch_tasks.append((indices, paths))

                print(f"Processing PDF with {len(batch_tasks)} batches of up to {images_per_batch} pages each")

                # Process batches in parallel
                with concurrent.futures.ThreadPoolExecutor(max_workers=max_concurrent) as executor:
                    # Submit all batch tasks
                    future_to_batch = {
                        executor.submit(self._process_multi_page_batch, path, indices, paths): (indices, paths)
                        for indices, paths in batch_tasks
                    }

                    # Process completed batch tasks
                    completed_batches = 0
                    for future in concurrent.futures.as_completed(future_to_batch):
                        indices, paths = future_to_batch[future]
                        try:
                            results = future.result()  # List of (section, markdown) tuples

                            # Store results in correct positions
                            for i, (idx, result) in enumerate(zip(indices, results)):
                                if result:
                                    section, markdown = result
                                    all_sections[idx] = section
                                    all_markdown[idx] = markdown

                            completed_batches += 1
                            processed_pages = min(completed_batches * images_per_batch, page_count)
                            print(
                                f"Completed batch {completed_batches}/{len(batch_tasks)} ({processed_pages}/{page_count} pages)")

                        except Exception as e:
                            print(f"Error processing batch {indices}: {str(e)}")
                            # Create error sections for failed pages
                            for idx in indices:
                                page_num = idx + 1
                                error_section = DocumentSection(title=f"Error on Page {page_num}")
                                error_section.add_element(DocumentElement(
                                    content=f"Failed to process page batch: {str(e)}",
                                    element_type="paragraph"
                                ))
                                all_sections[idx] = error_section
                                all_markdown[idx] = f"## Page {page_num}\n\nError processing page: {str(e)}"

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
        print(f"Started processing page {page_index + 1}")
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
                You should not include page numbers for better readability.
                When encountering chat history, use the format of chat message, clearly document the sender of each message.
                '''
            },
            {
                "role": "user",
                "content": [
                    {"type": "text",
                     "text": "Convert this document to well-formatted markdown with proper headers, lists, "
                             "code blocks, and tables. \n#Notice: \n##1. If there is a table in the image, it should "
                             "still be output as a table, and the format should be kept as much as possible. \n##2. If "
                             "there are people in the picture, try to identify their identities. \n##3. The output "
                             "language should be consistent with the content in the picture as much as possible"},
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

    def _call_api_multi_image(self, image_contents: list, page_numbers: list) -> str:
        """
        Call vision model API with multiple images
        
        Args:
            image_contents: List of base64 encoded images
            page_numbers: List of page numbers for these images
            
        Returns:
            str: Markdown representation of the document across multiple pages
        """
        from openai import OpenAI

        print(f"Sending {len(image_contents)} images to API in a single call")
        print(f"Using model: {self.model}")

        # Configure the client
        client = OpenAI(
            api_key=self.api_key,
            base_url=self.base_url,
        )

        # Build message with multiple images
        content = [{
            "type": "text",
            "text": f"These are consecutive pages from a document (pages {', '.join(map(str, page_numbers))}). "
                    f"Convert these document pages to well-formatted markdown.\n#Notice: \n##1. If there is a table in "
                    f"the image, it should still be output as a table. \n##2. If there are people in the picture, "
                    f"try to identify their identities. \n##3. The output language should be consistent with the "
                    f"content in the picture as much as possible. \n##4. You should not include page numbers for "
                    f"better readability."
        }]

        # Add each image to the content
        for i, img in enumerate(image_contents):
            page_num = page_numbers[i]
            content.append({
                "type": "image_url",
                "image_url": {"url": f"data:image/jpeg;base64,{img}"}
            })

        # Format complete message
        markdown_messages = [
            {
                "role": "system",
                "content": f'''You are an AI specialized in recognizing and extracting text. 
                Your mission is to analyze {len(image_contents)} pages from a document and generate the result in markdown format, use markdown syntax to preserve the title level of the original document.
                You should not include page numbers for better readability.
                When encountering chat history, use the format of chat message, clearly identify the sender of each message.
                '''
            },
            {
                "role": "user",
                "content": content
            }
        ]

        try:
            print("Requesting markdown conversion for multiple images...")
            # Use streaming for more reliable results with large responses
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
            print(f"\nReceived {len(markdown_response)} characters of markdown")
            return markdown_response

        except Exception as e:
            print(f"Error during multi-image API call: {str(e)}")
            return f"# API Error\n\nFailed to process document pages: {str(e)}"

    def _process_multi_page_batch(self, temp_dir: str, page_indices: list, image_paths: list) -> list:
        """
        Process multiple pages in a single API call
        
        Args:
            temp_dir: Temporary directory
            page_indices: List of page indices (0-based)
            image_paths: List of paths to image files
            
        Returns:
            list: List of (section, markdown) tuples for each page
        """
        page_numbers = [idx + 1 for idx in page_indices]  # Convert to 1-based page numbers
        print(f"Processing batch with pages: {page_numbers}")

        # Encode all images in the batch
        image_contents = []
        for path in image_paths:
            image_contents.append(self._prepare_image(path))

        # Call API with multiple images
        markdown_response = self._call_api_multi_image(image_contents, page_numbers)

        # Split response by page markers
        results = []

        try:
            # Check if it's a multi-page response with page markers
            if "## Page" in markdown_response:
                # Attempt to split by page markers
                parts = []
                current_part = []
                current_page_idx = None

                for line in markdown_response.split("\n"):
                    if line.strip().startswith("## Page"):
                        # If we already have content for a page, save it
                        if current_part and current_page_idx is not None:
                            parts.append((current_page_idx, "\n".join(current_part)))

                        # Start a new page
                        current_part = [line]

                        # Extract page number from heading
                        try:
                            page_text = line.strip().replace("## Page", "").strip()
                            current_page_idx = int(page_text) - 1  # Convert to 0-based index
                        except:
                            # If we can't extract page number, use position in batch
                            current_page_idx = len(parts)
                    else:
                        current_part.append(line)

                # Add the last part
                if current_part and current_page_idx is not None:
                    parts.append((current_page_idx, "\n".join(current_part)))

                # Create sections for each identified page
                for page_idx, content in parts:
                    if 0 <= page_idx < len(page_indices):
                        idx = page_indices[page_idx]
                        page_num = idx + 1

                        section = DocumentSection(title=f"Page {page_num}", level=1)
                        section.metadata["page"] = page_num
                        section.add_element(DocumentElement(
                            content=content,
                            element_type="markdown"
                        ))

                        # Include page heading if not already there
                        if not content.strip().startswith("## Page"):
                            page_markdown = f"## Page {page_num}\n\n{content}"
                        else:
                            page_markdown = content

                        results.append((section, page_markdown))
            else:
                # If no page markers are found, distribute content evenly
                # This is a fallback and likely to be less accurate
                for i, idx in enumerate(page_indices):
                    page_num = idx + 1

                    # Simple approach - split content by number of pages
                    chunk_size = max(1, len(markdown_response) // len(page_indices))
                    start_pos = i * chunk_size
                    end_pos = start_pos + chunk_size if i < len(page_indices) - 1 else len(markdown_response)

                    content = markdown_response[start_pos:end_pos]

                    section = DocumentSection(title=f"Page {page_num}", level=1)
                    section.metadata["page"] = page_num
                    section.add_element(DocumentElement(
                        content=content,
                        element_type="markdown"
                    ))

                    page_markdown = f"## Page {page_num}\n\n{content}"
                    results.append((section, page_markdown))
        except Exception as e:
            print(f"Error parsing multi-page response: {str(e)}")
            # Fallback - create an error section for each page
            for idx in page_indices:
                page_num = idx + 1
                error_content = f"Error parsing multi-page response: {str(e)}"

                section = DocumentSection(title=f"Page {page_num}", level=1)
                section.metadata["page"] = page_num
                section.add_element(DocumentElement(
                    content=error_content,
                    element_type="paragraph"
                ))

                page_markdown = f"## Page {page_num}\n\n{error_content}"
                results.append((section, page_markdown))

        # Ensure we have a result for each page in the batch
        while len(results) < len(page_indices):
            idx = page_indices[len(results)]
            page_num = idx + 1
            error_message = "No content was generated for this page in the batch."

            section = DocumentSection(title=f"Page {page_num}", level=1)
            section.metadata["page"] = page_num
            section.add_element(DocumentElement(
                content=error_message,
                element_type="paragraph"
            ))

            page_markdown = f"## Page {page_num}\n\n{error_message}"
            results.append((section, page_markdown))

        return results

    def _remove_page_markers(self, content: str) -> str:
        """
        Remove page marker headings from the markdown content
        
        Args:
            content: Markdown content with page markers
            
        Returns:
            str: Cleaned markdown without page markers
        """
        if not content:
            return content

        # Split by lines and filter out page marker headings
        lines = content.split('\n')
        filtered_lines = []
        skip_next_empty = False

        for line in lines:
            # Check if line is a page marker (## Page X)
            if line.strip().startswith('## Page '):
                skip_next_empty = True  # Skip the next empty line if it exists
                continue

            # Skip empty line after page marker
            if skip_next_empty and not line.strip():
                skip_next_empty = False
                continue

            filtered_lines.append(line)

        # Rejoin the filtered lines
        cleaned_content = '\n'.join(filtered_lines)

        return cleaned_content
