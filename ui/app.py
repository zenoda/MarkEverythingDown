import gradio as gr
import os
import tempfile
import shutil
from pathlib import Path
from processors.base import BaseDocumentProcessor, DocumentType
from processors.vision.vision_processor import VisionDocumentProcessor
import sys
import time

def detect_document_type(file_obj):
    """Detect document type from file extension"""
    if file_obj is None:
        return "No file selected", False
    
    doc_type = DocumentType.from_file_extension(file_obj.name)
    # Only show force vision option for PDFs
    show_force_vision = (doc_type == DocumentType.PDF)
    return f"{doc_type.value.upper()} Document", show_force_vision

def process_documents(file_objs, output_dir, force_vision, max_concurrent, images_per_batch, 
                    api_key, api_url, model, temperature, max_tokens, dynamic_batching, max_tokens_per_batch,
                    progress=gr.Progress()):
    """Process multiple documents and return summary"""
    try:
        if not file_objs:
            return "Please upload at least one document", None, "No files processed"
        
        # Ensure output directory exists
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
        else:
            output_dir = "output" # Default directory
            os.makedirs(output_dir, exist_ok=True)
            
        # Configure API globally
        if api_key:
            print(f"Configuring global API with key '{api_key}' at URL: {api_url}, model: {model}")
            VisionDocumentProcessor.configure_api(
                api_key=api_key,
                base_url=api_url,
                model=model
            )
        
        # Process files with progress updates
        processed_files = []
        errors = []
        last_content = None
        
        # Create a temporary directory for output files
        temp_output_dir = tempfile.mkdtemp()
        
        for i, file_obj in enumerate(file_objs):
            progress((i / len(file_objs)), desc=f"Processing {file_obj.name}...")
            temp_path = None
            
            try:
                # Create temp file with unique name
                temp_file_path = os.path.join(tempfile.gettempdir(), f"med_temp_{os.urandom(8).hex()}_{Path(file_obj.name).name}")
                
                # Write to temp file - simple direct approach
                if hasattr(file_obj, 'file'):
                    with open(temp_file_path, 'wb') as dest_file:
                        file_obj.file.seek(0)
                        shutil.copyfileobj(file_obj.file, dest_file)
                        file_obj.file.seek(0)
                elif hasattr(file_obj, 'name') and os.path.isfile(file_obj.name):
                    shutil.copy2(file_obj.name, temp_file_path)
                else:
                    with open(temp_file_path, 'wb') as dest_file:
                        if hasattr(file_obj, 'read'):
                            content = file_obj.read()
                            if hasattr(file_obj, 'seek'):
                                file_obj.seek(0)
                            dest_file.write(content)
                        else:
                            content = str(file_obj).encode('utf-8')
                            dest_file.write(content)
                
                temp_path = temp_file_path
                print(f"Created temp file: {temp_path}")
                
                # Detect document type
                doc_type = DocumentType.from_file_extension(temp_path)
                
                # Convert max_tokens from string to int or None
                max_tokens_value = None
                if max_tokens and max_tokens.strip():
                    try:
                        max_tokens_value = int(max_tokens)
                    except ValueError:
                        print(f"Invalid max_tokens value: {max_tokens}, using default")
                
                # Select appropriate processor - following test_processor.py pattern
                if force_vision and doc_type == DocumentType.PDF:
                    print(f"Using Vision processor for PDF: {file_obj.name}")
                    processor = VisionDocumentProcessor(temperature=temperature, max_tokens=max_tokens_value)
                elif doc_type == DocumentType.IMAGE:
                    print(f"Using Vision processor for image: {file_obj.name}")
                    processor = VisionDocumentProcessor(temperature=temperature, max_tokens=max_tokens_value)
                else:
                    processor = BaseDocumentProcessor.get_processor(temp_path)
                    print(f"Using {processor.__class__.__name__} for {file_obj.name}")
                
                # Add debug output to see what configuration is being used
                if isinstance(processor, VisionDocumentProcessor):
                    print(f"Vision processor configuration:")
                    print(f"- API Key: {processor.api_key[:4]}*** (length: {len(processor.api_key)})")
                    print(f"- Base URL: {processor.base_url}")
                    print(f"- Model: {processor.model}")
                    print(f"- Temperature: {processor.temperature}")
                    print(f"- Max Tokens: {processor.max_tokens}")
                    
                    # Process PDF with concurrent pages and multi-image batching
                    if doc_type == DocumentType.PDF:
                        document = processor.process(
                            temp_path, 
                            max_concurrent=max_concurrent, 
                            images_per_batch=images_per_batch,
                            dynamic_batching=dynamic_batching,
                            max_tokens_per_batch=max_tokens_per_batch
                        )
                        print(f"Processed PDF with {max_concurrent} concurrent workers, dynamic batching: {dynamic_batching}")
                    else:
                        # Direct processing for images
                        document = processor.process(temp_path)
                else:
                    # Normal processing for non-vision documents
                    document = processor.process(temp_path)
                
                # Generate output
                content = document.to_markdown()
                print(f"Generated markdown length: {len(content)}")
                
                # Use simplified filename for output
                output_filename = f"{Path(file_obj.name).stem}.md"
                
                # Save to output directory
                actual_output_path = os.path.join(output_dir, output_filename)
                with open(actual_output_path, "w", encoding="utf-8") as f:
                    f.write(content)
                
                # Copy to temp directory for Gradio
                temp_output_path = os.path.join(temp_output_dir, output_filename)
                with open(temp_output_path, "w", encoding="utf-8") as f:
                    f.write(content)
                
                # Remember the last content for preview
                last_content = content
                
                # Use actual path for processed file
                processed_files.append(actual_output_path)
                
            except Exception as e:
                error_msg = f"Error processing {file_obj.name}: {str(e)}"
                errors.append(error_msg)
                print(error_msg)
                import traceback
                traceback.print_exc()
            finally:
                # Clean up temp file
                if temp_path and os.path.exists(temp_path):
                    try:
                        os.unlink(temp_path)
                    except:
                        pass
        
        # Create summary
        if processed_files:
            processed_summary = f"Processed {len(processed_files)} files successfully to {output_dir}:"
            for path in processed_files:
                processed_summary += f"\n- {os.path.basename(path)}"
                
            if errors:
                processed_summary += f"\n\n{len(errors)} files had errors:"
                for error in errors:
                    processed_summary += f"\n- {error}"
        else:
            processed_summary = "No files were processed successfully."
            if errors:
                processed_summary += f"\n\nErrors:\n" + "\n".join(errors)
        
        # Return the processed results
        return last_content, processed_files, processed_summary
        
    except Exception as e:
        print(f"Overall error: {str(e)}")
        import traceback
        traceback.print_exc()
        return f"Error: {str(e)}", None, f"Error: {str(e)}"
    
def create_ui():
    """Create enhanced Gradio UI with batch processing"""
    # Set a modern theme and customize CSS with baby blue color
    custom_css = """
    :root {
        --baby-blue: #b3d8fd;
    }
    .title-text {
        text-align: center !important;
        margin-bottom: 10px !important;
    }
    .subtitle-text {
        text-align: center !important;
        margin-bottom: 20px !important;
        opacity: 0.8;
    }
    .supported-formats {
        text-align: center !important;
        font-style: italic;
        opacity: 0.7;
        margin-bottom: 30px !important;
    }
    .container {
        max-width: 1200px;
        margin: 0 auto;
    }
    .file-display {
        min-height: 100px;
        border-radius: 8px;
    }
    .file-container {
        border: 1px solid rgba(179, 216, 253, 0.3);
        padding: 15px;
        border-radius: 10px;
        margin-bottom: 15px;
    }
    .processing-options {
        padding: 15px;
        border-radius: 10px;
        background-color: rgba(179, 216, 253, 0.1);
    }
    .api-options {
        padding: 15px;
        border-radius: 10px;
        background-color: rgba(179, 216, 253, 0.1);
    }
    .primary-button {
        background-color: var(--baby-blue) !important;
        border-color: var(--baby-blue) !important;
    }
    """
    
    # Use "sky" as the primary hue which is closest to baby blue, then customize with CSS
    with gr.Blocks(title="MarkEverythingDown", theme=gr.themes.Soft(primary_hue="sky"), css=custom_css) as app:
        gr.Markdown("# MarkEverythingDown", elem_classes=["title-text"])
        gr.Markdown("Convert anything to structured markdown with AI", elem_classes=["subtitle-text"])
        gr.Markdown("*Supported formats: PDF, DOCX, PPTX, Excel, images, code files, notebooks, and markdown variants*", elem_classes=["supported-formats"])
        
        with gr.Column(elem_classes=["container"]):
            with gr.Row():
                with gr.Column(scale=2):
                    # File upload area with improved styling - using Column instead of Box
                    with gr.Column(elem_classes=["file-container"]):
                        file_input = gr.File(
                            label="Upload Document(s)",
                            file_count="multiple",
                            elem_classes=["file-display"]
                        )
                        
                        with gr.Row():
                            doc_type = gr.Textbox(
                                label="Detected Document Type",
                                interactive=False
                            )
                            
                            # Force vision option (only visible for PDFs)
                            force_vision = gr.Checkbox(
                                label="Use Vision Model for PDFs",
                                value=False,
                                info="Recommended for scanned PDFs"
                            )
                    
                    # Output directory with improved styling
                    output_dir = gr.Textbox(
                        label="Output Directory",
                        placeholder="Default: ./output",
                        value="output",
                        info="Where to save the processed markdown files"
                    )
                    
                    # Processing options with improved styling
                    with gr.Accordion("Processing Options", open=True, elem_classes=["processing-options"]):
                        with gr.Row():
                            max_concurrent = gr.Slider(
                                minimum=1, 
                                maximum=32, 
                                value=2, 
                                step=1, 
                                label="Concurrent Processing",
                                info="Higher values process faster but use more resources"
                            )
                            
                            temperature = gr.Slider(
                                minimum=0.0,
                                maximum=1.0,
                                value=0.0,
                                step=0.1,
                                label="Temperature",
                                info="0.0=deterministic, 1.0=creative"
                            )
                        
                        with gr.Row():
                            images_per_batch = gr.Slider(
                                minimum=1,
                                maximum=4,
                                value=1,
                                step=1,
                                label="Pages Per Batch",
                                info="Maximum number of pages to process in a single API call. Higher values provide better context between pages but use more resources (recommend 1 for smaller models)."
                            )
                            
                            max_tokens = gr.Number(
                                value=None,
                                label="Max Tokens",
                                precision=0,
                                info="Blank uses model default"
                            )
                        
                        with gr.Row():
                            dynamic_batching = gr.Checkbox(
                                label="Dynamic Batching",
                                value=True,
                                info="Automatically adjust batch sizes (≤ Pages Per Batch) based on page complexity. Helps optimize processing while staying within token limits."
                            )
                            
                            max_tokens_per_batch = gr.Slider(
                                minimum=1000,
                                maximum=32768,
                                value=4000,
                                step=500,
                                label="Max Tokens Per Batch",
                                info="Limit for dynamic batching",
                                visible=True
                            )
                    
                    # API configuration with improved styling
                    with gr.Accordion("API Configuration", open=False, elem_classes=["api-options"]):
                        with gr.Row():
                            api_key = gr.Textbox(
                                label="API Key", 
                                placeholder="lmstudio",
                                value="lmstudio",
                                info="API key for vision model"
                            )
                        
                        with gr.Row():
                            api_url = gr.Textbox(
                                label="API URL", 
                                placeholder="http://localhost:1234/v1",
                                value="http://localhost:1234/v1",
                                info="Base URL for API endpoint"
                            )
                            
                            model = gr.Textbox(
                                label="Model Name",
                                placeholder="qwen2.5-vl-7b-instruct",
                                value="qwen2.5-vl-7b-instruct",
                                info="Model to use for processing"
                            )
                    
                    # Process button with improved styling
                    process_btn = gr.Button("Process Document(s)", variant="primary", size="lg")
                
                with gr.Column(scale=3):
                    with gr.Tabs():
                        with gr.TabItem("Preview"):
                            output_text = gr.Code(
                                label="Markdown Output", 
                                language="markdown",
                                lines=25,
                                elem_classes=["markdown-preview"]
                            )
                        with gr.TabItem("Processing Summary"):
                            summary_text = gr.Markdown()
                    
                    output_files = gr.File(
                        label="Processed Files",
                        file_count="multiple",
                        type="filepath"
                    )
            
            # Status message at the bottom
            status = gr.Markdown("*Ready to process documents. Upload files to begin.*")
        
        # When file is uploaded, detect document type of first file
        file_input.change(
            fn=lambda files: detect_document_type(files[0] if files else None),
            inputs=[file_input],
            outputs=[doc_type, force_vision]
        )
        
        # Toggle visibility of max_tokens_per_batch based on dynamic_batching
        dynamic_batching.change(
            fn=lambda x: gr.update(visible=x),
            inputs=[dynamic_batching],
            outputs=[max_tokens_per_batch]
        )
        
        # Process documents when button is clicked
        process_btn.click(
            fn=lambda: gr.Markdown("*Processing documents... Please wait.*"),
            inputs=None,
            outputs=status
        ).then(
            fn=process_documents,
            inputs=[file_input, output_dir, force_vision, max_concurrent, images_per_batch, 
                   api_key, api_url, model, temperature, max_tokens, dynamic_batching, max_tokens_per_batch],
            outputs=[output_text, output_files, summary_text]
        ).then(
            fn=lambda: gr.Markdown("*Processing complete. Results are displayed above.*"),
            inputs=None,
            outputs=status
        )
        
    return app

if __name__ == "__main__":
    app = create_ui()

    app.launch(allowed_paths=["/"])