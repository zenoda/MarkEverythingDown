import os
import argparse
from pathlib import Path
from processors.base import BaseDocumentProcessor

def process_file(file_path, output_dir):
    """Process a single file"""
    try:
        print(f"Processing: {file_path}")
        
        # Get appropriate processor
        processor = BaseDocumentProcessor.get_processor(file_path)
        
        # Process document
        document = processor.process(file_path)
        
        # Generate output filename
        base_name = Path(file_path).stem
        output_base = os.path.join(output_dir, base_name)
        
        md_path = f"{output_base}.md"
        with open(md_path, "w", encoding="utf-8") as f:
            f.write(document.to_markdown())
        print(f"Markdown saved to: {md_path}")
            
    except Exception as e:
        print(f"Error processing {file_path}: {str(e)}")

def process_directory(input_dir, output_dir, format_):
    """Process all files in directory"""
    for root, _, files in os.walk(input_dir):
        for file in files:
            file_path = os.path.join(root, file)
            process_file(file_path, output_dir, format_)

def main():
    parser = argparse.ArgumentParser(description="Convert documents to structured markdown")
    
    parser.add_argument("input", nargs="?", help="Input file or directory path")
    parser.add_argument("--output", "-o", help="Output directory path", default="output")
    parser.add_argument("--ui", action="store_true", help="Launch graphical user interface")
    parser.add_argument("--api-key", help="API key for Qwen vision processor")
    parser.add_argument("--api-url", help="Base URL for API")
    parser.add_argument("--model", help="Model name to use")
    
    args = parser.parse_args()
    
    # Launch UI if requested
    if args.ui:
        from ui.app import create_ui
        app = create_ui()
        # Allow access to any path
        app.launch(allowed_paths=["/"])
        return
    
    # If no input is provided and not in UI mode, print help
    if not args.input:
        parser.print_help()
        return
    
    # Set API configuration if provided
    if args.api_key:
        from processors.vision.vision_processor import VisionDocumentProcessor
        VisionDocumentProcessor.configure_api(
            api_key=args.api_key,
            base_url=args.api_url,
            model=args.model
        )
    
    # Ensure output directory exists
    os.makedirs(args.output, exist_ok=True)
    
    if os.path.isdir(args.input):
        # Process directory
        process_directory(args.input, args.output, args.format)
    else:
        # Process single file
        process_file(args.input, args.output, args.format)

if __name__ == "__main__":
    main()