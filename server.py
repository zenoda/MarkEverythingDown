import os
from fastapi import FastAPI, File, UploadFile, HTTPException, Form
from pathlib import Path
from typing import Optional
from datetime import datetime
from processors.base import BaseDocumentProcessor, DocumentType
from processors.vision.vision_processor import VisionDocumentProcessor

# 创建 FastAPI 应用实例
app = FastAPI(title="MarkEverythingDown API", description="", version="1.0.0")

# 配置临时目录
TMP_DIR = Path("tmp")
TMP_DIR.mkdir(exist_ok=True)

# 允许的文件类型
ALLOWED_EXTENSIONS = {"pdf", "png", "jpg", "jpeg", "gif", "docx", "xlsx", "pptx"}

# 最大文件大小 (10MB)
MAX_FILE_SIZE = 10 * 1024 * 1024


def allowed_file(filename: str) -> bool:
    """检查文件扩展名是否允许"""
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


def validate_file_size(file_size: int) -> None:
    """验证文件大小是否超出限制"""
    if file_size > MAX_FILE_SIZE:
        raise HTTPException(
            status_code=413,
            detail=f"文件大小超过限制 ({MAX_FILE_SIZE / 1024 / 1024} MB)"
        )


def process_file(file_path, force_vision=False, max_concurrent=2, images_per_batch=1,
                 temperature=0.0, max_tokens=None, dynamic_batching=True, max_tokens_per_batch=4000):
    print(f"Processing: {file_path}")

    # Determine document type first
    doc_type = DocumentType.from_file_extension(file_path)

    # Select appropriate processor based on type and options
    if force_vision and doc_type == DocumentType.PDF:
        print(f"Using Vision processor for PDF: {file_path}")
        processor = VisionDocumentProcessor(temperature=temperature, max_tokens=max_tokens)
    elif doc_type == DocumentType.IMAGE:
        print(f"Using Vision processor for image: {file_path}")
        processor = VisionDocumentProcessor(temperature=temperature, max_tokens=max_tokens)
    else:
        # Use standard document detection
        processor = BaseDocumentProcessor.get_processor(file_path)

    print(f"Using {processor.__class__.__name__} for {file_path}")

    # Process document - with max_concurrent and images_per_batch for PDFs
    if isinstance(processor, VisionDocumentProcessor) and doc_type == DocumentType.PDF:
        document = processor.process(
            file_path,
            max_concurrent=max_concurrent,
            images_per_batch=images_per_batch,
            dynamic_batching=dynamic_batching,
            max_tokens_per_batch=max_tokens_per_batch
        )
        print(f"Processed PDF with {max_concurrent} concurrent workers, dynamic batching: {dynamic_batching}")
    else:
        document = processor.process(file_path)

    return {
        "content": document.to_markdown()
    }


# 定义 API 端点
@app.post("/process")
async def process(file: UploadFile = File(...),
                  api_key: str = Form(...),
                  base_url: str = Form(...),
                  model: str = Form(...),
                  max_concurrent: Optional[int] = Form(2),
                  images_per_batch: Optional[int] = Form(1),
                  temperature: Optional[float] = Form(0.0),
                  max_tokens: Optional[int] = Form(0),
                  dynamic_batching: Optional[bool] = Form(True),
                  max_tokens_per_batch: Optional[int] = Form(4000)
                  ):
    # 验证文件类型
    if not allowed_file(file.filename):
        raise HTTPException(
            status_code=400,
            detail=f"不支持的文件类型。允许的类型: {', '.join(ALLOWED_EXTENSIONS)}"
        )

    # 读取文件内容并验证大小
    contents = await file.read()
    validate_file_size(len(contents))

    # 生成唯一文件名
    timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
    file_ext = file.filename.rsplit('.', 1)[1] if '.' in file.filename else ""
    unique_filename = f"{timestamp}_{file.filename}"
    file_path = TMP_DIR / unique_filename

    # 保存文件
    with open(file_path, "wb") as f:
        f.write(contents)

    # 设置LLM api参数
    VisionDocumentProcessor.configure_api(
        api_key=api_key,
        base_url=base_url,
        model=model
    )
    try:
        return process_file(str(file_path), False, max_concurrent, images_per_batch, temperature, max_tokens,
                            dynamic_batching, max_tokens_per_batch)
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(
            status_code=500,
            detail=str(e)
        )
    finally:
        os.remove(file_path)


# 运行应用（开发环境）
if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
