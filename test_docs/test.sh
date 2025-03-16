# Test with image
python test_processor.py test_docs/test_image1.png \
    --api-key lm_studio \
    --base-url http://localhost:1234/v1 \
    --model qwen2.5-vl-7b-instruct\
    --output-dir /Users/SWRF/Downloads/test

python test_processor.py test_docs/test_image2.png \
    --api-key sk-0e696d7479284160b509173a1be148ff \
    --base-url https://dashscope.aliyuncs.com/compatible-mode/v1 \
    --model qwen2.5-vl-72b-instruct

python test_processor.py test_docs/test_image3.png \
    --api-key lm_studio \
    --base-url http://localhost:1234/v1 \
    --model qwen2.5-vl-7b-instruct

# Test with PDF using vision processing
python test_processor.py test_docs/sample.pdf \
    --api-key sk-0e696d7479284160b509173a1be148ff \
    --base-url https://dashscope.aliyuncs.com/compatible-mode/v1 \
    --model qwen2.5-vl-72b-instruct \
    --force-vision \
    --concurrent 2

# PDF with text processor
python test_processor.py test_docs/sample.pdf

# Word document
python test_processor.py test_docs/sample.docx

# PowerPoint document
python test_processor.py test_docs/sample.pptx

# Jupyter notebook
python test_processor.py test_docs/sample.ipynb

# Python file
python test_processor.py test_docs/sample.py

# R script
python test_processor.py test_docs/sample.r

# R Markdown
python test_processor.py test_docs/sample.rmd

# Plain text
python test_processor.py test_docs/sample.txt