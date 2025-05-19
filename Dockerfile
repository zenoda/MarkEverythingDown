FROM python:3.10

WORKDIR /app

COPY . .
RUN pip install --no-cache-dir -r requirements.txt  \
    && pip install gunicorn
EXPOSE 8000
CMD ["gunicorn", "-c", "gunicorn.conf.py", "server:app"]