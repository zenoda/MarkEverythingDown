# gunicorn.conf.py
workers = 4  # 工作进程数
worker_class = "uvicorn.workers.UvicornWorker"  # 工作进程类型
bind = "0.0.0.0:8000"  # 绑定地址和端口
timeout = 0  # 请求超时时间（秒）,0表示不限制
accesslog = "-"  # 访问日志输出到 stdout
errorlog = "-"  # 错误日志输出到 stderr
