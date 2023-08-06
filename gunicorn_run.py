import os
from app import fastapi_app
from gunicorn.app.base import BaseApplication

class StandaloneApplication(BaseApplication):
    def __init__(self, fastapi_app, options=None):
        self.options = options or {}
        self.application = fastapi_app
        super().__init__()

    def load_config(self):
        config = {key: value for key, value in self.options.items() if key in self.cfg.settings and value is not None}
        for key, value in config.items():
            self.cfg.set(key.lower(), value)

    def load(self):
        return self.application

if __name__ == "__main__":
    gunicorn_port = os.getenv("GUNICORN_PORT", "8080")
    workers = os.getenv("WORKERS", "1")
    
    options = {
        "bind": f"0.0.0.0:{gunicorn_port}",
        "workers": int(workers),
        "accesslog": "-",
        "errorlog": "-",
        "worker_class": "uvicorn.workers.UvicornWorker",
        "log_level": "info",
    }
    
    StandaloneApplication(fastapi_app, options).run()