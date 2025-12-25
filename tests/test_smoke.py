from fastapi import FastAPI

from serving.app.main import app


def test_app_is_fastapi_instance():
    assert isinstance(app, FastAPI)
