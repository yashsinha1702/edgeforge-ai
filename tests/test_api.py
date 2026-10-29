"""CPU-only tests for the FastAPI app (src/app/main.py): no GPU, no model downloads.

The GPU pipeline and the YOLO labeller are replaced by fakes; the prompt engine and the layout
augmenter are the real ones. Run with:  pytest tests/test_api.py
"""
import importlib.util
import io
import json
import sys
import threading
import time
import types
import zipfile
from pathlib import Path

import cv2
import numpy as np
import pytest
from PIL import Image

SRC = Path(__file__).resolve().parents[1] / "src"


def load(name, file):
    spec = importlib.util.spec_from_file_location(name, SRC / "efficient_diffusion_loader" / file)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class FakePipeline:
    """Stands in for EdgeForgePipeline: the real Canny step, a 0.3 s sleep instead of SDXL."""

    active = 0
    max_active = 0
    calls = 0
    lock = threading.Lock()

    def __init__(self, device="cpu"):
        pass

    def preprocess_canny(self, image_path):  # the same steps as pipeline.py
        image = Image.open(image_path).convert("RGB").resize((1024, 1024), Image.LANCZOS)
        edges = cv2.Canny(np.array(image), 100, 200)[:, :, None]
        return Image.fromarray(np.concatenate([edges] * 3, axis=2))

    def generate(self, prompt, control_image, seed=None):
        cls = type(self)
        with cls.lock:
            cls.active += 1
            cls.calls += 1
            cls.max_active = max(cls.max_active, cls.active)
        time.sleep(0.3)
        with cls.lock:
            cls.active -= 1
        return Image.new("RGB", (64, 64), "gray")


class FakeLabeler:
    def label_image(self, image_pil):
        return "2 0.5 0.5 0.4 0.3"


@pytest.fixture(scope="module")
def client(tmp_path_factory, monkeypatch_module):
    fake = types.ModuleType("efficient_diffusion_loader")
    fake.EdgeForgePipeline = FakePipeline
    fake.AutoLabeler = FakeLabeler
    fake.PromptExpander = load("prompt_expander", "prompt_expander.py").PromptExpander
    fake.LayoutAugmenter = load("layout_engine", "layout_engine.py").LayoutAugmenter
    monkeypatch_module.setitem(sys.modules, "efficient_diffusion_loader", fake)
    if importlib.util.find_spec("torch") is None:  # main.py only asks torch whether CUDA is there
        monkeypatch_module.setitem(sys.modules, "torch", types.SimpleNamespace(cuda=types.SimpleNamespace(is_available=lambda: False)))
    monkeypatch_module.syspath_prepend(str(SRC))
    monkeypatch_module.chdir(tmp_path_factory.mktemp("cwd"))
    from fastapi.testclient import TestClient

    main = importlib.import_module("app.main")
    with TestClient(main.app) as c:
        yield c


@pytest.fixture(scope="module")
def monkeypatch_module():
    with pytest.MonkeyPatch.context() as mp:
        yield mp


def layout_png():
    buf = io.BytesIO()
    Image.new("RGB", (200, 120), "white").save(buf, "PNG")
    return buf.getvalue()


def post(client, path, **data):
    return client.post(path, data=data, files={"control_image": ("layout.png", layout_png(), "image/png")})


def test_generate_returns_image_labels_and_prompt(client):
    r = post(client, "/generate", intent="a rusty car, hard to see")
    assert r.status_code == 200
    z = zipfile.ZipFile(io.BytesIO(r.content))
    assert sorted(z.namelist()) == ["generated_image.png", "generated_image.txt", "prompt.json"]
    assert json.loads(z.read("prompt.json"))["prompt"].startswith("a rusty car, hard to see")


def test_batch_returns_a_yolo_dataset_with_prompts(client):
    r = post(client, "/generate_batch", intent="a car on a highway", batch_size="3")
    assert r.status_code == 200
    z = zipfile.ZipFile(io.BytesIO(r.content))
    names = set(z.namelist())
    assert {f"images/train_000{i}.png" for i in range(3)} <= names
    assert {f"labels/train_000{i}.txt" for i in range(3)} <= names
    prompts = json.loads(z.read("prompts.json"))
    assert [p["image"] for p in prompts] == [f"images/train_000{i}.png" for i in range(3)]
    assert all(p["condition"] and p["prompt"].startswith("a car on a highway") for p in prompts)


def test_rejects_a_file_that_is_not_an_image(client):
    r = client.post("/generate", data={"intent": "x"}, files={"control_image": ("notes.txt", b"not an image", "text/plain")})
    assert r.status_code == 400


@pytest.mark.parametrize("size", ["0", "51"])
def test_rejects_batch_sizes_out_of_range(client, size):
    assert post(client, "/generate_batch", intent="x", batch_size=size).status_code == 422


def test_one_gpu_job_at_a_time_and_the_server_keeps_answering(client):
    FakePipeline.max_active = FakePipeline.calls = 0
    codes = []
    jobs = [threading.Thread(target=lambda: codes.append(post(client, "/generate_batch", intent="y", batch_size="3").status_code)) for _ in range(2)]
    for j in jobs:
        j.start()
    time.sleep(0.2)
    t = time.time()
    assert client.get("/openapi.json").status_code == 200
    assert time.time() - t < 0.3  # not stuck behind the generations
    for j in jobs:
        j.join()
    assert codes == [200, 200]
    assert FakePipeline.calls == 6 and FakePipeline.max_active == 1


def test_writes_no_temp_files(client):
    post(client, "/generate", intent="z")
    assert not list(Path.cwd().iterdir())
