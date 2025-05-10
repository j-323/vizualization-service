import subprocess
from pathlib import Path
from src.models.base import ModelClient

class LocalImageC(ModelClient):
    def __init__(self):
        super().__init__("ImageC_Local")

    def _call_model(self, prompt: str) -> str:
        out = Path("/tmp") / f"img_c_{abs(hash(prompt))}.png"
        subprocess.run(
            ["python", "run_local_img_c.py", "--prompt", prompt, "--out", str(out)],
            check=True
        )
        return str(out)

class LocalVideoD(ModelClient):
    def __init__(self):
        super().__init__("VideoD_Local")

    def _call_model(self, prompt: str) -> str:
        out = Path("/tmp") / f"vid_d_{abs(hash(prompt))}.mp4"
        subprocess.run(
            ["python", "run_local_vid_d.py", "--prompt", prompt, "--out", str(out)],
            check=True
        )
        return str(out)