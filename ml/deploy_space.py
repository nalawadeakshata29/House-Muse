# deploy_space.py
from huggingface_hub import HfApi
import os

api = HfApi(token=os.getenv("HF_API_TOKEN"))  # token needs WRITE access
for f in ["app.py", "requirements.txt", "README.md"]:
    api.upload_file(
        path_or_fileobj=f"hf_space/{f}",
        path_in_repo=f,
        repo_id="arnav1711/house-muse-interior-sdxl",
        repo_type="space",
    )
    print("uploaded", f)