"""Download ONNX models from the Hugging Face Hub into a plain local folder.

Models go into `<cache_dir>/<org>--<name>` as real files rather than the Hub
cache's symlinks: ONNX Runtime refuses external weight files (`model.onnx_data`)
whose resolved path is outside the model's own directory, which large models like
bge-m3 need. Already-downloaded files are reused.
"""

import logging
import time
from pathlib import Path

logger = logging.getLogger(__name__)


def download_onnx_model(repo_id: str, cache_dir: str, patterns: list[str]) -> Path:
    from huggingface_hub import snapshot_download

    target = Path(cache_dir).expanduser() / repo_id.replace("/", "--")
    start = time.perf_counter()
    path = Path(snapshot_download(repo_id, allow_patterns=patterns, local_dir=target))
    logger.info("model %s ready at %s (%.1fs)", repo_id, path, time.perf_counter() - start)
    return path
