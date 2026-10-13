"""Background subprocess jobs for GUI pseudo labels; no Torch dependency here."""

from copy import deepcopy
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import threading
import uuid

from core.io import ReviewError, atomic_bytes, json_bytes
from core.label_schema import DEFAULT
from pseudo_labeling.registry import MODELS, catalog

PROJECT = Path(__file__).resolve().parents[1]
ACTIVE = {"queued", "running", "cancelling"}


class PseudoJobs:
    def __init__(self, directory=None):
        self.directory = Path(directory or PROJECT / "outputs/pseudo_jobs")
        self.lock = threading.RLock()
        self.job = None
        self.process = None
        self.thread = None
        latest = self.directory / "latest.json"
        if latest.is_file():
            self.job = json.loads(latest.read_text(encoding="utf-8"))
            if self.job["status"] in ACTIVE:
                self.job.update(
                    status="interrupted",
                    error="Server restarted. Check the log/output before starting again.",
                )
                self.persist()

    def info(self):
        return dict(
            models=catalog(),
            python=os.environ.get("PSEUDO_PYTHON") or sys.executable,
            output_parent=str(PROJECT / "outputs"),
        )

    def get(self):
        with self.lock:
            return deepcopy(self.job)

    def persist(self):
        self.directory.mkdir(parents=True, exist_ok=True)
        atomic_bytes(self.directory / "latest.json", json_bytes(self.job))
        atomic_bytes(self.directory / (self.job["id"] + ".json"), json_bytes(self.job))

    def start(self, payload, store):
        if not isinstance(payload, dict):
            raise ReviewError("모델 설정이 필요합니다.")
        task = payload.get("task")
        if task not in MODELS:
            raise ReviewError("지원하지 않는 모델입니다.")
        config = {"task": task}
        # The selected workspace supplies immutable input/class metadata.
        from core.label_schema import load_schema

        config["schema"] = load_schema(store.root)
        source = store.dataset_root.resolve()
        output_text = payload.get("output_root", "")
        if not isinstance(output_text, str) or not output_text.strip():
            raise ReviewError("새 출력 폴더를 지정하세요.")
        output = Path(output_text).expanduser().resolve()
        if (
            output == source
            or output.is_relative_to(source)
            or source.is_relative_to(output)
            or output == store.root
        ):
            raise ReviewError("입력 데이터와 현재 편집 폴더 밖의 새 출력 폴더를 지정하세요.")
        if output.exists() and (not output.is_dir() or any(output.iterdir())):
            raise ReviewError("출력 폴더가 비어 있지 않습니다. 새 폴더를 지정하세요.", 409)
        for key in ("checkpoint", "model_template") if task == "multi_task" else ("checkpoint",):
            value = payload.get(key, "")
            if (
                not isinstance(value, str)
                or not value.strip()
                or not Path(value).expanduser().is_file()
            ):
                raise ReviewError(f"모델 파일이 없습니다: {key}")
            config[key] = str(Path(value).expanduser().resolve())
            if Path(config[key]).suffix.lower() != ".pt":
                raise ReviewError("이 모델은 .pt 체크포인트를 사용합니다.")
        interpreter = payload.get("python") or self.info()["python"]
        executable = shutil.which(interpreter) if isinstance(interpreter, str) else None
        if not executable:
            raise ReviewError("실행할 Python 경로를 확인하세요.")
        device = payload.get("device", "cpu")
        if not isinstance(device, str) or not re.fullmatch(r"cpu|(?:cuda:)?\d+", device):
            raise ReviewError("장치는 cpu 또는 GPU 번호로 입력하세요.")
        config["device"] = device.removeprefix("cuda:")
        for key, default, minimum, maximum in [
            ("height", 352, 32, 4096),
            ("width", 640, 32, 4096),
            ("limit", 0, 0, 10000000),
            ("max_det", 300, 1, 5000),
            ("batch", 8, 1, 128),
        ]:
            value = payload.get(key, default)
            if type(value) is not int or not minimum <= value <= maximum:
                raise ReviewError(f"잘못된 값: {key}")
            config[key] = value
        if config["height"] % 32 or config["width"] % 32:
            raise ReviewError("입력 높이·너비는 32의 배수로 지정하세요.")
        conf = payload.get("conf", 0.2)
        if type(conf) not in (int, float) or not 0 <= conf <= 1:
            raise ReviewError("confidence는 0~1이어야 합니다.")
        config["conf"] = conf
        mapping = payload.get("class_mapping", {})
        if not isinstance(mapping, dict) or not all(
            isinstance(k, str) and type(v) in (str, int) for k, v in mapping.items()
        ):
            raise ReviewError("클래스 매핑은 모델 이름/ID → GUI 이름/ID JSON 객체입니다.")
        config["class_mapping"] = mapping
        if task == "multi_task":
            for kind in ("detection", "segmentation"):
                by_id = {t["id"]: t for t in config["schema"][kind]}
                if any(
                    t["id"] not in by_id
                    or by_id[t["id"]]["name"] != t["name"]
                    or not by_id[t["id"]]["active"]
                    for t in DEFAULT[kind]
                ):
                    raise ReviewError(
                        "공동 모델은 기본 Golfcart/Person/Tree/Undef_obj 및 도로 클래스가 필요합니다."
                    )
            if mapping:
                raise ReviewError("공동 모델의 클래스 매핑은 학습 설정으로 고정됩니다.")
        config.update(dataset_root=str(source), output_root=str(output))
        with self.lock:
            if self.thread and self.thread.is_alive():
                raise ReviewError("이미 pseudo labeling 작업이 실행 중입니다.", 409)
            job_id = uuid.uuid4().hex
            self.directory.mkdir(parents=True, exist_ok=True)
            request = self.directory / (job_id + "_config.json")
            atomic_bytes(request, json_bytes(config))
            command = [
                executable,
                "-u",
                "-m",
                "scripts.generate_single_task",
                "--config",
                str(request),
            ]
            if task == "multi_task":
                command = [
                    executable,
                    "-u",
                    "-m",
                    "scripts.generate_pseudo_labels",
                    "--include-labeled",
                    "--workers",
                    "0",
                    "--no-previews",
                ]
                for key in (
                    "dataset_root",
                    "output_root",
                    "checkpoint",
                    "model_template",
                    "device",
                    "height",
                    "width",
                    "conf",
                    "max_det",
                    "limit",
                    "batch",
                ):
                    command.extend(["--" + key.replace("_", "-"), str(config[key])])
            self.job = dict(
                id=job_id,
                task=task,
                status="queued",
                input=str(source),
                output=str(output),
                created_utc=datetime.now(timezone.utc).isoformat(),
                done=0,
                total=0,
                log="",
                error="",
                checkpoint=config["checkpoint"],
                python=executable,
            )
            self.persist()
            self.thread = threading.Thread(target=self.run, args=(command,), daemon=True)
            self.thread.start()
            return deepcopy(self.job)

    def run(self, command):
        log_file = self.directory / (self.job["id"] + ".log")
        try:
            environment = dict(os.environ, PYTHONUTF8="1", PYTHONIOENCODING="utf-8")
            with self.lock:
                if self.job["status"] == "cancelling":
                    self.job["status"] = "cancelled"
                    self.persist()
                    return
                self.process = subprocess.Popen(
                    command,
                    cwd=PROJECT,
                    env=environment,
                    stdin=subprocess.DEVNULL,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True,
                    encoding="utf-8",
                    errors="replace",
                )
                self.job["status"] = "running"
                self.persist()
            with log_file.open("w", encoding="utf-8", newline="\n") as log:
                for line in self.process.stdout:
                    log.write(line)
                    log.flush()
                    with self.lock:
                        self.job["log"] = (self.job["log"] + line)[-10000:]
                        match = re.search(r"Generated (\d+)/(\d+)", line)
                        if match:
                            self.job.update(done=int(match[1]), total=int(match[2]))
                        match = re.search(r"Selected=(\d+) existing=(\d+) pending=(\d+)", line)
                        if match:
                            self.job.update(done=0, total=int(match[3]))
                        self.persist()
            code = self.process.wait()
            with self.lock:
                cancelled = self.job["status"] == "cancelling"
                self.job["status"] = (
                    "cancelled" if cancelled else "completed" if code == 0 else "error"
                )
                if code and not cancelled:
                    self.job["error"] = (
                        f"생성 실패 (exit {code}). 아래 로그에서 환경·체크포인트·매핑을 확인하세요."
                    )
                self.persist()
        except Exception as error:
            with self.lock:
                self.job.update(status="error", error=str(error))
                self.persist()
        finally:
            if self.process and self.process.stdout:
                self.process.stdout.close()
            self.process = None

    def cancel(self):
        with self.lock:
            if self.job and self.job["status"] in ACTIVE:
                self.job["status"] = "cancelling"
                if self.process and self.process.poll() is None:
                    self.process.terminate()
                self.persist()
            return deepcopy(self.job)

    def close(self):
        self.cancel()
        if self.thread:
            self.thread.join(5)
        if self.process and self.process.poll() is None:
            self.process.kill()
