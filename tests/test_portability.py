"""Real process locks, raw pipes and Unicode launch paths on Windows and Linux."""

import base64
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time
import unittest
from unittest.mock import patch
from urllib.request import urlopen

from core.file_lock import exclusive_lock
from core.io import inside
from sam.pipe_reader import LineReader
from sam.service import SamService, SamError
from run import PROJECT, launch_arguments


class FileLockTests(unittest.TestCase):
    def test_process_exclusion_release_and_persistent_lock_file(self):
        with tempfile.TemporaryDirectory() as tmp:
            lock = Path(tmp) / "한글 lock"
            code = """import sys
from core.file_lock import exclusive_lock
try:
    with exclusive_lock(sys.argv[1]):
        print('acquired')
except BlockingIOError:
    print('busy')
    raise SystemExit(9)
"""

            def attempt():
                return subprocess.run(
                    [sys.executable, "-c", code, str(lock)],
                    cwd=PROJECT,
                    capture_output=True,
                    text=True,
                    encoding="utf-8",
                    timeout=15,
                )

            with exclusive_lock(lock):
                with self.assertRaises(BlockingIOError):
                    with exclusive_lock(lock):
                        pass
                result = attempt()
                self.assertEqual(result.returncode, 9, result.stderr)
            self.assertEqual(attempt().returncode, 0)
            self.assertTrue(lock.is_file())
            with self.assertRaises(ValueError):
                with exclusive_lock(lock):
                    raise ValueError("intentional failure")
            self.assertEqual(attempt().returncode, 0)

    def test_windows_slashes_and_traversal_are_normalized(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp).resolve()
            self.assertEqual(inside(root, r"images\사진.png"), root / "images" / "사진.png")
            from core.io import ReviewError

            with self.assertRaises(ReviewError):
                inside(root, r"..\outside.png")


class PipeTests(unittest.TestCase):
    def spawn(self, code):
        process = subprocess.Popen(
            [sys.executable, "-u", "-c", code], stdout=subprocess.PIPE, stderr=subprocess.PIPE
        )
        self.addCleanup(self.cleanup, process)
        return process

    @staticmethod
    def cleanup(process):
        if process.poll() is None:
            process.kill()
        process.wait(timeout=10)
        for stream in [process.stdout, process.stderr]:
            stream.close()

    def test_fragmented_utf8_crlf_multiple_lines_and_eof(self):
        process = self.spawn(
            "import os,time;os.write(1,b'{\\\"note\\\":');time.sleep(.05);os.write(1,'\\\"한글\\\"}\\r\\n{}\\n'.encode('utf-8'))"
        )
        reader = LineReader(process.stdout)
        self.addCleanup(reader.close)
        self.assertEqual(json.loads(reader.read(10)), {"note": "한글"})
        self.assertEqual(json.loads(reader.read(10)), {})
        self.assertIsNone(reader.read(10))

    def test_deadline_incomplete_and_oversized_responses(self):
        for code, message, limit in [
            ("import time;time.sleep(10)", "timeout", 64),
            ('import os;os.write(1,b"incomplete")', "incomplete", 64),
            ('import os;os.write(1,b"x"*128)', "large", 64),
        ]:
            process = self.spawn(code)
            reader = LineReader(process.stdout, limit=limit)
            try:
                if message == "timeout":
                    with self.assertRaises(TimeoutError):
                        reader.read(0.05)
                else:
                    with self.assertRaisesRegex(RuntimeError, message):
                        reader.read(10)
            finally:
                if process.poll() is None:
                    process.kill()
                process.wait(timeout=10)
                reader.close()

    def test_worker_idle_exit_without_buffered_stdin_shutdown_error(self):
        process = subprocess.Popen(
            [
                sys.executable,
                "-u",
                str(PROJECT / "sam/worker.py"),
                "--checkpoint",
                "unused",
                "--config",
                "unused",
                "--idle-seconds",
                "1",
            ],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        try:
            process.wait(timeout=15)  # Keep stdin open to exercise the idle deadline.
            _, errors = process.communicate(timeout=5)
            self.assertEqual(process.returncode, 0, errors.decode(errors="replace"))
        finally:
            if process.poll() is None:
                process.kill()
                process.wait()
            for stream in [process.stdin, process.stdout, process.stderr]:
                stream.close()


class SamProcessTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name).resolve()
        self.script = self.root / "한글 fake worker.py"
        self.script.write_text(
            """import base64,json,sys,time
mode=sys.argv[sys.argv.index('--config')+1]
count=0
for line in sys.stdin.buffer:
    request=json.loads(line)
    if mode=='slow':time.sleep(10)
    if mode=='eof':break
    if mode=='error':
        print(json.dumps({'error':'worker failed'}),flush=True)
        break
    result=dict(width=request['width'],height=request['height'],
        mask=base64.b64encode(bytes([1])*request['width']*request['height']).decode(),
        cached_image=count>0,image_path=request['image_path'],note='한글')
    print(json.dumps(result,ensure_ascii=False),flush=True)
    count+=1
""",
            encoding="utf-8",
            newline="\n",
        )
        self.weight = self.root / "fake.pt"
        self.weight.write_bytes(b"fixture")
        self.service = SamService(
            python=sys.executable, checkpoint=self.weight, config="echo", timeout=10
        )
        original = subprocess.Popen

        def factory(command, **kwargs):
            return original([command[0], command[1], str(self.script), *command[3:]], **kwargs)

        self.patch = patch("sam.service.subprocess.Popen", side_effect=factory)
        self.patch.start()

    def tearDown(self):
        self.service.close()
        self.patch.stop()
        self.tmp.cleanup()

    def predict(self):
        return self.service.predict(
            self.root / "한글 image.png", 8, 6, [{"x": 1, "y": 1, "label": 1}]
        )

    def test_persistent_unicode_worker_and_cleanup(self):
        # Worker logs/protocol remain UTF-8 even if the parent locale is ASCII.
        with patch.dict(os.environ, {"PYTHONUTF8": "0", "PYTHONIOENCODING": "ascii"}):
            first = self.predict()
        process = self.service.process
        second = self.predict()
        self.assertEqual(base64.b64decode(first["mask"]), bytes([1]) * 48)
        self.assertEqual(first["note"], "한글")
        self.assertTrue(second["cached_image"])
        self.assertIs(process, self.service.process)
        self.service.close()
        self.assertIsNotNone(process.poll())
        self.assertIsNone(self.service.reader)

    def test_timeout_error_eof_and_restart(self):
        for mode in ["slow", "error", "eof"]:
            self.service.config = mode
            self.service.timeout = 0.1 if mode == "slow" else 10
            with self.assertRaises(SamError):
                self.predict()
            self.assertIsNone(self.service.process)
            self.service.config = "echo"
            self.service.timeout = 10
            self.assertEqual(self.predict()["width"], 8)
            self.service.close()


class LauncherTests(unittest.TestCase):
    def test_shared_environment_defaults_and_overrides(self):
        env = {"INPUT": r"C:\한글 data\images", "OUTPUT_DIR": r"D:\labels output", "PORT": "9001"}
        args = launch_arguments("gui", ["--port", "8766"], env)
        self.assertEqual(args[1], env["INPUT"])
        self.assertEqual(args[3], env["OUTPUT_DIR"])
        self.assertEqual(args[-2:], ["--port", "8766"])
        self.assertIn("cpu", launch_arguments("pseudo", [], {}))
        result = subprocess.run(
            [sys.executable, str(PROJECT / "run.py"), "gui", "--help"],
            cwd=tempfile.gettempdir(),
            capture_output=True,
            text=True,
            encoding="utf-8",
            timeout=20,
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("--dataset-root", result.stdout)

    def test_launcher_actual_gui_with_spaced_unicode_input_and_output(self):
        from http.server import HTTPServer

        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp) / "한글 data"
            source = base / "input images"
            source.mkdir(parents=True)
            # Reserve a currently free local port, release it for the child server.
            reserve = HTTPServer(("127.0.0.1", 0), type("Handler", (), {}))
            port = reserve.server_port
            reserve.server_close()
            env = dict(
                os.environ,
                INPUT=str(source),
                OUTPUT_DIR=str(base / "labels output"),
                PORT=str(port),
            )
            process = subprocess.Popen(
                [sys.executable, str(PROJECT / "run.py"), "gui"],
                cwd=tmp,
                env=env,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )
            try:
                deadline = time.monotonic() + 20
                while time.monotonic() < deadline:
                    if process.poll() is not None:
                        _, err = process.communicate()
                        self.fail(err.decode(errors="replace"))
                    try:
                        with urlopen(
                            f"http://127.0.0.1:{port}/api/config", timeout=0.3
                        ) as response:
                            config = json.load(response)
                        break
                    except OSError:
                        time.sleep(0.1)
                else:
                    self.fail("GUI startup deadline exceeded")
                self.assertEqual(Path(config["dataset_root"]), source.resolve())
                self.assertEqual(Path(config["root"]), (base / "labels output").resolve())
                self.assertEqual(Path(config["export_parent"]), (base / "exports").resolve())
            finally:
                if process.poll() is None:
                    process.terminate()
                process.communicate(timeout=10)

    @unittest.skipUnless(os.name == "nt", "native CMD launcher runs on Windows CI")
    def test_windows_batch_entrypoint(self):
        import shutil

        with tempfile.TemporaryDirectory() as tmp:
            checkout = Path(tmp) / "한글 project with spaces"
            checkout.mkdir()
            for name in ["run_gui.bat", "run.py"]:
                shutil.copy2(PROJECT / name, checkout / name)
            for name in ["core", "gui", "scripts", "sam", "pseudo_labeling"]:
                shutil.copytree(
                    PROJECT / name, checkout / name, ignore=shutil.ignore_patterns("__pycache__")
                )
            # CMD /s /c needs outer quotes around a quoted batch-file command.
            command = f'cmd.exe /d /s /c ""{checkout / "run_gui.bat"}" --help"'
            result = subprocess.run(
                command,
                cwd=tmp,
                env=dict(os.environ, PYTHON=sys.executable),
                capture_output=True,
                encoding="utf-8",
                timeout=20,
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertIn("--dataset-root", result.stdout)


if __name__ == "__main__":
    unittest.main()
