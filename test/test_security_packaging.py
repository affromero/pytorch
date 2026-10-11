"""Source metadata and detected Windows toolchain behavior without GPU execution."""

import email
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from packaging.requirements import Requirement
from packaging.version import Version


SOURCE = Path(__file__).resolve().parents[1]


class SourcePackagingTests(unittest.TestCase):
    def test_source_metadata_preserves_security_version_and_cuda_override(self):
        for override in (None, "2.3.0+cu121.security1"):
            with self.subTest(override=override), tempfile.TemporaryDirectory() as directory:
                environment = os.environ.copy()
                environment.pop("PYTORCH_BUILD_VERSION", None)
                environment.pop("PYTORCH_BUILD_NUMBER", None)
                shadow = Path(directory) / "shadow" / "tools"
                shadow.mkdir(parents=True)
                (shadow / "__init__.py").write_text("")
                environment["PYTHONPATH"] = str(shadow.parent)
                if override is not None:
                    environment["PYTORCH_BUILD_VERSION"] = override
                    environment["PYTORCH_BUILD_NUMBER"] = "1"
                code = (
                    "import sys; "
                    "sys.path.insert(0, " + repr(str(SOURCE)) + "); "
                    "sys.path.insert(0, " + repr(str(shadow.parent)) + "); "
                    "import setuptools.build_meta; "
                    "backend = setuptools.build_meta.__legacy__; "
                    "backend.get_requires_for_build_wheel(); "
                    "backend.prepare_metadata_for_build_wheel(" + repr(directory) + ")"
                )
                result = subprocess.run(
                    [sys.executable, "-c", code],
                    cwd=SOURCE,
                    env=environment,
                    text=True,
                    capture_output=True,
                    timeout=90,
                )
                self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
                metadata_files = list(Path(directory).glob("torch-*.dist-info/METADATA"))
                self.assertEqual(len(metadata_files), 1)
                metadata = email.message_from_string(metadata_files[0].read_text())
                self.assertEqual(Version(metadata["Version"]), Version(override or "2.3.0+security.1"))
                self.assertEqual(Version(metadata["Version"]).public, "2.3.0")
                self.assertIn(
                    Requirement("typing-extensions>=4.8.0"),
                    [Requirement(value) for value in metadata.get_all("Requires-Dist")],
                )

    def test_windows_source_requires_the_original_detected_cuda_toolchain(self):
        detected = {
            "WIN32": "TRUE",
            "MSVC": "TRUE",
            "USE_CUDA": "TRUE",
            "CAFFE2_USE_CUDA": "TRUE",
            "CAFFE2_USE_CUDNN": "TRUE",
            "CUDA_VERSION": "12.1",
            "CUDNN_VERSION": "8.9.2",
            "TORCH_BUILD_VERSION": "2.3.0+security.1",
        }
        cases = [("valid", {}, True)]
        for flag in ("MSVC", "USE_CUDA", "CAFFE2_USE_CUDA", "CAFFE2_USE_CUDNN"):
            cases.append((flag, {flag: "FALSE"}, False))
        for name, value in (
            ("CUDA_VERSION", "12.6"),
            ("CUDNN_VERSION", "9.0.0"),
            ("CUDA_VERSION", None),
            ("CUDNN_VERSION", None),
        ):
            cases.append((name + str(value), {name: value}, False))
        cases.extend(
            [
                ("non-Windows CPU", {"WIN32": "FALSE", "USE_CUDA": "FALSE"}, True),
                ("explicit wheel build", {"TORCH_BUILD_VERSION": "2.3.0+cu121.security1", "USE_CUDA": "FALSE"}, True),
            ]
        )
        for name, changes, succeeds in cases:
            with self.subTest(case=name), tempfile.TemporaryDirectory() as directory:
                values = {**detected, **changes}
                script = Path(directory) / "toolchain.cmake"
                script.write_text(
                    "\n".join(f'set({key} "{value}")' for key, value in values.items() if value is not None)
                    + f'\ninclude("{(SOURCE / "cmake/SecurityBackport.cmake").as_posix()}")\n'
                )
                result = subprocess.run(["cmake", "-P", str(script)], text=True, capture_output=True, timeout=15)
                if succeeds:
                    self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
                else:
                    self.assertNotEqual(result.returncode, 0)
                    self.assertIn("requires MSVC", result.stderr)


if __name__ == "__main__":
    unittest.main(verbosity=2)
