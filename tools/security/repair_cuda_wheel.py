"""Give CUDA wheels portable paths to their declared NVIDIA dependencies."""

import argparse
import hashlib
import json
import os
import stat
import struct
import subprocess
import tempfile
from pathlib import Path, PurePosixPath

from wheel.wheelfile import WheelFile


CUDA_PACKAGES = (
    "cublas",
    "cuda_cupti",
    "cuda_nvrtc",
    "cuda_runtime",
    "cudnn",
    "cufft",
    "curand",
    "cusolver",
    "cusparse",
    "nccl",
    "nvtx",
    "nvjitlink",
)


def sha256(path):
    with path.open("rb") as file:
        return hashlib.file_digest(file, "sha256").hexdigest()


def program_sections(path):
    """Hash every PROGBITS section, including compiled CUDA code and constants."""
    data = path.read_bytes()
    if data[:6] != b"\x7fELF\x02\x01" or struct.unpack_from("<H", data, 18)[0] != 62:
        raise ValueError(f"Expected an ELF64 little-endian x86_64 binary: {path}")
    offset = struct.unpack_from("<Q", data, 40)[0]
    size, count, names_index = struct.unpack_from("<HHH", data, 58)
    if size != 64 or not 0 < names_index < count:
        raise ValueError(f"Unexpected ELF section table: {path}")
    sections = [
        struct.unpack_from("<IIQQQQIIQQ", data, offset + i * size) for i in range(count)
    ]
    names = sections[names_index]
    strings = data[names[4] : names[4] + names[5]]
    hashes = {}
    for section in sections:
        if section[1] != 1:
            continue
        name = strings[section[0] :].split(b"\0", 1)[0].decode("ascii")
        hashes[name] = hashlib.sha256(
            data[section[4] : section[4] + section[5]]
        ).hexdigest()
    return hashes


def repair(wheel, output_directory):
    output_directory.mkdir(parents=True, exist_ok=True)
    output = output_directory / wheel.name
    if output.exists() or wheel.resolve() == output.resolve():
        raise ValueError("The repaired wheel must have a new output path")
    report = {"input_sha256": sha256(wheel), "binaries": []}
    with tempfile.TemporaryDirectory(prefix="torch-portable-wheel-") as directory:
        root = Path(directory)
        with WheelFile(wheel) as archive:
            entries = archive.infolist()
            if len({entry.filename for entry in entries}) != len(entries):
                raise ValueError("Wheel contains duplicate entries")
            for entry in entries:
                name = PurePosixPath(entry.filename)
                if (
                    name.is_absolute()
                    or ".." in name.parts
                    or stat.S_ISLNK(entry.external_attr >> 16)
                ):
                    raise ValueError(f"Unsafe wheel entry: {entry.filename}")
            archive.extractall(root)
            for entry in entries:
                mode = stat.S_IMODE(entry.external_attr >> 16)
                if mode:
                    (root / entry.filename).chmod(mode)
        for path in sorted(root.rglob("*")):
            if not path.is_file():
                continue
            with path.open("rb") as file:
                if file.read(4) != b"\x7fELF":
                    continue
            sections = program_sections(path)
            old = subprocess.check_output(
                ["patchelf", "--print-rpath", str(path)], text=True
            ).strip()
            paths = [
                item
                for item in old.split(":")
                if item.startswith(("$ORIGIN", "${ORIGIN}"))
            ]
            for package in CUDA_PACKAGES:
                relative = os.path.relpath(
                    root / "nvidia" / package / "lib", path.parent
                )
                paths.append(f"$ORIGIN/{relative}")
            new = ":".join(dict.fromkeys(paths))
            before = sha256(path)
            subprocess.run(
                ["patchelf", "--force-rpath", "--set-rpath", new, str(path)], check=True
            )
            if program_sections(path) != sections:
                raise ValueError(f"ELF program/data sections changed: {path}")
            report["binaries"].append(
                {
                    "path": path.relative_to(root).as_posix(),
                    "before_sha256": before,
                    "after_sha256": sha256(path),
                    "old_rpath": old,
                    "new_rpath": new,
                    "unchanged_program_sections": sections,
                }
            )
        if not report["binaries"]:
            raise ValueError("Wheel contains no native binaries")
        with WheelFile(output, "w") as archive:
            archive.write_files(str(root))
    with WheelFile(wheel) as original, WheelFile(output) as repaired:
        names = {entry.filename for entry in original.infolist() if not entry.is_dir()}
        if names != set(repaired.namelist()):
            raise ValueError("Repaired wheel entries differ")
        changed = {item["path"] for item in report["binaries"]} | {original.record_path}
        for name in sorted(names):
            with repaired.open(name) as file:
                actual = hashlib.file_digest(file, "sha256").digest()
            if name not in changed:
                with original.open(name) as file:
                    if hashlib.file_digest(file, "sha256").digest() != actual:
                        raise ValueError(f"Unexpected changed wheel entry: {name}")
    report["output_sha256"] = sha256(output)
    (output_directory / "wheel-repair.json").write_text(
        json.dumps(report, indent=2) + "\n"
    )
    print(json.dumps({"wheel": str(output), "sha256": report["output_sha256"]}))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("wheel", type=Path)
    parser.add_argument("output_directory", type=Path)
    args = parser.parse_args()
    repair(args.wheel, args.output_directory)
