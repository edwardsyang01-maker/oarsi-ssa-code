from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import os
from pathlib import Path
import struct
import time
import zipfile
import zlib
from concurrent.futures import ThreadPoolExecutor, as_completed

import requests

API = "https://data.mendeley.com/public-api/datasets/56rmx5bjcr/files?folder_id=root&version=1"
MODEL = "KneeXrayData/DetKneeData/best_models/1wd/det-epoch-470-0.000076.pth"
COMMIT = "dc19c2462111c3c5b27066e738fc11ddf57a6407"


def save_json(path, value):
    path = Path(path)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2))
    os.replace(tmp, path)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for b in iter(lambda: f.read(4 << 20), b""):
            h.update(b)
    return h.hexdigest()


class RangeZip(io.RawIOBase):
    def __init__(self, session, url, size):
        self.session, self.url, self.size, self.pos = session, url, size, 0

    def seekable(self): return True
    def readable(self): return True
    def tell(self): return self.pos

    def seek(self, offset, whence=0):
        self.pos = offset if whence == 0 else self.pos + offset if whence == 1 else self.size + offset
        return self.pos

    def response(self, start, length):
        end = start + length - 1
        r = self.session.get(self.url, headers={"Range": f"bytes={start}-{end}"}, stream=True, timeout=(20, 90))
        if r.status_code != 206 or r.headers.get("Content-Range") != f"bytes {start}-{end}/{self.size}":
            r.close()
            raise RuntimeError("Range response mismatch")
        self.url = r.url
        return r

    def read(self, n=-1):
        n = min(self.size - self.pos, n if n >= 0 else self.size - self.pos)
        if n <= 0: return b""
        if n > 8 << 20: raise RuntimeError("Unexpectedly large metadata read")
        with self.response(self.pos, n) as r:
            data = r.content
        if len(data) != n: raise RuntimeError("Incomplete metadata range")
        self.pos += n
        return data


def extract_member(reader, info, target, status_path):
    target.parent.mkdir(parents=True, exist_ok=True)
    receipt_path = target.with_suffix(target.suffix + ".receipt.json")
    if target.exists() and receipt_path.exists():
        receipt = json.loads(receipt_path.read_text())
        if target.stat().st_size == info.file_size and sha(target) == receipt["sha256"]:
            if receipt.get("path") != str(target):
                receipt["path"] = str(target)
                save_json(receipt_path, receipt)
            return receipt
        raise RuntimeError("Existing artifact differs from verified receipt")
    reader.seek(info.header_offset)
    head = reader.read(30)
    fields = struct.unpack("<4s5H3L2H", head)
    if fields[0] != b"PK\x03\x04": raise RuntimeError("Invalid ZIP local header")
    offset = info.header_offset + 30 + fields[-2] + fields[-1]
    if info.compress_type not in (0, 8): raise RuntimeError("Unsupported compression")
    dec = zlib.decompressobj(-15) if info.compress_type == 8 else None
    tmp = target.with_suffix(target.suffix + ".part")
    crc = total = received = 0
    h = hashlib.sha256()
    parts_dir = target.parent / (target.name + ".download_parts")
    parts_dir.mkdir(exist_ok=True)
    chunk_size = 8 << 20
    chunks = [(i, start, min(chunk_size, info.compress_size-start)) for i,start in enumerate(range(0, info.compress_size, chunk_size))]
    def fetch_chunk(spec):
        index, start, length = spec
        part = parts_dir / f"{index:04d}.bin"
        for attempt in range(3):
            present = part.stat().st_size if part.exists() else 0
            if present == length: return index
            if present > length: raise RuntimeError("Oversized download part")
            lo = offset + start + present; hi = offset + start + length - 1
            try:
                with requests.get(reader.url, headers={"Range":f"bytes={lo}-{hi}"}, stream=True, timeout=(20,60)) as r:
                    if r.status_code != 206 or r.headers.get("Content-Range") != f"bytes {lo}-{hi}/{reader.size}":raise RuntimeError("Invalid part range")
                    with part.open("ab") as f:
                        for data in r.iter_content(256 << 10): f.write(data)
                if part.stat().st_size == length:return index
            except requests.RequestException:
                if attempt == 2:raise
        raise RuntimeError("Incomplete download part")
    with ThreadPoolExecutor(max_workers=6) as pool:
        futures=[pool.submit(fetch_chunk,spec) for spec in chunks]
        for future in as_completed(futures):
            future.result()
            done=sum(p.stat().st_size for p in parts_dir.glob('*.bin'))
            save_json(status_path,{"state":"downloading_parts","member":info.filename,"received_compressed":done,"compressed_total":info.compress_size})
    with tmp.open("wb") as f:
        for index,start,length in chunks:
          with (parts_dir/f"{index:04d}.bin").open('rb') as part:
           for block in iter(lambda:part.read(1<<20),b''):
            received += len(block)
            data = dec.decompress(block) if dec else block
            f.write(data); h.update(data); crc = zlib.crc32(data, crc); total += len(data)
        if dec:
            data = dec.flush(); f.write(data); h.update(data); crc = zlib.crc32(data, crc); total += len(data)
        f.flush(); os.fsync(f.fileno())
    if received != info.compress_size or total != info.file_size or (crc & 0xffffffff) != info.CRC:
        raise RuntimeError("ZIP member size/CRC mismatch")
    os.replace(tmp, target)
    receipt = {"member": info.filename, "bytes": total, "compressed_bytes": received, "zip_crc32": f"{info.CRC:08x}", "crc_verified": True, "sha256": h.hexdigest(), "path": str(target)}
    save_json(receipt_path, receipt)
    return receipt


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True)
    parser.add_argument("--project", required=True)
    parser.add_argument("--model-only", action="store_true", help="relay for downloading the public model; no patient list is read")
    a = parser.parse_args()
    root, project = Path(a.root), Path(a.project)
    assets = root / "assets"; assets.mkdir(parents=True, exist_ok=True)
    status = root / "download_status.json"
    s = requests.Session()
    response = s.get(API, timeout=45); response.raise_for_status()
    metadata = next(x for x in response.json() if x["filename"] == "KneeXrayData.zip")
    save_json(assets / "mendeley_metadata.json", metadata)
    details = metadata["content_details"]
    reader = RangeZip(s, details["download_url"], details["size"])
    with zipfile.ZipFile(reader) as z:
        infos = z.infolist()
        model = z.getinfo(MODEL)
        detector_inventory = [{"name": i.filename, "size": i.file_size, "compressed": i.compress_size} for i in infos if "/DetKneeData/best_models/" in i.filename and not i.is_dir()]
        save_json(assets / "detector_inventory.json", detector_inventory)
        receipts = [extract_member(reader, model, assets / "chen_yolov2_1wd.pth", status)]
        manifest = project / "outputs/downstream/oa_kl_wholefilm_v2/preparation_20261025/wholefilm_train.csv"
        train_ids = set()
        if not a.model_only:
            with manifest.open() as f: train_ids = {r["patient_id"] for r in csv.DictReader(f)}
        controls = [i for i in infos if "/DetKneeData/H5/trainH5/" in i.filename and i.filename.endswith(".h5") and Path(i.filename).stem in train_ids]
        controls.sort(key=lambda i: hashlib.sha256(("chen-adapter-control-v1|"+i.filename).encode()).hexdigest())
        for index, info in enumerate(controls[:3], 1):
            receipts.append(extract_member(reader, info, assets / f"author_train_control_{index}.h5", status))
    # Pin the source version and use it as read-only reference; no third-party install scripts are run.
    url = f"https://codeload.github.com/PingjunChen/GradingKneeOA/zip/{COMMIT}"
    source = assets / "author_source.zip"
    if not source.exists():
        r = s.get(url, timeout=90); r.raise_for_status(); source.write_bytes(r.content)
    with zipfile.ZipFile(source) as z:
        for info in z.infolist():
            parts = Path(info.filename).parts[1:]
            if not parts or info.is_dir(): continue
            if ".." in parts: raise RuntimeError("Unsafe source member")
            dst = assets / "author_source" / Path(*parts)
            dst.parent.mkdir(parents=True, exist_ok=True)
            dst.write_bytes(z.read(info))
    save_json(assets / "download_receipt.json", {"dataset": "10.17632/56rmx5bjcr.1", "archive_sha256_published": details["sha256_hash"], "whole_archive_downloaded": False, "whole_archive_sha_verified": False, "source_commit": COMMIT, "source_zip_sha256": sha(source), "files": receipts, "author_train_controls": len(receipts)-1})
    save_json(status, {"state": "completed", "model_sha256": receipts[0]["sha256"], "files": len(receipts), "source_commit": COMMIT})
    print(json.dumps({"state": "completed", "model_bytes": receipts[0]["bytes"], "model_sha256": receipts[0]["sha256"], "controls": len(receipts)-1}), flush=True)


if __name__ == "__main__":
    main()
