#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import os
import re
import sys
import tarfile
import urllib.request
import xml.etree.ElementTree as ET
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))

from PIL import Image

from radiotriage import CHEST_LABELS


REPORTS_URL = "https://openi.nlm.nih.gov/imgs/collections/NLMCXR_reports.tgz"
IMAGES_URL = "https://openi.nlm.nih.gov/imgs/collections/NLMCXR_png.tgz"

LABEL_PATTERNS = {
    "atelectasis": [r"\batelect"],
    "cardiomegaly": [r"\bcardiomeg", r"enlarged cardiac", r"enlarged heart"],
    "effusion": [r"\beffusion", r"pleural fluid"],
    "infiltration": [r"\binfiltrat", r"airspace opacity", r"interstitial opacity"],
    "mass": [r"\bmass\b"],
    "nodule": [r"\bnodule", r"nodular"],
    "pneumonia": [r"\bpneumonia"],
    "pneumothorax": [r"\bpneumothorax"],
    "consolidation": [r"\bconsolidat"],
    "edema": [r"\bedema", r"pulmonary vascular congestion"],
    "emphysema": [r"\bemphysema", r"hyperinflation", r"hyperinflated"],
    "fibrosis": [r"\bfibrosis", r"fibrotic", r"scarring", r"scar\b"],
    "pleural_thickening": [r"pleural thickening", r"pleural scar"],
    "hernia": [r"\bhernia"],
}

NEGATION_PREFIX = re.compile(
    r"\b(no|without|negative for|absence of|no evidence of|there is no|there are no|free of)\b"
    r"[^.]{0,80}$",
    flags=re.IGNORECASE,
)


def download(url: str, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if out_path.exists() and out_path.stat().st_size > 0:
        return
    print(f"Downloading {url} -> {out_path}")
    urllib.request.urlretrieve(url, out_path)


def extract(archive: Path, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    marker = out_dir / ".extracted"
    if marker.exists():
        return
    print(f"Extracting {archive} -> {out_dir}")
    with tarfile.open(archive, "r:gz") as tar:
        tar.extractall(out_dir)
    marker.write_text("ok\n", encoding="utf-8")


def text_of(elem: ET.Element | None) -> str:
    if elem is None or elem.text is None:
        return ""
    return elem.text.strip()


def report_sections(root: ET.Element) -> dict[str, str]:
    sections: dict[str, str] = {}
    for node in root.findall(".//AbstractText"):
        label = node.attrib.get("Label", "").upper()
        sections[label] = text_of(node)
    return sections


def mesh_terms(root: ET.Element) -> list[str]:
    terms = []
    for tag in ("major", "minor"):
        for node in root.findall(f".//MeSH/{tag}"):
            if node.text:
                terms.append(node.text.strip())
    return terms


def image_ids(root: ET.Element) -> list[str]:
    ids = []
    for node in root.findall(".//parentImage"):
        img_id = node.attrib.get("id")
        if img_id:
            ids.append(img_id)
    return ids


def find_image(images_root: Path, img_id: str) -> Path | None:
    candidates = [
        images_root / f"{img_id}.png",
        images_root / "NLMCXR_png" / f"{img_id}.png",
        images_root / "png" / f"{img_id}.png",
    ]
    for path in candidates:
        if path.exists():
            return path
    matches = list(images_root.rglob(f"{img_id}.png"))
    return matches[0] if matches else None


def valid_image(path: Path) -> bool:
    try:
        with Image.open(path) as img:
            img.verify()
        with Image.open(path) as img:
            img.convert("L").load()
        return True
    except Exception:
        return False


def is_negated(text: str, start: int) -> bool:
    prefix = text[max(0, start - 100) : start]
    return bool(NEGATION_PREFIX.search(prefix))


def label_report(text: str, mesh: list[str]) -> dict[str, int]:
    haystack = " ".join([text, " ".join(mesh)]).lower()
    labels: dict[str, int] = {}
    for label, patterns in LABEL_PATTERNS.items():
        value = 0
        for pattern in patterns:
            for match in re.finditer(pattern, haystack, flags=re.IGNORECASE):
                if not is_negated(haystack, match.start()):
                    value = 1
                    break
            if value:
                break
        labels[label] = value
    return labels


def build_manifest(args: argparse.Namespace) -> Path:
    reports_dir = Path(args.reports_dir)
    images_dir = Path(args.images_dir)
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    rows = []
    missing_images = 0
    broken_images = 0
    for xml_path in sorted(reports_dir.rglob("*.xml")):
        root = ET.parse(xml_path).getroot()
        sections = report_sections(root)
        findings = sections.get("FINDINGS", "")
        impression = sections.get("IMPRESSION", "")
        indication = sections.get("INDICATION", "")
        report = " ".join(part for part in [indication, findings, impression] if part).replace("\n", " ").strip()
        if not report:
            continue
        labels = label_report(report, mesh_terms(root))
        for img_id in image_ids(root):
            image_path = find_image(images_dir, img_id)
            if image_path is None:
                missing_images += 1
                continue
            if not valid_image(image_path):
                broken_images += 1
                continue
            row = {
                "image_path": os.path.relpath(image_path, out_path.parent),
                "report": report,
                "source_xml": os.path.relpath(xml_path, out_path.parent),
            }
            row.update(labels)
            rows.append(row)
            if args.max_rows and len(rows) >= args.max_rows:
                break
        if args.max_rows and len(rows) >= args.max_rows:
            break

    with out_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["image_path", "report", "source_xml", *CHEST_LABELS])
        writer.writeheader()
        writer.writerows(rows)

    print(f"Wrote {len(rows)} rows to {out_path}")
    print(f"Skipped image references without local PNG: {missing_images}")
    print(f"Skipped broken PNG files: {broken_images}")
    positives = {label: sum(row[label] for row in rows) for label in CHEST_LABELS}
    print("Positive labels:", positives)
    return out_path


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--download-reports", action="store_true")
    p.add_argument("--download-images", action="store_true", help="Downloads the official 1.3 GB PNG archive.")
    p.add_argument("--extract", action="store_true")
    p.add_argument("--reports-url", default=REPORTS_URL)
    p.add_argument("--images-url", default=IMAGES_URL)
    p.add_argument("--raw-dir", default="data/openi/raw")
    p.add_argument("--reports-dir", default="data/openi/reports")
    p.add_argument("--images-dir", default="data/openi/images")
    p.add_argument("--output", default="data/openi/manifest.csv")
    p.add_argument("--max-rows", type=int, default=None)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    raw_dir = Path(args.raw_dir)
    reports_archive = raw_dir / "NLMCXR_reports.tgz"
    images_archive = raw_dir / "NLMCXR_png.tgz"
    if args.download_reports:
        download(args.reports_url, reports_archive)
    if args.download_images:
        download(args.images_url, images_archive)
    if args.extract:
        if reports_archive.exists():
            extract(reports_archive, Path(args.reports_dir))
        if images_archive.exists():
            extract(images_archive, Path(args.images_dir))
    build_manifest(args)


if __name__ == "__main__":
    main()
