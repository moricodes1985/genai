
"""
Load a small, mixed-format dataset for a license-focused RAG POC.

Returns LlamaIndex `Document` objects with helpful metadata:
- license_id: SPDX identifier if known (e.g., "MIT", "Apache-2.0")
- source: where the content came from (file path or URL snapshot)
- source_type: "plaintext" | "markdown" | "html" | "csv" | "json"
- section_hint: for long licenses we store a best-effort top-level section name
- title: human-friendly title for UI/citations

You can extend/replace these loaders as your corpus grows.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Iterable, List, Optional

from llama_index.core import Document

# ---------- Helpers ----------

def _read_text(path: Path, encoding: str = "utf-8") -> str:
    return path.read_text(encoding=encoding, errors="ignore")


def _norm_title(stem: str) -> str:
    # Make a nice title from filename stem
    return (
        stem.replace("_", " ")
        .replace("-", " ")
        .replace("license", "")
        .strip()
        .title()
    )


def _guess_license_id_from_name(name: str) -> Optional[str]:
    name_low = name.lower()
    for spdx in ["mit", "apache-2.0", "gpl-3.0", "lgpl-3.0", "mpl-2.0", "bsd-3-clause", "bsd-2-clause", "agpl-3.0"]:
        if spdx in name_low:
            return spdx.upper() if spdx == "mit" else spdx
    # Try loose matches
    if "apache" in name_low:
        return "Apache-2.0"
    if "gpl" in name_low and "3" in name_low:
        return "GPL-3.0"
    if "mit" in name_low:
        return "MIT"
    return None


def _mk_doc(text: str, *, source: str, source_type: str, title: str, license_id: Optional[str] = None, extra_meta: Optional[dict] = None) -> Document:
    metadata = {
        "source": source,
        "source_type": source_type,
        "title": title,
    }
    if license_id:
        metadata["license_id"] = license_id
    if extra_meta:
        metadata.update(extra_meta)
    return Document(text=text, metadata=metadata)


# ---------- Plaintext / Markdown licenses ----------

def load_license_files(directory: str | Path, patterns: Iterable[str] = ("*.txt", "*.md")) -> List[Document]:
    """
    Load plaintext/markdown license documents from a folder.
    Adds SPDX-ish license_id if it can be inferred from filename.
    """
    dir_path = Path(directory)
    docs: List[Document] = []
    for pattern in patterns:
        for p in sorted(dir_path.rglob(pattern)):
            text = _read_text(p)
            license_id = _guess_license_id_from_name(p.name)
            # crude section hint: first top-level heading or first line
            first_line = next((ln.strip() for ln in text.splitlines() if ln.strip()), "")
            section_hint = first_line[:120]
            docs.append(
                _mk_doc(
                    text,
                    source=str(p),
                    source_type="markdown" if p.suffix.lower() == ".md" else "plaintext",
                    title=_norm_title(p.stem),
                    license_id=license_id,
                    extra_meta={"section_hint": section_hint},
                )
            )
    return docs


# ---------- Saved HTML snapshots (e.g., OSI pages) ----------

def load_html_pages(directory: str | Path) -> List[Document]:
    """
    Load HTML files saved locally (e.g., OSI overview pages).
    We ingest raw HTML for the POC; LlamaIndex will still index text nodes.
    If you later want cleaner HTML → text, you can preprocess with BeautifulSoup.
    """
    dir_path = Path(directory)
    docs: List[Document] = []
    for p in sorted(dir_path.rglob("*.html")):
        html = _read_text(p)
        title = _norm_title(p.stem)
        docs.append(
            _mk_doc(
                html,
                source=str(p),
                source_type="html",
                title=title,
                license_id=_guess_license_id_from_name(p.name),
            )
        )
    return docs


# ---------- SPDX sample (CSV/JSON) ----------

def load_spdx_csv(path: str | Path, limit: Optional[int] = None) -> List[Document]:
    """
    Load a *small* CSV with a handful of SPDX rows (id, name, osiApproved, referenceUrl, etc.)
    Creates one Document per row to keep retrieval granular.
    """
    p = Path(path)
    docs: List[Document] = []
    with p.open(newline="", encoding="utf-8", errors="ignore") as f:
        reader = csv.DictReader(f)
        for i, row in enumerate(reader):
            if limit is not None and i >= limit:
                break
            license_id = row.get("licenseId") or row.get("id") or _guess_license_id_from_name(json.dumps(row))
            title = f"SPDX: {row.get('name') or license_id or 'License'}"
            text = json.dumps(row, indent=2, ensure_ascii=False)
            docs.append(
                _mk_doc(
                    text,
                    source=str(p),
                    source_type="csv",
                    title=title,
                    license_id=license_id,
                    extra_meta={"spdx_row_index": i},
                )
            )
    return docs


def load_spdx_json(path: str | Path, limit: Optional[int] = None) -> List[Document]:
    """
    Load a small SPDX JSON snapshot. Expects a dict with a 'licenses' array (common in SPDX dumps).
    """
    p = Path(path)
    data = json.loads(_read_text(p))
    licenses = data.get("licenses") or data  # support either a list or {'licenses': [...]}
    docs: List[Document] = []
    for i, row in enumerate(licenses):
        if limit is not None and i >= limit:
            break
        license_id = row.get("licenseId") or _guess_license_id_from_name(json.dumps(row))
        title = f"SPDX: {row.get('name') or license_id or 'License'}"
        text = json.dumps(row, indent=2, ensure_ascii=False)
        docs.append(
            _mk_doc(
                text,
                source=str(p),
                source_type="json",
                title=title,
                license_id=license_id,
                extra_meta={"spdx_item_index": i},
            )
        )
    return docs


# ---------- Turnkey POC loader ----------

def load_poc_corpus(base_dir: str | Path = "data/poc") -> List[Document]:
    """
    Load a minimal, diverse dataset for the POC:
      - licenses/*.txt|.md  (MIT, Apache-2.0, GPL-3.0 ...)
      - web/*.html           (e.g., OSI overview pages)
      - spdx/spdx_sample.csv (a tiny CSV with a few licenses)
    """
    base = Path(base_dir)
    docs: List[Document] = []

    # 1) Plaintext/Markdown licenses
    docs += load_license_files(base / "licenses")

    # 2) HTML snapshots
    web_dir = base / "web"
    if web_dir.exists():
        docs += load_html_pages(web_dir)

    # 3) SPDX sample
    spdx_csv = base / "spdx" / "spdx_sample.csv"
    if spdx_csv.exists():
        docs += load_spdx_csv(spdx_csv, limit=25)

    return docs
