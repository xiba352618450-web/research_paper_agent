from __future__ import annotations

import hashlib
import re
import sys
import unicodedata
from dataclasses import dataclass
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]

EXPECTED_PDFS = {
    "01 Attention Is All You Need.pdf": 15,
    "02 Language Models are Few-Shot Learners.pdf": 75,
    "03 InstructGPT RLHF - Training Language Models to Follow Instructions with Human Feedback.pdf": 68,
    "04 RAG - Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks.pdf": 19,
    "05 LoRA - Low-Rank Adaptation of Large Language Models.pdf": 26,
    "06 ReAct - Synergizing Reasoning and Acting in Language Models.pdf": 33,
}

EXCLUDED_DIR_NAMES = {".git", "eval", "__pycache__", ".pytest_cache", "tmp", "temp"}


@dataclass(frozen=True)
class Anchor:
    label: str
    any_of: tuple[str, ...]
    mode: str = "contains"


@dataclass
class PdfInfo:
    path: Path
    page_count: int
    sha256: str


ANCHORS: dict[str, dict[int, list[Anchor]]] = {
    "05 LoRA - Low-Rank Adaptation of Large Language Models.pdf": {
        19: [
            Anchor("D.4 GPT-3", ("D.4 GPT-3",)),
            Anchor("AdamW", ("AdamW",)),
            Anchor("2 epochs", ("2 epochs",)),
            Anchor("batch size of 128", ("batch size of 128",)),
            Anchor("weight decay factor of 0.1", ("weight decay factor of 0.1",)),
        ],
        20: [
            Anchor("4.7M", ("4.7M", "4.7 M")),
            Anchor("37.7M", ("37.7M", "37.7 M")),
            Anchor("rq = rv = 1", ("rq=rv=1",), "compact"),
            Anchor("rq = rv = 8", ("rq=rv=8",), "compact"),
            Anchor("Table 12 reference", ("Table 12",)),
            Anchor("not actual Table 12 table page", ("hyperparametersfinetunepreembedprelayerbitfitadapterhlora",), "not_compact"),
        ],
        21: [
            Anchor("Table 12", ("Table 12",)),
            Anchor("Optimizer", ("Optimizer",)),
            Anchor("Batch Size", ("Batch Size",)),
            Anchor("# Epoch", ("# Epoch",)),
            Anchor("Warmup Tokens", ("Warmup Tokens",)),
            Anchor("LR Schedule", ("LR Schedule",)),
            Anchor("LoRA learning rate", ("2.00E-04", "0.0002")),
        ],
        23: [
            Anchor("Table 15", ("Table 15",)),
            Anchor("rank/config content", ("rq=rv=8", "rq=rk=rv=ro=2", "rv=2"), "compact"),
        ],
    },
    "01 Attention Is All You Need.pdf": {
        4: [
            Anchor("scaled dot-product attention", ("scaled dot-product attention",)),
            Anchor("softmax", ("softmax",)),
            Anchor("Q/K/V formula", ("attention(q,k,v)", "qkt", "querywithallkeys"), "compact"),
            Anchor("sqrt(d_k)", ("√dk", "sqrt(dk)", "1√dk"), "compact"),
        ],
        5: [
            Anchor("h = 8", ("h = 8", "h=8")),
            Anchor("dmodel/h = 64", ("dmodel/h=64", "d model/h=64"), "compact"),
        ],
        9: [
            Anchor("Table 3", ("Table 3",)),
            Anchor("base", ("base",)),
            Anchor("512", ("512",)),
            Anchor("8", ("8",)),
        ],
    },
    "04 RAG - Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks.pdf": {
        2: [
            Anchor("retriever", ("retriever",)),
            Anchor("generator", ("generator",)),
            Anchor("marginalize", ("marginalize", "margin-alize")),
        ],
        3: [
            Anchor("RAG-Sequence", ("rag-sequence",)),
            Anchor("RAG-Token", ("rag-token",)),
        ],
        4: [
            Anchor(
                "negative marginal log likelihood",
                ("negative marginal log-likelihood", "negative log likelihood", "negativemarginalloglikelihood"),
                "contains_or_compact",
            ),
            Anchor("decoding", ("decoding",)),
        ],
    },
    "06 ReAct - Synergizing Reasoning and Acting in Language Models.pdf": {
        5: [
            Anchor("3,000 trajectories", ("3,000 trajectories", "3000 trajectories", "3000trajectories"), "contains_or_compact"),
            Anchor("finetune smaller language models", ("finetune smaller language models",)),
        ],
        15: [
            Anchor("HOTPOTQA FINETUNING DETAILS", ("hotpotqafinetuningdetails",), "compact"),
            Anchor("batch size of 64", ("batch size of 64",)),
            Anchor("4,000 steps", ("4,000 steps", "4000 steps", "4000steps"), "contains_or_compact"),
        ],
    },
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def is_excluded(path: Path) -> bool:
    rel_parts = path.relative_to(PROJECT_ROOT).parts
    return any(part in EXCLUDED_DIR_NAMES for part in rel_parts[:-1])


def locate_pdf(filename: str) -> Path:
    matches = [path for path in PROJECT_ROOT.rglob(filename) if path.is_file() and not is_excluded(path)]
    if not matches:
        raise FileNotFoundError(f"Missing PDF: {filename}")
    if len(matches) > 1:
        joined = "\n".join(str(path) for path in matches)
        raise RuntimeError(f"Multiple PDFs named {filename}:\n{joined}")
    return matches[0]


def get_pdf_reader(path: Path) -> Any:
    try:
        import fitz  # type: ignore

        return ("fitz", fitz.open(path))
    except Exception:
        try:
            from pypdf import PdfReader

            return ("pypdf", PdfReader(str(path)))
        except Exception as exc:
            raise RuntimeError("No supported PDF reader found. Install PyMuPDF or pypdf.") from exc


def page_count(reader_kind: str, reader: Any) -> int:
    if reader_kind == "fitz":
        return int(reader.page_count)
    return len(reader.pages)


def extract_page_text(reader_kind: str, reader: Any, page_number: int) -> str:
    index = page_number - 1
    if reader_kind == "fitz":
        return reader.load_page(index).get_text("text") or ""
    return reader.pages[index].extract_text() or ""


def normalize_text(text: str) -> dict[str, str]:
    normalized = unicodedata.normalize("NFKC", text)
    normalized = normalized.replace("\u2212", "-")
    folded = re.sub(r"\s+", " ", normalized).strip().casefold()
    no_commas = folded.replace(",", "")
    compact = re.sub(r"[\s,]+", "", folded)
    return {"folded": folded, "no_commas": no_commas, "compact": compact}


def anchor_matches(anchor: Anchor, normalized: dict[str, str]) -> bool:
    candidates = [unicodedata.normalize("NFKC", item).casefold() for item in anchor.any_of]
    if anchor.mode == "contains":
        return any(item in normalized["folded"] or item in normalized["no_commas"] for item in candidates)
    if anchor.mode == "compact":
        compact_candidates = [re.sub(r"[\s,]+", "", item) for item in candidates]
        return any(item in normalized["compact"] for item in compact_candidates)
    if anchor.mode == "contains_or_compact":
        compact_candidates = [re.sub(r"[\s,]+", "", item) for item in candidates]
        return any(item in normalized["folded"] or item in normalized["no_commas"] for item in candidates) or any(
            item in normalized["compact"] for item in compact_candidates
        )
    if anchor.mode == "not_compact":
        compact_candidates = [re.sub(r"[\s,]+", "", item) for item in candidates]
        return not any(item in normalized["compact"] for item in compact_candidates)
    raise ValueError(f"Unknown anchor mode: {anchor.mode}")


def verify_all() -> tuple[list[str], dict[str, PdfInfo]]:
    errors: list[str] = []
    infos: dict[str, PdfInfo] = {}

    for filename, expected_page_count in EXPECTED_PDFS.items():
        try:
            path = locate_pdf(filename)
            reader_kind, reader = get_pdf_reader(path)
            actual_page_count = page_count(reader_kind, reader)
            infos[filename] = PdfInfo(path=path, page_count=actual_page_count, sha256=sha256_file(path))
            if actual_page_count != expected_page_count:
                errors.append(f"{filename}: expected {expected_page_count} pages, found {actual_page_count}")

            for page_number, anchors in ANCHORS.get(filename, {}).items():
                if page_number > actual_page_count:
                    errors.append(f"{filename} page {page_number}: page is out of range")
                    continue
                text = extract_page_text(reader_kind, reader, page_number)
                normalized = normalize_text(text)
                missing = [anchor.label for anchor in anchors if not anchor_matches(anchor, normalized)]
                if missing:
                    snippet = normalized["folded"][:700]
                    errors.append(
                        f"{filename} page {page_number}: missing anchors {missing}; extracted snippet: {snippet}"
                    )
        except Exception as exc:
            errors.append(f"{filename}: {exc}")

    return errors, infos


def main() -> int:
    errors, infos = verify_all()
    if errors:
        print("PDF anchor validation failed:", file=sys.stderr)
        for error in errors:
            print(f"- {error}", file=sys.stderr)
        return 1

    print("OK: PDF anchors validated.")
    for filename in sorted(infos):
        info = infos[filename]
        print(f"- {filename}: pages={info.page_count}, sha256={info.sha256}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
