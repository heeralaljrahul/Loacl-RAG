"""Ingestion: walk, extract, chunk, embed, store — incrementally.

Re-running ingest on the same folder is cheap and safe.  Each file is
checked in three widening steps: size+mtime (a stat call), then a SHA-256 of
the bytes, and only then is it actually parsed and embedded.  Editing one
file in a folder of two thousand re-embeds one file.

Roots you ingest are remembered, so ``ingest`` with no arguments rescans
everything you've ever added and picks up whatever changed.
"""

from __future__ import annotations

import hashlib
import time
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from pathlib import Path

from .chunk import chunk_blocks, embed_text
from .config import Config
from .embed import Embedder
from .loaders import LoaderError, is_supported, load
from .store import Store


@dataclass
class IngestReport:
    added: int = 0
    updated: int = 0
    unchanged: int = 0
    skipped: int = 0
    removed: int = 0
    chunks: int = 0
    failures: list[tuple[str, str]] = field(default_factory=list)
    seconds: float = 0.0

    def summary(self) -> str:
        parts = [
            f"{self.added} added", f"{self.updated} updated",
            f"{self.unchanged} unchanged", f"{self.chunks} chunks",
        ]
        if self.removed:
            parts.append(f"{self.removed} removed")
        if self.skipped:
            parts.append(f"{self.skipped} skipped")
        if self.failures:
            parts.append(f"{len(self.failures)} failed")
        return ", ".join(parts) + f" in {self.seconds:.1f}s"


class Ingestor:
    def __init__(self, cfg: Config, store: Store, embedder: Embedder):
        self.cfg = cfg
        self.store = store
        self.embedder = embedder

    def ingest(
        self,
        roots: Iterable[str | Path],
        *,
        force: bool = False,
        prune: bool = True,
        progress: Callable[[str], None] = lambda _: None,
    ) -> IngestReport:
        started = time.perf_counter()
        report = IngestReport()

        roots = [Path(r).expanduser().resolve() for r in roots]
        self._remember(roots)

        files = list(self._walk(roots, report, progress))
        known = self.store.all_document_paths()
        seen: set[str] = set()

        self._check_model_consistency()

        for index, path in enumerate(files, start=1):
            key = str(path)
            seen.add(key)
            try:
                changed = self._ingest_one(path, known.get(key), force, report)
            except LoaderError as exc:
                report.failures.append((key, str(exc)))
                progress(f"  ! {path.name}: {exc}")
                continue
            except Exception as exc:  # noqa: BLE001 - one bad file must not stop a run
                report.failures.append((key, f"{type(exc).__name__}: {exc}"))
                progress(f"  ! {path.name}: {type(exc).__name__}: {exc}")
                continue
            if changed:
                progress(f"  [{index}/{len(files)}] {changed} {path.name}")

        if prune:
            in_roots = [
                doc for path, doc in known.items()
                if path not in seen and self._under(Path(path), roots)
                and not Path(path).exists()
            ]
            for doc in in_roots:
                self.store.delete_document(int(doc["id"]))
                report.removed += 1

        report.seconds = time.perf_counter() - started
        return report

    def rescan(self, **kwargs) -> IngestReport:
        roots = self.store.get_meta("roots", []) or []
        if not roots:
            raise RuntimeError(
                "no folders have been ingested yet — run `ingest <folder>` first"
            )
        return self.ingest(roots, **kwargs)

    def reindex(self, *, progress: Callable[[str], None] = lambda _: None) -> IngestReport:
        """Wipe and rebuild from the remembered roots (use after changing the
        embedding model — old and new vectors are not comparable)."""
        self.store.db.execute("DELETE FROM chunks")
        self.store.db.execute("DELETE FROM documents")
        self.store.db.commit()
        self.store.bump_generation()
        self.store.set_meta("embed_model", self.embedder.name)
        return self.rescan(force=True, progress=progress)

    # -- internals ------------------------------------------------------
    def _check_model_consistency(self):
        stored = self.store.get_meta("embed_model")
        if stored is None:
            self.store.set_meta("embed_model", self.embedder.name)
        elif stored != self.embedder.name:
            raise RuntimeError(
                f"index was built with embedding model '{stored}' but the current "
                f"model is '{self.embedder.name}'. Vectors from different models are "
                f"not comparable — run `python cli.py reindex` to rebuild."
            )

    def _remember(self, roots: list[Path]):
        known = {str(r) for r in (self.store.get_meta("roots", []) or [])}
        known |= {str(r) for r in roots}
        self.store.set_meta("roots", sorted(known))

    @staticmethod
    def _under(path: Path, roots: list[Path]) -> bool:
        return any(path == root or root in path.parents for root in roots)

    def _walk(self, roots: list[Path], report: IngestReport,
              progress: Callable[[str], None]) -> Iterable[Path]:
        ignore = self.cfg.ignore_set
        limit = self.cfg.max_file_mb * 1e6
        for root in roots:
            if root.is_file():
                yield root
                continue
            if not root.exists():
                progress(f"  ! {root} does not exist")
                continue
            for path in sorted(root.rglob("*")):
                if path.is_dir():
                    continue
                if any(part in ignore or part.startswith("~$") for part in path.parts):
                    continue
                if not is_supported(path):
                    report.skipped += 1
                    continue
                try:
                    if path.stat().st_size > limit:
                        report.skipped += 1
                        progress(f"  ~ {path.name} skipped (> {self.cfg.max_file_mb:g} MB)")
                        continue
                except OSError:
                    report.skipped += 1
                    continue
                yield path

    def _ingest_one(self, path: Path, existing, force: bool,
                    report: IngestReport) -> str | None:
        stat = path.stat()
        if existing is not None and not force:
            if (int(existing["bytes"] or 0) == stat.st_size
                    and abs(float(existing["mtime"] or 0) - stat.st_mtime) < 1e-6
                    and int(existing["n_chunks"] or 0) > 0):
                report.unchanged += 1
                report.chunks += int(existing["n_chunks"])
                return None

        digest = _sha256(path)
        if existing is not None and not force and existing["sha256"] == digest \
                and int(existing["n_chunks"] or 0) > 0:
            # Touched but not edited: refresh the stat so the next run is a
            # single stat call again.
            self.store.db.execute(
                "UPDATE documents SET mtime=?, bytes=? WHERE id=?",
                (stat.st_mtime, stat.st_size, existing["id"]),
            )
            self.store.db.commit()
            report.unchanged += 1
            report.chunks += int(existing["n_chunks"])
            return None

        blocks = load(path)
        title = _title(path, blocks)
        chunks = chunk_blocks(
            blocks,
            chunk_chars=self.cfg.chunk_chars,
            overlap_chars=self.cfg.overlap_chars,
            min_chunk_chars=self.cfg.min_chunk_chars,
        )
        if not chunks:
            report.skipped += 1
            return None

        vectors = self.embedder.embed_documents([embed_text(title, c) for c in chunks])

        doc_id = self.store.upsert_document(
            path=str(path), title=title, ext=path.suffix.lower(),
            size=stat.st_size, mtime=stat.st_mtime, sha256=digest,
        )
        self.store.add_chunks(
            doc_id,
            [(c.ord, c.heading, c.page, c.text) for c in chunks],
            vectors,
        )
        report.chunks += len(chunks)
        if existing is None:
            report.added += 1
            return "+"
        report.updated += 1
        return "*"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _title(path: Path, blocks) -> str:
    """Prefer the document's own first heading over the filename — a file
    called ``doc1.pdf`` retrieves badly, "Employee Handbook 2026" does not."""
    for block in blocks[:1]:
        for line in block.text.splitlines()[:12]:
            line = line.strip()
            if line.startswith("#"):
                candidate = line.lstrip("#").strip()
                if 3 <= len(candidate) <= 120:
                    return candidate
            elif 8 <= len(line) <= 90 and line[0].isupper() and not line.endswith("."):
                return line
    return path.stem.replace("_", " ").replace("-", " ").strip()
