"""Air-Gapped Hybrid Literature Index v0.2.0 (Inverted BM25 + Local MiniLM + Numeric Provenance)."""

from __future__ import annotations

import math
import re
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple


_ARXIV_ID_RE = re.compile(r"(2[456]\d{2}\.\d{4,5})")
_ACL_ID_RE = re.compile(r"(202\d\.[a-z-]+\.\d+(?:v\d+)?)", re.IGNORECASE)
_NUM_TOKEN_RE = re.compile(r"(?<![a-zA-Z0-9.])\d[\d,]*(?:\.\d+)?%?(?![a-zA-Z0-9])")


@dataclass
class PassageChunk:
    chunk_id: str
    doc_id: str
    page_num: int
    text: str


@dataclass
class RetrievalHit:
    doc_id: str
    page_num: int
    score: float
    bm25_score: float
    dense_score: float
    snippet: str

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def _tokenize(text: str) -> List[str]:
    return re.findall(r"[a-zA-Z0-9%.+-]{2,}", (text or "").lower())


def extract_doc_id(pdf_path: Path, first_page_text: str = "") -> str:
    """Extracts a canonical arXiv/ACL/stem document identifier from filename or page 1."""
    m = _ARXIV_ID_RE.search(pdf_path.name)
    if m:
        return m.group(1)
    m_acl = _ACL_ID_RE.search(pdf_path.name)
    if m_acl:
        return m_acl.group(1)
    if first_page_text:
        m_txt = _ARXIV_ID_RE.search(first_page_text[:2000])
        if m_txt:
            return m_txt.group(1)
    return pdf_path.stem


def extract_significant_numbers(text: str) -> List[str]:
    """Extracts quantitative metrics while ignoring publication years and arXiv IDs."""
    cleaned = _ARXIV_ID_RE.sub(" ", text or "")
    out: List[str] = []
    for raw in _NUM_TOKEN_RE.findall(cleaned):
        stripped = raw.replace(",", "").rstrip("%")
        try:
            val = float(stripped)
        except ValueError:
            continue
        # Skip plain 4-digit publication years (1990-2029) without decimal or %
        if "." not in raw and "%" not in raw and 1990 <= val <= 2029:
            continue
        # Skip tiny single-digit counts unless formatted with % or decimal
        if "." not in raw and "%" not in raw and val < 10:
            continue
        out.append(raw)
    return list(dict.fromkeys(out))


def _number_variants(num_tok: str) -> List[str]:
    """Generates comma-invariant and percentage<->proportion variants (matching Paper 1 ucai-verify)."""
    raw = num_tok.strip()
    no_comma = raw.replace(",", "")
    variants = {raw, no_comma}
    if raw.endswith("%"):
        try:
            pct = float(no_comma[:-1])
            variants.add(f"{pct / 100.0:.3f}".rstrip("0").rstrip("."))
            variants.add(f"{pct / 100.0:.4f}".rstrip("0").rstrip("."))
            variants.add(no_comma[:-1])
            variants.add(f"{pct:.1f}")
            variants.add(f"{pct:.1f}%")
            if pct == int(pct):
                variants.add(str(int(pct)))
                variants.add(f"{int(pct)}%")
                variants.add(f"{int(pct)}.0")
                variants.add(f"{int(pct)}.0%")
        except ValueError:
            pass
    elif "." not in no_comma and len(no_comma) >= 4:
        try:
            variants.add(f"{int(no_comma):,}")
        except ValueError:
            pass
    return list(variants)


class SovereignHybridIndex:
    """Zero-cloud literature index combining inverted Okapi BM25, local embeddings, and numeric checks."""

    def __init__(self, embed_model_name: str = "sentence-transformers/all-MiniLM-L6-v2") -> None:
        self.embed_model_name = embed_model_name
        self.chunks: List[PassageChunk] = []
        self._doc_lengths: List[int] = []
        self._postings: Dict[str, List[Tuple[int, int]]] = defaultdict(list)
        self._df: Counter[str] = Counter()
        self._doc_full_text: Dict[str, str] = {}
        self._indexed_docs: Set[str] = set()
        self._avgdl: float = 1.0
        self._embedder: Any = None
        self._embeddings: Any = None
        self._reranker: Any = None
        self.device: str = "cpu"
        self.model_vram_mb: float = 0.0

    def build_reranker(self, rerank_model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2") -> Dict[str, Any]:
        """Loads an offline Cross-Encoder reranker on the active GPU/CPU for Stage-2 precision."""
        import torch
        from sentence_transformers import CrossEncoder

        device = "cuda" if torch.cuda.is_available() else "cpu"
        self._reranker = CrossEncoder(rerank_model_name, device=device)
        return {"rerank_model": rerank_model_name, "device": device}

    def has_doc(self, doc_id: str) -> bool:
        return doc_id in self._indexed_docs

    def doc_contains_number(self, doc_id: str, num_tok: str) -> bool:
        txt = self._doc_full_text.get(doc_id, "")
        if not txt:
            return False
        for var in _number_variants(num_tok):
            if var and var in txt:
                return True
        return False

    def ingest_pdf(
        self,
        doc_id: Optional[str],
        pdf_path: str | Path,
        max_pages: int = 20,
        max_chunks: int = 120,
    ) -> Tuple[str, int]:
        import fitz

        p = Path(pdf_path)
        count = 0
        page_texts: List[str] = []
        resolved_id = doc_id or ""

        with fitz.open(str(p)) as doc:
            for p_idx, page in enumerate(doc, start=1):
                if p_idx > max_pages or count >= max_chunks:
                    break
                raw = " ".join((page.get_text("text") or "").split())
                if not raw:
                    continue
                if p_idx == 1 and not resolved_id:
                    resolved_id = extract_doc_id(p, raw)
                page_texts.append(raw)
                step = 550
                for off in range(0, len(raw), step):
                    if count >= max_chunks:
                        break
                    seg = raw[off : off + 750]
                    if len(seg) < 60:
                        continue
                    self._add_chunk(f"{resolved_id}:p{p_idx}:{off}", resolved_id, p_idx, seg)
                    count += 1

        if not resolved_id:
            resolved_id = extract_doc_id(p, "")
        if count > 0:
            self._indexed_docs.add(resolved_id)
            prev = self._doc_full_text.get(resolved_id, "")
            self._doc_full_text[resolved_id] = (prev + "\n" + "\n".join(page_texts)).strip()
            self._avgdl = sum(self._doc_lengths) / max(1, len(self._doc_lengths))
        return resolved_id, count

    def ingest_chunks(self, doc_id: str, pages: List[Tuple[int, str]]) -> int:
        count = 0
        page_texts: List[str] = []
        for p_idx, text in pages:
            seg = " ".join((text or "").split())
            if not seg:
                continue
            page_texts.append(seg)
            self._add_chunk(f"{doc_id}:p{p_idx}", doc_id, p_idx, seg)
            count += 1
        if count > 0:
            self._indexed_docs.add(doc_id)
            prev = self._doc_full_text.get(doc_id, "")
            self._doc_full_text[doc_id] = (prev + "\n" + "\n".join(page_texts)).strip()
            self._avgdl = sum(self._doc_lengths) / max(1, len(self._doc_lengths))
        return count

    def _add_chunk(self, chunk_id: str, doc_id: str, page_num: int, text: str) -> None:
        idx = len(self.chunks)
        self.chunks.append(PassageChunk(chunk_id, doc_id, page_num, text))
        toks = _tokenize(text)
        self._doc_lengths.append(len(toks))
        tf_map = Counter(toks)
        for term, tf in tf_map.items():
            self._postings[term].append((idx, tf))
            self._df[term] += 1

    def build_dense_index(self, batch_size: int = 64) -> Dict[str, Any]:
        import torch
        from sentence_transformers import SentenceTransformer

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        if self.device == "cuda":
            torch.cuda.reset_peak_memory_stats()
        self._embedder = SentenceTransformer(self.embed_model_name, device=self.device)
        texts = [c.text for c in self.chunks]
        self._embeddings = self._embedder.encode(
            texts,
            batch_size=batch_size,
            convert_to_tensor=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        if self.device == "cuda":
            self.model_vram_mb = round(torch.cuda.max_memory_allocated() / (1024 * 1024), 2)
        return {
            "device": self.device,
            "gpu_name": torch.cuda.get_device_name(0) if self.device == "cuda" else "CPU-Fallback",
            "pdfs_indexed": len(self._indexed_docs),
            "chunks_indexed": len(self.chunks),
            "bm25_vocab_size": len(self._df),
            "peak_vram_mb": self.model_vram_mb,
            "embed_model": self.embed_model_name,
        }

    def _bm25_scores(self, query: str, k1: float = 1.5, b: float = 0.75) -> List[float]:
        q_toks = _tokenize(query)
        n_docs = len(self.chunks)
        scores = [0.0] * n_docs
        if not n_docs or not q_toks:
            return scores
        for qt in set(q_toks):
            postings = self._postings.get(qt)
            if not postings:
                continue
            df = self._df[qt]
            idf = math.log((n_docs - df + 0.5) / (df + 0.5) + 1.0)
            for doc_idx, tf in postings:
                dl = self._doc_lengths[doc_idx]
                denom = tf + k1 * (1.0 - b + b * dl / self._avgdl)
                scores[doc_idx] += idf * (tf * (k1 + 1.0)) / denom
        max_s = max(scores) if scores else 1.0
        if max_s > 0:
            scores = [s / max_s for s in scores]
        return scores

    def search(self, query: str, mode: str = "hybrid", top_k: int = 3) -> List[RetrievalHit]:
        if not self.chunks:
            return []
        bm25 = self._bm25_scores(query)
        if mode == "bm25" or self._embedder is None or self._embeddings is None:
            dense = [0.0] * len(self.chunks)
            combined = bm25
        else:
            import torch

            q_emb = self._embedder.encode(
                query, convert_to_tensor=True, normalize_embeddings=True, show_progress_bar=False
            )
            sims = torch.matmul(self._embeddings, q_emb).detach().cpu().tolist()
            dense = [max(0.0, float(x)) for x in sims]
            if mode == "dense":
                combined = dense
            else:
                max_b = max(bm25) if bm25 else 1.0
                norm_bm25 = [b / max_b for b in bm25] if max_b > 0 else bm25
                combined = [0.55 * b + 0.45 * d for b, d in zip(norm_bm25, dense)]

        cand_k = max(top_k, 18) if (mode == "hybrid" and self._reranker is not None) else top_k
        ranked_idx = sorted(range(len(self.chunks)), key=lambda i: combined[i], reverse=True)[:cand_k]
        if mode == "hybrid" and self._reranker is not None and ranked_idx:
            pairs = [(query, self.chunks[i].text[:512]) for i in ranked_idx]
            rr_scores = self._reranker.predict(pairs)
            ranked_pairs = sorted(zip(ranked_idx, rr_scores), key=lambda x: float(x[1]), reverse=True)[:top_k]
            ranked_idx = [i for i, _ in ranked_pairs]
            for i, s in ranked_pairs:
                combined[i] = float(s)
        return [
            RetrievalHit(
                doc_id=self.chunks[i].doc_id,
                page_num=self.chunks[i].page_num,
                score=round(combined[i], 4),
                bm25_score=round(bm25[i], 4),
                dense_score=round(dense[i], 4),
                snippet=self.chunks[i].text[:320],
            )
            for i in ranked_idx
        ]
