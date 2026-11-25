"""UCAI Sovereign Box v0.2.0 — Single-GPU Air-Gapped Research-in-a-Box."""

__version__ = "0.2.0"

from .bench_builder import SovereignBenchItem, build_sovereign_benchmark
from .crate_packager import generate_ro_crate_bundle, verify_ro_crate_bundle
from .evaluator import SovereignBoxEvaluator
from .sovereign_index import PassageChunk, RetrievalHit, SovereignHybridIndex

__all__ = [
    "__version__",
    "PassageChunk",
    "RetrievalHit",
    "SovereignHybridIndex",
    "SovereignBenchItem",
    "build_sovereign_benchmark",
    "SovereignBoxEvaluator",
    "generate_ro_crate_bundle",
    "verify_ro_crate_bundle",
]
