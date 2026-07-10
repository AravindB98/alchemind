"""Alchemind — invent, predict, and validate novel chemical compounds.

Public API:
    from alchemind import DiscoveryPipeline
    pipe = DiscoveryPipeline()
    result = pipe.run(seed_smiles="CC(=O)Oc1ccccc1C(=O)O", objective="solubility", n=20)
"""

from .pipeline.discovery import DiscoveryPipeline, DiscoveryResult, Candidate

__version__ = "0.1.0"
__all__ = ["DiscoveryPipeline", "DiscoveryResult", "Candidate", "__version__"]
