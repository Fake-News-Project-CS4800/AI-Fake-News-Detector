"""Blockchain integration - proof packet generation only.

The AI team is responsible for:
- Computing SHA-256 hashes of analyzed text
- Packaging analysis results (label, confidence, reasons)
- Creating standardized proof packets

The Blockchain team is responsible for:
- Digital signing (EIP-712 or other methods)
- Blockchain storage and transactions
- Verification and retrieval
"""
from .proof_packet import ProofPacket

__all__ = ["ProofPacket"]
