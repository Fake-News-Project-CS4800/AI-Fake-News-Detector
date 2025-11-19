"""Proof packet data structure for blockchain team handoff.

This module provides ONLY the data packet format that the AI team generates.
The blockchain team is responsible for signing, verification, and blockchain storage.
"""
import json
import time
from typing import Dict, Optional


class ProofPacket:
    """Data packet format for analysis results to be sent to blockchain team."""

    @staticmethod
    def create_packet(
        text_hash: str,
        label: str,
        confidence: float,
        model_version: str,
        reasons: list = None,
        metadata: Optional[Dict] = None
    ) -> Dict:
        """Create a proof packet with analysis results.

        This is the data format the AI team provides to the blockchain team.
        The blockchain team will handle signing and storage.

        Args:
            text_hash: SHA-256 hash of analyzed text
            label: Predicted label (Human/AI/Inconclusive)
            confidence: Confidence score (0-1)
            model_version: Model version identifier
            reasons: List of human-readable reasons for the prediction
            metadata: Optional additional metadata

        Returns:
            Proof packet dictionary ready for blockchain team
        """
        timestamp = int(time.time())

        packet = {
            "version": "1.0",
            "timestamp": timestamp,
            "analysis": {
                "text_hash": text_hash,
                "label": label,
                "confidence": confidence,
                "model_version": model_version,
                "reasons": reasons or []
            }
        }

        # Add optional metadata
        if metadata:
            packet["metadata"] = metadata

        return packet

    @staticmethod
    def save_packet(packet: Dict, filepath: str):
        """Save proof packet to file.

        Args:
            packet: Proof packet to save
            filepath: Path to save to
        """
        with open(filepath, 'w') as f:
            json.dump(packet, f, indent=2)

        print(f"Proof packet saved to {filepath}")

    @staticmethod
    def load_packet(filepath: str) -> Dict:
        """Load proof packet from file.

        Args:
            filepath: Path to load from

        Returns:
            Proof packet dictionary
        """
        with open(filepath, 'r') as f:
            packet = json.load(f)

        return packet

    @staticmethod
    def to_json(packet: Dict) -> str:
        """Convert proof packet to JSON string.

        Args:
            packet: Proof packet

        Returns:
            JSON string
        """
        return json.dumps(packet, indent=2)


# Example usage
if __name__ == "__main__":
    # Example: Create a proof packet for blockchain team
    packet = ProofPacket.create_packet(
        text_hash="e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855",
        label="AI",
        confidence=0.95,
        model_version="1.0.0",
        reasons=[
            "Low lexical diversity (0.45)",
            "Repetitive phrase patterns (12 duplicates)",
            "High frequency of AI-typical transitional phrases"
        ],
        metadata={
            "source": "api",
            "processing_time_ms": 123.45
        }
    )

    print("Proof Packet for Blockchain Team:")
    print(ProofPacket.to_json(packet))
    print("\n" + "="*60)
    print("NOTE: This packet will be sent to the blockchain team,")
    print("who will handle signing and blockchain storage.")
