"""
Checkpoint model for persisting Genetic Algorithm state.
Enables resume capability for interrupted runs.
"""

from typing import List, Dict, Any, Optional
from dataclasses import dataclass, field, asdict
from datetime import datetime
import json
from pathlib import Path


@dataclass
class GACheckpoint:
    """
    Represents the complete state of the Genetic Algorithm at a point in time.

    This checkpoint can be serialized to JSON and restored to resume execution.
    """

    version: str = "1.0"

    # Timestamp when checkpoint was created
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())

    # Current generation number
    generation: int = 0

    # Current population of scenarios (serialized)
    population: List[Dict[str, Any]] = field(default_factory=list)

    # All previously seen scenario configurations (to avoid duplicates)
    seen_population: List[Dict[str, Any]] = field(default_factory=list)

    # Best scenario from each generation
    best_of_generation: List[Dict[str, Any]] = field(default_factory=list)

    # Random number generator state (for reproducibility)
    rng_state: Optional[Dict[str, Any]] = None

    # Configuration used for this run
    config_snapshot: Optional[Dict[str, Any]] = None

    # Metadata
    total_scenarios_evaluated: int = 0
    run_id: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert checkpoint to dictionary."""
        return asdict(self)

    def to_json(self) -> str:
        """Serialize checkpoint to JSON string."""
        return json.dumps(self.to_dict(), indent=2, default=str)

    def save(self, filepath: Path) -> None:
        """
        Save checkpoint to file.

        Args:
            filepath: Path where checkpoint should be saved
        """
        filepath.parent.mkdir(parents=True, exist_ok=True)

        with open(filepath, "w") as f:
            f.write(self.to_json())

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "GACheckpoint":
        """
        Create checkpoint from dictionary.

        Args:
            data: Dictionary containing checkpoint data

        Returns:
            GACheckpoint instance
        """
        return cls(**data)

    @classmethod
    def from_json(cls, json_str: str) -> "GACheckpoint":
        """
        Deserialize checkpoint from JSON string.

        Args:
            json_str: JSON string containing checkpoint data

        Returns:
            GACheckpoint instance
        """
        data = json.loads(json_str)
        return cls.from_dict(data)

    @classmethod
    def load(cls, filepath: Path) -> "GACheckpoint":
        """
        Load checkpoint from file.

        Args:
            filepath: Path to checkpoint file

        Returns:
            GACheckpoint instance

        Raises:
            FileNotFoundError: If checkpoint file doesn't exist
            ValueError: If checkpoint file is invalid
        """
        if not filepath.exists():
            raise FileNotFoundError(f"Checkpoint file not found: {filepath}")

        try:
            with open(filepath, "r") as f:
                return cls.from_json(f.read())
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid checkpoint file: {e}") from e

    def validate(self) -> bool:
        """
        Validate checkpoint data integrity.

        Returns:
            True if checkpoint is valid

        Raises:
            ValueError: If checkpoint data is invalid
        """
        if self.generation < 0:
            raise ValueError(f"Invalid generation number: {self.generation}")

        if not self.population:
            raise ValueError("Population cannot be empty")

        if self.total_scenarios_evaluated < len(self.seen_population):
            raise ValueError("Inconsistent scenario count")

        return True


@dataclass
class CheckpointManager:
    """
    Manages checkpoint creation, loading, and validation.
    """

    output_dir: Path
    checkpoint_filename: str = "checkpoint.json"
    auto_save: bool = True
    keep_history: bool = False  # Keep checkpoints from each generation

    def get_checkpoint_path(self, generation: Optional[int] = None) -> Path:
        """
        Get path to checkpoint file.

        Args:
            generation: Optional generation number for historical checkpoints

        Returns:
            Path to checkpoint file
        """
        if generation is not None and self.keep_history:
            filename = f"checkpoint_gen_{generation}.json"
        else:
            filename = self.checkpoint_filename

        return self.output_dir / filename

    def save_checkpoint(
        self, checkpoint: GACheckpoint, generation: Optional[int] = None
    ) -> Path:
        """
        Save checkpoint to disk.

        Args:
            checkpoint: Checkpoint to save
            generation: Optional generation number for historical checkpoints

        Returns:
            Path where checkpoint was saved
        """
        filepath = self.get_checkpoint_path(generation)
        checkpoint.save(filepath)
        return filepath

    def load_checkpoint(self, filepath: Optional[Path] = None) -> GACheckpoint:
        """
        Load checkpoint from disk.

        Args:
            filepath: Optional custom path to checkpoint file

        Returns:
            Loaded checkpoint
        """
        if filepath is None:
            filepath = self.get_checkpoint_path()

        checkpoint = GACheckpoint.load(filepath)
        checkpoint.validate()
        return checkpoint

    def checkpoint_exists(self) -> bool:
        """Check if a checkpoint file exists."""
        return self.get_checkpoint_path().exists()

    def list_checkpoints(self) -> List[Path]:
        """List all checkpoint files in the output directory."""
        if self.keep_history:
            return sorted(self.output_dir.glob("checkpoint_gen_*.json"))
        else:
            checkpoint_path = self.get_checkpoint_path()
            return [checkpoint_path] if checkpoint_path.exists() else []
