import os
import json
import datetime
from typing import Dict, Any


class GeneticCheckpoint:
    """
    Exact checkpoint save/load logic extracted from GeneticAlgorithm.
    NO logic change. NO schema change.
    """

    def __init__(self, output_dir: str, checkpoint_path: str = None):
        self.checkpoint_file = checkpoint_path or os.path.join(
            output_dir, "checkpoint.json"
        )

    def save(
        self,
        *,
        generation: int,
        run_uuid: str,
        seed,
        population,
        seen_population,
        best_of_generation,
        rng_state,
        stagnant_generations: int,
        scenario_mutation_rate: float,
        start_time,
        serialize_scenarios,
        scenario_to_key,
    ):
        checkpoint_data = {
            "version": "1.0",
            "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
            "generation": generation,
            "completed_generations": generation,
            "run_uuid": run_uuid,
            "seed": seed,
            "population": serialize_scenarios(population),
            "seen_population": {
                scenario_to_key(s): r.model_dump(mode="json")
                for s, r in seen_population.items()
            },
            "best_of_generation": [
                r.model_dump(mode="json") for r in best_of_generation
            ],
            "rng_state": rng_state,
            "stagnant_generations": stagnant_generations,
            "scenario_mutation_rate": scenario_mutation_rate,
            "start_time": start_time.isoformat() if start_time else None,
        }

        temp_file = f"{self.checkpoint_file}.tmp"
        with open(temp_file, "w", encoding="utf-8") as f:
            json.dump(checkpoint_data, f, indent=2)

        os.replace(temp_file, self.checkpoint_file)

    def load(self) -> Dict[str, Any]:
        if not os.path.exists(self.checkpoint_file):
            raise FileNotFoundError(
                f"Checkpoint file not found: {self.checkpoint_file}"
            )

        try:
            with open(self.checkpoint_file, "r", encoding="utf-8") as f:
                return json.load(f)
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid checkpoint file: {e}") from e
