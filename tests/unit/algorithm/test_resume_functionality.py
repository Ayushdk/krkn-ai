"""
Resume functionality tests for GeneticAlgorithm
"""

import os
import json
import pytest
from unittest.mock import Mock, patch

from krkn_ai.algorithm.genetic import GeneticAlgorithm
from krkn_ai.models.scenario.base import CompositeScenario, CompositeDependency
from krkn_ai.models.scenario.scenario_dummy import DummyScenario
from krkn_ai.models.cluster_components import ClusterComponents


class TestResumeCheckpointSaving:
    """Test checkpoint saving functionality"""

    def test_save_checkpoint_creates_file(self, genetic_algorithm, temp_output_dir):
        """Test that checkpoint file is created when saving"""
        genetic_algorithm.population = [
            DummyScenario(cluster_components=ClusterComponents())
        ]
        genetic_algorithm.best_of_generation = []
        genetic_algorithm.seen_population = {}

        # Save checkpoint
        genetic_algorithm._save_checkpoint(1)

        # Verify checkpoint file exists
        checkpoint_path = os.path.join(temp_output_dir, "checkpoint.json")
        assert os.path.exists(checkpoint_path)

    def test_save_checkpoint_contains_required_fields(
        self, genetic_algorithm, temp_output_dir
    ):
        """Test that checkpoint contains all required fields"""
        genetic_algorithm.population = [
            DummyScenario(cluster_components=ClusterComponents())
        ]
        genetic_algorithm.best_of_generation = []
        genetic_algorithm.seen_population = {}

        # Save checkpoint
        genetic_algorithm._save_checkpoint(2)

        # Read and verify checkpoint content
        checkpoint_path = os.path.join(temp_output_dir, "checkpoint.json")
        with open(checkpoint_path, "r") as f:
            checkpoint_data = json.load(f)

        required_fields = [
            "version",
            "timestamp",
            "generation",
            "completed_generations",
            "run_uuid",
            "seed",
            "population",
            "seen_population",
            "best_of_generation",
            "rng_state",
            "stagnant_generations",
            "scenario_mutation_rate",
            "start_time",
        ]

        for field in required_fields:
            assert field in checkpoint_data, f"Missing required field: {field}"

        assert checkpoint_data["generation"] == 2
        assert checkpoint_data["completed_generations"] == 2

    def test_save_checkpoint_with_composite_scenario(
        self, genetic_algorithm, temp_output_dir
    ):
        """Test checkpoint saving with CompositeScenario in population"""
        dummy_scenario_a = DummyScenario(cluster_components=ClusterComponents())
        dummy_scenario_b = DummyScenario(cluster_components=ClusterComponents())
        composite_scenario = CompositeScenario(
            scenario_a=dummy_scenario_a,
            scenario_b=dummy_scenario_b,
            dependency=CompositeDependency.A_ON_B,
        )

        genetic_algorithm.population = [composite_scenario]
        genetic_algorithm.best_of_generation = []
        genetic_algorithm.seen_population = {}

        # Save checkpoint
        genetic_algorithm._save_checkpoint(1)

        # Verify checkpoint file exists and contains CompositeScenario
        checkpoint_path = os.path.join(temp_output_dir, "checkpoint.json")
        assert os.path.exists(checkpoint_path)

        with open(checkpoint_path, "r") as f:
            checkpoint_data = json.load(f)

        assert len(checkpoint_data["population"]) == 1
        assert checkpoint_data["population"][0]["__class__"] == "CompositeScenario"


class TestResumeCheckpointLoading:
    """Test checkpoint loading functionality"""

    def test_load_checkpoint_file_not_found(self, genetic_algorithm):
        """Test loading checkpoint when file doesn't exist"""
        genetic_algorithm.checkpoint_file = "/nonexistent/checkpoint.json"

        with pytest.raises(FileNotFoundError, match="Checkpoint file not found"):
            genetic_algorithm._load_checkpoint()

    def test_load_checkpoint_restores_generation(
        self, genetic_algorithm, temp_output_dir
    ):
        """Test that checkpoint loading restores generation correctly"""
        # Create a checkpoint file
        checkpoint_data = {
            "version": "1.0",
            "timestamp": "2024-01-01T00:00:00+00:00",
            "generation": 3,
            "completed_generations": 3,
            "run_uuid": "test-uuid",
            "seed": None,
            "population": [],
            "seen_population": {},
            "best_of_generation": [],
            "rng_state": {
                "seed": None,
                "bit_generator": "PCG64",
                "state": {
                    "bit_generator": "PCG64",
                    "state": {"state": 123, "inc": 456},
                    "has_uint32": 0,
                    "uinteger": 789,
                },
                "timestamp": "2024-01-01T00:00:00+00:00",
            },
            "stagnant_generations": 0,
            "scenario_mutation_rate": 0.6,
            "start_time": "2024-01-01T00:00:00+00:00",
        }

        checkpoint_path = os.path.join(temp_output_dir, "checkpoint.json")
        with open(checkpoint_path, "w") as f:
            json.dump(checkpoint_data, f)

        genetic_algorithm.checkpoint_file = checkpoint_path

        # Load checkpoint
        genetic_algorithm._load_checkpoint()

        # Verify generation is restored
        assert genetic_algorithm.completed_generations == 3

    def test_load_checkpoint_restores_population(
        self, genetic_algorithm, temp_output_dir
    ):
        """Test that checkpoint loading restores population"""
        # Create checkpoint with population data
        population_data = [
            {
                "name": "dummy-scenario",
                "krknctl_name": "dummy",
                "krknhub_image": "dummy:latest",
                "__class__": "DummyScenario",
                "__str__": "dummy-scenario",
            }
        ]

        checkpoint_data = {
            "version": "1.0",
            "timestamp": "2024-01-01T00:00:00+00:00",
            "generation": 2,
            "completed_generations": 2,
            "run_uuid": "test-uuid",
            "seed": None,
            "population": population_data,
            "seen_population": {},
            "best_of_generation": [],
            "rng_state": {
                "seed": None,
                "bit_generator": "PCG64",
                "state": {
                    "bit_generator": "PCG64",
                    "state": {"state": 123, "inc": 456},
                    "has_uint32": 0,
                    "uinteger": 789,
                },
                "timestamp": "2024-01-01T00:00:00+00:00",
            },
            "stagnant_generations": 0,
            "scenario_mutation_rate": 0.6,
            "start_time": "2024-01-01T00:00:00+00:00",
        }

        checkpoint_path = os.path.join(temp_output_dir, "checkpoint.json")
        with open(checkpoint_path, "w") as f:
            json.dump(checkpoint_data, f)

        genetic_algorithm.checkpoint_file = checkpoint_path

        # Mock DummyScenario in allowed_scenario_classes
        genetic_algorithm.allowed_scenario_classes["DummyScenario"] = DummyScenario

        # Load checkpoint
        genetic_algorithm._load_checkpoint()

        # Verify population is restored
        assert len(genetic_algorithm.population) == 1
        assert isinstance(genetic_algorithm.population[0], DummyScenario)

    def test_load_checkpoint_restores_composite_scenario(
        self, genetic_algorithm, temp_output_dir
    ):
        """Test that checkpoint loading restores CompositeScenario correctly"""
        # Create checkpoint with CompositeScenario data
        population_data = [
            {
                "name": "composite",
                "krknctl_name": "",
                "krknhub_image": "",
                "scenario_a": {
                    "name": "dummy-scenario",
                    "krknctl_name": "dummy",
                    "krknhub_image": "dummy:latest",
                },
                "scenario_b": {
                    "name": "dummy-scenario",
                    "krknctl_name": "dummy",
                    "krknhub_image": "dummy:latest",
                },
                "dependency": 1,
                "__class__": "CompositeScenario",
                "__str__": "composite",
            }
        ]

        checkpoint_data = {
            "version": "1.0",
            "timestamp": "2024-01-01T00:00:00+00:00",
            "generation": 2,
            "completed_generations": 2,
            "run_uuid": "test-uuid",
            "seed": None,
            "population": population_data,
            "seen_population": {},
            "best_of_generation": [],
            "rng_state": {
                "seed": None,
                "bit_generator": "PCG64",
                "state": {
                    "bit_generator": "PCG64",
                    "state": {"state": 123, "inc": 456},
                    "has_uint32": 0,
                    "uinteger": 789,
                },
                "timestamp": "2024-01-01T00:00:00+00:00",
            },
            "stagnant_generations": 0,
            "scenario_mutation_rate": 0.6,
            "start_time": "2024-01-01T00:00:00+00:00",
        }

        checkpoint_path = os.path.join(temp_output_dir, "checkpoint.json")
        with open(checkpoint_path, "w") as f:
            json.dump(checkpoint_data, f)

        genetic_algorithm.checkpoint_file = checkpoint_path

        # Mock ScenarioFactory for nested scenario generation
        with patch(
            "krkn_ai.algorithm.genetic.ScenarioFactory.generate_random_scenario"
        ) as mock_gen:
            mock_gen.return_value = DummyScenario(
                cluster_components=ClusterComponents()
            )

            # Load checkpoint
            genetic_algorithm._load_checkpoint()

        # Verify CompositeScenario is restored
        assert len(genetic_algorithm.population) == 1
        assert isinstance(genetic_algorithm.population[0], CompositeScenario)

    def test_load_checkpoint_handles_corrupted_json(
        self, genetic_algorithm, temp_output_dir
    ):
        """Test that loading handles corrupted JSON gracefully"""
        checkpoint_path = os.path.join(temp_output_dir, "checkpoint.json")
        with open(checkpoint_path, "w") as f:
            f.write("invalid json content {")

        genetic_algorithm.checkpoint_file = checkpoint_path

        with pytest.raises(ValueError, match="Invalid checkpoint file"):
            genetic_algorithm._load_checkpoint()


class TestResumeInitialization:
    """Test resume initialization functionality"""

    def test_resume_initialization_loads_checkpoint(
        self, minimal_config, temp_output_dir
    ):
        """Test that resume=True loads checkpoint during initialization"""
        # Create a checkpoint file
        checkpoint_data = {
            "version": "1.0",
            "timestamp": "2024-01-01T00:00:00+00:00",
            "generation": 2,
            "completed_generations": 2,
            "run_uuid": "test-uuid",
            "seed": None,
            "population": [],
            "seen_population": {},
            "best_of_generation": [],
            "rng_state": {
                "seed": None,
                "bit_generator": "PCG64",
                "state": {
                    "bit_generator": "PCG64",
                    "state": {"state": 123, "inc": 456},
                    "has_uint32": 0,
                    "uinteger": 789,
                },
                "timestamp": "2024-01-01T00:00:00+00:00",
            },
            "stagnant_generations": 0,
            "scenario_mutation_rate": 0.6,
            "start_time": "2024-01-01T00:00:00+00:00",
        }

        checkpoint_path = os.path.join(temp_output_dir, "checkpoint.json")
        with open(checkpoint_path, "w") as f:
            json.dump(checkpoint_data, f)

        with patch("krkn_ai.algorithm.genetic.KrknRunner"):
            with patch(
                "krkn_ai.algorithm.genetic.ScenarioFactory.generate_valid_scenarios"
            ) as mock_gen:
                mock_gen.return_value = [("pod_scenarios", Mock)]

                # Initialize with resume=True
                ga = GeneticAlgorithm(
                    config=minimal_config,
                    output_dir=temp_output_dir,
                    format="yaml",
                    resume=True,
                )

                # Verify checkpoint was loaded
                assert ga.completed_generations == 2
                assert ga.resume is True

    def test_resume_initialization_with_custom_checkpoint_path(
        self, minimal_config, temp_output_dir
    ):
        """Test resume with custom checkpoint path"""
        # Create checkpoint in custom location
        custom_checkpoint_dir = os.path.join(temp_output_dir, "custom")
        os.makedirs(custom_checkpoint_dir)
        custom_checkpoint_path = os.path.join(
            custom_checkpoint_dir, "my_checkpoint.json"
        )

        checkpoint_data = {
            "version": "1.0",
            "timestamp": "2024-01-01T00:00:00+00:00",
            "generation": 1,
            "completed_generations": 1,
            "run_uuid": "test-uuid",
            "seed": None,
            "population": [],
            "seen_population": {},
            "best_of_generation": [],
            "rng_state": {
                "seed": None,
                "bit_generator": "PCG64",
                "state": {
                    "bit_generator": "PCG64",
                    "state": {"state": 123, "inc": 456},
                    "has_uint32": 0,
                    "uinteger": 789,
                },
                "timestamp": "2024-01-01T00:00:00+00:00",
            },
            "stagnant_generations": 0,
            "scenario_mutation_rate": 0.6,
            "start_time": "2024-01-01T00:00:00+00:00",
        }

        with open(custom_checkpoint_path, "w") as f:
            json.dump(checkpoint_data, f)

        with patch("krkn_ai.algorithm.genetic.KrknRunner"):
            with patch(
                "krkn_ai.algorithm.genetic.ScenarioFactory.generate_valid_scenarios"
            ) as mock_gen:
                mock_gen.return_value = [("pod_scenarios", Mock)]

                # Initialize with custom checkpoint path
                ga = GeneticAlgorithm(
                    config=minimal_config,
                    output_dir=temp_output_dir,
                    format="yaml",
                    resume=True,
                    checkpoint_path=custom_checkpoint_path,
                )

                # Verify custom checkpoint was loaded
                assert ga.completed_generations == 1
                assert ga.checkpoint_path == custom_checkpoint_path


class TestResumeSimulation:
    """Test resume simulation functionality"""

    def test_simulate_resumes_from_correct_generation(self, genetic_algorithm):
        """Test that simulate() resumes from the correct generation"""
        # Set up resume state
        genetic_algorithm.resume = True
        genetic_algorithm.completed_generations = 2
        genetic_algorithm.population = [
            DummyScenario(cluster_components=ClusterComponents())
        ]
        genetic_algorithm.config.generations = 4

        # Mock the fitness calculation and other methods
        with patch.object(genetic_algorithm, "calculate_fitness") as mock_fitness:
            with patch.object(genetic_algorithm, "_save_checkpoint") as mock_save:
                with patch.object(genetic_algorithm, "crossover") as mock_crossover:
                    with patch.object(genetic_algorithm, "mutate") as mock_mutate:
                        # Create proper mock result with scenario
                        mock_result = Mock()
                        mock_result.fitness_result.fitness_score = 0.5
                        mock_result.scenario = DummyScenario(
                            cluster_components=ClusterComponents()
                        )
                        mock_fitness.return_value = mock_result

                        # Mock crossover and mutate to return simple scenarios
                        dummy_scenario = DummyScenario(
                            cluster_components=ClusterComponents()
                        )
                        mock_crossover.return_value = (dummy_scenario, dummy_scenario)
                        mock_mutate.return_value = dummy_scenario

                        # Run simulation
                        genetic_algorithm.simulate()

                        # Verify it ran at least one generation and saved checkpoint
                        assert (
                            mock_save.call_count >= 1
                        )  # Should save at least one checkpoint
                        # Verify it completed more generations than it started with
                        assert genetic_algorithm.completed_generations > 2

    def test_simulate_handles_empty_restored_population(self, genetic_algorithm):
        """Test that simulate handles empty restored population by exiting gracefully"""
        # Set up resume state with empty population
        genetic_algorithm.resume = True
        genetic_algorithm.completed_generations = 1
        genetic_algorithm.population = []  # Empty population
        genetic_algorithm.config.generations = 3

        # The simulate method should exit early when population is empty
        # Let's test that it logs the warning and exits gracefully
        with patch.object(genetic_algorithm, "_save_checkpoint"):
            # Run simulation
            genetic_algorithm.simulate()

            # Verify it incremented completed_generations by 1 before stopping
            # (this happens because cur_generation is incremented before the population check)
            assert genetic_algorithm.completed_generations == 2

    def test_load_checkpoint_rebuilds_empty_population(
        self, genetic_algorithm, temp_output_dir
    ):
        """Test that checkpoint loading rebuilds empty population"""
        # Create checkpoint with empty population
        checkpoint_data = {
            "version": "1.0",
            "timestamp": "2024-01-01T00:00:00+00:00",
            "generation": 2,
            "completed_generations": 2,
            "run_uuid": "test-uuid",
            "seed": None,
            "population": [],  # Empty population
            "seen_population": {},
            "best_of_generation": [],
            "rng_state": {
                "seed": None,
                "bit_generator": "PCG64",
                "state": {
                    "bit_generator": "PCG64",
                    "state": {"state": 123, "inc": 456},
                    "has_uint32": 0,
                    "uinteger": 789,
                },
                "timestamp": "2024-01-01T00:00:00+00:00",
            },
            "stagnant_generations": 0,
            "scenario_mutation_rate": 0.6,
            "start_time": "2024-01-01T00:00:00+00:00",
        }

        checkpoint_path = os.path.join(temp_output_dir, "checkpoint.json")
        with open(checkpoint_path, "w") as f:
            json.dump(checkpoint_data, f)

        genetic_algorithm.checkpoint_file = checkpoint_path

        # Mock create_population to return dummy scenarios
        with patch.object(genetic_algorithm, "create_population") as mock_create:
            mock_create.return_value = [
                DummyScenario(cluster_components=ClusterComponents())
            ]

            # Load checkpoint
            genetic_algorithm._load_checkpoint()

            # Verify population was rebuilt
            mock_create.assert_called_once()
            assert len(genetic_algorithm.population) == 1


class TestResumeAllowedScenarioClasses:
    """Test that CompositeScenario is properly allowed for resume"""

    def test_composite_scenario_in_allowed_classes(self, genetic_algorithm):
        """Test that CompositeScenario is in allowed_scenario_classes"""
        assert "CompositeScenario" in genetic_algorithm.allowed_scenario_classes
        assert (
            genetic_algorithm.allowed_scenario_classes["CompositeScenario"]
            == CompositeScenario
        )

    def test_regular_scenarios_in_allowed_classes(self, genetic_algorithm):
        """Test that regular scenarios are also in allowed_scenario_classes"""
        # The fixture should have at least one regular scenario
        assert (
            len(genetic_algorithm.allowed_scenario_classes) >= 2
        )  # At least CompositeScenario + 1 regular


class TestResumeIntegration:
    """Integration tests for the complete resume workflow"""

    def test_full_resume_workflow(self, minimal_config, temp_output_dir):
        """Test complete resume workflow: save checkpoint, then resume from it"""
        with patch("krkn_ai.algorithm.genetic.KrknRunner"):
            with patch(
                "krkn_ai.algorithm.genetic.ScenarioFactory.generate_valid_scenarios"
            ) as mock_gen:
                mock_gen.return_value = [("dummy_scenarios", DummyScenario)]

                # First run: create and save checkpoint
                ga1 = GeneticAlgorithm(
                    config=minimal_config, output_dir=temp_output_dir, format="yaml"
                )
                ga1.population = [DummyScenario(cluster_components=ClusterComponents())]
                ga1.best_of_generation = []
                ga1.seen_population = {}
                ga1._save_checkpoint(1)

                # Second run: resume from checkpoint
                minimal_config.generations = 3  # Extend generations for resume
                ga2 = GeneticAlgorithm(
                    config=minimal_config,
                    output_dir=temp_output_dir,
                    format="yaml",
                    resume=True,
                )

                # Verify resume state
                assert ga2.resume is True
                assert ga2.completed_generations == 1
                assert len(ga2.population) == 1

    def test_resume_with_different_config_generations(
        self, minimal_config, temp_output_dir
    ):
        """Test resume with different generation count in config"""
        with patch("krkn_ai.algorithm.genetic.KrknRunner"):
            with patch(
                "krkn_ai.algorithm.genetic.ScenarioFactory.generate_valid_scenarios"
            ) as mock_gen:
                mock_gen.return_value = [("dummy_scenarios", DummyScenario)]

                # Create checkpoint with 2 completed generations
                checkpoint_data = {
                    "version": "1.0",
                    "timestamp": "2024-01-01T00:00:00+00:00",
                    "generation": 2,
                    "completed_generations": 2,
                    "run_uuid": "test-uuid",
                    "seed": None,
                    "population": [
                        {
                            "name": "dummy-scenario",
                            "krknctl_name": "dummy",
                            "krknhub_image": "dummy:latest",
                            "__class__": "DummyScenario",
                            "__str__": "dummy-scenario",
                        }
                    ],
                    "seen_population": {},
                    "best_of_generation": [],
                    "rng_state": {
                        "seed": None,
                        "bit_generator": "PCG64",
                        "state": {
                            "bit_generator": "PCG64",
                            "state": {"state": 123, "inc": 456},
                            "has_uint32": 0,
                            "uinteger": 789,
                        },
                        "timestamp": "2024-01-01T00:00:00+00:00",
                    },
                    "stagnant_generations": 0,
                    "scenario_mutation_rate": 0.6,
                    "start_time": "2024-01-01T00:00:00+00:00",
                }

                checkpoint_path = os.path.join(temp_output_dir, "checkpoint.json")
                with open(checkpoint_path, "w") as f:
                    json.dump(checkpoint_data, f)

                # Resume with higher generation count
                minimal_config.generations = 5
                ga = GeneticAlgorithm(
                    config=minimal_config,
                    output_dir=temp_output_dir,
                    format="yaml",
                    resume=True,
                )

                # Verify it can continue beyond original checkpoint
                assert ga.completed_generations == 2
                assert (
                    ga.config.generations == 5
                )  # Should be able to run 3 more generations
