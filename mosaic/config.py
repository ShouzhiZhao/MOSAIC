"""Shared, paper-aligned experiment configuration."""

from dataclasses import asdict, dataclass, field
from typing import Dict

IN_DISTRIBUTION_MOVIES = (
    "Heat",
    "Jumanji",
    "Waiting_to_Exhale",
)

OUT_OF_DISTRIBUTION_MOVIES = (
    "GoldenEye",
    "Sabrina",
    "Nixon",
)

APPENDIX_OUT_OF_DISTRIBUTION_MOVIES = (
    *OUT_OF_DISTRIBUTION_MOVIES,
    "Father_of_the_Bride_Part_II",
    "Tom_and_Huck",
    "Sudden_Death",
    "Dracula:_Dead_and_Loving_It",
    "Balto",
    "Cutthroat_Island",
    "Casino",
    "Sense_and_Sensibility",
    "Get_Shorty",
    "Copycat",
    "The_City_of_Lost_Children",
    "Pocahontas",
)

NEW_ITEM_MOVIE = "The_Effect_of_Gamma_Rays_on_Man-in-the-Moon_Marigolds"


@dataclass(frozen=True)
class DataSplit:
    """Movie-level split used by CASO.

    OOD audit movies and the new-item movie are excluded from pretraining. The
    three reporting titles are the compact main-text subset of the 15-title
    appendix audit panel. Every OOD title exists in the item catalog but has
    zero PromoSim seed conditions.
    """

    in_distribution: tuple[str, ...] = IN_DISTRIBUTION_MOVIES
    out_of_distribution: tuple[str, ...] = OUT_OF_DISTRIBUTION_MOVIES
    appendix_out_of_distribution: tuple[str, ...] = APPENDIX_OUT_OF_DISTRIBUTION_MOVIES
    new_item: str = NEW_ITEM_MOVIE

    @property
    def held_out(self) -> frozenset[str]:
        return frozenset((*self.appendix_out_of_distribution, self.new_item))


@dataclass(frozen=True)
class ModelConfig:
    """Serialized architecture defaults preserve legacy checkpoint compatibility.

    Public two-stage training imports its flow architecture from train-flow.
    Never change these fallbacks to reinterpret archived checkpoints.
    """

    num_nodes: int = 1000
    input_dim: int = 2
    hidden_dim: int = 64
    num_layers: int = 2
    num_heads: int = 4
    positional_dim: int = 8
    dropout: float = 0.1
    flow_hidden_dim: int = 256
    flow_time_dim: int = 128
    flow_steps: int = 6
    logit_epsilon: float = 1e-4
    predictor_name: str = "Dual-Branch-Acceptance-Predictor"
    predictor_variant: str = "dual_branch"
    acceptance_output: str = "node"
    flow_velocity_variant: str = "mlp"
    flow_shared_hidden_dim: int = 32
    flow_context_topk: int = 0
    flow_budget_conditioned: bool = False
    flow_dequantization_width: float = 0.5
    flow_solver: str = "euler"


@dataclass(frozen=True)
class TrainingConfig:
    epochs: int = 200
    patience: int = 40
    batch_size: int = 8
    learning_rate: float = 3e-4
    minimum_learning_rate: float = 1e-6
    prediction_weight: float = 1.0
    gradient_clip: float = 2.0
    validation_fraction: float = 0.1
    random_seed: int = 2026
    minimum_replays: int = 1


@dataclass(frozen=True)
class SearchConfig:
    max_budget: int = 10
    latent_iterations: int = 1000
    latent_learning_rate: float = 5e-4
    budget_penalty: float = 0.5
    random_restarts: int = 8
    random_seed: int = 2026
    evaluation_batch_size: int = 64
    max_swap_passes: int = 5
    latent_flow_steps: int = 8
    latent_stability_patience: int = 50


@dataclass(frozen=True)
class MosaicConfig:
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    search: SearchConfig = field(default_factory=SearchConfig)
    split: DataSplit = field(default_factory=DataSplit)

    def to_dict(self) -> Dict[str, object]:
        return asdict(self)
