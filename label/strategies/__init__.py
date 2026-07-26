"""Pluggable reward labeling strategies for automated experiment search.

Re-exports all public names so ``from label.strategies import X`` still works.
"""

from label.strategies._base import (
    LabelingStrategy,
    StrategyConfig,
    extract_actions,
    extract_states,
    normalize_rewards,
    smooth,
)
from label.strategies.advanced import (
    DiscriminativeStrategy,
    OptimalTransportStrategy,
    PotentialBasedStrategy,
)
from label.strategies.attribution import (
    ProgressEstimatorStrategy,
    ReturnDecompositionStrategy,
)
from label.strategies.contrastive import (
    ContrastiveDistanceStrategy,
    TemporalContrastiveStrategy,
)
from label.strategies.globally_consistent import GloballyConsistentStrategy
from label.strategies.hmm import HMMBaselineStrategy
from label.strategies.hybrid import (
    EnsembleStrategy,
    HMMContrastiveHybridStrategy,
    ProgressContrastiveHybridStrategy,
)
from label.strategies.registry import build_strategy_pool

__all__ = [
    "LabelingStrategy",
    "StrategyConfig",
    "extract_actions",
    "extract_states",
    "normalize_rewards",
    "smooth",
    "HMMBaselineStrategy",
    "ContrastiveDistanceStrategy",
    "TemporalContrastiveStrategy",
    "ProgressEstimatorStrategy",
    "ReturnDecompositionStrategy",
    "HMMContrastiveHybridStrategy",
    "ProgressContrastiveHybridStrategy",
    "EnsembleStrategy",
    "PotentialBasedStrategy",
    "DiscriminativeStrategy",
    "OptimalTransportStrategy",
    "GloballyConsistentStrategy",
    "build_strategy_pool",
]
