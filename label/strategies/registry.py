"""Strategy pool builder for automated experiment search."""

from label.strategies._base import LabelingStrategy, StrategyConfig
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


def build_strategy_pool(config: StrategyConfig | None = None) -> list[LabelingStrategy]:
    """Build the full pool of strategies to search over."""
    cfg = config or StrategyConfig()

    return [
        HMMBaselineStrategy(cfg),
        ContrastiveDistanceStrategy(cfg, n_clusters=10, distance_weight=0.3, temporal_weight=0.9),
        ContrastiveDistanceStrategy(cfg, n_clusters=20, distance_weight=0.5, temporal_weight=0.9),
        ContrastiveDistanceStrategy(cfg, n_clusters=15, distance_weight=0.5, temporal_weight=0.95),
        TemporalContrastiveStrategy(cfg, time_decay=0.9, n_neighbors=3),
        TemporalContrastiveStrategy(cfg, time_decay=0.95, n_neighbors=5),
        TemporalContrastiveStrategy(cfg, time_decay=0.99, n_neighbors=10),
        ProgressEstimatorStrategy(cfg, fail_ceiling=0.3),
        ProgressEstimatorStrategy(cfg, fail_ceiling=0.5),
        ReturnDecompositionStrategy(cfg, hidden_dim=32, n_iters=200),
        ReturnDecompositionStrategy(cfg, hidden_dim=64, n_iters=300),
        HMMContrastiveHybridStrategy(cfg, hmm_weight=0.3, n_clusters=15),
        HMMContrastiveHybridStrategy(cfg, hmm_weight=0.5, n_clusters=15),
        HMMContrastiveHybridStrategy(cfg, hmm_weight=0.7, n_clusters=15),
        ProgressContrastiveHybridStrategy(cfg, progress_weight=0.4),
        ProgressContrastiveHybridStrategy(cfg, progress_weight=0.6),
        PotentialBasedStrategy(cfg, gamma=0.99, n_neighbors=10),
        PotentialBasedStrategy(cfg, gamma=0.95, n_neighbors=5),
        DiscriminativeStrategy(cfg, window=5),
        DiscriminativeStrategy(cfg, window=10),
        OptimalTransportStrategy(cfg, n_time_bins=10),
        OptimalTransportStrategy(cfg, n_time_bins=20),
        EnsembleStrategy(cfg),
        GloballyConsistentStrategy(cfg, time_decay=0.9, margin=1.0),
        GloballyConsistentStrategy(cfg, time_decay=0.95, margin=2.0),
        GloballyConsistentStrategy(cfg, time_decay=0.85, margin=1.5),
    ]
