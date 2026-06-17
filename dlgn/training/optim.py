# Source: ported from Older_Imp.ipynb cell 5
# Optimizer builder. Gradient clipping then adamw — semantics must not change.
from __future__ import annotations

import optax


def create_optimizer(config: dict) -> optax.GradientTransformation:
    """Build the optimizer chain: clip gradients then AdamW.

    Reads: clip_value, learning_rate, weight_decay from config.
    """
    return optax.chain(
        optax.clip(config['clip_value']),
        optax.adamw(
            learning_rate=config['learning_rate'],
            b1=0.9,
            b2=0.99,
            weight_decay=config['weight_decay'],
        ),
    )
