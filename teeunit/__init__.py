"""
TeeUnit: Multi-Agent Arena Environment

A turn-based multi-agent deathmatch environment wrapping real Teeworlds,
for RL training. Built on the OpenEnv framework.

OpenEnv-compatible API:
    from teeunit import TeeEnv, TeeAction
    
    # Async usage
    async with TeeEnv(base_url="http://localhost:7860") as env:
        result = await env.reset()
        obs = await env.step(TeeAction(agent_id=0, direction=1, fire=True))
    
    # Sync usage
    with TeeEnv(base_url="http://localhost:7860").sync() as env:
        result = env.reset()
        obs = env.step(TeeAction(agent_id=0, direction=1, fire=True))

For RL training, use the tensor conversion methods:
    obs_tensor = observation.to_tensor()  # Shape: (195,)
    action = TeeAction.from_discrete_action(action_idx, agent_id=0)
"""

__version__ = "0.3.0"

# =============================================================================
# OpenEnv-Compatible Classes (Primary API)
# =============================================================================

from .openenv_models import (
    # Core OpenEnv types
    Action,
    Observation,
    State,
    # TeeUnit Action
    TeeAction,
    TeeMultiAction,
    # TeeUnit Observation
    TeeObservation,
    TeeMultiObservation,
    VisiblePlayer,
    VisibleProjectile,
    VisiblePickup,
    KillEvent,
    # TeeUnit State
    TeeState,
    TeeStepResult,
    TeeMultiStepResult,
    # Config
    RewardConfig,
    # Constants
    WeaponType,
    WEAPON_NAMES,
    MAX_OTHER_PLAYERS,
    MAX_PICKUPS,
    MAX_PROJECTILES,
)

from .openenv_client import (
    TeeEnv,
    SyncTeeEnv,
    TeeEnvClient,
    create_client,
)

from .openenv_environment import (
    TeeEnvironment,
    TeeConfig,
)

# =============================================================================
# Legacy Classes (for backwards compatibility)
# =============================================================================

from .models import (
    TeeInput,
    TeeObservation as LegacyTeeObservation,
    TeeState as LegacyTeeState,
    StepResult,
    GameConfig,
    AgentScore,
)

from .client import (
    TeeEnv as LegacyTeeEnv,
    SyncTeeEnv as LegacySyncTeeEnv,
    LocalTeeEnv,
    TeeEnvError,
    make_env,
)

__all__ = [
    # Version
    "__version__",
    
    # OpenEnv Base Types
    "Action",
    "Observation", 
    "State",
    
    # TeeUnit Action
    "TeeAction",
    "TeeMultiAction",
    
    # TeeUnit Observation
    "TeeObservation",
    "TeeMultiObservation",
    "VisiblePlayer",
    "VisibleProjectile",
    "VisiblePickup",
    "KillEvent",
    
    # TeeUnit State
    "TeeState",
    "TeeStepResult",
    "TeeMultiStepResult",
    
    # Environment
    "TeeEnvironment",
    "TeeConfig",
    "RewardConfig",
    
    # Client
    "TeeEnv",
    "SyncTeeEnv",
    "TeeEnvClient",
    "create_client",
    
    # Constants
    "WeaponType",
    "WEAPON_NAMES",
    "MAX_OTHER_PLAYERS",
    "MAX_PICKUPS",
    "MAX_PROJECTILES",
    
    # Legacy (backwards compatibility)
    "TeeInput",
    "StepResult",
    "GameConfig",
    "AgentScore",
    "LocalTeeEnv",
    "TeeEnvError",
    "make_env",
]
