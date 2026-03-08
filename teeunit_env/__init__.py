# Copyright (c) 2024 TeeUnit Project
# SPDX-License-Identifier: MIT

"""
TeeUnit OpenEnv Environment

An OpenEnv-compatible multi-agent arena environment wrapping the real Teeworlds 0.7.5 game
for LLM-based reinforcement learning training.

Example:
    >>> from teeunit_env import TeeEnv
    >>> 
    >>> with TeeEnv(base_url="http://localhost:8000") as env:
    ...     env.reset()
    ...     tools = env.list_tools()
    ...     print([t.name for t in tools])
    ...     # ['move', 'jump', 'aim', 'shoot', 'hook', 'get_status']
    ...     
    ...     # Get current game state
    ...     status = env.call_tool("get_status")
    ...     print(status)
    ...     
    ...     # Take actions
    ...     env.call_tool("move", direction="right")
    ...     env.call_tool("aim", x=500, y=300)
    ...     env.call_tool("shoot", weapon=1)
"""

from .client import TeeEnv, TeeAction

__all__ = ["TeeEnv", "TeeAction"]
__version__ = "0.1.0"
