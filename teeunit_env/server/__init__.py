# Copyright (c) 2024 TeeUnit Project
# SPDX-License-Identifier: MIT

"""
TeeUnit OpenEnv Server

This module provides the server-side implementation of the TeeUnit environment,
exposing Teeworlds game control through MCP tools.
"""

from .tee_environment import TeeEnvironment

__all__ = ["TeeEnvironment"]
