# ----------------------------------------------------------------------------
#  File:        __init__.py
#  Project:     Celaya Solutions Ollama Simulator
#  Created by:  Celaya Solutions, 2025
#  Author:      Christopher Celaya <chris@celayasolutions.com>
#  Description: Initialization file for agents module
#  Version:     1.0.0
#  License:     MIT (SPDX-Identifier: MIT)
#  Last Update: June 22, 2025
# ----------------------------------------------------------------------------

from .mother import MotherLLM
from .baby import BabyLLM
from .evaluator import Evaluator

__all__ = ["MotherLLM", "BabyLLM", "Evaluator"] 