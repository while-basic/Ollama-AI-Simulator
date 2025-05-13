# ----------------------------------------------------------------------------
#  File:        ollama_simulator.py
#  Project:     Celaya Solutions Ollama Simulator
#  Created by:  Celaya Solutions, 2025
#  Author:      Christopher Celaya <chris@celayasolutions.com>
#  Description: Entry point script for the Ollama Simulator
#  Version:     1.0.0
#  License:     MIT (SPDX-Identifier: MIT)
#  Last Update: June 22, 2025
# ----------------------------------------------------------------------------

import sys
import os
from pathlib import Path

# Add the parent directory to sys.path to allow importing the package
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Import the main function from the package
from ollama_simulator.main import main

if __name__ == "__main__":
    main() 