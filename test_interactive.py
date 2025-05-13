# ----------------------------------------------------------------------------
#  File:        test_interactive.py
#  Project:     Celaya Solutions Ollama Simulator
#  Created by:  Celaya Solutions, 2025
#  Author:      Christopher Celaya <chris@celayasolutions.com>
#  Description: Test script for interactive commands in Ollama Simulator
#  Version:     1.0.0
#  License:     MIT (SPDX-Identifier: MIT)
#  Last Update: June 22, 2025
# ----------------------------------------------------------------------------

import os
import sys
from pathlib import Path

# Add the parent directory to the path so we can import the ollama_simulator package
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from ollama_simulator.runtime.simulation_loop import SimulationLoop
from ollama_simulator.agents.mother import MotherLLM
from ollama_simulator.agents.baby import BabyLLM

def test_interactive_commands():
    """Test the interactive commands."""
    print("\n" + "="*80)
    print("🧠 OLLAMA SIMULATOR - Interactive Commands Test")
    print("="*80 + "\n")
    
    # Initialize components
    mother = MotherLLM()
    baby = BabyLLM()
    
    # Load states
    mother._load_state()
    baby_state = baby._load_state()
    
    # Test Mother's answer_user_question
    print("Testing Mother's answer_user_question method...")
    question = "What have you been teaching the Baby recently?"
    print(f"\nQuestion to Mother: {question}")
    mother.answer_user_question(question, baby_state, stream=True)
    
    print("\n" + "-"*80 + "\n")
    
    # Test Baby's answer_user_question
    print("Testing Baby's answer_user_question method...")
    question = "What's your favorite toy?"
    print(f"\nQuestion to Baby: {question}")
    baby.answer_user_question(question, stream=True)
    
    print("\n" + "-"*80 + "\n")
    
    # Test help command
    print("Testing help command...")
    sim = SimulationLoop()
    sim._print_help()
    
    print("\n" + "-"*80 + "\n")
    
    # Test status command
    print("Testing status command...")
    sim._print_status()
    
    print("\n" + "="*80)
    print("🎓 TEST COMPLETED")
    print("="*80 + "\n")

if __name__ == "__main__":
    test_interactive_commands() 