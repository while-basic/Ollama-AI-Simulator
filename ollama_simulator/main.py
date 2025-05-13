# ----------------------------------------------------------------------------
#  File:        main.py
#  Project:     Celaya Solutions Ollama Simulator
#  Created by:  Celaya Solutions, 2025
#  Author:      Christopher Celaya <chris@celayasolutions.com>
#  Description: Entry point for the Ollama Simulator application
#  Version:     1.0.0
#  License:     MIT (SPDX-Identifier: MIT)
#  Last Update: June 22, 2025
# ----------------------------------------------------------------------------

import os
import sys
import argparse
from pathlib import Path
from loguru import logger
from dotenv import load_dotenv

# Use relative imports for the package structure
from .runtime.simulation_loop import SimulationLoop
from .dashboard.ui_flask import start_web_dashboard
from .dashboard.ui_textual import start_terminal_ui

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Ollama Simulator - Mother and Baby LLM Simulation")
    
    parser.add_argument(
        "--config", 
        type=str, 
        default="config/settings.yaml",
        help="Path to config file (default: config/settings.yaml)"
    )
    
    parser.add_argument(
        "--mode", 
        type=str, 
        choices=["cli", "web", "tui"], 
        default="cli",
        help="Interface mode: cli (command line), web (browser), or tui (terminal UI)"
    )
    
    parser.add_argument(
        "--days", 
        type=int, 
        default=None,
        help="Number of days to simulate (overrides config)"
    )
    
    parser.add_argument(
        "--obsidian", 
        type=str, 
        default=None,
        help="Path to Obsidian vault for logging (overrides config)"
    )
    
    parser.add_argument(
        "--mother", 
        type=str, 
        default=None,
        help="Mother LLM model to use (overrides config)"
    )
    
    parser.add_argument(
        "--baby", 
        type=str, 
        default=None,
        help="Baby LLM model to use (overrides config)"
    )
    
    parser.add_argument(
        "--verbose", 
        action="store_true",
        help="Enable verbose logging"
    )
    
    return parser.parse_args()

def setup_environment():
    """Set up the environment for the simulation."""
    # Load environment variables from .env file
    load_dotenv()
    
    # Set up base directory
    base_dir = Path(__file__).parent
    os.chdir(base_dir)
    
    # Ensure data directories exist
    data_dir = base_dir / "data"
    os.makedirs(data_dir, exist_ok=True)
    os.makedirs(data_dir / "logs", exist_ok=True)
    os.makedirs(data_dir / "obsidian_vault", exist_ok=True)
    os.makedirs(data_dir / "faiss_index", exist_ok=True)
    
    # Configure logger
    logger.remove()
    logger.add(sys.stderr, level="INFO")
    logger.add(data_dir / "logs" / "main.log", rotation="10 MB", level="DEBUG")
    
    logger.info("Environment setup complete")

def run_cli_mode(config_path, args):
    """Run the simulation in CLI mode."""
    # Initialize simulation loop
    simulation = SimulationLoop(config_path=config_path)
    
    # Override config with command line arguments
    if args.days:
        simulation.config["simulation"]["days_to_simulate"] = args.days
    if args.mother:
        simulation.config["models"]["mother"] = args.mother
    if args.baby:
        simulation.config["models"]["baby"] = args.baby
    if args.obsidian:
        simulation.config["paths"]["obsidian_vault"] = args.obsidian
    
    # Start the simulation
    logger.info("Starting simulation in CLI mode")
    simulation.start()
    
    return simulation

def main():
    """Main entry point for the application."""
    # Parse command line arguments
    args = parse_args()
    
    # Set up environment
    setup_environment()
    
    # Resolve config path
    config_path = Path(args.config)
    if not config_path.is_absolute():
        config_path = Path(__file__).parent / args.config
    
    # Run in the selected mode
    try:
        if args.mode == "cli":
            simulation = run_cli_mode(config_path, args)
        elif args.mode == "web":
            start_web_dashboard(config_path, args)
        elif args.mode == "tui":
            start_terminal_ui(config_path, args)
        else:
            logger.error(f"Unknown mode: {args.mode}")
            sys.exit(1)
    except KeyboardInterrupt:
        logger.info("Simulation interrupted by user")
    except Exception as e:
        logger.error(f"Error in simulation: {e}")
        raise
    
    logger.info("Simulation completed")

if __name__ == "__main__":
    main() 