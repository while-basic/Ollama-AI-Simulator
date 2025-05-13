# ----------------------------------------------------------------------------
#  File:        logger.py
#  Project:     Celaya Solutions Ollama Simulator
#  Created by:  Celaya Solutions, 2025
#  Author:      Christopher Celaya <chris@celayasolutions.com>
#  Description: Implementation of logging for the simulation
#  Version:     1.0.0
#  License:     MIT (SPDX-Identifier: MIT)
#  Last Update: June 22, 2025
# ----------------------------------------------------------------------------

import os
import json
import logging
from loguru import logger
from pathlib import Path
from datetime import datetime

class SimulationLogger:
    """
    Logger for the simulation that records interactions, evaluations, and progress.
    Writes logs to files and provides methods for querying log data.
    """
    
    def __init__(self, log_path=None, verbose=True):
        """
        Initialize the simulation logger.
        
        Args:
            log_path: Path to store log files
            verbose: Whether to print verbose logs to console
        """
        self.log_path = log_path or Path(__file__).parent.parent / "data" / "logs"
        self.verbose = verbose
        self.simulation_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Create log directories
        self._ensure_log_dirs()
        
        # Configure loguru
        self._configure_loguru()
        
        logger.info(f"Simulation logger initialized with ID {self.simulation_id}")
    
    def _ensure_log_dirs(self):
        """Ensure log directories exist."""
        os.makedirs(self.log_path, exist_ok=True)
        os.makedirs(self.log_path / "interactions", exist_ok=True)
        os.makedirs(self.log_path / "dreams", exist_ok=True)
        os.makedirs(self.log_path / "progress", exist_ok=True)
        os.makedirs(self.log_path / "milestones", exist_ok=True)
    
    def _configure_loguru(self):
        """Configure loguru logger."""
        # Remove default handler
        logger.remove()
        
        # Add console handler if verbose
        if self.verbose:
            logger.add(
                lambda msg: print(msg, end=""),
                format="<green>{time:HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>",
                level="INFO"
            )
        
        # Add file handler
        log_file = self.log_path / f"simulation_{self.simulation_id}.log"
        logger.add(
            str(log_file),
            rotation="10 MB",
            format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function}:{line} - {message}",
            level="DEBUG"
        )
    
    def log_lesson(self, lesson_content, topic, day, interaction_number):
        """
        Log a lesson from Mother to Baby.
        
        Args:
            lesson_content: Content of the lesson
            topic: Topic of the lesson
            day: Simulation day
            interaction_number: Interaction number within the day
        """
        log_data = {
            "type": "lesson",
            "content": lesson_content,
            "topic": topic,
            "day": day,
            "interaction": interaction_number,
            "timestamp": datetime.now().isoformat()
        }
        
        # Stub implementation - just log to console
        if self.verbose:
            logger.info(f"Day {day}, Lesson {interaction_number}: {topic}")
            logger.debug(f"Content: {lesson_content[:50]}...")
    
    def log_response(self, response_content, day, interaction_number):
        """
        Log a response from Baby to Mother.
        
        Args:
            response_content: Content of the response
            day: Simulation day
            interaction_number: Interaction number within the day
        """
        # Stub implementation - just log to console
        if self.verbose:
            logger.info(f"Day {day}, Response {interaction_number}: {response_content[:50]}...")
    
    def log_evaluation(self, evaluation, day, interaction_number):
        """
        Log an evaluation of Baby's response.
        
        Args:
            evaluation: Evaluation data
            day: Simulation day
            interaction_number: Interaction number within the day
        """
        # Stub implementation - just log to console
        if self.verbose:
            score = evaluation.get("overall_score", 0.0)
            logger.info(f"Day {day}, Evaluation {interaction_number}: Score = {score:.2f}")
    
    def log_feedback(self, feedback_content, day, interaction_number):
        """
        Log feedback from Mother to Baby.
        
        Args:
            feedback_content: Content of the feedback
            day: Simulation day
            interaction_number: Interaction number within the day
        """
        # Stub implementation - just log to console
        if self.verbose:
            logger.info(f"Day {day}, Feedback {interaction_number}: {feedback_content[:50]}...")
    
    def log_dream(self, dream_content, day):
        """
        Log a dream sequence.
        
        Args:
            dream_content: Content of the dream
            day: Simulation day
        """
        log_data = {
            "type": "dream",
            "content": dream_content,
            "day": day,
            "timestamp": datetime.now().isoformat()
        }
        
        self._write_dream_log(log_data)
        
        if self.verbose:
            logger.info(f"Day {day}, Dream: {dream_content[:50]}...")
    
    def log_daydream(self, dream_content, day, interaction_number):
        """
        Log a daydream sequence.
        
        Args:
            dream_content: Content of the daydream
            day: Simulation day
            interaction_number: Interaction number within the day
        """
        log_data = {
            "type": "daydream",
            "content": dream_content,
            "day": day,
            "interaction": interaction_number,
            "timestamp": datetime.now().isoformat()
        }
        
        self._write_dream_log(log_data)
        
        if self.verbose:
            logger.info(f"Day {day}, Daydream at interaction {interaction_number}: {dream_content[:50]}...")
    
    def log_dream_results(self, dream_results, day):
        """
        Log results of dream processing.
        
        Args:
            dream_results: Results of dream processing
            day: Simulation day
        """
        log_data = {
            "type": "dream_results",
            "results": dream_results,
            "day": day,
            "timestamp": datetime.now().isoformat()
        }
        
        self._write_dream_log(log_data)
        
        if self.verbose:
            concepts = dream_results.get("reinforced_concepts", [])
            logger.info(f"Day {day}, Dream reinforced concepts: {', '.join(concepts[:3])}")
    
    def log_milestones(self, achieved_milestones, day, interaction_number):
        """
        Log achieved milestones.
        
        Args:
            achieved_milestones: List of achieved milestone IDs
            day: Simulation day
            interaction_number: Interaction number within the day
        """
        log_data = {
            "type": "milestones",
            "achieved_milestones": achieved_milestones,
            "day": day,
            "interaction": interaction_number,
            "timestamp": datetime.now().isoformat()
        }
        
        self._write_milestone_log(log_data)
        
        if self.verbose:
            logger.info(f"Day {day}, Milestones achieved: {', '.join(achieved_milestones)}")
    
    def log_progress_report(self, progress_report):
        """
        Log a progress report.
        
        Args:
            progress_report: Progress report data
        """
        log_data = {
            "type": "progress_report",
            "report": progress_report,
            "timestamp": datetime.now().isoformat()
        }
        
        self._write_progress_log(log_data)
        
        if self.verbose:
            day = progress_report.get("day", 0)
            vocab_size = progress_report.get("baby_state", {}).get("vocabulary_size", 0)
            avg_score = progress_report.get("evaluator_progress", {}).get("average_score", 0.0)
            logger.info(f"Day {day} Progress: Vocabulary = {vocab_size}, Average Score = {avg_score:.2f}")
    
    def log_simulation_end(self, total_days, total_interactions):
        """
        Log the end of simulation.
        
        Args:
            total_days: Total days simulated
            total_interactions: Total interactions
        """
        log_data = {
            "type": "simulation_end",
            "total_days": total_days,
            "total_interactions": total_interactions,
            "timestamp": datetime.now().isoformat()
        }
        
        self._write_progress_log(log_data)
        
        if self.verbose:
            logger.info(f"Simulation ended after {total_days} days and {total_interactions} interactions")
    
    def _write_interaction_log(self, log_data):
        """
        Write interaction log to file.
        
        Args:
            log_data: Log data to write
        """
        day = log_data.get("day", 0)
        interaction = log_data.get("interaction", 0)
        log_file = self.log_path / "interactions" / f"day_{day:03d}_interaction_{interaction:03d}_{log_data['type']}.json"
        
        with open(log_file, "w") as f:
            json.dump(log_data, f, indent=2)
    
    def _write_dream_log(self, log_data):
        """
        Write dream log to file.
        
        Args:
            log_data: Log data to write
        """
        day = log_data.get("day", 0)
        dream_type = log_data.get("type", "dream")
        interaction = log_data.get("interaction", "")
        
        if interaction:
            log_file = self.log_path / "dreams" / f"day_{day:03d}_interaction_{interaction:03d}_{dream_type}.json"
        else:
            log_file = self.log_path / "dreams" / f"day_{day:03d}_{dream_type}.json"
        
        with open(log_file, "w") as f:
            json.dump(log_data, f, indent=2)
    
    def _write_milestone_log(self, log_data):
        """
        Write milestone log to file.
        
        Args:
            log_data: Log data to write
        """
        day = log_data.get("day", 0)
        log_file = self.log_path / "milestones" / f"day_{day:03d}_milestones.json"
        
        with open(log_file, "w") as f:
            json.dump(log_data, f, indent=2)
    
    def _write_progress_log(self, log_data):
        """
        Write progress log to file.
        
        Args:
            log_data: Log data to write
        """
        day = log_data.get("report", {}).get("day", 0) if "report" in log_data else 0
        log_type = log_data.get("type", "progress")
        log_file = self.log_path / "progress" / f"day_{day:03d}_{log_type}.json"
        
        with open(log_file, "w") as f:
            json.dump(log_data, f, indent=2) 