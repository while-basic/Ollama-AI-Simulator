# ----------------------------------------------------------------------------
#  File:        milestones.py
#  Project:     Celaya Solutions Ollama Simulator
#  Created by:  Celaya Solutions, 2025
#  Author:      Christopher Celaya <chris@celayasolutions.com>
#  Description: Implementation of milestone tracking for Baby LLM development
#  Version:     1.0.0
#  License:     MIT (SPDX-Identifier: MIT)
#  Last Update: June 22, 2025
# ----------------------------------------------------------------------------

import os
import json
import re
from loguru import logger
from pathlib import Path
from datetime import datetime

class MilestoneTracker:
    """
    Milestone tracker that monitors Baby LLM's development progress.
    Tracks achievements and developmental milestones.
    """
    
    def __init__(self, milestones_def_path=None, milestones_state_path=None):
        """
        Initialize the milestone tracker.
        
        Args:
            milestones_def_path: Path to milestones.json definition file
            milestones_state_path: Path to store milestone state data
        """
        self.milestones_def_path = milestones_def_path or Path(__file__).parent / "milestones.json"
        self.milestones_state_path = milestones_state_path or Path(__file__).parent.parent / "data" / "milestone_state.json"
        
        # Load milestone definitions and state
        self.milestone_definitions = self._load_milestone_definitions()
        self.milestone_state = self._load_milestone_state()
        
        # Count total milestones
        total_milestones = sum(len(stage_milestones) for stage_milestones in self.milestone_definitions.values())
        logger.info(f"Milestone tracker initialized with {total_milestones} milestones across {len(self.milestone_definitions)} stages")
    
    def _load_milestone_definitions(self):
        """Load milestone definitions from the milestones.json file."""
        try:
            with open(self.milestones_def_path, "r") as f:
                return json.load(f)
        except (FileNotFoundError, json.JSONDecodeError) as e:
            logger.error(f"Error loading milestone definitions: {e}")
            return {}
    
    def _load_milestone_state(self):
        """Load milestone state from the milestone_state.json file."""
        if os.path.exists(self.milestones_state_path):
            try:
                with open(self.milestones_state_path, "r") as f:
                    return json.load(f)
            except (FileNotFoundError, json.JSONDecodeError) as e:
                logger.error(f"Error loading milestone state: {e}")
        
        # Initialize empty state if file doesn't exist or has errors
        return {
            "achieved_milestones": {},
            "current_growth_stage": "infant"
        }
    
    def save_milestone_state(self):
        """Save milestone state to disk."""
        os.makedirs(os.path.dirname(self.milestones_state_path), exist_ok=True)
        
        with open(self.milestones_state_path, "w") as f:
            json.dump(self.milestone_state, f, indent=2)
            
        logger.info(f"Milestone state saved to {self.milestones_state_path}")
    
    def save_milestones(self):
        """Save milestone state to disk. Alias for save_milestone_state for compatibility."""
        self.save_milestone_state()
    
    def check_milestones(self, baby_response, baby_state, evaluation):
        """
        Check if any milestones have been achieved.
        
        Args:
            baby_response: Baby's response text
            baby_state: Current state of the Baby LLM
            evaluation: Evaluation of the response
            
        Returns:
            list: List of newly achieved milestone IDs
        """
        newly_achieved = []
        
        # Get current growth stage
        current_stage = baby_state.get("growth_stage", "infant")
        
        # Check milestones for the current stage
        if current_stage in self.milestone_definitions:
            for milestone in self.milestone_definitions[current_stage]:
                milestone_id = milestone["id"]
                
                # Skip already achieved milestones
                if milestone_id in self.milestone_state["achieved_milestones"]:
                    continue
                
                # Check if milestone is achieved
                if self._check_milestone_trigger(milestone, baby_response, baby_state, evaluation):
                    self._achieve_milestone(milestone_id, milestone, baby_response)
                    newly_achieved.append(milestone_id)
        
        # Check if we need to update the growth stage
        if newly_achieved:
            self._update_growth_stage(baby_state)
            self.save_milestone_state()
        
        return newly_achieved
    
    def _check_milestone_trigger(self, milestone, baby_response, baby_state, evaluation):
        """
        Check if a milestone's trigger condition is met.
        
        Args:
            milestone: Milestone definition
            baby_response: Baby's response text
            baby_state: Current state of the Baby LLM
            evaluation: Evaluation of the response
            
        Returns:
            bool: True if milestone is achieved, False otherwise
        """
        trigger = milestone.get("trigger", {})
        trigger_type = trigger.get("type", "")
        
        # Check if evaluation score is high enough
        if evaluation["overall_score"] < milestone.get("reward", 0.5):
            return False
        
        # Check trigger based on type
        if trigger_type == "response_contains":
            # Check if response contains any of the values
            values = trigger.get("value", [])
            return any(value.lower() in baby_response.lower() for value in values)
        
        elif trigger_type == "response_pattern":
            # Check if response matches the regex pattern
            pattern = trigger.get("value", "")
            return bool(re.search(pattern, baby_response, re.IGNORECASE))
        
        elif trigger_type == "response_length_and_contains":
            # Check if response is long enough and contains any of the values
            min_length = trigger.get("min_length", 0)
            values = trigger.get("value", [])
            return (len(baby_response) >= min_length and 
                    any(value.lower() in baby_response.lower() for value in values))
        
        elif trigger_type == "consecutive_responses":
            # Check if the pattern has appeared in consecutive responses
            pattern = trigger.get("pattern", "")
            count = trigger.get("count", 2)
            
            # This would require tracking previous responses, which we don't have here
            # For now, just check if the current response matches the pattern
            return bool(re.search(pattern, baby_response, re.IGNORECASE))
        
        return False
    
    def _achieve_milestone(self, milestone_id, milestone, context=""):
        """
        Mark a milestone as achieved.
        
        Args:
            milestone_id: ID of the milestone
            milestone: Milestone definition
            context: Context in which the milestone was achieved
        """
        now = datetime.now().isoformat()
        
        self.milestone_state["achieved_milestones"][milestone_id] = {
            "achieved_at": now,
            "title": milestone.get("title", milestone_id),
            "description": milestone.get("description", ""),
            "context": context[:200] if context else "",  # Truncate long context
            "reward": milestone.get("reward", 0.0)
        }
        
        logger.info(f"Milestone achieved: {milestone.get('title', milestone_id)}")
    
    def _update_growth_stage(self, baby_state):
        """
        Update the growth stage based on achieved milestones.
        
        Args:
            baby_state: Current state of the Baby LLM
        """
        # Count milestones by stage
        milestone_counts = {
            "infant": 0,
            "toddler": 0,
            "child": 0,
            "teenager": 0,
            "adult": 0,
            "elder": 0
        }
        
        # Count achieved milestones by stage prefix
        for milestone_id in self.milestone_state["achieved_milestones"]:
            for stage in milestone_counts:
                if milestone_id.startswith(f"{stage}_") or milestone_id in self.milestone_definitions.get(stage, []):
                    milestone_counts[stage] += 1
                    break
        
        # Determine appropriate stage based on milestone counts and day count
        day = baby_state.get("day", 0)
        current_stage = self.milestone_state["current_growth_stage"]
        
        # Growth stage progression logic
        if day >= 60 and milestone_counts["adult"] >= 2:
            new_stage = "elder"
        elif day >= 30 and milestone_counts["teenager"] >= 2:
            new_stage = "adult"
        elif day >= 16 and milestone_counts["child"] >= 2:
            new_stage = "teenager"
        elif day >= 6 and milestone_counts["toddler"] >= 2:
            new_stage = "child"
        elif day >= 2 and milestone_counts["infant"] >= 1:
            new_stage = "toddler"
        else:
            new_stage = "infant"
        
        # Update if stage changed
        if new_stage != current_stage:
            logger.info(f"Growth stage changed from {current_stage} to {new_stage}")
            self.milestone_state["current_growth_stage"] = new_stage
    
    def get_milestone_summary(self):
        """
        Get a summary of achieved milestones.
        
        Returns:
            dict: Milestone summary
        """
        # Count milestones by stage
        milestone_counts = {stage: 0 for stage in self.milestone_definitions}
        total_milestones = {stage: len(milestones) for stage, milestones in self.milestone_definitions.items()}
        
        # Count achieved milestones by stage
        for milestone_id in self.milestone_state["achieved_milestones"]:
            for stage, milestones in self.milestone_definitions.items():
                if any(m["id"] == milestone_id for m in milestones):
                    milestone_counts[stage] += 1
                    break
        
        # Calculate completion percentage
        completion = {
            stage: (count / total_milestones[stage] * 100) if total_milestones[stage] > 0 else 0
            for stage, count in milestone_counts.items()
        }
        
        return {
            "achieved_count": len(self.milestone_state["achieved_milestones"]),
            "total_count": sum(total_milestones.values()),
            "by_stage": {
                stage: {
                    "achieved": milestone_counts[stage],
                    "total": total_milestones[stage],
                    "completion_percentage": completion[stage]
                }
                for stage in milestone_counts
            },
            "current_growth_stage": self.milestone_state["current_growth_stage"]
        }
    
    def get_next_milestones(self, baby_state):
        """
        Get upcoming milestones that are not yet achieved.
        
        Args:
            baby_state: Current state of the Baby LLM
            
        Returns:
            list: List of upcoming milestone definitions
        """
        current_stage = baby_state.get("growth_stage", "infant")
        next_stage = self._get_next_growth_stage(current_stage)
        
        upcoming_milestones = []
        
        # Add unachieved milestones from current stage
        if current_stage in self.milestone_definitions:
            for milestone in self.milestone_definitions[current_stage]:
                if milestone["id"] not in self.milestone_state["achieved_milestones"]:
                    upcoming_milestones.append(milestone)
        
        # Add a few milestones from the next stage
        if next_stage in self.milestone_definitions:
            for milestone in self.milestone_definitions[next_stage][:3]:  # First 3 milestones
                if milestone["id"] not in self.milestone_state["achieved_milestones"]:
                    upcoming_milestones.append(milestone)
        
        return upcoming_milestones[:5]  # Return at most 5 upcoming milestones
    
    def _get_next_growth_stage(self, current_stage):
        """
        Get the next growth stage after the current one.
        
        Args:
            current_stage: Current growth stage
            
        Returns:
            str: Next growth stage or None if there is no next stage
        """
        stages = ["infant", "toddler", "child", "teenager", "adult", "elder"]
        try:
            current_index = stages.index(current_stage)
            if current_index < len(stages) - 1:
                return stages[current_index + 1]
        except ValueError:
            pass
        
        return None 