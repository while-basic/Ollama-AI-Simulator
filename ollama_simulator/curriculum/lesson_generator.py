# ----------------------------------------------------------------------------
#  File:        lesson_generator.py
#  Project:     Celaya Solutions Ollama Simulator
#  Created by:  Celaya Solutions, 2025
#  Author:      Christopher Celaya <chris@celayasolutions.com>
#  Description: Implementation of dynamic lesson generation for Baby LLM
#  Version:     1.0.0
#  License:     MIT (SPDX-Identifier: MIT)
#  Last Update: June 22, 2025
# ----------------------------------------------------------------------------

import os
import json
import random
from loguru import logger
from pathlib import Path

class LessonGenerator:
    """
    Lesson generator that creates appropriate lessons based on Baby's development stage.
    Manages curriculum progression and topic selection.
    """
    
    def __init__(self, topics_path=None, growth_topics_path=None):
        """
        Initialize the lesson generator.
        
        Args:
            topics_path: Path to the topics.json file (legacy)
            growth_topics_path: Path to the growth_topics.json file
        """
        self.topics_path = topics_path or Path(__file__).parent / "topics.json"
        self.growth_topics_path = growth_topics_path or Path(__file__).parent / "growth_topics.json"
        
        # Load both topic structures
        self.legacy_topics = self._load_topics(self.topics_path)
        self.growth_topics = self._load_topics(self.growth_topics_path)
        
        # Initialize state
        self.current_stage = "stage1"  # Legacy
        self.current_growth_stage = "infant"  # New growth-based
        self.completed_topics = set()
        self.current_topic = None
        
        logger.info(f"Lesson generator initialized with growth stages: {', '.join(self.growth_topics.keys())}")
    
    def _load_topics(self, path):
        """
        Load topics from a JSON file.
        
        Args:
            path: Path to the JSON file
            
        Returns:
            dict: Loaded topics
        """
        try:
            with open(path, "r") as f:
                return json.load(f)
        except (FileNotFoundError, json.JSONDecodeError) as e:
            logger.error(f"Error loading topics from {path}: {e}")
            return {}
    
    def select_next_topic(self, baby_state):
        """
        Select the next topic based on Baby's current state.
        
        Args:
            baby_state: Current state of the Baby LLM
            
        Returns:
            dict: Selected topic
        """
        # Determine appropriate growth stage based on Baby's development
        appropriate_stage = self._determine_growth_stage(baby_state)
        
        # If stage changed, log it
        if appropriate_stage != self.current_growth_stage:
            logger.info(f"Baby progressed from {self.current_growth_stage} to {appropriate_stage}")
            self.current_growth_stage = appropriate_stage
        
        # Get available topics for this stage
        available_topics = self._get_available_growth_topics(self.current_growth_stage)
        
        if not available_topics:
            # If all topics in this stage are completed, either move to next stage or recycle
            next_stage = self._get_next_growth_stage(self.current_growth_stage)
            if next_stage:
                logger.info(f"All topics in {self.current_growth_stage} completed, moving to {next_stage}")
                self.current_growth_stage = next_stage
                available_topics = self._get_available_growth_topics(self.current_growth_stage)
            else:
                # Recycle completed topics with lowest completion count
                logger.info("All topics completed, recycling topics")
                self.completed_topics = set()
                available_topics = self._get_available_growth_topics(self.current_growth_stage)
        
        # Select a topic
        if available_topics:
            self.current_topic = random.choice(available_topics)
            topic_id = self.current_topic.get("id", "unknown")
            self.completed_topics.add(topic_id)
            return self.current_topic
        else:
            # Fallback topic if something went wrong
            logger.warning("No topics available, using fallback topic")
            return {
                "id": "fallback",
                "title": "Fallback",
                "description": "A simple fallback topic",
                "expected_concepts": ["hello", "world"],
                "difficulty": 0.1
            }
    
    def _determine_growth_stage(self, baby_state):
        """
        Determine the appropriate growth stage based on Baby's state.
        
        Args:
            baby_state: Current state of the Baby LLM
            
        Returns:
            str: Appropriate growth stage
        """
        # Get the day count from baby state
        day = baby_state.get("day", 0)
        avg_score = baby_state.get("average_score", 0.0)
        milestones = baby_state.get("achieved_milestones", [])
        
        # Determine stage based on day count and other factors
        if day >= 60 and avg_score >= 0.8 and len([m for m in milestones if m.startswith("adult_")]) >= 2:
            return "elder"
        elif day >= 30 and avg_score >= 0.7 and len([m for m in milestones if m.startswith("teenager_")]) >= 2:
            return "adult"
        elif day >= 16 and avg_score >= 0.6 and len([m for m in milestones if m.startswith("child_")]) >= 2:
            return "teenager"
        elif day >= 6 and avg_score >= 0.5 and len([m for m in milestones if m.startswith("toddler_")]) >= 2:
            return "child"
        elif day >= 2 and avg_score >= 0.4 and len([m for m in milestones if m.startswith("infant_")]) >= 1:
            return "toddler"
        else:
            return "infant"
    
    def _get_available_growth_topics(self, growth_stage):
        """
        Get available topics for the given growth stage.
        
        Args:
            growth_stage: Growth stage name
            
        Returns:
            list: Available topics
        """
        if growth_stage not in self.growth_topics:
            logger.warning(f"Growth stage {growth_stage} not found in topics")
            return []
        
        # Collect all topics from all categories in this growth stage
        all_topics = []
        for category, topics in self.growth_topics[growth_stage].items():
            for topic in topics:
                if topic["id"] not in self.completed_topics:
                    all_topics.append(topic)
        
        return all_topics
    
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
    
    def generate_lesson(self, topic, difficulty_modifier=0.0):
        """
        Generate a lesson for the given topic.
        
        Args:
            topic: Topic to generate a lesson for
            difficulty_modifier: Modifier to adjust difficulty (-1.0 to 1.0)
            
        Returns:
            dict: Generated lesson
        """
        # Adjust difficulty
        base_difficulty = topic.get("difficulty", 0.5)
        adjusted_difficulty = max(0.1, min(1.0, base_difficulty + difficulty_modifier))
        
        # Generate lesson content
        lesson = {
            "title": topic.get("title", "Unknown Topic"),
            "topic": topic.get("title", "Unknown"),
            "stage": self.current_growth_stage,
            "difficulty": adjusted_difficulty,
            "concepts": topic.get("expected_concepts", []),
            "expected_concepts": topic.get("expected_concepts", []),
            "content": self._generate_lesson_content(topic, adjusted_difficulty)
        }
        
        return lesson
    
    def _generate_lesson_content(self, topic, difficulty):
        """
        Generate lesson content for the given topic and difficulty.
        
        Args:
            topic: Topic to generate content for
            difficulty: Difficulty level (0.1-1.0)
            
        Returns:
            str: Generated lesson content
        """
        # This is a simplified implementation
        # In a real system, you'd use a more sophisticated approach
        
        title = topic.get("title", "Unknown Topic")
        description = topic.get("description", "")
        concepts = topic.get("expected_concepts", [])
        concepts_str = ", ".join(concepts)
        
        if difficulty <= 0.2:
            return f"Today we're learning about {title}. {description} Can you say these words? {concepts_str}"
        elif difficulty <= 0.4:
            return f"Let's talk about {title}. {description} These are important words: {concepts_str}. Can you use them in a simple sentence?"
        elif difficulty <= 0.6:
            return f"We're going to learn about {title}. {description} Think about these concepts: {concepts_str}. Can you ask a question about them?"
        elif difficulty <= 0.8:
            return f"Today's lesson is about {title}. {description} Let's explore these ideas: {concepts_str}. How do they relate to your understanding?"
        else:
            return f"Let's have an advanced lesson on {title}. {description} Consider these concepts: {concepts_str}. Can you explain how they relate to each other and to broader philosophical ideas?" 