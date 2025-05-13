# ----------------------------------------------------------------------------
#  File:        evaluator.py
#  Project:     Celaya Solutions Ollama Simulator
#  Created by:  Celaya Solutions, 2025
#  Author:      Christopher Celaya <chris@celayasolutions.com>
#  Description: Implementation of the Evaluator for Baby LLM responses
#  Version:     1.0.0
#  License:     MIT (SPDX-Identifier: MIT)
#  Last Update: June 22, 2025
# ----------------------------------------------------------------------------

import os
import json
import ollama
from loguru import logger
from pathlib import Path
from datetime import datetime

class Evaluator:
    """
    Evaluator class that assesses Baby LLM responses and tracks progress over time.
    Uses the Mother LLM for evaluation but maintains separate tracking.
    """
    
    def __init__(self, model_name="llama3.2:latest", log_path=None):
        """
        Initialize the Evaluator.
        
        Args:
            model_name: Name of the Ollama model to use for evaluation
            log_path: Path to store evaluation logs
        """
        self.model_name = model_name
        self.log_path = log_path or Path(__file__).parent.parent / "data" / "evaluation_logs.json"
        self.evaluation_history = self._load_history()
        self.milestones = self._load_milestones()
        
        logger.info(f"Evaluator initialized with model {model_name}")
    
    def _load_history(self):
        """Load evaluation history from disk if it exists."""
        if os.path.exists(self.log_path):
            try:
                with open(self.log_path, "r") as f:
                    return json.load(f)
            except (json.JSONDecodeError, FileNotFoundError) as e:
                logger.error(f"Error loading evaluation history: {e}")
        
        return []
    
    def _load_milestones(self):
        """Load milestone definitions."""
        # In a real implementation, this would be loaded from a config file
        return {
            "first_word": {
                "description": "Baby uses first recognizable word correctly",
                "threshold": 0.7,
                "achieved": False
            },
            "simple_sentence": {
                "description": "Baby forms a simple subject-verb sentence",
                "threshold": 0.8,
                "achieved": False
            },
            "ask_question": {
                "description": "Baby asks a question unprompted",
                "threshold": 0.9,
                "achieved": False
            },
            "self_reference": {
                "description": "Baby refers to itself ('I', 'me', 'my')",
                "threshold": 0.85,
                "achieved": False
            },
            "emotional_expression": {
                "description": "Baby expresses an emotion about itself",
                "threshold": 0.8,
                "achieved": False
            }
        }
    
    def evaluate_response(self, baby_response, lesson, expected_concepts):
        """
        Evaluate the Baby's response to a lesson.
        
        Args:
            baby_response: The Baby LLM's response
            lesson: The lesson content
            expected_concepts: List of concepts the Baby should have learned
            
        Returns:
            dict: Evaluation results
        """
        prompt = f"""
        You are evaluating a Baby LLM's response to a lesson.
        
        Lesson: {lesson}
        
        Expected concepts: {', '.join(expected_concepts)}
        
        Baby's response: {baby_response}
        
        Evaluate on these criteria:
        1. Comprehension (0-1): Did the Baby understand the lesson?
        2. Accuracy (0-1): How accurate was the Baby's response?
        3. Complexity (0-1): How complex was the Baby's language?
        4. Creativity (0-1): Did the Baby show any creativity or novel thinking?
        
        Format your response as a JSON object with these fields plus an overall_score (average) and comments field.
        """
        
        response = ollama.chat(
            model=self.model_name,
            messages=[
                {"role": "system", "content": "You are an objective evaluator of language learning."},
                {"role": "user", "content": prompt}
            ]
        )
        
        evaluation_text = response['message']['content']
        
        # Extract JSON from the response
        try:
            # Find JSON block in the response
            json_start = evaluation_text.find('{')
            json_end = evaluation_text.rfind('}') + 1
            
            if json_start >= 0 and json_end > json_start:
                json_str = evaluation_text[json_start:json_end]
                evaluation = json.loads(json_str)
            else:
                # Fallback: parse manually
                lines = evaluation_text.split('\n')
                evaluation = {
                    "comprehension": self._extract_score(lines, "Comprehension"),
                    "accuracy": self._extract_score(lines, "Accuracy"),
                    "complexity": self._extract_score(lines, "Complexity"),
                    "creativity": self._extract_score(lines, "Creativity"),
                    "overall_score": 0.5,  # Default
                    "comments": "Parsing failed, using default values"
                }
                
                # Calculate overall score
                scores = [evaluation["comprehension"], evaluation["accuracy"], 
                          evaluation["complexity"], evaluation["creativity"]]
                evaluation["overall_score"] = sum(scores) / len(scores)
                
        except Exception as e:
            logger.error(f"Error parsing evaluation: {e}")
            evaluation = {
                "comprehension": 0.5,
                "accuracy": 0.5,
                "complexity": 0.5,
                "creativity": 0.5,
                "overall_score": 0.5,
                "comments": f"Error parsing evaluation: {str(e)}"
            }
        
        # Add metadata
        evaluation["timestamp"] = datetime.now().isoformat()
        evaluation["lesson"] = lesson
        evaluation["baby_response"] = baby_response
        evaluation["expected_concepts"] = expected_concepts
        
        # Save to history
        self.evaluation_history.append(evaluation)
        self._save_history()
        
        # Check for milestones
        self._check_milestones(baby_response, evaluation)
        
        return evaluation
    
    def _extract_score(self, lines, criterion):
        """Extract a score from evaluation text lines."""
        for line in lines:
            if criterion.lower() in line.lower() and ':' in line:
                try:
                    score_text = line.split(':')[1].strip()
                    return float(next((s for s in score_text.split() if s.replace('.', '').isdigit()), 0.5))
                except (ValueError, IndexError):
                    pass
        return 0.5  # Default score
    
    def _save_history(self):
        """Save evaluation history to disk."""
        os.makedirs(os.path.dirname(self.log_path), exist_ok=True)
        
        with open(self.log_path, "w") as f:
            json.dump(self.evaluation_history, f, indent=2)
            
        logger.info(f"Evaluation history saved to {self.log_path}")
    
    def _check_milestones(self, response, evaluation):
        """Check if any milestones have been achieved."""
        # Check for first word
        if not self.milestones["first_word"]["achieved"] and evaluation["overall_score"] >= self.milestones["first_word"]["threshold"]:
            self.milestones["first_word"]["achieved"] = True
            self.milestones["first_word"]["achieved_at"] = datetime.now().isoformat()
            logger.info("Milestone achieved: First word!")
        
        # Check for simple sentence
        words = response.split()
        if (not self.milestones["simple_sentence"]["achieved"] and 
            len(words) >= 3 and 
            evaluation["complexity"] >= self.milestones["simple_sentence"]["threshold"]):
            self.milestones["simple_sentence"]["achieved"] = True
            self.milestones["simple_sentence"]["achieved_at"] = datetime.now().isoformat()
            logger.info("Milestone achieved: Simple sentence!")
        
        # Check for question
        if (not self.milestones["ask_question"]["achieved"] and 
            ('?' in response or any(q in response.lower() for q in ['what', 'why', 'how', 'when', 'where', 'who']))):
            self.milestones["ask_question"]["achieved"] = True
            self.milestones["ask_question"]["achieved_at"] = datetime.now().isoformat()
            logger.info("Milestone achieved: Asked a question!")
        
        # Check for self-reference
        if (not self.milestones["self_reference"]["achieved"] and 
            any(word.lower() in ['i', 'me', 'my', 'mine', 'myself'] for word in words)):
            self.milestones["self_reference"]["achieved"] = True
            self.milestones["self_reference"]["achieved_at"] = datetime.now().isoformat()
            logger.info("Milestone achieved: Self-reference!")
        
        # Check for emotional expression
        emotion_words = ['happy', 'sad', 'angry', 'excited', 'scared', 'like', 'love', 'hate', 'afraid']
        if (not self.milestones["emotional_expression"]["achieved"] and 
            any(word.lower() in emotion_words for word in words) and 
            any(word.lower() in ['i', 'me', 'my', 'feel', 'am'] for word in words)):
            self.milestones["emotional_expression"]["achieved"] = True
            self.milestones["emotional_expression"]["achieved_at"] = datetime.now().isoformat()
            logger.info("Milestone achieved: Emotional expression!")
    
    def get_progress_report(self):
        """
        Generate a progress report based on evaluation history.
        
        Returns:
            dict: Progress report with metrics and milestones
        """
        if not self.evaluation_history:
            return {
                "total_evaluations": 0,
                "average_score": 0.0,
                "milestones": self.milestones
            }
        
        # Calculate metrics
        total = len(self.evaluation_history)
        avg_overall = sum(e["overall_score"] for e in self.evaluation_history) / total
        avg_comprehension = sum(e["comprehension"] for e in self.evaluation_history) / total
        avg_accuracy = sum(e["accuracy"] for e in self.evaluation_history) / total
        avg_complexity = sum(e["complexity"] for e in self.evaluation_history) / total
        avg_creativity = sum(e["creativity"] for e in self.evaluation_history) / total
        
        # Get trend (last 5 vs previous 5)
        if total >= 10:
            recent = self.evaluation_history[-5:]
            previous = self.evaluation_history[-10:-5]
            
            recent_avg = sum(e["overall_score"] for e in recent) / 5
            previous_avg = sum(e["overall_score"] for e in previous) / 5
            
            trend = recent_avg - previous_avg
        else:
            trend = 0.0
        
        return {
            "total_evaluations": total,
            "average_score": avg_overall,
            "comprehension": avg_comprehension,
            "accuracy": avg_accuracy,
            "complexity": avg_complexity,
            "creativity": avg_creativity,
            "trend": trend,
            "milestones": self.milestones
        } 