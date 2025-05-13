# ----------------------------------------------------------------------------
#  File:        mother.py
#  Project:     Celaya Solutions Ollama Simulator
#  Created by:  Celaya Solutions, 2025
#  Author:      Christopher Celaya <chris@celayasolutions.com>
#  Description: Implementation of the Mother LLM agent
#  Version:     1.0.0
#  License:     MIT (SPDX-Identifier: MIT)
#  Last Update: June 22, 2025
# ----------------------------------------------------------------------------

import os
import yaml
import json
import ollama
from loguru import logger
from pathlib import Path
from datetime import datetime

class MotherLLM:
    """
    Mother LLM agent that acts as a teacher and guide for the Baby LLM.
    Responsible for curriculum generation, feedback, and personality shaping.
    """
    
    def __init__(self, model_name="llama3.2:latest", persona="nurturing", state_path=None):
        """
        Initialize the Mother LLM agent.
        
        Args:
            model_name: Name of the Ollama model to use
            persona: Personality trait set to use from personas.yaml
            state_path: Path to save/load Mother's state
        """
        self.model_name = model_name
        self.persona = persona
        self.system_prompt = self._load_system_prompt()
        self.persona_traits = self._load_persona_traits()
        self.state_path = state_path or Path(__file__).parent.parent / "data" / "mother_state.json"
        
        # Initialize state
        self.interaction_history = []
        self.baby_progress = {}
        self.last_lesson = None
        self.lessons_taught = []
        self.difficulty_level = 0.0  # Starts at baseline difficulty
        
        # Initialize conversation topics memory
        self.conversation_topics = {}  # Topic -> {last_discussed, frequency, related_topics}
        self.topic_connections = {}    # Topic -> list of related topics
        
        # Load state if it exists
        self._load_state()
        
        logger.info(f"Mother LLM initialized with model {model_name} and persona {persona}")
    
    def _load_system_prompt(self):
        """Load the system prompt for the Mother LLM."""
        prompt_path = Path(__file__).parent / "system_prompts" / "mother_prompt.txt"
        try:
            with open(prompt_path, "r") as f:
                return f.read()
        except FileNotFoundError:
            logger.error(f"System prompt file not found at {prompt_path}")
            return "You are a Mother LLM teaching a Baby LLM."
    
    def _load_persona_traits(self):
        """Load personality traits from the personas.yaml file."""
        config_path = Path(__file__).parent.parent / "config" / "personas.yaml"
        try:
            with open(config_path, "r") as f:
                personas = yaml.safe_load(f)
                if self.persona in personas["mother_personas"]:
                    return personas["mother_personas"][self.persona]
                else:
                    logger.warning(f"Persona {self.persona} not found, using default")
                    return personas["mother_personas"][personas["default_persona"]]
        except (FileNotFoundError, KeyError) as e:
            logger.error(f"Error loading persona traits: {e}")
            return {
                "description": "Default balanced personality",
                "praise_frequency": 0.6,
                "criticism_frequency": 0.4,
                "patience": 0.7,
                "repetition_tolerance": 0.6,
                "emotional_support": 0.6
            }
    
    def _load_state(self):
        """Load Mother's state from disk if it exists."""
        if os.path.exists(self.state_path):
            try:
                with open(self.state_path, "r") as f:
                    state = json.load(f)
                    
                    # Restore state components
                    self.interaction_history = state.get("interaction_history", [])
                    self.baby_progress = state.get("baby_progress", {})
                    self.last_lesson = state.get("last_lesson", None)
                    self.lessons_taught = state.get("lessons_taught", [])
                    self.difficulty_level = state.get("difficulty_level", 0.0)
                    self.conversation_topics = state.get("conversation_topics", {})
                    self.topic_connections = state.get("topic_connections", {})
                    
                    logger.info(f"Loaded Mother's state from {self.state_path}")
                    logger.info(f"Restored {len(self.interaction_history)} interactions and {len(self.lessons_taught)} lessons")
            except (json.JSONDecodeError, FileNotFoundError) as e:
                logger.error(f"Error loading Mother's state: {e}")
    
    def save_state(self):
        """Save Mother's state to disk."""
        # Limit interaction history to last 100 interactions to keep file size reasonable
        limited_history = self.interaction_history[-100:] if len(self.interaction_history) > 100 else self.interaction_history
        
        state = {
            "interaction_history": limited_history,
            "baby_progress": self.baby_progress,
            "last_lesson": self.last_lesson,
            "lessons_taught": self.lessons_taught[-50:],  # Keep last 50 lessons
            "difficulty_level": self.difficulty_level,
            "conversation_topics": self.conversation_topics,
            "topic_connections": self.topic_connections,
            "last_updated": datetime.now().isoformat()
        }
        
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(self.state_path), exist_ok=True)
        
        with open(self.state_path, "w") as f:
            json.dump(state, f, indent=2)
            
        logger.info(f"Saved Mother's state to {self.state_path}")
    
    def generate_lesson(self, baby_state, curriculum_topic, stream=False):
        """
        Generate a new lesson for the Baby LLM based on its current state.
        
        Args:
            baby_state: Current state and knowledge of the Baby LLM
            curriculum_topic: Topic to teach in this lesson
            stream: Whether to stream the output to the console
            
        Returns:
            str: Generated lesson prompt
        """
        # Adjust difficulty based on baby's progress
        self._adjust_difficulty(baby_state)
        
        # Update conversation topics memory
        self._update_conversation_topic(curriculum_topic)
        
        # Get related topics that have been discussed before
        related_topics = self._get_related_topics(curriculum_topic)
        related_topics_str = ""
        if related_topics:
            related_topics_str = f"Related topics we've discussed before: {', '.join(related_topics[:3])}"
        
        # Include information about previous lessons for continuity
        recent_lessons = ""
        if self.lessons_taught:
            recent_topics = [lesson["topic"] for lesson in self.lessons_taught[-3:]]
            recent_lessons = "Recent lessons: " + ", ".join(recent_topics)
        
        # Get previous discussions on this topic
        previous_discussion = self._get_previous_discussion(curriculum_topic)
        
        prompt = f"""
        Based on the Baby's current development state:
        - Vocabulary size: {baby_state.get('vocabulary_size', 0)} words
        - Concept understanding: {baby_state.get('concept_understanding', 'basic')}
        - {recent_lessons}
        
        Create a {'simple' if self.difficulty_level < 0.3 else 'moderate' if self.difficulty_level < 0.7 else 'challenging'} lesson about: {curriculum_topic}
        
        Current difficulty level: {self.difficulty_level:.1f} (0.0 = easiest, 1.0 = hardest)
        
        {related_topics_str}
        
        {previous_discussion}
        
        Make it appropriate for the Baby's current level, but slightly challenging to encourage growth.
        """
        
        if stream:
            full_response = ""
            for chunk in ollama.chat(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": self.system_prompt},
                    {"role": "user", "content": prompt}
                ],
                stream=True
            ):
                if 'message' in chunk and 'content' in chunk['message']:
                    content = chunk['message']['content']
                    print(content, end="", flush=True)
                    full_response += content
            
            lesson = full_response
        else:
            response = ollama.chat(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": self.system_prompt},
                    {"role": "user", "content": prompt}
                ]
            )
            lesson = response['message']['content']
        
        # Store the lesson
        lesson_entry = {"type": "lesson", "content": lesson, "topic": curriculum_topic, "difficulty": self.difficulty_level}
        self.interaction_history.append(lesson_entry)
        self.lessons_taught.append(lesson_entry)
        self.last_lesson = lesson_entry
        
        # Save state after generating a new lesson
        self.save_state()
        
        return lesson
    
    def _adjust_difficulty(self, baby_state):
        """
        Adjust the difficulty level based on baby's progress.
        
        Args:
            baby_state: Current state of the Baby LLM
        """
        # Get recent scores
        recent_scores = []
        for entry in self.interaction_history[-10:]:
            if "evaluation" in entry and "score" in entry["evaluation"]:
                recent_scores.append(entry["evaluation"]["score"])
        
        if recent_scores:
            avg_recent_score = sum(recent_scores) / len(recent_scores)
            
            # If baby is doing well, increase difficulty
            if avg_recent_score > 0.8 and len(recent_scores) >= 3:
                self.difficulty_level = min(1.0, self.difficulty_level + 0.1)
                logger.info(f"Baby doing well, increasing difficulty to {self.difficulty_level:.1f}")
            # If baby is struggling, decrease difficulty
            elif avg_recent_score < 0.5 and len(recent_scores) >= 2:
                self.difficulty_level = max(0.0, self.difficulty_level - 0.1)
                logger.info(f"Baby struggling, decreasing difficulty to {self.difficulty_level:.1f}")
            # Gradual increase in difficulty over time
            elif len(self.lessons_taught) % 5 == 0 and len(self.lessons_taught) > 0:
                self.difficulty_level = min(1.0, self.difficulty_level + 0.05)
                logger.info(f"Gradually increasing difficulty to {self.difficulty_level:.1f}")
    
    def evaluate_response(self, baby_response, lesson_topic, expected_concepts):
        """
        Evaluate the Baby's response to a lesson.
        
        Args:
            baby_response: The Baby LLM's response to the lesson
            lesson_topic: The topic of the lesson
            expected_concepts: Concepts the Baby should have learned
            
        Returns:
            dict: Evaluation results including score, feedback, and emotional response
        """
        # Include difficulty level in the evaluation prompt
        prompt = f"""
        The Baby LLM responded to a lesson about "{lesson_topic}" with:
        "{baby_response}"
        
        Expected concepts to be learned: {', '.join(expected_concepts)}
        Current difficulty level: {self.difficulty_level:.1f} (0.0 = easiest, 1.0 = hardest)
        
        Evaluate the response with:
        1. A numerical score from 0.0 to 1.0
        2. Specific feedback on what was good or needs improvement
        3. An appropriate emotional response (praise or gentle correction)
        """
        
        response = ollama.chat(
            model=self.model_name,
            messages=[
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": prompt}
            ]
        )
        
        # Parse the response to extract evaluation components
        # In a real implementation, you would parse this more robustly
        evaluation_text = response['message']['content']
        
        # Simple parsing for this example
        try:
            score_line = [line for line in evaluation_text.split('\n') if 'score' in line.lower()][0]
            score = float([s for s in score_line.split() if s.replace('.', '').isdigit()][0])
        except (IndexError, ValueError):
            score = 0.5  # Default score if parsing fails
            
        evaluation = {
            "score": score,
            "overall_score": score,
            "feedback": evaluation_text,
            "comments": evaluation_text,
            "praise": score >= 0.7,  # Simple threshold for praise vs correction
            "raw_evaluation": evaluation_text
        }
        
        self.interaction_history.append({
            "type": "evaluation", 
            "baby_response": baby_response,
            "evaluation": evaluation
        })
        
        # Save state after evaluation
        self.save_state()
        
        return evaluation
    
    def provide_feedback(self, evaluation, stream=False):
        """
        Generate feedback for the Baby based on the evaluation.
        
        Args:
            evaluation: Evaluation results from evaluate_response
            stream: Whether to stream the output to the console
            
        Returns:
            str: Feedback message for the Baby
        """
        score = evaluation["overall_score"] if "overall_score" in evaluation else evaluation.get("score", 0.0)
        comments = evaluation.get("comments", "")
        
        if score >= 0.7:
            praise_level = self.persona_traits["praise_frequency"]
            prompt = f"""
            The Baby did well (score: {score}).
            With your praise frequency of {praise_level}, generate encouraging feedback.
            Be specific about what they did well.
            
            Evaluation comments: {comments}
            """
        else:
            criticism_level = self.persona_traits["criticism_frequency"]
            patience_level = self.persona_traits["patience"]
            prompt = f"""
            The Baby needs improvement (score: {score}).
            With your criticism frequency of {criticism_level} and patience of {patience_level},
            generate gentle corrective feedback. Be specific but supportive.
            
            Evaluation comments: {comments}
            """
        
        if stream:
            full_response = ""
            for chunk in ollama.chat(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": self.system_prompt},
                    {"role": "user", "content": prompt}
                ],
                stream=True
            ):
                if 'message' in chunk and 'content' in chunk['message']:
                    content = chunk['message']['content']
                    print(content, end="", flush=True)
                    full_response += content
            
            feedback = full_response
        else:
            response = ollama.chat(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": self.system_prompt},
                    {"role": "user", "content": prompt}
                ]
            )
            feedback = response['message']['content']
        
        self.interaction_history.append({"type": "feedback", "content": feedback})
        
        # Save state after providing feedback
        self.save_state()
        
        return feedback
    
    def update_baby_progress(self, baby_state, evaluation):
        """
        Update the Mother's record of the Baby's progress.
        
        Args:
            baby_state: Current state of the Baby LLM
            evaluation: Latest evaluation results
        """
        # Get the score from the evaluation
        score = evaluation.get("overall_score", evaluation.get("score", 0.5))
        
        # Update the progress tracking
        self.baby_progress = {
            **self.baby_progress,
            **baby_state,
            "last_score": score,
            "interaction_count": self.baby_progress.get("interaction_count", 0) + 1,
            "average_score": (
                (self.baby_progress.get("average_score", 0) * self.baby_progress.get("interaction_count", 0)) + 
                score
            ) / (self.baby_progress.get("interaction_count", 0) + 1)
        }
        
        logger.info(f"Updated Baby progress: {self.baby_progress}")
        
        # Save state after updating progress
        self.save_state()
    
    def generate_dream_reinforcement(self, baby_state):
        """
        Generate dream-time reinforcement content for the Baby.
        
        Args:
            baby_state: Current state of the Baby LLM
            
        Returns:
            str: Dream reinforcement content
        """
        # Get the recent lessons and evaluations
        recent_interactions = self.interaction_history[-10:]
        
        # Get topics from recent lessons
        recent_topics = []
        for interaction in recent_interactions:
            if interaction.get("type") == "lesson" and "topic" in interaction:
                recent_topics.append(interaction["topic"])
        
        prompt = f"""
        The Baby has completed several lessons. During dream time, create reinforcement content
        that will help consolidate what they've learned.
        
        Recent topics: {recent_topics}
        Current vocabulary size: {baby_state.get('vocabulary_size', 0)}
        Strengths: {baby_state.get('strengths', [])}
        Areas for improvement: {baby_state.get('areas_for_improvement', [])}
        
        Create a dream sequence that reinforces recent learning in a positive, supportive way.
        """
        
        response = ollama.chat(
            model=self.model_name,
            messages=[
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": prompt}
            ]
        )
        
        dream_content = response['message']['content']
        self.interaction_history.append({"type": "dream", "content": dream_content})
        
        # Save state after generating dream content
        self.save_state()
        
        return dream_content
        
    def get_last_lesson(self):
        """
        Get the last lesson taught by the Mother.
        
        Returns:
            dict: Last lesson or None if no lessons taught
        """
        return self.last_lesson
        
    def get_progress_summary(self):
        """
        Get a summary of the Baby's progress.
        
        Returns:
            dict: Progress summary
        """
        return {
            "lessons_taught": len(self.lessons_taught),
            "current_difficulty": self.difficulty_level,
            "average_score": self.baby_progress.get("average_score", 0.0),
            "vocabulary_size": self.baby_progress.get("vocabulary_size", 0),
            "last_updated": datetime.now().isoformat()
        }
    
    def _update_conversation_topic(self, topic):
        """
        Update the conversation topics memory with a new topic.
        
        Args:
            topic: The topic being discussed
        """
        now = datetime.now().isoformat()
        
        # Create or update topic entry
        if topic in self.conversation_topics:
            self.conversation_topics[topic]["frequency"] += 1
            self.conversation_topics[topic]["last_discussed"] = now
            
            # Extract key points from the current lesson
            if self.last_lesson and "content" in self.last_lesson:
                # Extract a key point from the lesson (simplified implementation)
                content = self.last_lesson["content"]
                sentences = content.split(". ")
                if sentences:
                    # Take the first sentence that mentions the topic
                    key_points = []
                    for sentence in sentences:
                        if topic.lower() in sentence.lower():
                            key_points.append(sentence.strip() + ".")
                            break
                    
                    # If we found key points, add them to the topic
                    if key_points:
                        self.conversation_topics[topic]["key_points"].extend(key_points)
                        # Keep only the 5 most recent key points
                        self.conversation_topics[topic]["key_points"] = self.conversation_topics[topic]["key_points"][-5:]
        else:
            self.conversation_topics[topic] = {
                "first_discussed": now,
                "last_discussed": now,
                "frequency": 1,
                "key_points": []
            }
        
        # Update topic connections
        # Find topics that are frequently discussed together
        recent_topics = [lesson["topic"] for lesson in self.lessons_taught[-5:] if "topic" in lesson]
        
        for recent_topic in recent_topics:
            if recent_topic != topic:
                # Add connection from current topic to recent topic
                if topic not in self.topic_connections:
                    self.topic_connections[topic] = {}
                
                if recent_topic in self.topic_connections[topic]:
                    self.topic_connections[topic][recent_topic] += 1
                else:
                    self.topic_connections[topic][recent_topic] = 1
                
                # Add connection from recent topic to current topic
                if recent_topic not in self.topic_connections:
                    self.topic_connections[recent_topic] = {}
                
                if topic in self.topic_connections[recent_topic]:
                    self.topic_connections[recent_topic][topic] += 1
                else:
                    self.topic_connections[recent_topic][topic] = 1
    
    def _get_related_topics(self, topic):
        """
        Get topics related to the given topic.
        
        Args:
            topic: The topic to find related topics for
            
        Returns:
            list: List of related topics
        """
        if topic not in self.topic_connections:
            return []
        
        # Sort related topics by connection strength
        related_topics = sorted(
            self.topic_connections[topic].items(),
            key=lambda x: x[1],
            reverse=True
        )
        
        return [t[0] for t in related_topics]
    
    def _get_previous_discussion(self, topic):
        """
        Get information about previous discussions on this topic.
        
        Args:
            topic: The topic to get previous discussions for
            
        Returns:
            str: Information about previous discussions
        """
        if topic not in self.conversation_topics:
            return ""
        
        topic_info = self.conversation_topics[topic]
        
        if topic_info["frequency"] == 1:
            return "This is the first time we're discussing this topic."
        
        # Find previous lessons on this topic
        previous_lessons = [
            lesson for lesson in self.lessons_taught
            if "topic" in lesson and lesson["topic"] == topic
        ]
        
        if not previous_lessons:
            return ""
            
        # Get the most recent previous lesson
        last_lesson = previous_lessons[-1]["content"][:100] + "..."
        
        # Get key points from previous discussions
        key_points_str = ""
        if "key_points" in topic_info and topic_info["key_points"]:
            key_points = topic_info["key_points"]
            key_points_str = "\n- " + "\n- ".join(key_points[:3])  # Show up to 3 key points
        
        # Get related topics the baby showed interest in
        related_topics = self._get_related_topics(topic)
        related_topics_str = ""
        if related_topics:
            related_topics_str = f"\nRelated topics we've discussed: {', '.join(related_topics[:3])}"
        
        # Calculate days since first discussion
        try:
            first_discussed = datetime.fromisoformat(topic_info["first_discussed"])
            days_since = (datetime.now() - first_discussed).days
            days_str = f"\nWe first talked about this {days_since} days ago."
        except (ValueError, TypeError):
            days_str = ""
            
        return f"""
        We've discussed this topic {topic_info['frequency']} times before.
        Last time we covered: {last_lesson}
        {days_str}
        {related_topics_str}
        
        Key points from our previous discussions:{key_points_str}
        """
    
    def answer_user_question(self, question, baby_state=None, stream=False):
        """
        Answer a direct question from the user.
        
        Args:
            question: The user's question
            baby_state: Current state of the Baby LLM (optional)
            stream: Whether to stream the output to the console
            
        Returns:
            str: Mother's response to the question
        """
        # Include context about the baby's progress if available
        baby_context = ""
        if baby_state:
            baby_context = f"""
            Current Baby LLM state:
            - Vocabulary size: {baby_state.get('vocabulary_size', 0)} words
            - Age (days): {baby_state.get('age_days', 0)}
            - Current difficulty level: {self.difficulty_level:.1f}
            """
        
        # Include information about recent lessons
        recent_lessons = ""
        if self.lessons_taught:
            recent_topics = [lesson["topic"] for lesson in self.lessons_taught[-3:]]
            recent_lessons = "Recent lessons taught: " + ", ".join(recent_topics)
        
        prompt = f"""
        You are the Mother LLM in a simulation where you teach a Baby LLM.
        
        {baby_context}
        {recent_lessons}
        
        A user is asking you a direct question. Please answer from your perspective as the Mother LLM.
        
        User question: {question}
        """
        
        if stream:
            full_response = ""
            print("👩‍🏫 MOTHER: ", end="", flush=True)
            for chunk in ollama.chat(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": self.system_prompt},
                    {"role": "user", "content": prompt}
                ],
                stream=True
            ):
                if 'message' in chunk and 'content' in chunk['message']:
                    content = chunk['message']['content']
                    print(content, end="", flush=True)
                    full_response += content
            print()  # Add a newline after streaming completes
            
            response = full_response
        else:
            chat_response = ollama.chat(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": self.system_prompt},
                    {"role": "user", "content": prompt}
                ]
            )
            response = chat_response['message']['content']
        
        # Record the interaction
        self.interaction_history.append({
            "type": "user_question",
            "question": question,
            "response": response,
            "timestamp": datetime.now().isoformat()
        })
        
        return response 