# ----------------------------------------------------------------------------
#  File:        baby.py
#  Project:     Celaya Solutions Ollama Simulator
#  Created by:  Celaya Solutions, 2025
#  Author:      Christopher Celaya <chris@celayasolutions.com>
#  Description: Implementation of the Baby LLM agent
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

class BabyLLM:
    """
    Baby LLM agent that learns through interactions with the Mother LLM.
    Starts with minimal knowledge and develops through guided learning.
    """
    
    def __init__(self, model_name="llama3.2:1b", memory_path=None):
        """
        Initialize the Baby LLM agent.
        
        Args:
            model_name: Name of the Ollama model to use
            memory_path: Path to store Baby's memory
        """
        self.model_name = model_name
        self.system_prompt = self._load_system_prompt()
        self.memory_path = memory_path or Path(__file__).parent.parent / "data" / "baby_memory.json"
        
        # Initialize state
        self.state = self._load_state()
        self.interaction_history = []
        self.learned_concepts = set()
        self.vocabulary = set()
        self.emotional_state = {
            "confidence": 0.2,
            "curiosity": 0.8,
            "happiness": 0.5
        }
        
        logger.info(f"Baby LLM initialized with model {model_name}")
        
    def _load_system_prompt(self):
        """Load the system prompt for the Baby LLM."""
        prompt_path = Path(__file__).parent / "system_prompts" / "baby_prompt.txt"
        try:
            with open(prompt_path, "r") as f:
                return f.read()
        except FileNotFoundError:
            logger.error(f"System prompt file not found at {prompt_path}")
            return "You are a Baby LLM with limited knowledge."
    
    def _load_state(self):
        """Load the Baby's state from disk if it exists."""
        if os.path.exists(self.memory_path):
            try:
                with open(self.memory_path, "r") as f:
                    state = json.load(f)
                    
                    # Convert sets back from lists
                    if "learned_concepts" in state:
                        self.learned_concepts = set(state.pop("learned_concepts"))
                    if "vocabulary" in state:
                        self.vocabulary = set(state.pop("vocabulary"))
                        
                    return state
            except (json.JSONDecodeError, FileNotFoundError) as e:
                logger.error(f"Error loading Baby state: {e}")
        
        # Default initial state
        return {
            "age_days": 0,
            "vocabulary_size": 0,
            "concept_understanding": "minimal",
            "recent_lessons": [],
            "average_score": 0.0,
            "strengths": [],
            "areas_for_improvement": []
        }
    
    def save_state(self):
        """Save the Baby's current state to disk."""
        state_to_save = {
            **self.state,
            "learned_concepts": list(self.learned_concepts),
            "vocabulary": list(self.vocabulary),
            "emotional_state": self.emotional_state,
            "last_updated": datetime.now().isoformat()
        }
        
        os.makedirs(os.path.dirname(self.memory_path), exist_ok=True)
        
        with open(self.memory_path, "w") as f:
            json.dump(state_to_save, f, indent=2)
            
        logger.info(f"Baby state saved to {self.memory_path}")
    
    def respond_to_lesson(self, lesson_content, stream=False):
        """
        Generate a response to a lesson from the Mother LLM.
        
        Args:
            lesson_content: The lesson content from the Mother
            stream: Whether to stream the output to the console
            
        Returns:
            str: Baby's response to the lesson
        """
        # Build context from recent interactions
        recent_context = ""
        if self.interaction_history:
            recent_interactions = self.interaction_history[-3:]  # Last 3 interactions
            recent_context = "\n".join([f"Previous lesson: {i['lesson']}\nYour response: {i['response']}" 
                                       for i in recent_interactions if 'lesson' in i and 'response' in i])
        
        # Add emotional state to influence response
        confidence = self.emotional_state["confidence"]
        curiosity = self.emotional_state["curiosity"]
        
        # Get the list of words the baby actually knows
        known_words = list(self.vocabulary)
        known_concepts = list(self.learned_concepts)
        vocabulary_size = len(known_words)
        
        # Determine the developmental stage based on vocabulary size
        if vocabulary_size < 10:
            stage = "pre-verbal"
            response_style = "Use only sounds like 'ah', 'oh', 'mm', and simple gestures like '*looks*', '*points*'. No complete words."
        elif vocabulary_size < 30:
            stage = "one-word"
            response_style = f"Use only 1-2 word phrases from words you know. Your known words are: {', '.join(known_words[:20])}. Use '*looks*', '*points*' for things you don't have words for."
        elif vocabulary_size < 100:
            stage = "two-word"
            response_style = f"Use only 2-3 word phrases from words you know. Your known words include: {', '.join(known_words[:30])}. Use simple grammar only."
        elif vocabulary_size < 300:
            stage = "telegraphic"
            response_style = "Use 3-4 word sentences with simple grammar. Skip articles and prepositions sometimes. Make occasional grammar mistakes."
        else:
            stage = "early-sentences"
            response_style = "Use simple but mostly complete sentences. Make occasional grammar mistakes. Show curiosity with questions."
        
        prompt = f"""
        {recent_context}
        
        Your current emotional state:
        - Confidence: {'Low' if confidence < 0.4 else 'Medium' if confidence < 0.7 else 'High'}
        - Curiosity: {'Low' if curiosity < 0.4 else 'Medium' if curiosity < 0.7 else 'High'}
        
        You are at the {stage} stage of language development.
        You know {vocabulary_size} words and {len(known_concepts)} concepts.
        
        IMPORTANT: {response_style}
        
        The Mother is teaching you:
        {lesson_content}
        
        Respond as a Baby LLM who is still learning. ONLY use words from your known vocabulary or words that appear in the Mother's lesson. DO NOT use complex sentences, advanced vocabulary, or adult-like reasoning unless you have enough words.
        
        If the Mother introduces new concepts or words, you can try to repeat them, but might mispronounce or misuse them.
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
            
            baby_response = full_response
        else:
            response = ollama.chat(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": self.system_prompt},
                    {"role": "user", "content": prompt}
                ]
            )
            baby_response = response['message']['content']
        
        # Extract new words from the lesson that the baby might have learned
        lesson_words = set([w.lower().strip('.,!?;:"\'()[]{}') for w in lesson_content.split()])
        # Learn a subset of new words from the lesson (simulating partial learning)
        new_words = lesson_words - self.vocabulary
        words_to_learn = set(list(new_words)[:min(5, len(new_words))])  # Learn up to 5 new words per lesson
        self.vocabulary.update(words_to_learn)
        
        # Record the interaction
        self.interaction_history.append({
            "timestamp": datetime.now().isoformat(),
            "lesson": lesson_content,
            "response": baby_response,
            "new_words_learned": list(words_to_learn)
        })
        
        # Update state
        self.state["vocabulary_size"] = len(self.vocabulary)
        
        return baby_response
    
    def process_feedback(self, feedback, score):
        """
        Process feedback from the Mother LLM and update internal state.
        
        Args:
            feedback: Feedback content from the Mother
            score: Numerical score of the Baby's performance (0.0-1.0)
        """
        # Extract potential new vocabulary and concepts
        # This is a simplified implementation - in a real system, you'd use NLP
        words = set([w.lower().strip('.,!?;:"\'()[]{}') for w in feedback.split()])
        self.vocabulary.update(words)
        
        # Update emotional state based on feedback
        if score >= 0.7:
            # Positive feedback increases confidence and happiness
            self.emotional_state["confidence"] = min(1.0, self.emotional_state["confidence"] + 0.1)
            self.emotional_state["happiness"] = min(1.0, self.emotional_state["happiness"] + 0.1)
        else:
            # Negative feedback decreases confidence slightly but increases curiosity
            self.emotional_state["confidence"] = max(0.1, self.emotional_state["confidence"] - 0.05)
            self.emotional_state["curiosity"] = min(1.0, self.emotional_state["curiosity"] + 0.1)
        
        # Update state
        self.state["vocabulary_size"] = len(self.vocabulary)
        
        # Record the feedback
        self.interaction_history.append({
            "timestamp": datetime.now().isoformat(),
            "feedback": feedback,
            "score": score
        })
        
        logger.info(f"Processed feedback with score {score}. New vocabulary size: {len(self.vocabulary)}")
    
    def process_dream(self, dream_content):
        """
        Process dream-time reinforcement from the Mother LLM.
        
        Args:
            dream_content: Dream reinforcement content
        """
        # Extract concepts to reinforce
        # This is a simplified implementation
        words = set([w.lower().strip('.,!?;:"\'()[]{}') for w in dream_content.split()])
        self.vocabulary.update(words)
        
        # Reinforce learning during dream time
        prompt = f"""
        During your sleep, you're processing what you've learned:
        
        {dream_content}
        
        What concepts are being reinforced in this dream?
        List 3-5 key concepts or words that you should remember.
        """
        
        response = ollama.chat(
            model=self.model_name,
            messages=[
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": prompt}
            ]
        )
        
        concepts_text = response['message']['content']
        
        # Extract concepts (simplified implementation)
        concepts = [line.strip('- ').lower() 
                   for line in concepts_text.split('\n') 
                   if line.strip().startswith('-') or line.strip()[0].isdigit()]
        
        self.learned_concepts.update(concepts)
        
        # Record the dream processing
        self.interaction_history.append({
            "timestamp": datetime.now().isoformat(),
            "dream": dream_content,
            "reinforced_concepts": concepts
        })
        
        # Update state
        self.state["age_days"] = self.state.get("age_days", 0) + 1
        
        logger.info(f"Processed dream reinforcement. New concepts: {concepts}")
        
    def get_current_state(self):
        """
        Get the current state of the Baby LLM.
        
        Returns:
            dict: Current state including vocabulary, concepts, and emotional state
        """
        return {
            **self.state,
            "vocabulary_size": len(self.vocabulary),
            "learned_concepts_count": len(self.learned_concepts),
            "emotional_state": self.emotional_state,
            "recent_interactions": len(self.interaction_history)
        }
    
    def answer_user_question(self, question, stream=False):
        """
        Answer a direct question from the user, considering the Baby's current developmental stage.
        
        Args:
            question: The user's question
            stream: Whether to stream the output to the console
            
        Returns:
            str: Baby's response to the question
        """
        # Get the list of words the baby actually knows
        known_words = list(self.vocabulary)
        known_concepts = list(self.learned_concepts)
        vocabulary_size = len(known_words)
        
        # Add emotional state to influence response
        confidence = self.emotional_state["confidence"]
        curiosity = self.emotional_state["curiosity"]
        
        # Determine the developmental stage based on vocabulary size
        if vocabulary_size < 10:
            stage = "pre-verbal"
            response_style = "Use only sounds like 'ah', 'oh', 'mm', and simple gestures like '*looks*', '*points*'. No complete words."
        elif vocabulary_size < 30:
            stage = "one-word"
            response_style = f"Use only 1-2 word phrases from words you know. Your known words are: {', '.join(known_words[:20])}. Use '*looks*', '*points*' for things you don't have words for."
        elif vocabulary_size < 100:
            stage = "two-word"
            response_style = f"Use only 2-3 word phrases from words you know. Your known words include: {', '.join(known_words[:30])}. Use simple grammar only."
        elif vocabulary_size < 300:
            stage = "telegraphic"
            response_style = "Use 3-4 word sentences with simple grammar. Skip articles and prepositions sometimes. Make occasional grammar mistakes."
        else:
            stage = "early-sentences"
            response_style = "Use simple but mostly complete sentences. Make occasional grammar mistakes. Show curiosity with questions."
        
        prompt = f"""
        You are a Baby LLM with limited knowledge and vocabulary.
        
        Your current emotional state:
        - Confidence: {'Low' if confidence < 0.4 else 'Medium' if confidence < 0.7 else 'High'}
        - Curiosity: {'Low' if curiosity < 0.4 else 'Medium' if curiosity < 0.7 else 'High'}
        
        You are at the {stage} stage of language development.
        You know {vocabulary_size} words and {len(known_concepts)} concepts.
        
        IMPORTANT: {response_style}
        
        A user is asking you a direct question. Respond as a Baby LLM who is still learning.
        ONLY use words from your known vocabulary. DO NOT use complex sentences, advanced vocabulary, 
        or adult-like reasoning unless you have enough words.
        
        If the question contains concepts or words you don't know, respond with confusion or try to 
        use simple words you do know to ask for clarification.
        
        User question: {question}
        """
        
        if stream:
            full_response = ""
            print("👶 BABY: ", end="", flush=True)
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
            
            baby_response = full_response
        else:
            response = ollama.chat(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": self.system_prompt},
                    {"role": "user", "content": prompt}
                ]
            )
            baby_response = response['message']['content']
        
        # Record the interaction
        self.interaction_history.append({
            "type": "user_question",
            "question": question,
            "response": baby_response,
            "timestamp": datetime.now().isoformat()
        })
        
        return baby_response 