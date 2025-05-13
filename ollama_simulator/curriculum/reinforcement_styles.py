 # ----------------------------------------------------------------------------
#  File:        reinforcement_styles.py
#  Project:     Celaya Solutions Ollama Simulator
#  Created by:  Celaya Solutions, 2025
#  Author:      Christopher Celaya <chris@celayasolutions.com>
#  Description: Implementation of reinforcement styles for Baby LLM learning
#  Version:     1.0.0
#  License:     MIT (SPDX-Identifier: MIT)
#  Last Update: June 22, 2025
# ----------------------------------------------------------------------------

import yaml
import random
from loguru import logger
from pathlib import Path

class ReinforcementStyler:
    """
    Reinforcement styler that defines how feedback is provided to the Baby LLM.
    Adapts based on Mother's personality and Baby's needs.
    """
    
    def __init__(self, personas_path=None):
        """
        Initialize the reinforcement styler.
        
        Args:
            personas_path: Path to the personas.yaml file
        """
        self.personas_path = personas_path or Path(__file__).parent.parent / "config" / "personas.yaml"
        self.personas = self._load_personas()
        self.current_persona = self.personas.get("default_persona", "nurturing")
        
        logger.info(f"Reinforcement styler initialized with persona {self.current_persona}")
    
    def _load_personas(self):
        """Load personas from the personas.yaml file."""
        try:
            with open(self.personas_path, "r") as f:
                return yaml.safe_load(f)
        except (FileNotFoundError, yaml.YAMLError) as e:
            logger.error(f"Error loading personas: {e}")
            return {
                "mother_personas": {
                    "nurturing": {
                        "description": "A warm, encouraging teacher who emphasizes positive reinforcement",
                        "praise_frequency": 0.8,
                        "criticism_frequency": 0.2,
                        "patience": 0.9,
                        "repetition_tolerance": 0.8,
                        "emotional_support": 0.9
                    }
                },
                "default_persona": "nurturing"
            }
    
    def set_persona(self, persona_name):
        """
        Set the current persona.
        
        Args:
            persona_name: Name of the persona to use
            
        Returns:
            bool: True if persona was set successfully
        """
        if persona_name in self.personas.get("mother_personas", {}):
            self.current_persona = persona_name
            logger.info(f"Set persona to {persona_name}")
            return True
        else:
            logger.warning(f"Persona {persona_name} not found, keeping {self.current_persona}")
            return False
    
    def get_current_persona_traits(self):
        """
        Get traits of the current persona.
        
        Returns:
            dict: Persona traits
        """
        return self.personas.get("mother_personas", {}).get(self.current_persona, {})
    
    def should_praise(self, score):
        """
        Determine if praise should be given based on score and persona.
        
        Args:
            score: Evaluation score (0.0-1.0)
            
        Returns:
            bool: True if praise should be given
        """
        traits = self.get_current_persona_traits()
        praise_threshold = 0.5  # Default threshold
        
        # Adjust threshold based on persona traits
        if traits:
            # Lower threshold means more likely to praise
            praise_threshold = 0.7 - (traits.get("praise_frequency", 0.5) * 0.4)
        
        return score >= praise_threshold
    
    def should_criticize(self, score):
        """
        Determine if criticism should be given based on score and persona.
        
        Args:
            score: Evaluation score (0.0-1.0)
            
        Returns:
            bool: True if criticism should be given
        """
        traits = self.get_current_persona_traits()
        criticism_threshold = 0.5  # Default threshold
        
        # Adjust threshold based on persona traits
        if traits:
            # Higher threshold means less likely to criticize
            criticism_threshold = 0.3 + (traits.get("criticism_frequency", 0.5) * 0.4)
        
        return score < criticism_threshold
    
    def generate_praise_template(self, score):
        """
        Generate a praise template based on score and persona.
        
        Args:
            score: Evaluation score (0.0-1.0)
            
        Returns:
            str: Praise template
        """
        traits = self.get_current_persona_traits()
        
        # Basic templates
        basic_templates = [
            "Good job! {specific}",
            "Well done! {specific}",
            "That's right! {specific}",
            "Very good! {specific}",
            "Excellent! {specific}"
        ]
        
        # Emotional templates (used with high emotional support)
        emotional_templates = [
            "I'm so proud of you! {specific}",
            "Wonderful job! I'm happy to see you learning! {specific}",
            "That makes me so happy! {specific}",
            "You're doing amazing! {specific}",
            "I love how you're learning! {specific}"
        ]
        
        # Detailed templates (used with high patience)
        detailed_templates = [
            "That's correct! {specific} You're understanding this very well.",
            "Great work! {specific} I can see you're really thinking about this.",
            "Perfect! {specific} You've really grasped this concept.",
            "Excellent answer! {specific} You're making great progress.",
            "You got it right! {specific} This shows you're learning well."
        ]
        
        # Select template pool based on persona traits
        templates = basic_templates
        
        if traits:
            emotional_support = traits.get("emotional_support", 0.5)
            patience = traits.get("patience", 0.5)
            
            if emotional_support > 0.7 and random.random() < emotional_support:
                templates = emotional_templates
            elif patience > 0.7 and random.random() < patience:
                templates = detailed_templates
        
        # Select a template based on score
        if score > 0.9:
            # For very high scores, prefer more enthusiastic templates
            enthusiastic = ["Excellent!", "Perfect!", "Amazing!", "Wonderful!"]
            return random.choice(enthusiastic) + " {specific}"
        else:
            return random.choice(templates)
    
    def generate_criticism_template(self, score):
        """
        Generate a criticism template based on score and persona.
        
        Args:
            score: Evaluation score (0.0-1.0)
            
        Returns:
            str: Criticism template
        """
        traits = self.get_current_persona_traits()
        
        # Basic templates
        basic_templates = [
            "Not quite. {specific}",
            "Let's try again. {specific}",
            "That's not right. {specific}",
            "That's not correct. {specific}",
            "That's not it. {specific}"
        ]
        
        # Gentle templates (used with high patience and emotional support)
        gentle_templates = [
            "That's a good try, but not quite right. {specific}",
            "You're getting there, but not exactly. {specific}",
            "I see what you're trying to say, but {specific}",
            "That's close, but let's think about it differently. {specific}",
            "I like your effort, but let's try another way. {specific}"
        ]
        
        # Direct templates (used with low patience and high criticism frequency)
        direct_templates = [
            "No. {specific}",
            "Incorrect. {specific}",
            "That's wrong. {specific}",
            "Not right. {specific}",
            "That's not correct at all. {specific}"
        ]
        
        # Select template pool based on persona traits
        templates = basic_templates
        
        if traits:
            patience = traits.get("patience", 0.5)
            emotional_support = traits.get("emotional_support", 0.5)
            criticism_frequency = traits.get("criticism_frequency", 0.5)
            
            if patience > 0.7 and emotional_support > 0.6:
                templates = gentle_templates
            elif patience < 0.4 and criticism_frequency > 0.7:
                templates = direct_templates
        
        # Select a template based on score
        if score < 0.3:
            # For very low scores, be more direct but still within persona
            if traits and traits.get("patience", 0.5) > 0.7:
                return "That's not quite right, but it's okay to make mistakes. {specific}"
            else:
                return "That's incorrect. {specific}"
        else:
            return random.choice(templates)
    
    def adjust_for_repetition(self, is_repeated_error):
        """
        Adjust reinforcement style for repeated errors.
        
        Args:
            is_repeated_error: Whether this is a repeated error
            
        Returns:
            dict: Adjustments to make to reinforcement
        """
        traits = self.get_current_persona_traits()
        
        if not is_repeated_error:
            return {"patience_modifier": 0.0, "detail_modifier": 0.0}
        
        # Default adjustments
        patience_modifier = -0.2  # Decrease patience slightly
        detail_modifier = 0.2  # Increase detail slightly
        
        if traits:
            repetition_tolerance = traits.get("repetition_tolerance", 0.5)
            
            # Adjust based on repetition tolerance
            patience_modifier = -0.3 + (repetition_tolerance * 0.2)  # Higher tolerance = less patience reduction
            detail_modifier = 0.3  # Always increase detail for repetition
        
        return {"patience_modifier": patience_modifier, "detail_modifier": detail_modifier}
    
    def get_reinforcement_style(self, score, is_repeated_error=False):
        """
        Get the overall reinforcement style based on score and context.
        
        Args:
            score: Evaluation score (0.0-1.0)
            is_repeated_error: Whether this is a repeated error
            
        Returns:
            dict: Reinforcement style parameters
        """
        traits = self.get_current_persona_traits()
        
        # Determine if we should praise or criticize
        should_praise = self.should_praise(score)
        should_criticize = self.should_criticize(score)
        
        # Get template
        if should_praise:
            template = self.generate_praise_template(score)
            tone = "positive"
        elif should_criticize:
            template = self.generate_criticism_template(score)
            tone = "negative"
        else:
            # Neutral tone
            template = "I see. {specific}"
            tone = "neutral"
        
        # Get repetition adjustments
        repetition_adjustments = self.adjust_for_repetition(is_repeated_error)
        
        # Base style from persona traits
        base_style = {
            "patience": traits.get("patience", 0.5) + repetition_adjustments["patience_modifier"],
            "detail": 0.5 + repetition_adjustments["detail_modifier"],
            "emotional_support": traits.get("emotional_support", 0.5),
            "tone": tone,
            "template": template
        }
        
        # Ensure values are in valid range
        for key in ["patience", "detail", "emotional_support"]:
            base_style[key] = max(0.0, min(1.0, base_style[key]))
        
        return base_style 