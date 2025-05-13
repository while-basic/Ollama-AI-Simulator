# ----------------------------------------------------------------------------
#  File:        context_manager.py
#  Project:     Celaya Solutions Ollama Simulator
#  Created by:  Celaya Solutions, 2025
#  Author:      Christopher Celaya <chris@celayasolutions.com>
#  Description: Implementation of context management for the simulation
#  Version:     1.0.0
#  License:     MIT (SPDX-Identifier: MIT)
#  Last Update: June 22, 2025
# ----------------------------------------------------------------------------

from loguru import logger

class ContextManager:
    """
    Context manager that orchestrates memory routing and history management.
    Provides relevant context for lessons and interactions.
    """
    
    def __init__(self, memory_store=None):
        """
        Initialize the context manager.
        
        Args:
            memory_store: HebbianMemoryStore instance
        """
        self.memory_store = memory_store
        self.interaction_history = []
        
        logger.info("Context manager initialized")
    
    def get_context_for_lesson(self, lesson_content, baby_state):
        """
        Get relevant context for a lesson.
        
        Args:
            lesson_content: Content of the lesson
            baby_state: Current state of the Baby LLM
            
        Returns:
            dict: Context information
        """
        # Stub implementation
        return {
            "baby_state": baby_state,
            "relevant_memories": [],
            "memory_summary": "",
            "recent_interactions": self.get_recent_interactions(3)
        }
    
    def record_interaction(self, interaction_type, content, metadata=None):
        """
        Record an interaction in the history.
        
        Args:
            interaction_type: Type of interaction (e.g., 'lesson', 'response', 'feedback')
            content: Content of the interaction
            metadata: Additional metadata
            
        Returns:
            int: Index of the recorded interaction
        """
        interaction = {
            "type": interaction_type,
            "content": content,
            "metadata": metadata or {}
        }
        
        self.interaction_history.append(interaction)
        return len(self.interaction_history) - 1
    
    def get_recent_interactions(self, count=5):
        """
        Get recent interactions from history.
        
        Args:
            count: Number of recent interactions to retrieve
            
        Returns:
            list: Recent interactions
        """
        return self.interaction_history[-count:] if self.interaction_history else []
    
    def get_interaction_by_index(self, index):
        """
        Get an interaction by its index.
        
        Args:
            index: Index of the interaction
            
        Returns:
            dict: Interaction data or None if not found
        """
        if 0 <= index < len(self.interaction_history):
            return self.interaction_history[index]
        return None
    
    def get_interactions_by_type(self, interaction_type, count=5):
        """
        Get recent interactions of a specific type.
        
        Args:
            interaction_type: Type of interactions to retrieve
            count: Maximum number of interactions to retrieve
            
        Returns:
            list: Filtered interactions
        """
        filtered = [i for i in self.interaction_history if i["type"] == interaction_type]
        return filtered[-count:] if filtered else []
    
    def clear_history(self):
        """Clear the interaction history."""
        self.interaction_history = []
        logger.info("Interaction history cleared")
    
    def get_context_summary(self):
        """
        Get a summary of the current context.
        
        Returns:
            dict: Context summary
        """
        # Count interaction types
        type_counts = {}
        for interaction in self.interaction_history:
            interaction_type = interaction["type"]
            type_counts[interaction_type] = type_counts.get(interaction_type, 0) + 1
        
        return {
            "total_interactions": len(self.interaction_history),
            "type_counts": type_counts,
            "recent_interactions": self.get_recent_interactions(3)
        } 