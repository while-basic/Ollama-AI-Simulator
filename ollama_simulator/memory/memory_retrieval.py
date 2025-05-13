# ----------------------------------------------------------------------------
#  File:        memory_retrieval.py
#  Project:     Celaya Solutions Ollama Simulator
#  Created by:  Celaya Solutions, 2025
#  Author:      Christopher Celaya <chris@celayasolutions.com>
#  Description: Implementation of memory retrieval for the Baby LLM
#  Version:     1.0.0
#  License:     MIT (SPDX-Identifier: MIT)
#  Last Update: June 22, 2025
# ----------------------------------------------------------------------------

import numpy as np
import ollama
from loguru import logger
from .hebbian_store import HebbianMemoryStore

class MemoryRetrieval:
    """
    Memory retrieval system for the Baby LLM.
    Handles semantic search and context-based memory retrieval.
    """
    
    def __init__(self, memory_store=None, embedding_model="llama3.2:1b"):
        """
        Initialize the memory retrieval system.
        
        Args:
            memory_store: HebbianMemoryStore instance
            embedding_model: Model to use for generating embeddings
        """
        self.memory_store = memory_store or HebbianMemoryStore()
        self.embedding_model = embedding_model
        
        logger.info(f"Memory retrieval initialized with embedding model {embedding_model}")
    
    def get_embedding(self, text):
        """
        Get embedding vector for a text using Ollama.
        
        Args:
            text: Text to embed
            
        Returns:
            numpy.ndarray: Embedding vector
        """
        try:
            response = ollama.embeddings(
                model=self.embedding_model,
                prompt=text
            )
            
            if 'embedding' in response:
                return np.array(response['embedding'], dtype=np.float32)
            else:
                logger.error("No embedding in response")
                return np.zeros(384, dtype=np.float32)  # Default dimension
                
        except Exception as e:
            logger.error(f"Error getting embedding: {e}")
            return np.zeros(384, dtype=np.float32)  # Default dimension
    
    def retrieve_by_content(self, query, k=5):
        """
        Retrieve memories by semantic similarity to the query.
        
        Args:
            query: Query text
            k: Number of results to return
            
        Returns:
            list: List of memory dictionaries
        """
        # Get query embedding
        query_embedding = self.get_embedding(query)
        
        # Search in memory store
        memory_ids = self.memory_store.retrieve_similar_memories(query_embedding, k)
        
        # Get full memory data
        memories = []
        for memory_id, distance in memory_ids:
            memory = self.memory_store.get_memory_by_id(memory_id)
            if memory:
                memory["relevance_score"] = 1.0 / (1.0 + distance)  # Convert distance to similarity
                memories.append(memory)
        
        return memories
    
    def retrieve_by_association(self, memory_id, min_strength=0.3):
        """
        Retrieve memories associated with the given memory.
        
        Args:
            memory_id: ID of the memory to find associations for
            min_strength: Minimum association strength
            
        Returns:
            list: List of associated memory dictionaries
        """
        # Get associated memory IDs
        associations = self.memory_store.get_associated_memories(memory_id, min_strength)
        
        # Get full memory data
        memories = []
        for associated_id, strength in associations:
            memory = self.memory_store.get_memory_by_id(associated_id)
            if memory:
                memory["association_strength"] = strength
                memories.append(memory)
        
        # Sort by association strength
        memories.sort(key=lambda m: m["association_strength"], reverse=True)
        
        return memories
    
    def retrieve_recent_memories(self, limit=10):
        """
        Retrieve the most recently accessed memories.
        
        Args:
            limit: Maximum number of memories to return
            
        Returns:
            list: List of recent memory dictionaries
        """
        # This is a simplified implementation
        # In a real system, you'd query the database directly
        
        cursor = self.memory_store.conn.cursor()
        
        cursor.execute('''
        SELECT id FROM memories
        ORDER BY last_accessed DESC
        LIMIT ?
        ''', (limit,))
        
        memory_ids = [row[0] for row in cursor.fetchall()]
        
        # Get full memory data
        memories = []
        for memory_id in memory_ids:
            memory = self.memory_store.get_memory_by_id(memory_id)
            if memory:
                memories.append(memory)
        
        return memories
    
    def retrieve_emotional_memories(self, emotion, limit=5):
        """
        Retrieve memories with the specified emotional tag.
        
        Args:
            emotion: Emotion to search for
            limit: Maximum number of memories to return
            
        Returns:
            list: List of emotional memory dictionaries
        """
        # This is a simplified implementation
        # In a real system, you'd use a more sophisticated query
        
        cursor = self.memory_store.conn.cursor()
        
        # Search for emotion in JSON emotional_tags field
        cursor.execute('''
        SELECT id FROM memories
        WHERE emotional_tags LIKE ?
        ORDER BY confidence DESC
        LIMIT ?
        ''', (f'%"{emotion}"%', limit))
        
        memory_ids = [row[0] for row in cursor.fetchall()]
        
        # Get full memory data
        memories = []
        for memory_id in memory_ids:
            memory = self.memory_store.get_memory_by_id(memory_id)
            if memory:
                memories.append(memory)
        
        return memories
    
    def retrieve_context_for_lesson(self, lesson_content, limit=3):
        """
        Retrieve relevant context for a new lesson.
        
        Args:
            lesson_content: Content of the new lesson
            limit: Maximum number of context memories to return
            
        Returns:
            list: List of relevant memory dictionaries
        """
        # Get semantic memories
        semantic_memories = self.retrieve_by_content(lesson_content, k=limit)
        
        # Get recent memories
        recent_memories = self.retrieve_recent_memories(limit=limit)
        
        # Combine and deduplicate
        memory_ids = set()
        context_memories = []
        
        # First add semantic matches
        for memory in semantic_memories:
            if memory["id"] not in memory_ids:
                memory_ids.add(memory["id"])
                context_memories.append(memory)
        
        # Then add recent memories if not already included
        for memory in recent_memories:
            if memory["id"] not in memory_ids and len(context_memories) < limit * 2:
                memory_ids.add(memory["id"])
                context_memories.append(memory)
        
        return context_memories
    
    def generate_memory_summary(self, memories):
        """
        Generate a natural language summary of the given memories.
        
        Args:
            memories: List of memory dictionaries
            
        Returns:
            str: Natural language summary
        """
        if not memories:
            return "You don't have any relevant memories."
        
        # Create a summary prompt
        memory_texts = [f"Memory {i+1}: {m['content']}" for i, m in enumerate(memories)]
        memory_text = "\n".join(memory_texts)
        
        prompt = f"""
        Summarize these memories in a simple, child-like way:
        
        {memory_text}
        
        Summary:
        """
        
        try:
            response = ollama.chat(
                model=self.embedding_model,
                messages=[
                    {"role": "system", "content": "You are a helpful assistant that summarizes memories for a learning AI."},
                    {"role": "user", "content": prompt}
                ]
            )
            
            return response['message']['content']
        except Exception as e:
            logger.error(f"Error generating memory summary: {e}")
            return "You remember something about " + ", ".join([m["content"][:20] + "..." for m in memories[:3]]) 