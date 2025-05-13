# ----------------------------------------------------------------------------
#  File:        dream_engine.py
#  Project:     Celaya Solutions Ollama Simulator
#  Created by:  Celaya Solutions, 2025
#  Author:      Christopher Celaya <chris@celayasolutions.com>
#  Description: Implementation of dream-time memory consolidation
#  Version:     1.0.0
#  License:     MIT (SPDX-Identifier: MIT)
#  Last Update: June 22, 2025
# ----------------------------------------------------------------------------

import ollama
import random
from loguru import logger
from datetime import datetime
from .hebbian_store import HebbianMemoryStore
from .memory_retrieval import MemoryRetrieval
from .memory_writer import MemoryWriter

class DreamEngine:
    """
    Dream engine that simulates nighttime memory consolidation.
    Reinforces important memories, creates new associations, and prunes weak connections.
    """
    
    def __init__(self, memory_store=None, mother_model="llama3.2:latest", baby_model="llama3.2:1b"):
        """
        Initialize the dream engine.
        
        Args:
            memory_store: HebbianMemoryStore instance
            mother_model: Model name for the Mother LLM
            baby_model: Model name for the Baby LLM
        """
        self.memory_store = memory_store or HebbianMemoryStore()
        self.memory_retrieval = MemoryRetrieval(memory_store, baby_model)
        self.memory_writer = MemoryWriter(memory_store, baby_model)
        self.mother_model = mother_model
        self.baby_model = baby_model
        
        logger.info(f"Dream engine initialized with models: Mother={mother_model}, Baby={baby_model}")
    
    def generate_dream(self, baby_state, stream=False):
        """
        Generate dream content based on baby's current state.
        
        Args:
            baby_state: Current state of the Baby LLM
            stream: Whether to stream the output to the console
            
        Returns:
            str: Generated dream content
        """
        # Get recent memories to incorporate into the dream
        recent_memories = self.memory_store.get_recent_memories(limit=5)
        memory_content = "\n".join([m.get("content", "")[:100] + "..." for m in recent_memories])
        
        # Create dream prompt for Mother
        prompt = f"""
        You are creating a dream sequence for a Baby LLM that will help consolidate its learning.
        The dream should be simple, positive, and reinforce recent memories in a creative way.
        
        Baby's current state:
        - Vocabulary size: {baby_state.get('vocabulary_size', 0)} words
        - Concept understanding: {baby_state.get('concept_understanding', 'basic')}
        
        Recent memories:
        {memory_content}
        
        Create a dream sequence that:
        1. Reinforces key concepts from recent learning
        2. Is appropriate for the Baby's current development level
        3. Has a positive, supportive tone
        4. Uses simple language and imagery
        5. Includes some repetition of important words or ideas
        
        Dream content:
        """
        
        # Generate dream content using Mother model
        try:
            if stream:
                full_response = ""
                for chunk in ollama.chat(
                    model=self.mother_model,
                    messages=[
                        {"role": "system", "content": "You are creating a dream sequence for a learning AI."},
                        {"role": "user", "content": prompt}
                    ],
                    stream=True
                ):
                    if 'message' in chunk and 'content' in chunk['message']:
                        content = chunk['message']['content']
                        # Don't print here - let the caller handle printing
                        full_response += content
                
                dream_content = full_response
            else:
                response = ollama.chat(
                    model=self.mother_model,
                    messages=[
                        {"role": "system", "content": "You are creating a dream sequence for a learning AI."},
                        {"role": "user", "content": prompt}
                    ]
                )
                dream_content = response['message']['content']
            
            logger.info("Generated dream content")
            return dream_content
            
        except Exception as e:
            logger.error(f"Error generating dream content: {e}")
            # Fallback dream content
            return "I dreamed about learning new words and concepts. It was peaceful and reinforcing."
    
    def process_dream(self, dream_content, baby_state):
        """
        Process a dream sequence to consolidate learning.
        
        Args:
            dream_content: The dream content
            baby_state: Current state of the Baby LLM
            
        Returns:
            dict: Results of dream processing
        """
        # Extract concepts from dream
        reinforced_concepts = self._extract_concepts_from_dream(dream_content)
        
        # Store the dream as a memory
        dream_memory_id = self.memory_writer.store_dream_memory(dream_content, reinforced_concepts)
        
        # Find related memories
        related_memories = self._find_related_memories(dream_content)
        related_memory_ids = [m["id"] for m in related_memories]
        
        # Create associations between dream and related memories
        if related_memory_ids:
            all_memory_ids = [dream_memory_id] + related_memory_ids
            self.memory_writer.create_associations_between_memories(all_memory_ids)
        
        # Apply memory decay
        decay_count = self.memory_store.decay_memories(decay_factor=0.98)
        
        # Save memory store
        self.memory_store.save()
        
        # Get updated stats
        stats = self.memory_store.get_stats()
        
        logger.info(f"Processed dream with {len(reinforced_concepts)} concepts, created associations with {len(related_memory_ids)} memories")
        
        return {
            "dream_memory_id": dream_memory_id,
            "reinforced_concepts": reinforced_concepts,
            "related_memory_count": len(related_memory_ids),
            "decay_count": decay_count,
            "memory_stats": stats
        }
    
    def _extract_concepts_from_dream(self, dream_content):
        """
        Extract key concepts from dream content.
        
        Args:
            dream_content: Dream content
            
        Returns:
            list: Extracted concepts
        """
        prompt = f"""
        Extract the key concepts from this dream sequence:
        
        {dream_content}
        
        List 3-5 key concepts or words that should be reinforced in the Baby's memory.
        Format each concept as a single word or short phrase on a new line starting with a hyphen.
        """
        
        try:
            response = ollama.chat(
                model=self.baby_model,
                messages=[
                    {"role": "system", "content": "You extract key concepts from text."},
                    {"role": "user", "content": prompt}
                ]
            )
            
            concept_text = response['message']['content']
            
            # Parse concepts from response
            concepts = []
            for line in concept_text.split('\n'):
                line = line.strip()
                if line.startswith('-') or line.startswith('*'):
                    concept = line[1:].strip()
                    if concept:
                        concepts.append(concept)
                elif line and not any(c in line.lower() for c in ['concept', 'key', 'reinforced']):
                    # Might be a plain list without bullets
                    concepts.append(line)
            
            return concepts
            
        except Exception as e:
            logger.error(f"Error extracting concepts from dream: {e}")
            # Extract simple word-based concepts as fallback
            words = dream_content.split()
            # Filter to words of reasonable length, exclude common words
            common_words = {'the', 'and', 'a', 'to', 'of', 'in', 'that', 'is', 'was', 'it', 'for', 'on', 'with'}
            concepts = [w for w in words if len(w) > 3 and w.lower() not in common_words]
            # Take unique concepts
            return list(set(concepts))[:5]
    
    def _get_important_memories(self, limit=5):
        """
        Get important memories to reinforce.
        
        Args:
            limit: Maximum number of memories to return
            
        Returns:
            list: Important memories
        """
        # Get memories with high confidence
        cursor = self.memory_store.conn.cursor()
        
        cursor.execute('''
        SELECT id FROM memories
        WHERE confidence > 0.7
        ORDER BY RANDOM()
        LIMIT ?
        ''', (limit,))
        
        high_confidence_ids = [row[0] for row in cursor.fetchall()]
        
        # Get memories with high access count
        cursor.execute('''
        SELECT id FROM memories
        WHERE access_count > 1
        ORDER BY access_count DESC
        LIMIT ?
        ''', (limit,))
        
        high_access_ids = [row[0] for row in cursor.fetchall()]
        
        # Get recent memories
        cursor.execute('''
        SELECT id FROM memories
        ORDER BY last_accessed DESC
        LIMIT ?
        ''', (limit,))
        
        recent_ids = [row[0] for row in cursor.fetchall()]
        
        # Combine and deduplicate
        all_ids = list(set(high_confidence_ids + high_access_ids + recent_ids))
        random.shuffle(all_ids)  # Randomize order
        
        # Get full memory data
        memories = []
        for memory_id in all_ids[:limit]:
            memory = self.memory_store.get_memory_by_id(memory_id)
            if memory:
                memories.append(memory)
        
        return memories
    
    def _find_related_memories(self, content, limit=5):
        """
        Find memories related to the content.
        
        Args:
            content: Content to find related memories for
            limit: Maximum number of memories to return
            
        Returns:
            list: Related memories
        """
        # Use semantic search to find related memories
        return self.memory_retrieval.retrieve_by_content(content, k=limit) 