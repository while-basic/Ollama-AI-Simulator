# ----------------------------------------------------------------------------
#  File:        memory_writer.py
#  Project:     Celaya Solutions Ollama Simulator
#  Created by:  Celaya Solutions, 2025
#  Author:      Christopher Celaya <chris@celayasolutions.com>
#  Description: Implementation of memory writing for the Baby LLM
#  Version:     1.0.0
#  License:     MIT (SPDX-Identifier: MIT)
#  Last Update: June 22, 2025
# ----------------------------------------------------------------------------

import os
import json
import ollama
import numpy as np
from loguru import logger
from pathlib import Path
from datetime import datetime
from .hebbian_store import HebbianMemoryStore
import faiss

class MemoryWriter:
    """
    Memory writer system for the Baby LLM.
    Handles storing memories, creating associations, and writing to Obsidian.
    """
    
    def __init__(self, memory_store=None, embedding_model="llama3.2:1b", obsidian_path=None):
        """
        Initialize the memory writer system.
        
        Args:
            memory_store: HebbianMemoryStore instance
            embedding_model: Model to use for generating embeddings
            obsidian_path: Path to Obsidian vault
        """
        self.memory_store = memory_store or HebbianMemoryStore()
        self.embedding_model = embedding_model
        self.obsidian_path = obsidian_path or Path(__file__).parent.parent / "data" / "obsidian_vault"
        
        # Ensure Obsidian directories exist
        self._ensure_obsidian_dirs()
        
        logger.info(f"Memory writer initialized with embedding model {embedding_model}")
    
    def _ensure_obsidian_dirs(self):
        """Ensure Obsidian directory structure exists."""
        # Create main directories
        os.makedirs(self.obsidian_path, exist_ok=True)
        os.makedirs(self.obsidian_path / "baby_growth", exist_ok=True)
        os.makedirs(self.obsidian_path / "mother_logs", exist_ok=True)
        os.makedirs(self.obsidian_path / "memory_visualizations", exist_ok=True)
        
        # Create index files if they don't exist
        baby_index = self.obsidian_path / "baby_growth" / "index.md"
        if not baby_index.exists():
            with open(baby_index, "w") as f:
                f.write("# Baby LLM Growth Journal\n\nThis folder contains the developmental journey of the Baby LLM.\n\n## Recent Entries\n\n")
        
        mother_index = self.obsidian_path / "mother_logs" / "index.md"
        if not mother_index.exists():
            with open(mother_index, "w") as f:
                f.write("# Mother LLM Teaching Journal\n\nThis folder contains the teaching logs of the Mother LLM.\n\n## Recent Entries\n\n")
        
        vis_index = self.obsidian_path / "memory_visualizations" / "index.md"
        if not vis_index.exists():
            with open(vis_index, "w") as f:
                f.write("# Memory Network Visualizations\n\nThis folder contains visualizations of the Baby LLM's memory network.\n\n## Visualizations\n\n")
        
        # Create metadata file for Obsidian
        obsidian_config = self.obsidian_path / ".obsidian" / "config"
        os.makedirs(os.path.dirname(obsidian_config), exist_ok=True)
        if not obsidian_config.exists():
            with open(obsidian_config, "w") as f:
                f.write('{"enabledPlugins":["obsidian-git","dataview"]}')
        
        logger.info(f"Ensured Obsidian directories at {self.obsidian_path}")
    
    def get_embedding(self, text):
        """
        Get embedding vector for text.
        
        Args:
            text: Text to get embedding for
            
        Returns:
            numpy.ndarray: Embedding vector
        """
        try:
            # Use Ollama to get embeddings
            response = ollama.embeddings(
                model=self.embedding_model,
                prompt=text
            )
            
            if 'embedding' in response:
                # Get the embedding vector
                embedding = np.array(response['embedding'], dtype=np.float32)
                
                # Check if the dimension matches what we expect
                if self.memory_store.vector_dim != len(embedding):
                    logger.warning(f"Embedding dimension mismatch: expected {self.memory_store.vector_dim}, got {len(embedding)}")
                    
                    # Initialize a new FAISS index with the correct dimension
                    self.memory_store.vector_dim = len(embedding)
                    self.memory_store.index = faiss.IndexFlatL2(len(embedding))
                    logger.info(f"Created new FAISS index with dimension {len(embedding)}")
                
                return embedding
            else:
                logger.error("No embedding found in response")
                # Return a zero vector of the expected dimension
                return np.zeros(self.memory_store.vector_dim, dtype=np.float32)
                
        except Exception as e:
            logger.error(f"Error getting embedding: {e}")
            # Return a zero vector of the expected dimension
            return np.zeros(self.memory_store.vector_dim, dtype=np.float32)
    
    def extract_emotional_tags(self, content):
        """
        Extract emotional tags from content using LLM.
        
        Args:
            content: Text content to analyze
            
        Returns:
            dict: Emotional tags with confidence scores
        """
        prompt = f"""
        Extract emotional tags from this text:
        
        "{content}"
        
        Return a JSON object with emotion names as keys and confidence scores (0.0-1.0) as values.
        Only include emotions that are clearly present in the text.
        Example: {{"happy": 0.8, "curious": 0.6}}
        """
        
        try:
            response = ollama.chat(
                model=self.embedding_model,
                messages=[
                    {"role": "system", "content": "You are a helpful assistant that extracts emotional content from text."},
                    {"role": "user", "content": prompt}
                ]
            )
            
            response_text = response['message']['content']
            
            # Extract JSON from the response
            try:
                # Find JSON block
                json_start = response_text.find('{')
                json_end = response_text.rfind('}') + 1
                
                if json_start >= 0 and json_end > json_start:
                    json_str = response_text[json_start:json_end]
                    emotional_tags = json.loads(json_str)
                    return emotional_tags
                else:
                    return {}
            except json.JSONDecodeError:
                logger.error("Failed to parse emotional tags JSON")
                return {}
                
        except Exception as e:
            logger.error(f"Error extracting emotional tags: {e}")
            return {}
    
    def store_lesson_memory(self, lesson_content, baby_response, evaluation):
        """
        Store a lesson memory.
        
        Args:
            lesson_content: Content of the lesson
            baby_response: Baby's response to the lesson
            evaluation: Evaluation of the response
            
        Returns:
            int: ID of the stored memory
        """
        try:
            # Combine lesson and response for the memory
            combined_content = f"LESSON: {lesson_content}\n\nRESPONSE: {baby_response}"
            
            # Get embedding
            embedding = self.get_embedding(combined_content)
            
            # Extract emotional tags
            emotional_tags = self.extract_emotional_tags(baby_response)
            
            # Get confidence from evaluation
            score = evaluation.get("overall_score", evaluation.get("score", 0.5))
            
            # Store in memory store
            memory_id = self.memory_store.store_memory(
                content=combined_content,
                vector=embedding,
                emotional_tags=emotional_tags,
                confidence=score,
                source="lesson"
            )
            
            # Write to Obsidian
            self._write_baby_memory_to_obsidian(
                memory_id=memory_id,
                memory_type="lesson",
                content=combined_content,
                emotional_tags=emotional_tags,
                confidence=score
            )
            
            logger.info(f"Stored lesson memory with ID {memory_id}")
            return memory_id
        except Exception as e:
            logger.error(f"Error storing lesson memory: {e}")
            return None
    
    def store_feedback_memory(self, feedback_content, score):
        """
        Store a feedback memory.
        
        Args:
            feedback_content: Content of the feedback
            score: Score associated with the feedback
            
        Returns:
            int: ID of the stored memory
        """
        try:
            # Get embedding
            embedding = self.get_embedding(feedback_content)
            
            # Extract emotional tags
            emotional_tags = self.extract_emotional_tags(feedback_content)
            
            # Store in memory store
            memory_id = self.memory_store.store_memory(
                content=feedback_content,
                vector=embedding,
                emotional_tags=emotional_tags,
                confidence=score,
                source="feedback"
            )
            
            # Write to Obsidian
            self._write_baby_memory_to_obsidian(
                memory_id=memory_id,
                memory_type="feedback",
                content=feedback_content,
                emotional_tags=emotional_tags,
                confidence=score
            )
            
            logger.info(f"Stored feedback memory with ID {memory_id}")
            return memory_id
        except Exception as e:
            logger.error(f"Error storing feedback memory: {e}")
            return None
            
    def store_dream_memory(self, dream_content, reinforced_concepts=None):
        """
        Store a dream memory.
        
        Args:
            dream_content: Content of the dream
            reinforced_concepts: Concepts reinforced in the dream
            
        Returns:
            int: ID of the stored memory
        """
        try:
            # Format the content with reinforced concepts
            if reinforced_concepts:
                concepts_str = "\n".join([f"- {concept}" for concept in reinforced_concepts])
                formatted_content = f"{dream_content}\n\nREINFORCED CONCEPTS:\n{concepts_str}"
            else:
                formatted_content = dream_content
            
            # Get embedding
            embedding = self.get_embedding(formatted_content)
            
            # Extract emotional tags
            emotional_tags = self.extract_emotional_tags(dream_content)
            
            # Store in memory store
            memory_id = self.memory_store.store_memory(
                content=formatted_content,
                vector=embedding,
                emotional_tags=emotional_tags,
                confidence=0.7,  # Dreams are fairly confident memories
                source="dream"
            )
            
            # Write to Obsidian
            self._write_baby_memory_to_obsidian(
                memory_id=memory_id,
                memory_type="dream",
                content=formatted_content,
                emotional_tags=emotional_tags,
                confidence=0.7
            )
            
            logger.info(f"Stored dream memory with ID {memory_id}")
            return memory_id
        except Exception as e:
            logger.error(f"Error storing dream memory: {e}")
            return None
    
    def create_associations_between_memories(self, memory_ids, strength_matrix=None):
        """
        Create associations between multiple memories.
        
        Args:
            memory_ids: List of memory IDs to associate
            strength_matrix: Optional matrix of association strengths
            
        Returns:
            list: IDs of created associations
        """
        try:
            if not memory_ids or len(memory_ids) < 2:
                logger.warning("Need at least 2 memory IDs to create associations")
                return []
            
            # Filter out None values (failed memory creations)
            memory_ids = [mid for mid in memory_ids if mid is not None]
            if len(memory_ids) < 2:
                logger.warning("Not enough valid memory IDs to create associations")
                return []
            
            association_ids = []
            
            # Create associations between all pairs
            for i in range(len(memory_ids)):
                for j in range(i+1, len(memory_ids)):
                    # Get strength if provided
                    if strength_matrix is not None:
                        strength = strength_matrix[i][j]
                    else:
                        strength = 0.5  # Default strength
                    
                    # Create bidirectional associations
                    assoc_id1 = self.memory_store.create_association(
                        memory_ids[i], memory_ids[j], strength
                    )
                    association_ids.append(assoc_id1)
                    
                    # Create reverse association (with same strength)
                    assoc_id2 = self.memory_store.create_association(
                        memory_ids[j], memory_ids[i], strength
                    )
                    association_ids.append(assoc_id2)
            
            logger.info(f"Created {len(association_ids)} associations between {len(memory_ids)} memories")
            return association_ids
        except Exception as e:
            logger.error(f"Error creating associations: {e}")
            return []
    
    def write_mother_log(self, log_type, content, metadata=None):
        """
        Write a log entry for the Mother LLM to Obsidian.
        
        Args:
            log_type: Type of log entry (e.g., 'lesson', 'evaluation', 'milestone')
            content: Log content
            metadata: Additional metadata to include
            
        Returns:
            str: Path to the created file
        """
        now = datetime.now()
        date_str = now.strftime("%Y-%m-%d")
        time_str = now.strftime("%H-%M-%S")
        
        # Create a descriptive title based on content
        title_base = content.split("\n")[0][:30].strip()
        if not title_base:
            title_base = f"{log_type.capitalize()} Entry"
        
        # Create unique ID for this log
        log_id = f"{date_str}-{time_str}"
        
        # Create filename
        filename = f"{date_str}-{log_id}-{log_type}.md"
        file_path = self.obsidian_path / "mother_logs" / filename
        
        # Create YAML frontmatter
        frontmatter = f"""---
id: {log_id}
type: {log_type}
created: {now.isoformat()}
---

"""
        
        # Create title
        title = f"# {log_type.capitalize()}: {title_base}"
        
        # Format content with proper markdown
        formatted_content = content.replace("\n", "\n\n")
        
        # Add metadata section
        metadata_str = "\n## Metadata\n\n"
        if metadata:
            for key, value in metadata.items():
                metadata_str += f"- **{key}**: {value}\n"
        else:
            metadata_str += "- **Created**: " + now.strftime("%Y-%m-%d %H:%M:%S") + "\n"
        
        # Combine all parts
        file_content = frontmatter + title + "\n\n" + formatted_content + "\n\n" + metadata_str
        
        # Write to file
        with open(file_path, "w") as f:
            f.write(file_content)
        
        logger.info(f"Wrote Mother log to {file_path}")
        
        # Update index
        self._update_mother_index(log_type, filename, title_base)
        
        return str(file_path)
    
    def _write_baby_memory_to_obsidian(self, memory_id, memory_type, content, emotional_tags, confidence):
        """
        Write a Baby memory to Obsidian.
        
        Args:
            memory_id: ID of the memory
            memory_type: Type of memory
            content: Memory content
            emotional_tags: Emotional tags
            confidence: Confidence score
            
        Returns:
            str: Path to the created file
        """
        now = datetime.now()
        date_str = now.strftime("%Y-%m-%d")
        time_str = now.strftime("%H-%M-%S")
        
        # Create a descriptive title based on content
        title_base = content.split("\n")[0][:30].strip()
        if not title_base:
            title_base = f"{memory_type.capitalize()} Entry"
        
        # Create filename with date and ID for uniqueness
        filename = f"{date_str}-{memory_id}-{memory_type}.md"
        file_path = self.obsidian_path / "baby_growth" / filename
        
        # Format emotional tags for display
        emotions_formatted = ""
        if emotional_tags:
            emotions_list = []
            for emotion, score in emotional_tags.items():
                emotions_list.append(f"{emotion}: {score:.2f}")
            emotions_formatted = ", ".join(emotions_list)
        else:
            emotions_formatted = "None detected"
        
        # Create YAML frontmatter
        frontmatter = f"""---
id: {memory_id}
type: {memory_type}
created: {now.isoformat()}
confidence: {confidence:.2f}
emotional_tags: {emotions_formatted}
---

"""
        
        # Create title
        title = f"# {memory_type.capitalize()}: {title_base}"
        
        # Format content with proper markdown
        formatted_content = content.replace("\n", "\n\n")
        
        # Add metadata section
        metadata = f"""
## Metadata

- **ID**: {memory_id}
- **Type**: {memory_type}
- **Created**: {now.strftime("%Y-%m-%d %H:%M:%S")}
- **Confidence**: {confidence:.2f}
- **Emotional Tags**: {emotions_formatted}

"""
        
        # Combine all parts
        file_content = frontmatter + title + "\n\n" + formatted_content + "\n\n" + metadata
        
        # Write to file
        with open(file_path, "w") as f:
            f.write(file_content)
        
        logger.info(f"Wrote Baby memory to {file_path}")
        
        # Update index
        self._update_baby_index(memory_type, filename, title_base)
        
        return str(file_path)
    
    def _update_baby_index(self, memory_type, filename, snippet):
        """Update the Baby growth index."""
        index_path = self.obsidian_path / "baby_growth" / "index.md"
        now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        with open(index_path, "r") as f:
            content = f.read()
        
        # Add new entry at the top of the Recent Entries section
        entry_line = f"- [{now} - {memory_type.capitalize()}: {snippet}](baby_growth/{filename})\n"
        
        if "## Recent Entries" in content:
            parts = content.split("## Recent Entries")
            new_content = parts[0] + "## Recent Entries\n\n" + entry_line + parts[1].split("\n", 1)[1]
        else:
            new_content = content + "\n## Recent Entries\n\n" + entry_line
        
        with open(index_path, "w") as f:
            f.write(new_content)
    
    def _update_mother_index(self, log_type, filename, snippet):
        """Update the Mother logs index."""
        index_path = self.obsidian_path / "mother_logs" / "index.md"
        now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        with open(index_path, "r") as f:
            content = f.read()
        
        # Add new entry at the top of the Recent Entries section
        entry_line = f"- [{now} - {log_type.capitalize()}: {snippet}](mother_logs/{filename})\n"
        
        if "## Recent Entries" in content:
            parts = content.split("## Recent Entries")
            new_content = parts[0] + "## Recent Entries\n\n" + entry_line + parts[1].split("\n", 1)[1]
        else:
            new_content = content + "\n## Recent Entries\n\n" + entry_line
        
        with open(index_path, "w") as f:
            f.write(new_content)
    
    def create_memory_visualization(self):
        """
        Create a visual representation of the memory network in Obsidian.
        Uses Obsidian's graph view and Markdown links to visualize connections.
        
        Returns:
            str: Path to the created visualization file
        """
        now = datetime.now()
        date_str = now.strftime("%Y-%m-%d")
        
        # Create filename
        filename = f"{date_str}-memory-network.md"
        file_path = self.obsidian_path / "memory_visualizations" / filename
        
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        
        # Get recent memories
        recent_memories = self.memory_store.get_recent_memories(limit=20)
        
        # Create YAML frontmatter
        frontmatter = f"""---
id: memory-network-{date_str}
type: visualization
created: {now.isoformat()}
---

"""
        
        # Create title
        title = f"# Memory Network Visualization: {date_str}"
        
        # Create mermaid diagram for memory network
        mermaid_content = """
```mermaid
graph TD
    classDef lesson fill:#f9d5e5,stroke:#333,stroke-width:1px
    classDef feedback fill:#eeeeee,stroke:#333,stroke-width:1px
    classDef dream fill:#d5f9e5,stroke:#333,stroke-width:1px
    
"""
        
        # Add nodes for memories
        memory_nodes = {}
        for i, memory in enumerate(recent_memories):
            memory_id = memory.get("id", f"unknown-{i}")
            memory_type = memory.get("source", "unknown")
            content_preview = memory.get("content", "")[:30].replace("\n", " ").strip()
            
            node_id = f"M{memory_id}"
            memory_nodes[memory_id] = node_id
            
            # Add node to diagram
            mermaid_content += f"    {node_id}[\"{content_preview}...\"]"
            
            # Add class based on memory type
            if memory_type == "lesson":
                mermaid_content += ":::lesson"
            elif memory_type == "feedback":
                mermaid_content += ":::feedback"
            elif memory_type == "dream":
                mermaid_content += ":::dream"
            
            mermaid_content += "\n"
        
        # Add connections between memories
        for memory_id in memory_nodes:
            # Get associated memories
            associations = self.memory_store.get_associated_memories(memory_id, min_strength=0.3)
            
            for assoc_id, strength in associations:
                if assoc_id in memory_nodes:
                    # Calculate line thickness based on strength
                    thickness = "normal"
                    if strength > 0.7:
                        thickness = "thick"
                    elif strength < 0.4:
                        thickness = "dotted"
                    
                    # Add connection to diagram
                    if thickness == "thick":
                        mermaid_content += f"    {memory_nodes[memory_id]} ==> {memory_nodes[assoc_id]}\n"
                    elif thickness == "dotted":
                        mermaid_content += f"    {memory_nodes[memory_id]} -.-> {memory_nodes[assoc_id]}\n"
                    else:
                        mermaid_content += f"    {memory_nodes[memory_id]} --> {memory_nodes[assoc_id]}\n"
        
        # Close mermaid diagram
        mermaid_content += "```\n"
        
        # Create emotional state visualization
        emotional_mermaid = """
```mermaid
flowchart LR
    classDef high fill:#d4f7a4,stroke:#333,stroke-width:1px
    classDef medium fill:#ffe599,stroke:#333,stroke-width:1px
    classDef low fill:#f8cecc,stroke:#333,stroke-width:1px
"""
        
        # Extract emotional tags from recent memories
        all_emotions = {}
        for memory in recent_memories:
            if "emotional_tags" in memory:
                for emotion, strength in memory["emotional_tags"].items():
                    if emotion in all_emotions:
                        all_emotions[emotion] = max(all_emotions[emotion], strength)
                    else:
                        all_emotions[emotion] = strength
        
        # Create emotion nodes
        if all_emotions:
            emotional_mermaid += "    E[Emotions]\n"
            
            for emotion, strength in all_emotions.items():
                emotion_id = f"E_{emotion.replace(' ', '_')}"
                emotional_mermaid += f"    {emotion_id}[\"{emotion}: {strength:.2f}\"]\n"
                
                # Add class based on strength
                if strength > 0.7:
                    emotional_mermaid += f"    {emotion_id}:::high\n"
                elif strength > 0.4:
                    emotional_mermaid += f"    {emotion_id}:::medium\n"
                else:
                    emotional_mermaid += f"    {emotion_id}:::low\n"
                
                # Connect to center
                emotional_mermaid += f"    E --- {emotion_id}\n"
        else:
            emotional_mermaid += "    E[No emotions detected]\n"
        
        # Close emotional mermaid diagram
        emotional_mermaid += "```\n"
        
        # Create concept map visualization
        concept_mermaid = """
```mermaid
mindmap
    root((Memory Concepts))
"""
        
        # Extract concepts from recent memories
        concepts = {}
        for memory in recent_memories:
            content = memory.get("content", "")
            
            # Extract potential concepts (simplified approach)
            words = [w.strip('.,!?;:"\'()[]{}') for w in content.split()]
            for word in words:
                if len(word) > 4 and word.lower() not in ["lesson", "response", "feedback", "dream"]:
                    if word in concepts:
                        concepts[word] += 1
                    else:
                        concepts[word] = 1
        
        # Sort concepts by frequency
        sorted_concepts = sorted(concepts.items(), key=lambda x: x[1], reverse=True)
        
        # Create concept hierarchy (simplified)
        top_concepts = sorted_concepts[:5]  # Top 5 concepts
        secondary_concepts = sorted_concepts[5:15]  # Next 10 concepts
        
        # Add top concepts
        for concept, count in top_concepts:
            concept_mermaid += f"    {concept} [{count}]\n"
            
            # Add related secondary concepts
            related = []
            for sec_concept, sec_count in secondary_concepts:
                if sec_concept.lower() in concept.lower() or concept.lower() in sec_concept.lower():
                    related.append((sec_concept, sec_count))
            
            # Add up to 3 related concepts
            for rel_concept, rel_count in related[:3]:
                concept_mermaid += f"        {rel_concept} [{rel_count}]\n"
        
        # Close concept mermaid diagram
        concept_mermaid += "```\n"
        
        # Create memory list with links
        memory_list = "\n## Memory Details\n\n"
        for memory in recent_memories:
            memory_id = memory.get("id", "unknown")
            memory_type = memory.get("source", "unknown")
            content_preview = memory.get("content", "")[:100].replace("\n", " ").strip()
            confidence = memory.get("confidence", 0.0)
            
            memory_list += f"- **Memory {memory_id}** ({memory_type}): {content_preview}... (Confidence: {confidence:.2f})\n"
        
        # Add statistics
        stats = self.memory_store.get_stats()
        stats_section = f"""
## Memory Statistics

- Total Memories: {stats.get('total_memories', 0)}
- Total Associations: {stats.get('total_associations', 0)}
- Average Confidence: {stats.get('average_confidence', 0.0):.2f}
- Average Association Strength: {stats.get('average_association_strength', 0.0):.2f}
- Reinforced Memories: {stats.get('reinforced_memories', 0)}
- Forgotten Memories: {stats.get('forgotten_memories', 0)}

"""
        
        # Combine all parts
        file_content = (
            frontmatter + 
            title + "\n\n" + 
            "## Memory Network\n\n" + mermaid_content + "\n" +
            "## Emotional State\n\n" + emotional_mermaid + "\n" +
            "## Concept Map\n\n" + concept_mermaid + "\n" +
            stats_section + "\n" + 
            memory_list
        )
        
        # Write to file
        with open(file_path, "w") as f:
            f.write(file_content)
        
        logger.info(f"Created memory visualization at {file_path}")
        
        # Update index
        self._update_visualization_index(filename, date_str)
        
        return str(file_path)
    
    def _update_visualization_index(self, filename, date_str):
        """Update the visualization index."""
        # Create directory if it doesn't exist
        vis_dir = self.obsidian_path / "memory_visualizations"
        os.makedirs(vis_dir, exist_ok=True)
        
        index_path = vis_dir / "index.md"
        
        # Create index if it doesn't exist
        if not index_path.exists():
            with open(index_path, "w") as f:
                f.write("# Memory Network Visualizations\n\nThis folder contains visualizations of the Baby LLM's memory network.\n\n## Visualizations\n\n")
        
        # Read existing index
        with open(index_path, "r") as f:
            content = f.read()
        
        # Add new entry
        entry_line = f"- [{date_str} Memory Network](memory_visualizations/{filename})\n"
        
        if "## Visualizations" in content:
            parts = content.split("## Visualizations")
            new_content = parts[0] + "## Visualizations\n\n" + entry_line + parts[1].split("\n", 1)[1]
        else:
            new_content = content + "\n## Visualizations\n\n" + entry_line
        
        # Write updated index
        with open(index_path, "w") as f:
            f.write(new_content) 