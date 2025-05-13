# ----------------------------------------------------------------------------
#  File:        hebbian_store.py
#  Project:     Celaya Solutions Ollama Simulator
#  Created by:  Celaya Solutions, 2025
#  Author:      Christopher Celaya <chris@celayasolutions.com>
#  Description: Implementation of Hebbian learning-based memory store
#  Version:     1.0.0
#  License:     MIT (SPDX-Identifier: MIT)
#  Last Update: June 22, 2025
# ----------------------------------------------------------------------------

import os
import json
import numpy as np
import faiss
from loguru import logger
from pathlib import Path
from datetime import datetime
import sqlite3

class HebbianMemoryStore:
    """
    Memory store that implements Hebbian learning principles.
    Uses a combination of vector embeddings (FAISS) and relational storage (SQLite).
    """
    
    def __init__(self, vector_dim=384, db_path=None, index_path=None):
        """
        Initialize the Hebbian memory store.
        
        Args:
            vector_dim: Dimension of the vector embeddings
            db_path: Path to the SQLite database
            index_path: Path to the FAISS index
        """
        self.vector_dim = vector_dim
        self.db_path = db_path or Path(__file__).parent.parent / "data" / "baby_memory.db"
        self.index_path = index_path or Path(__file__).parent.parent / "data" / "faiss_index" / "memory.index"
        
        # Initialize vector store
        self.index = self._initialize_faiss()
        
        # Initialize relational store
        self.conn = self._initialize_sqlite()
        
        # Track memory statistics
        self.stats = {
            "total_memories": 0,
            "reinforced_memories": 0,
            "forgotten_memories": 0,
            "last_updated": datetime.now().isoformat()
        }
        
        logger.info(f"Hebbian memory store initialized with vector dimension {vector_dim}")
    
    def _initialize_faiss(self):
        """Initialize the FAISS vector index."""
        os.makedirs(os.path.dirname(self.index_path), exist_ok=True)
        
        if os.path.exists(self.index_path):
            try:
                index = faiss.read_index(str(self.index_path))
                logger.info(f"Loaded existing FAISS index with {index.ntotal} vectors")
                return index
            except Exception as e:
                logger.error(f"Error loading FAISS index: {e}")
        
        # Create a new index
        index = faiss.IndexFlatL2(self.vector_dim)
        logger.info("Created new FAISS index")
        return index
    
    def _initialize_sqlite(self):
        """Initialize the SQLite database."""
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
        
        conn = sqlite3.connect(str(self.db_path))
        cursor = conn.cursor()
        
        # Create tables if they don't exist
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS memories (
            id INTEGER PRIMARY KEY,
            content TEXT NOT NULL,
            created_at TEXT NOT NULL,
            last_accessed TEXT NOT NULL,
            access_count INTEGER DEFAULT 1,
            emotional_tags TEXT,
            confidence REAL DEFAULT 0.5,
            source TEXT
        )
        ''')
        
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS associations (
            id INTEGER PRIMARY KEY,
            memory_id INTEGER,
            associated_memory_id INTEGER,
            strength REAL DEFAULT 0.5,
            created_at TEXT NOT NULL,
            last_reinforced TEXT,
            FOREIGN KEY (memory_id) REFERENCES memories(id),
            FOREIGN KEY (associated_memory_id) REFERENCES memories(id)
        )
        ''')
        
        conn.commit()
        logger.info("SQLite database initialized")
        return conn
    
    def store_memory(self, content, vector, emotional_tags=None, confidence=0.5, source=None):
        """
        Store a new memory.
        
        Args:
            content: Text content of the memory
            vector: Vector embedding of the memory
            emotional_tags: Emotional tags associated with the memory
            confidence: Confidence score (0.0-1.0)
            source: Source of the memory (e.g., 'lesson', 'feedback', 'dream')
            
        Returns:
            int: ID of the stored memory
        """
        # Convert to numpy array if needed
        if not isinstance(vector, np.ndarray):
            vector = np.array([vector], dtype=np.float32)
        elif len(vector.shape) == 1:
            vector = vector.reshape(1, -1).astype(np.float32)
        
        # Store in FAISS
        self.index.add(vector)
        vector_id = self.index.ntotal - 1
        
        # Store in SQLite
        cursor = self.conn.cursor()
        now = datetime.now().isoformat()
        
        emotional_tags_json = json.dumps(emotional_tags) if emotional_tags else None
        
        cursor.execute('''
        INSERT INTO memories (content, created_at, last_accessed, emotional_tags, confidence, source)
        VALUES (?, ?, ?, ?, ?, ?)
        ''', (content, now, now, emotional_tags_json, confidence, source))
        
        memory_id = cursor.lastrowid
        self.conn.commit()
        
        # Update stats
        self.stats["total_memories"] += 1
        self.stats["last_updated"] = now
        
        logger.info(f"Stored new memory with ID {memory_id}")
        return memory_id
    
    def create_association(self, memory_id, associated_memory_id, strength=0.5):
        """
        Create an association between two memories.
        
        Args:
            memory_id: ID of the first memory
            associated_memory_id: ID of the second memory
            strength: Association strength (0.0-1.0)
            
        Returns:
            int: ID of the created association
        """
        cursor = self.conn.cursor()
        now = datetime.now().isoformat()
        
        # Check if association already exists
        cursor.execute('''
        SELECT id, strength FROM associations 
        WHERE memory_id = ? AND associated_memory_id = ?
        ''', (memory_id, associated_memory_id))
        
        result = cursor.fetchone()
        
        if result:
            # Update existing association
            assoc_id, old_strength = result
            new_strength = min(1.0, old_strength + strength * 0.5)  # Hebbian reinforcement
            
            cursor.execute('''
            UPDATE associations 
            SET strength = ?, last_reinforced = ?
            WHERE id = ?
            ''', (new_strength, now, assoc_id))
            
            self.conn.commit()
            logger.info(f"Reinforced association {assoc_id} from {old_strength} to {new_strength}")
            
            # Update stats
            self.stats["reinforced_memories"] += 1
            self.stats["last_updated"] = now
            
            return assoc_id
        else:
            # Create new association
            cursor.execute('''
            INSERT INTO associations (memory_id, associated_memory_id, strength, created_at, last_reinforced)
            VALUES (?, ?, ?, ?, ?)
            ''', (memory_id, associated_memory_id, strength, now, now))
            
            assoc_id = cursor.lastrowid
            self.conn.commit()
            
            logger.info(f"Created new association {assoc_id} with strength {strength}")
            return assoc_id
    
    def retrieve_similar_memories(self, vector, k=5):
        """
        Retrieve memories similar to the given vector.
        
        Args:
            vector: Query vector
            k: Number of results to return
            
        Returns:
            list: List of (memory_id, distance) tuples
        """
        # Convert to numpy array if needed
        if not isinstance(vector, np.ndarray):
            vector = np.array([vector], dtype=np.float32)
        elif len(vector.shape) == 1:
            vector = vector.reshape(1, -1).astype(np.float32)
        
        # Search in FAISS
        if self.index.ntotal == 0:
            logger.warning("No vectors in the index yet")
            return []
        
        k = min(k, self.index.ntotal)  # Can't retrieve more than we have
        distances, indices = self.index.search(vector, k)
        
        # Get memory IDs from SQLite
        cursor = self.conn.cursor()
        results = []
        
        for i, idx in enumerate(indices[0]):
            # Get memory ID for this vector index
            cursor.execute('SELECT id FROM memories LIMIT 1 OFFSET ?', (idx,))
            result = cursor.fetchone()
            
            if result:
                memory_id = result[0]
                distance = distances[0][i]
                results.append((memory_id, distance))
                
                # Update access count and timestamp
                now = datetime.now().isoformat()
                cursor.execute('''
                UPDATE memories 
                SET access_count = access_count + 1, last_accessed = ?
                WHERE id = ?
                ''', (now, memory_id))
        
        self.conn.commit()
        return results
    
    def get_memory_by_id(self, memory_id):
        """
        Get a memory by its ID.
        
        Args:
            memory_id: ID of the memory to retrieve
            
        Returns:
            dict: Memory data or None if not found
        """
        cursor = self.conn.cursor()
        
        cursor.execute('''
        SELECT id, content, created_at, last_accessed, access_count, emotional_tags, confidence, source
        FROM memories WHERE id = ?
        ''', (memory_id,))
        
        result = cursor.fetchone()
        
        if not result:
            return None
        
        # Update access count and timestamp
        now = datetime.now().isoformat()
        cursor.execute('''
        UPDATE memories 
        SET access_count = access_count + 1, last_accessed = ?
        WHERE id = ?
        ''', (now, memory_id))
        
        self.conn.commit()
        
        # Parse emotional tags
        emotional_tags = json.loads(result[5]) if result[5] else {}
        
        return {
            "id": result[0],
            "content": result[1],
            "created_at": result[2],
            "last_accessed": result[3],
            "access_count": result[4],
            "emotional_tags": emotional_tags,
            "confidence": result[6],
            "source": result[7]
        }
    
    def get_associated_memories(self, memory_id, min_strength=0.0):
        """
        Get memories associated with the given memory.
        
        Args:
            memory_id: ID of the memory
            min_strength: Minimum association strength to include
            
        Returns:
            list: List of (associated_memory_id, strength) tuples
        """
        cursor = self.conn.cursor()
        
        cursor.execute('''
        SELECT associated_memory_id, strength
        FROM associations
        WHERE memory_id = ? AND strength >= ?
        ORDER BY strength DESC
        ''', (memory_id, min_strength))
        
        return cursor.fetchall()
    
    def update_memory_confidence(self, memory_id, confidence_delta):
        """
        Update the confidence of a memory.
        
        Args:
            memory_id: ID of the memory
            confidence_delta: Change in confidence (-1.0 to 1.0)
            
        Returns:
            float: New confidence value
        """
        cursor = self.conn.cursor()
        
        # Get current confidence
        cursor.execute('SELECT confidence FROM memories WHERE id = ?', (memory_id,))
        result = cursor.fetchone()
        
        if not result:
            logger.warning(f"Memory {memory_id} not found")
            return None
        
        current_confidence = result[0]
        new_confidence = max(0.0, min(1.0, current_confidence + confidence_delta))
        
        # Update confidence
        cursor.execute('''
        UPDATE memories SET confidence = ? WHERE id = ?
        ''', (new_confidence, memory_id))
        
        self.conn.commit()
        
        logger.info(f"Updated memory {memory_id} confidence from {current_confidence} to {new_confidence}")
        return new_confidence
    
    def decay_memories(self, decay_factor=0.99):
        """
        Apply decay to all memories based on last access time.
        
        Args:
            decay_factor: Factor to decay memory strengths (0.0-1.0)
            
        Returns:
            int: Number of memories affected
        """
        cursor = self.conn.cursor()
        now = datetime.now()
        
        # Get all associations
        cursor.execute('SELECT id, strength, last_reinforced FROM associations')
        associations = cursor.fetchall()
        
        affected = 0
        for assoc_id, strength, last_reinforced in associations:
            if not last_reinforced:
                continue
                
            # Calculate days since last reinforcement
            last_date = datetime.fromisoformat(last_reinforced)
            days_since = (now - last_date).days
            
            if days_since > 0:
                # Apply decay based on days since last reinforcement
                decay = decay_factor ** days_since
                new_strength = strength * decay
                
                cursor.execute('''
                UPDATE associations SET strength = ? WHERE id = ?
                ''', (new_strength, assoc_id))
                
                affected += 1
        
        self.conn.commit()
        
        # Update stats
        self.stats["forgotten_memories"] += affected
        self.stats["last_updated"] = now.isoformat()
        
        logger.info(f"Applied decay to {affected} associations")
        return affected
    
    def save(self):
        """Save the current state of the memory store."""
        # Save FAISS index
        os.makedirs(os.path.dirname(self.index_path), exist_ok=True)
        faiss.write_index(self.index, str(self.index_path))
        
        # SQLite is saved automatically
        
        logger.info(f"Memory store saved to {self.index_path} and {self.db_path}")
    
    def get_stats(self):
        """
        Get statistics about the memory store.
        
        Returns:
            dict: Memory statistics
        """
        cursor = self.conn.cursor()
        
        # Count memories
        cursor.execute('SELECT COUNT(*) FROM memories')
        total_memories = cursor.fetchone()[0]
        
        # Count associations
        cursor.execute('SELECT COUNT(*) FROM associations')
        total_associations = cursor.fetchone()[0]
        
        # Get average confidence
        cursor.execute('SELECT AVG(confidence) FROM memories')
        avg_confidence = cursor.fetchone()[0] or 0.0
        
        # Get average association strength
        cursor.execute('SELECT AVG(strength) FROM associations')
        avg_strength = cursor.fetchone()[0] or 0.0
        
        return {
            "total_memories": total_memories,
            "total_associations": total_associations,
            "average_confidence": avg_confidence,
            "average_association_strength": avg_strength,
            "reinforced_memories": self.stats["reinforced_memories"],
            "forgotten_memories": self.stats["forgotten_memories"],
            "last_updated": datetime.now().isoformat()
        }
    
    def get_recent_memories(self, limit=5):
        """
        Get the most recently accessed memories.
        
        Args:
            limit: Maximum number of memories to return
            
        Returns:
            list: List of memory dictionaries
        """
        cursor = self.conn.cursor()
        
        cursor.execute('''
        SELECT id, content, created_at, last_accessed, access_count, emotional_tags, confidence, source
        FROM memories
        ORDER BY last_accessed DESC
        LIMIT ?
        ''', (limit,))
        
        results = cursor.fetchall()
        
        memories = []
        for result in results:
            # Parse emotional tags
            emotional_tags = json.loads(result[5]) if result[5] else {}
            
            memories.append({
                "id": result[0],
                "content": result[1],
                "created_at": result[2],
                "last_accessed": result[3],
                "access_count": result[4],
                "emotional_tags": emotional_tags,
                "confidence": result[6],
                "source": result[7]
            })
        
        return memories
        
    def consolidate_memories(self):
        """
        Consolidate memories by strengthening important associations
        and weakening less important ones.
        
        Returns:
            dict: Statistics about the consolidation process
        """
        cursor = self.conn.cursor()
        
        # Get memories with high access counts
        cursor.execute('''
        SELECT id FROM memories
        WHERE access_count > 1
        ORDER BY access_count DESC
        LIMIT 10
        ''')
        
        high_access_ids = [row[0] for row in cursor.fetchall()]
        
        # Strengthen associations between frequently accessed memories
        strengthened = 0
        for i, memory_id in enumerate(high_access_ids):
            for j in range(i+1, len(high_access_ids)):
                associated_id = high_access_ids[j]
                
                # Check if association exists
                cursor.execute('''
                SELECT id, strength FROM associations
                WHERE (memory_id = ? AND associated_memory_id = ?) OR
                      (memory_id = ? AND associated_memory_id = ?)
                ''', (memory_id, associated_id, associated_id, memory_id))
                
                result = cursor.fetchone()
                
                if result:
                    # Strengthen existing association
                    assoc_id, strength = result
                    new_strength = min(1.0, strength + 0.1)
                    cursor.execute('''
                    UPDATE associations SET strength = ?, last_reinforced = ?
                    WHERE id = ?
                    ''', (new_strength, datetime.now().isoformat(), assoc_id))
                    strengthened += 1
                else:
                    # Create new association
                    self.create_association(memory_id, associated_id, strength=0.5)
                    strengthened += 1
        
        # Prune very weak associations
        cursor.execute('''
        DELETE FROM associations WHERE strength < 0.1
        ''')
        pruned = cursor.rowcount
        
        self.conn.commit()
        
        return {
            "strengthened": strengthened,
            "pruned": pruned
        } 