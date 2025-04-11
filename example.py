"""
Narrative Threading Module

A system for maintaining coherent narrative context across LLM interactions by:
- Storing and retrieving memories
- Connecting conversation fragments
- Maintaining character/agent consistency
- Providing long-term context for more meaningful interactions
"""

import json
import datetime
from typing import List, Dict, Optional, Union
import uuid
import hashlib
import os

# Choose your inference backend (OpenAI or Ollama)
INFERENCE_BACKEND = "openai"  # or "ollama"

if INFERENCE_BACKEND == "openai":
    from openai import OpenAI
    client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))
elif INFERENCE_BACKEND == "ollama":
    import ollama

class Memory:
    """Class representing a single memory unit"""
    
    def __init__(self, content: str, importance: float = 0.5, 
                 timestamp: Optional[datetime.datetime] = None,
                 metadata: Optional[Dict] = None):
        """
        Initialize a memory
        
        Args:
            content: The content of the memory
            importance: Importance score (0.0-1.0)
            timestamp: When the memory was created
            metadata: Additional metadata about the memory
        """
        self.id = str(uuid.uuid4())
        self.content = content
        self.importance = max(0.0, min(1.0, importance))
        self.timestamp = timestamp or datetime.datetime.now()
        self.metadata = metadata or {}
        self.embedding = None  # Will store vector embedding if computed
    
    def to_dict(self) -> Dict:
        """Convert memory to dictionary"""
        return {
            'id': self.id,
            'content': self.content,
            'importance': self.importance,
            'timestamp': self.timestamp.isoformat(),
            'metadata': self.metadata,
            'embedding': self.embedding
        }
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'Memory':
        """Create memory from dictionary"""
        memory = cls(
            content=data['content'],
            importance=data['importance'],
            timestamp=datetime.datetime.fromisoformat(data['timestamp']),
            metadata=data['metadata']
        )
        memory.id = data['id']
        memory.embedding = data.get('embedding')
        return memory

class NarrativeThread:
    """Class representing a coherent narrative thread"""
    
    def __init__(self, name: str, description: str, 
                 memories: Optional[List[Memory]] = None):
        """
        Initialize a narrative thread
        
        Args:
            name: Name/identifier for the thread
            description: Description of what this thread represents
            memories: Initial memories in this thread
        """
        self.id = str(uuid.uuid4())
        self.name = name
        self.description = description
        self.memories = memories or []
        self.created_at = datetime.datetime.now()
        self.last_accessed = self.created_at
    
    def add_memory(self, memory: Memory) -> None:
        """Add a memory to this thread"""
        self.memories.append(memory)
        self.last_accessed = datetime.datetime.now()
    
    def get_context(self, max_tokens: int = 2000, recency_weight: float = 0.7,
                   importance_weight: float = 0.3) -> str:
        """
        Get contextual summary of this thread
        
        Args:
            max_tokens: Maximum length of context to return
            recency_weight: Weight for recency in memory selection
            importance_weight: Weight for importance in memory selection
            
        Returns:
            Formatted context string
        """
        if not self.memories:
            return ""
        
        # Score and sort memories
        now = datetime.datetime.now()
        scored_memories = []
        
        for mem in self.memories:
            # Normalized recency (0-1 where 1 is most recent)
            recency = 1 - (now - mem.timestamp).total_seconds() / (now - self.created_at).total_seconds()
            recency = max(0, min(1, recency))
            
            # Combined score
            score = (recency_weight * recency) + (importance_weight * mem.importance)
            scored_memories.append((score, mem))
        
        # Sort by score descending
        scored_memories.sort(reverse=True, key=lambda x: x[0])
        
        # Build context string within token limit
        context = f"Narrative Thread: {self.name}\nDescription: {self.description}\n\nRelevant Memories:\n"
        remaining_tokens = max_tokens - len(context.split())  # Rough estimate
        
        for score, mem in scored_memories:
            mem_str = f"- {mem.content} (importance: {mem.importance:.2f}, {mem.timestamp.strftime('%Y-%m-%d')})\n"
            mem_tokens = len(mem_str.split())
            
            if remaining_tokens - mem_tokens > 0:
                context += mem_str
                remaining_tokens -= mem_tokens
            else:
                break
        
        return context
    
    def to_dict(self) -> Dict:
        """Convert thread to dictionary"""
        return {
            'id': self.id,
            'name': self.name,
            'description': self.description,
            'memories': [m.to_dict() for m in self.memories],
            'created_at': self.created_at.isoformat(),
            'last_accessed': self.last_accessed.isoformat()
        }
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'NarrativeThread':
        """Create thread from dictionary"""
        thread = cls(
            name=data['name'],
            description=data['description'],
            memories=[Memory.from_dict(m) for m in data['memories']]
        )
        thread.id = data['id']
        thread.created_at = datetime.datetime.fromisoformat(data['created_at'])
        thread.last_accessed = datetime.datetime.fromisoformat(data['last_accessed'])
        return thread

class NarrativeThreading:
    """Main class for narrative threading system"""
    
    def __init__(self, agent_name: str, agent_description: str, 
                 persistence_file: Optional[str] = None):
        """
        Initialize narrative threading system
        
        Args:
            agent_name: Name of the agent/character
            agent_description: Description of the agent's role/personality
            persistence_file: File to load/save state (None for no persistence)
        """
        self.agent_name = agent_name
        self.agent_description = agent_description
        self.persistence_file = persistence_file
        self.threads: Dict[str, NarrativeThread] = {}
        self.active_threads: List[str] = []
        
        # Core memories that always influence the agent
        self.core_memories = [
            Memory(f"My name is {agent_name}", importance=1.0),
            Memory(agent_description, importance=1.0)
        ]
        
        # Load from persistence file if exists
        if persistence_file and os.path.exists(persistence_file):
            self.load_state()
    
    def create_thread(self, name: str, description: str) -> NarrativeThread:
        """Create a new narrative thread"""
        if name in self.threads:
            raise ValueError(f"Thread with name '{name}' already exists")
        
        thread = NarrativeThread(name, description)
        self.threads[thread.id] = thread
        self.save_state()
        return thread
    
    def add_to_thread(self, thread_id: str, content: str, 
                     importance: float = 0.5, metadata: Optional[Dict] = None) -> None:
        """Add a memory to a thread"""
        if thread_id not in self.threads:
            raise ValueError(f"Thread with ID '{thread_id}' not found")
        
        memory = Memory(content, importance, metadata=metadata)
        self.threads[thread_id].add_memory(memory)
        self.save_state()
    
    def get_thread_context(self, thread_id: str, max_tokens: int = 1000) -> str:
        """Get context for a specific thread"""
        if thread_id not in self.threads:
            raise ValueError(f"Thread with ID '{thread_id}' not found")
        
        return self.threads[thread_id].get_context(max_tokens)
    
    def get_full_context(self, max_tokens: int = 3000) -> str:
        """
        Get full contextual summary including:
        - Agent identity
        - Core memories
        - Active threads
        """
        # Start with agent identity
        context = f"Agent: {self.agent_name}\nDescription: {self.agent_description}\n\n"
        
        # Add core memories
        context += "Core Memories:\n"
        for mem in self.core_memories:
            context += f"- {mem.content}\n"
        
        # Add active threads
        context += "\nActive Narrative Threads:\n"
        remaining_tokens = max_tokens - len(context.split())
        
        for thread_id in self.active_threads:
            if thread_id in self.threads:
                thread_context = self.threads[thread_id].get_context(
                    max_tokens=remaining_tokens//len(self.active_threads)
                )
                context += thread_context + "\n"
                remaining_tokens -= len(thread_context.split())
        
        return context
    
    def generate_response(self, prompt: str, max_tokens: int = 1500) -> str:
        """
        Generate a response using the narrative context
        
        Args:
            prompt: User input/prompt
            max_tokens: Maximum response length
            
        Returns:
            Generated response
        """
        # Get relevant context
        context = self.get_full_context(max_tokens=3000)
        
        # Format the prompt with context
        full_prompt = f"""{context}
        
        Current Conversation:
        User: {prompt}
        
        Based on the above context and memories, generate an appropriate response as {self.agent_name}:
        """
        
        # Call the appropriate inference backend
        if INFERENCE_BACKEND == "openai":
            response = client.chat.completions.create(
                model="gpt-4",
                messages=[
                    {"role": "system", "content": "You are a helpful AI assistant."},
                    {"role": "user", "content": full_prompt}
                ],
                max_tokens=max_tokens,
                temperature=0.7
            )
            return response.choices[0].message.content
        elif INFERENCE_BACKEND == "ollama":
            response = ollama.generate(
                model="llama2",
                prompt=full_prompt,
                options={
                    "temperature": 0.7,
                    "num_predict": max_tokens
                }
            )
            return response['response']
    
    def find_relevant_threads(self, query: str, top_n: int = 3) -> List[NarrativeThread]:
        """
        Find threads relevant to the query (simplified version using keyword matching)
        
        Note: In production, you'd want to use proper vector similarity search
        """
        # This is a simplified approach - consider using proper embeddings/similarity search
        query_words = set(query.lower().split())
        scored_threads = []
        
        for thread in self.threads.values():
            # Simple keyword matching
            thread_text = f"{thread.name} {thread.description} {' '.join(m.content for m in thread.memories)}"
            thread_words = set(thread_text.lower().split())
            common_words = query_words.intersection(thread_words)
            score = len(common_words)
            
            if score > 0:
                scored_threads.append((score, thread))
        
        # Sort by score and return top N
        scored_threads.sort(reverse=True, key=lambda x: x[0])
        return [t for (s, t) in scored_threads[:top_n]]
    
    def save_state(self) -> None:
        """Save current state to persistence file"""
        if not self.persistence_file:
            return
            
        state = {
            'agent_name': self.agent_name,
            'agent_description': self.agent_description,
            'threads': [t.to_dict() for t in self.threads.values()],
            'active_threads': self.active_threads,
            'core_memories': [m.to_dict() for m in self.core_memories]
        }
        
        with open(self.persistence_file, 'w') as f:
            json.dump(state, f, indent=2)
    
    def load_state(self) -> None:
        """Load state from persistence file"""
        if not self.persistence_file or not os.path.exists(self.persistence_file):
            return
            
        with open(self.persistence_file, 'r') as f:
            state = json.load(f)
        
        self.agent_name = state['agent_name']
        self.agent_description = state['agent_description']
        self.threads = {t['id']: NarrativeThread.from_dict(t) for t in state['threads']}
        self.active_threads = state['active_threads']
        self.core_memories = [Memory.from_dict(m) for m in state['core_memories']]

# Example usage
if __name__ == "__main__":
    # Initialize the narrative threading system
    narrative = NarrativeThreading(
        agent_name="Athena",
        agent_description="A wise and knowledgeable AI assistant with a focus on history and philosophy",
        persistence_file="athena_memories.json"
    )
    
    # Create some narrative threads
    history_thread = narrative.create_thread(
        name="Ancient History Discussions",
        description="Conversations about ancient civilizations and their impact on modern society"
    )
    
    # Add memories to threads
    narrative.add_to_thread(
        history_thread.id,
        "The user expressed particular interest in comparing Greek and Roman political systems",
        importance=0.8
    )
    
    # Set active threads
    narrative.active_threads = [history_thread.id]
    
    # Generate a response using narrative context
    response = narrative.generate_response(
        "What can we learn from how ancient Athens handled civic participation?"
    )
    print(response)
