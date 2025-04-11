csharp
Copy
using UnityEngine;
using LLMUnity;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Threading.Tasks;

[System.Serializable]
public class Memory
{
    public string id;
    public string content;
    public float importance; // 0-1 scale
    public DateTime timestamp;
    public Dictionary<string, string> metadata;

    public Memory(string content, float importance = 0.5f, Dictionary<string, string> metadata = null)
    {
        this.id = Guid.NewGuid().ToString();
        this.content = content;
        this.importance = Mathf.Clamp(importance, 0f, 1f);
        this.timestamp = DateTime.Now;
        this.metadata = metadata ?? new Dictionary<string, string>();
    }
}

[System.Serializable]
public class NarrativeThread
{
    public string id;
    public string name;
    public string description;
    public List<Memory> memories;
    public DateTime lastAccessed;

    public NarrativeThread(string name, string description)
    {
        this.id = Guid.NewGuid().ToString();
        this.name = name;
        this.description = description;
        this.memories = new List<Memory>();
        this.lastAccessed = DateTime.Now;
    }

    public void AddMemory(Memory memory)
    {
        memories.Add(memory);
        lastAccessed = DateTime.Now;
    }

    public string GetContext(int maxMemories = 5)
    {
        // Sort memories by relevance (importance + recency)
        var sortedMemories = memories.OrderByDescending(m => 
            (m.importance * 0.7f) + // Weight importance more
            (0.3f * (float)(DateTime.Now - m.timestamp).TotalHours / 24f) // Normalize recency
            .Take(maxMemories);

        string context = $"Thread: {name}\n{description}\n\nRecent Memories:\n";
        foreach (var memory in sortedMemories)
        {
            context += $"- {memory.content} (importance: {memory.importance:F1})\n";
        }
        return context;
    }
}

public class NarrativeThreadingSystem : MonoBehaviour
{
    [Header("LLM Configuration")]
    public LLMCharacter llmCharacter;
    public string characterName = "AI Companion";
    public string characterDescription = "A helpful AI assistant with long-term memory";

    [Header("Narrative Settings")]
    [Tooltip("How many recent threads to keep in context")]
    public int maxActiveThreads = 3;
    [Tooltip("How many memories per thread to include")]
    public int memoriesPerThread = 3;

    private List<NarrativeThread> threads = new List<NarrativeThread>();
    private List<string> activeThreadIds = new List<string>();

    // Core memories that always influence the character
    private List<Memory> coreMemories = new List<Memory>();

    private void Awake()
    {
        InitializeCoreMemories();
    }

    private void InitializeCoreMemories()
    {
        coreMemories.Add(new Memory($"My name is {characterName}", 1f));
        coreMemories.Add(new Memory(characterDescription, 1f));
        coreMemories.Add(new Memory("I'm designed to maintain coherent narratives across conversations", 0.8f));
    }

    public NarrativeThread CreateThread(string name, string description)
    {
        var thread = new NarrativeThread(name, description);
        threads.Add(thread);
        UpdateActiveThreads(thread.id);
        return thread;
    }

    public void AddMemoryToThread(string threadId, string content, float importance = 0.5f, Dictionary<string, string> metadata = null)
    {
        var thread = threads.FirstOrDefault(t => t.id == threadId);
        if (thread != null)
        {
            thread.AddMemory(new Memory(content, importance, metadata));
            UpdateActiveThreads(threadId);
        }
    }

    private void UpdateActiveThreads(string accessedThreadId)
    {
        // Move accessed thread to front
        activeThreadIds.Remove(accessedThreadId);
        activeThreadIds.Insert(0, accessedThreadId);

        // Trim if over limit
        if (activeThreadIds.Count > maxActiveThreads)
        {
            activeThreadIds = activeThreadIds.Take(maxActiveThreads).ToList();
        }
    }

    public string GetFullContext()
    {
        string context = $"Character: {characterName}\nDescription: {characterDescription}\n\n";

        // Add core memories
        context += "Core Knowledge:\n";
        foreach (var memory in coreMemories)
        {
            context += $"- {memory.content}\n";
        }

        // Add active threads
        context += "\nCurrent Narrative Context:\n";
        foreach (var threadId in activeThreadIds)
        {
            var thread = threads.FirstOrDefault(t => t.id == threadId);
            if (thread != null)
            {
                context += thread.GetContext(memoriesPerThread) + "\n";
            }
        }

        return context;
    }

    public async Task<string> GenerateResponse(string userInput)
    {
        // Get full context
        string context = GetFullContext();

        // Format the prompt
        string prompt = $"{context}\n\nCurrent conversation:\nPlayer: {userInput}\n{characterName}:";

        // Generate response using LLMUnity
        string response = await llmCharacter.Complete(prompt);

        // Process the response to extract any new memories
        ProcessResponseForMemories(userInput, response);

        return response;
    }

    private void ProcessResponseForMemories(string userInput, string aiResponse)
    {
        // Simple heuristic - if the response is substantive, create a memory
        if (aiResponse.Split(' ').Length > 5) // More than 5 words
        {
            // Find the most relevant active thread or create a new one
            NarrativeThread relevantThread = activeThreadIds.Count > 0 ? 
                threads.First(t => t.id == activeThreadIds[0]) : 
                CreateThread("General Conversation", "Miscellaneous discussions");

            // Create memory with moderate importance
            AddMemoryToThread(relevantThread.id, $"Conversation about: {userInput}", 0.6f);
        }
    }

    // Helper method to find relevant threads based on keywords
    public List<NarrativeThread> FindRelevantThreads(string query, int maxResults = 3)
    {
        var queryKeywords = query.ToLower().Split(new[] { ' ', ',', '.', '!', '?' }, StringSplitOptions.RemoveEmptyEntries);

        return threads
            .Select(thread => {
                var threadText = $"{thread.name} {thread.description} {string.Join(" ", thread.memories.Select(m => m.content))}";
                var threadKeywords = threadText.ToLower().Split(new[] { ' ', ',', '.', '!', '?' }, StringSplitOptions.RemoveEmptyEntries);
                var score = queryKeywords.Sum(qk => threadKeywords.Count(tk => tk.Contains(qk) || qk.Contains(tk)));
                return (thread, score);
            })
            .Where(x => x.score > 0)
            .OrderByDescending(x => x.score)
            .Take(maxResults)
            .Select(x => x.thread)
            .ToList();
    }

    // Example usage in a MonoBehaviour
    public async void OnUserInput(string userMessage)
    {
        // First, find if this relates to any existing threads
        var relevantThreads = FindRelevantThreads(userMessage);
        
        if (relevantThreads.Count > 0)
        {
            // If we found relevant threads, use the most relevant one
            UpdateActiveThreads(relevantThreads[0].id);
        }
        else if (userMessage.Split(' ').Length > 8) // Longer messages might start new threads
        {
            // Create a new thread for substantial new topics
            CreateThread(ExtractTopic(userMessage), "User-initiated discussion");
        }

        // Generate and display response
        string response = await GenerateResponse(userMessage);
        Debug.Log(response);
    }

    private string ExtractTopic(string message)
    {
        // Very simple topic extraction - in practice you'd want something more sophisticated
        var words = message.Split(' ');
        if (words.Length > 3)
        {
            return $"{words[0]} {words[1]} {words[2]}...";
        }
        return "New Discussion";
    }
}
