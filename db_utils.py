import os
from pymongo import MongoClient
from datetime import datetime

# Global client cache
_client = None

def get_db():
    global _client
    if _client is None:
        mongo_uri = os.getenv("MONGO_URI", "mongodb://localhost:27017/")
        _client = MongoClient(mongo_uri)
    
    # Use default database from URI if provided, otherwise "rag_chatbot_db"
    return _client.get_default_database(default="rag_chatbot_db")

def save_chat_thread(thread_id: str, chat_history: list):
    """
    Saves or updates a chat thread in the 'threads' collection.
    """
    if not chat_history:
        return

    db = get_db()
    
    # Extract a simple snippet from the user's first question for the sidebar preview
    preview = "New Chat"
    for msg in chat_history:
        if msg.get("role") == "user":
            preview = msg.get("content", "")[:30] + ("..." if len(msg.get("content", "")) > 30 else "")
            break

    db.threads.update_one(
        {"thread_id": thread_id},
        {"$set": {
            "thread_id": thread_id,
            "preview": preview,
            "messages": chat_history,
            "updated_at": datetime.now()
        }},
        upsert=True
    )

def get_all_threads():
    """
    Returns a list of all threads, sorted by most recent first.
    Returns: list of dicts [{"thread_id": ..., "preview": ..., "updated_at": ...}]
    """
    db = get_db()
    cursor = db.threads.find(
        {}, 
        {"thread_id": 1, "preview": 1, "updated_at": 1, "_id": 0}
    ).sort("updated_at", -1)
    
    return list(cursor)

def get_thread_history(thread_id: str):
    """
    Returns the message history list for a specific thread_id.
    """
    db = get_db()
    doc = db.threads.find_one({"thread_id": thread_id}, {"messages": 1, "_id": 0})
    if doc and "messages" in doc:
        return doc["messages"]
    return []
