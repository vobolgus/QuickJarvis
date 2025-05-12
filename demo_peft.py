"""
Demo script to showcase the Predictive Efficiently Indexed Feature Tracking (PEFT)
memory system for the Voice Assistant.

This script allows for:
1. Simulating conversations
2. Testing memory persistance
3. Viewing memory statistics
4. Testing context retrieval and response prediction
"""
import os
import argparse
import json
import time
from datetime import datetime

from analysis import GemmaAnalyzer
from utils import logger, setup_logging

# Set up more verbose logging for the demo
logger = setup_logging(console_level=20)  # INFO level


def simulate_conversation(analyzer, conversation_data):
    """Simulate a conversation using provided data."""
    print("\n===== Simulating Conversation =====")
    responses = []

    for i, message in enumerate(conversation_data, 1):
        print(f"\nMessage {i}/{len(conversation_data)}:")
        print(f"USER: {message}")

        response = analyzer.analyze_text(message)
        responses.append(response)

        print(f"ASSISTANT: {response}")
        time.sleep(0.5)  # Brief pause for readability

    return responses


def test_memory_persistence(memory_path):
    """Test if memory is being saved correctly."""
    print("\n===== Testing Memory Persistence =====")

    if os.path.exists(memory_path):
        stats = os.stat(memory_path)
        size_kb = stats.st_size / 1024
        modified_time = datetime.fromtimestamp(stats.st_mtime).strftime("%Y-%m-%d %H:%M:%S")

        print(f"Memory database exists at: {memory_path}")
        print(f"Size: {size_kb:.2f} KB")
        print(f"Last modified: {modified_time}")
        return True
    else:
        print(f"Memory database not found at: {memory_path}")
        return False


def show_memory_stats(analyzer):
    """Show statistics about the memory database."""
    print("\n===== Memory Statistics =====")

    stats = analyzer.get_memory_stats()

    print(f"Total conversations: {stats.get('conversation_count', 0)}")
    print(f"Total messages: {stats.get('message_count', 0)}")
    print(f"Total features indexed: {stats.get('feature_count', 0)}")

    # Show sentiment distribution if available
    sentiment_dist = stats.get('sentiment_distribution', {})
    if sentiment_dist:
        print("\nSentiment Distribution:")
        for sentiment, count in sentiment_dist.items():
            print(f"  - {sentiment}: {count}")

    # Show top intents if available
    top_intents = stats.get('top_intents', {})
    if top_intents:
        print("\nTop Intents:")
        for intent, count in top_intents.items():
            print(f"  - {intent}: {count}")

    return stats


def test_context_retrieval(analyzer, query):
    """Test retrieving relevant context from memory."""
    print("\n===== Testing Context Retrieval =====")
    print(f"Query: {query}")

    # Manually retrieve context using the memory object
    relevant_context, response_suggestions = analyzer.memory.get_conversation_context(query)

    print(f"\nFound {len(relevant_context)} relevant messages from past conversations:")
    for i, msg in enumerate(relevant_context, 1):
        print(f"{i}. [{msg['role']}]: {msg['content'][:100]}...")

    print(f"\nGenerated {len(response_suggestions)} response suggestions:")
    for i, suggestion in enumerate(response_suggestions, 1):
        print(f"{i}. {suggestion[:100]}...")

    # Now actually analyze the text with the full PEFT system
    print("\nFull analysis with PEFT context enhancement:")
    response = analyzer.analyze_text(query)
    print(f"RESPONSE: {response}")

    return {
        "relevant_context": relevant_context,
        "response_suggestions": response_suggestions,
        "final_response": response
    }


def main():
    parser = argparse.ArgumentParser(description="Demo for PEFT Memory System")
    parser.add_argument("--memory-path", type=str, default="memory/memory.sqlite",
                        help="Path to memory database")
    parser.add_argument("--simulate", action="store_true",
                        help="Simulate sample conversations")
    parser.add_argument("--stats", action="store_true",
                        help="Show memory database statistics")
    parser.add_argument("--query", type=str,
                        help="Test context retrieval with a specific query")
    parser.add_argument("--new-conversation", action="store_true",
                        help="Start a new conversation")
    parser.add_argument("--prune", type=int, metavar="DAYS",
                        help="Prune conversations older than specified days")

    args = parser.parse_args()

    # Create memory directory if needed
    os.makedirs(os.path.dirname(os.path.abspath(args.memory_path)), exist_ok=True)

    # Initialize analyzer with PEFT memory
    analyzer = GemmaAnalyzer(memory_path=args.memory_path)

    # Check LMStudio connection
    lmstudio_available = analyzer.check_connection()
    if not lmstudio_available:
        print("⚠️ WARNING: Could not connect to LMStudio API.")
        print("Gemma analysis will be unavailable. Ensure LMStudio is running with a model loaded.")
        # Continue anyway for testing purposes

    # Check if we should start a new conversation
    if args.new_conversation:
        conv_id = analyzer.start_new_conversation()
        print(f"Started a new conversation with ID: {conv_id}")

    # Check memory persistence
    test_memory_persistence(args.memory_path)

    # Show statistics if requested
    if args.stats:
        show_memory_stats(analyzer)

    # Simulate sample conversations if requested
    if args.simulate:
        sample_conversations = [
            # Weather conversation
            [
                "What's the weather today?",
                "Will it rain this weekend?",
                "Thanks for the information"
            ],
            # Technical support conversation
            [
                "My microphone isn't working",
                "I've checked the cables, everything seems connected",
                "How do I test if it's a hardware or software issue?"
            ],
            # Reminder conversation
            [
                "Remind me to call John tomorrow",
                "Also add a reminder for the team meeting at 3 PM",
                "What reminders do I have set?"
            ]
        ]

        for i, conversation in enumerate(sample_conversations, 1):
            print(f"\n\n======== SAMPLE CONVERSATION {i} ========")
            # Start a new conversation for each sample
            analyzer.start_new_conversation()
            simulate_conversation(analyzer, conversation)

    # Test context retrieval if query provided
    if args.query:
        test_context_retrieval(analyzer, args.query)

    # Prune old conversations if requested
    if args.prune is not None:
        pruned_count = analyzer.prune_old_conversations(args.prune)
        print(f"Pruned {pruned_count} conversations older than {args.prune} days")
        if args.stats:
            # Show updated stats after pruning
            print("\nUpdated statistics after pruning:")
            show_memory_stats(analyzer)

    print("\nDemo completed.")


if __name__ == "__main__":
    main()