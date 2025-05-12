"""
Text analysis functionality using LMStudio's Gemma model for the Voice Assistant.
Enhanced with PEFT memory system for improved context handling.
"""
import json
import os
from typing import Dict, Any, Optional, List, Tuple

import requests

from utils import logger
from memory import PEFTMemory

# Default LMStudio configuration
DEFAULT_LMSTUDIO_API_URL = "http://localhost:1234/v1/chat/completions"
# Enhanced system prompt for better context handling, memory, and exit cues
DEFAULT_SYSTEM_PROMPT = (
    "You are a helpful, context-aware AI assistant that remembers previous user interactions "
    "and your own responses. Provide concise, accurate, and informative answers. For math statements spell the equations, rather than typesetting them (eq. not \\frac{1}{2}, but one over two). "
    "If you determine from the user's query or the context that the user wishes to end the conversation, "
    "respond with a polite closing statement that includes a phrase like 'Goodbye', 'Session ended', or 'Farewell'. /no_think"
)

class GemmaAnalyzer:
    """
    Handles text analysis using LMStudio's Gemma model via API.
    Enhanced with PEFT memory system for better context handling and response prediction.
    """

    def __init__(self,
                 api_url: str = DEFAULT_LMSTUDIO_API_URL,
                 system_prompt: str = DEFAULT_SYSTEM_PROMPT,
                 memory_path: str = "memory/memory.sqlite"):
        """
        Initialize the Gemma text analyzer with PEFT memory.

        Args:
            api_url: URL for the LMStudio API endpoint
            system_prompt: System prompt to guide the model's behavior
            memory_path: Path to the persistent memory database
        """
        self.api_url = api_url
        self.system_prompt = system_prompt

        # Initialize PEFT memory system
        self.memory = PEFTMemory(storage_path=memory_path)
        self.current_conversation_id = self.memory.start_conversation()

        # Add system prompt to memory
        self.memory.add_message("system", self.system_prompt, self.current_conversation_id)

        logger.debug(f"Initialized Gemma analyzer with API URL: {api_url} and PEFT memory at {memory_path}")

    def check_connection(self) -> bool:
        """
        Check if LMStudio is running and accessible.

        Returns:
            True if LMStudio is available, False otherwise
        """
        try:
            # Simple request to check if the server is running
            base_url_parts = self.api_url.split('/')[:-2] # e.g. http://localhost:1234/v1/chat/completions -> http://localhost:1234
            base_url = "/".join(base_url_parts)
            # Try hitting a common endpoint like /v1/models if available, or just base URL
            # For LMStudio, the base server itself might not respond to GET /
            # Let's try /v1/models or similar. If API URL is specific, just check its base.
            target_url_for_check = base_url + "/v1/models" # A common OpenAI compatible endpoint
            try:
                response = requests.get(target_url_for_check, timeout=3)
            except requests.exceptions.ConnectionError: # If server itself is not running
                 response = requests.get(base_url, timeout=3) # Try base URL as fallback check

            if response.status_code < 400:  # 2xx or 3xx are usually okay for a health check
                logger.debug(f"Successfully connected to LMStudio API (checked {target_url_for_check or base_url}, status: {response.status_code})")
                return True
            logger.warning(f"LMStudio API check returned status code: {response.status_code} from {target_url_for_check or base_url}")
            return False
        except requests.exceptions.RequestException as e:
            logger.warning(f"Failed to connect to LMStudio API: {e}")
            return False

    def analyze_text(self,
                     text: str,
                     temperature: float = 0.7,
                     max_tokens: int = 300) -> str:
        """
        Send text to LMStudio's Gemma model for analysis, enhanced with PEFT memory.

        Args:
            text: The text to analyze
            temperature: Sampling temperature (0-1)
            max_tokens: Maximum number of tokens to generate

        Returns:
            Analysis result as text. This will be an error message string if analysis failed.
        """
        try:
            logger.debug(f"Analyzing text with Gemma (length: {len(text)} chars)")

            # Add user message to memory
            self.memory.add_message("user", text, self.current_conversation_id)

            # Get relevant context and response suggestions from past conversations
            relevant_context, response_suggestions = self.memory.get_conversation_context(text)

            # Get current conversation for the API request
            conversation_history = self.memory.get_conversation_history(self.current_conversation_id)

            # Prepare API payload with enhanced context
            payload = {
                "model": "gemma-3-1b-it", # Ensure this model is loaded in LMStudio
                "messages": conversation_history.copy(),
                "temperature": temperature,
                "max_tokens": max_tokens,
                "stream": False # Ensure not streaming for this simple case
            }

            # Add contextual information from previous conversations if available
            if relevant_context:
                context_summary = self._format_context_summary(relevant_context)
                # Insert context hint just before the last user message to preserve recency
                payload["messages"].insert(len(payload["messages"])-1, {
                    "role": "system",
                    "content": context_summary
                })
                logger.debug(f"Added context from {len(relevant_context)} historical messages")

            # Add potential response suggestions if available
            if response_suggestions:
                suggestion_hint = self._format_response_suggestions(response_suggestions)
                # Add suggestions after the user message
                payload["messages"].append({
                    "role": "system",
                    "content": suggestion_hint
                })
                logger.debug(f"Added {len(response_suggestions)} response suggestions")

            # Send request to API
            headers = {"Content-Type": "application/json"}
            logger.debug(f"Sending request to LMStudio API with {len(payload['messages'])} messages")
            response = requests.post(self.api_url, json=payload, headers=headers, timeout=60) # Increased timeout

            if response.status_code == 200:
                result = response.json()
                logger.debug(f"LMStudio API raw response received")
                if "choices" in result and len(result["choices"]) > 0 and "message" in result["choices"][0] and "content" in result["choices"][0]["message"]:
                    analysis = result["choices"][0]["message"]["content"]
                    logger.debug("Analysis completed successfully")

                    # Store assistant response in memory
                    self.memory.add_message("assistant", analysis, self.current_conversation_id)

                    return analysis
                else:
                    err_msg = "LMStudio API response in unexpected format."
                    logger.error(f"{err_msg} Full response: {result}")
                    self.memory.add_message("assistant", f"Error: {err_msg}", self.current_conversation_id)
                    return f"Error: {err_msg}"
            else:
                err = f"Error connecting to LMStudio: {response.status_code} - {response.text}"
                logger.error(f"Error from LMStudio API: {response.status_code} - {response.text}")
                self.memory.add_message("assistant", err, self.current_conversation_id)
                return err # Return error message to be potentially spoken or handled
        except requests.exceptions.Timeout:
            err = "Error analyzing with Gemma: Request timed out."
            logger.error(err)
            self.memory.add_message("assistant", err, self.current_conversation_id)
            return err
        except Exception as e:
            err = f"Error analyzing with Gemma: {str(e)}"
            logger.error(err, exc_info=True)
            self.memory.add_message("assistant", err, self.current_conversation_id)
            return err # Return error message

    def _format_context_summary(self, relevant_context: List[Dict[str, Any]]) -> str:
        """Format the context summary from previous conversations."""
        context_summary = "Prior relevant conversation context (for reference):\n"

        for i, msg in enumerate(relevant_context, 1):
            role = msg["role"]
            content = msg["content"]
            # Truncate very long messages
            if len(content) > 200:
                content = content[:197] + "..."
            context_summary += f"{i}. {role.capitalize()}: {content}\n"

        context_summary += "\nIncorporate this context when relevant, but prioritize the current conversation."
        return context_summary

    def _format_response_suggestions(self, suggestions: List[str]) -> str:
        """Format the response suggestions."""
        suggestion_text = "Here are response patterns from similar past conversations that may be helpful:\n"

        for i, suggestion in enumerate(suggestions, 1):
            # Truncate very long suggestions
            if len(suggestion) > 150:
                suggestion = suggestion[:147] + "..."
            suggestion_text += f"{i}. {suggestion}\n"

        suggestion_text += "\nUse these as inspiration if relevant, but prioritize creating a fresh response."
        return suggestion_text

    def get_memory_stats(self) -> Dict[str, Any]:
        """
        Get statistics about the PEFT memory system.

        Returns:
            Dictionary of statistics
        """
        return self.memory.get_conversation_stats()

    def start_new_conversation(self) -> str:
        """
        Start a new conversation in the memory system.

        Returns:
            ID of the new conversation
        """
        # Start new conversation in memory
        self.current_conversation_id = self.memory.start_conversation()

        # Add system prompt to new conversation
        self.memory.add_message("system", self.system_prompt, self.current_conversation_id)

        logger.debug(f"Started new conversation with ID: {self.current_conversation_id}")
        return self.current_conversation_id

    def prune_old_conversations(self, max_age_days: int = 30) -> int:
        """
        Remove conversations older than the specified age.

        Args:
            max_age_days: Maximum age of conversations to keep in days

        Returns:
            Number of pruned conversations
        """
        # Get conversation count before pruning
        stats_before = self.memory.get_conversation_stats()
        conversation_count_before = stats_before.get("conversation_count", 0)

        # Prune old conversations
        self.memory.prune_old_conversations(max_age_days)

        # Get conversation count after pruning
        stats_after = self.memory.get_conversation_stats()
        conversation_count_after = stats_after.get("conversation_count", 0)

        # Calculate number of pruned conversations
        pruned_count = max(0, conversation_count_before - conversation_count_after)

        logger.info(f"Pruned {pruned_count} conversations older than {max_age_days} days")
        return pruned_count