"""
Text analysis functionality using LMStudio's Gemma model for the Voice Assistant.
"""
import json
from typing import Dict, Any, Optional

import requests

from utils import logger

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
    """

    def __init__(self,
                 api_url: str = DEFAULT_LMSTUDIO_API_URL,
                 system_prompt: str = DEFAULT_SYSTEM_PROMPT):
        """
        Initialize the Gemma text analyzer.

        Args:
            api_url: URL for the LMStudio API endpoint
            system_prompt: System prompt to guide the model's behavior
        """
        self.api_url = api_url
        self.system_prompt = system_prompt
        # Initialize conversation history with system prompt
        self.history = [{"role": "system", "content": self.system_prompt}]
        logger.debug(f"Initialized Gemma analyzer with API URL: {api_url}")

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
                     max_tokens: int = 300) -> str: # Changed Optional[str] to str
        """
        Send text to LMStudio's Gemma model for analysis.

        Args:
            text: The text to analyze
            temperature: Sampling temperature (0-1)
            max_tokens: Maximum number of tokens to generate

        Returns:
            Analysis result as text. This will be an error message string if analysis failed.
        """
        # Add user message to history and send full conversation
        try:
            logger.debug(f"Analyzing text with Gemma (length: {len(text)} chars)")
            self.history.append({"role": "user", "content": text})

            payload = {
                "model": "gemma-3-1b-it", # Ensure this model is loaded in LMStudio
                "messages": self.history,
                "temperature": temperature,
                "max_tokens": max_tokens,
                "stream": False # Ensure not streaming for this simple case
            }
            headers = {"Content-Type": "application/json"}
            logger.debug(f"Sending request to LMStudio API: {json.dumps(payload, indent=2)}")
            response = requests.post(self.api_url, json=payload, headers=headers, timeout=60) # Increased timeout

            if response.status_code == 200:
                result = response.json()
                logger.debug(f"LMStudio API raw response: {json.dumps(result, indent=2)}")
                if "choices" in result and len(result["choices"]) > 0 and "message" in result["choices"][0] and "content" in result["choices"][0]["message"]:
                    analysis = result["choices"][0]["message"]["content"]
                    logger.debug("Analysis completed successfully")
                    # Store assistant response in history
                    self.history.append({"role": "assistant", "content": analysis})
                    return analysis
                else:
                    err_msg = "LMStudio API response in unexpected format."
                    logger.error(f"{err_msg} Full response: {result}")
                    self.history.append({"role": "assistant", "content": f"Error: {err_msg}"})
                    return f"Error: {err_msg}"
            else:
                err = f"Error connecting to LMStudio: {response.status_code} - {response.text}"
                logger.error(f"Error from LMStudio API: {response.status_code} - {response.text}")
                self.history.append({"role": "assistant", "content": err})
                return err # Return error message to be potentially spoken or handled
        except requests.exceptions.Timeout:
            err = "Error analyzing with Gemma: Request timed out."
            logger.error(err)
            self.history.append({"role": "assistant", "content": err})
            return err
        except Exception as e:
            err = f"Error analyzing with Gemma: {str(e)}"
            logger.error(err, exc_info=True)
            self.history.append({"role": "assistant", "content": err})
            return err # Return error message