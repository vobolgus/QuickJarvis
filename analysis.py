"""
Text analysis functionality using LMStudio's Gemma model for the Voice Assistant.
"""
import json
from typing import Dict, Any, Optional

import requests

from utils import logger

# Default LMStudio configuration
DEFAULT_LMSTUDIO_API_URL = "http://localhost:1234/v1/chat/completions"
# Enhanced system prompt for better context handling and memory
DEFAULT_SYSTEM_PROMPT = (
    "You are a helpful, context-aware AI assistant that remembers previous user interactions "
    "and your own responses. Provide concise, accurate, and informative answers. For math statements spell the equations, rather than typesetting them (eq. not \\frac{1}{2}, but one over two) /no_think"
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
        logger.info(f"Initialized Gemma analyzer with API URL: {api_url}")

    def check_connection(self) -> bool:
        """
        Check if LMStudio is running and accessible.

        Returns:
            True if LMStudio is available, False otherwise
        """
        try:
            # Simple request to check if the server is running
            base_url = self.api_url.rsplit('/', 1)[0]
            response = requests.get(base_url, timeout=5)
            if response.status_code < 500:  # Any response that's not a server error
                logger.info("Successfully connected to LMStudio API")
                return True
            logger.warning(f"LMStudio API returned status code: {response.status_code}")
            return False
        except requests.exceptions.RequestException as e:
            logger.warning(f"Failed to connect to LMStudio API: {e}")
            return False

    def analyze_text(self,
                     text: str,
                     temperature: float = 0.7,
                     max_tokens: int = 300) -> Optional[str]:
        """
        Send text to LMStudio's Gemma model for analysis.

        Args:
            text: The text to analyze
            temperature: Sampling temperature (0-1)
            max_tokens: Maximum number of tokens to generate

        Returns:
            Analysis result as text, or None if the analysis failed
        """
        # Add user message to history and send full conversation
        try:
            logger.info(f"Analyzing text with Gemma (length: {len(text)} chars)")
            self.history.append({"role": "user", "content": text})

            payload = {
                "model": "gemma-3-1b-it",
                "messages": self.history,
                "temperature": temperature,
                "max_tokens": max_tokens
            }
            headers = {"Content-Type": "application/json"}
            logger.debug(f"Sending request to LMStudio API: {json.dumps(payload)}")
            response = requests.post(self.api_url, json=payload, headers=headers, timeout=30)

            if response.status_code == 200:
                result = response.json()
                analysis = result["choices"][0]["message"]["content"]
                logger.info("Analysis completed successfully")
                # Store assistant response in history
                self.history.append({"role": "assistant", "content": analysis})
                return analysis
            else:
                err = f"Error connecting to LMStudio: {response.status_code} - {response.text}"
                logger.error(f"Error from LMStudio API: {response.status_code} - {response.text}")
                self.history.append({"role": "assistant", "content": err})
                return err
        except Exception as e:
            err = f"Error analyzing with Gemma: {str(e)}"
            logger.error(err)
            self.history.append({"role": "assistant", "content": err})
            return err
