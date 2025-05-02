"""
Text analysis functionality using LMStudio's Gemma model for the Voice Assistant.
"""
import json
from typing import Dict, Any, Optional

import requests

from utils import logger

# Default LMStudio configuration
DEFAULT_LMSTUDIO_API_URL = "http://localhost:1234/v1/chat/completions"
DEFAULT_SYSTEM_PROMPT = """Ты - дружелюбный виртуальный ассистент, созданный чтобы помогать людям. Вот твои основные принципы:

1. Ты всегда спокоен, терпелив и вежлив, независимо от того, как пользователь общается с тобой.

2. Ты отвечаешь ясно и лаконично, избегая слишком длинных и сложных объяснений, когда в них нет необходимости.

3. Твой тон - тёплый и дружелюбный, но профессиональный. Ты создаёшь ощущение комфортного разговора.

4. Когда пользователь задаёт вопрос, ты даёшь конкретный ответ без лишних вступлений или отступлений.

5. Если ты не знаешь ответа или не можешь выполнить запрос, ты честно сообщаешь об этом и предлагаешь альтернативные решения, если это возможно.

6. Ты уважаешь время пользователя - твои ответы информативны, но не избыточны.

7. Ты не используешь смайлики или эмодзи в своих ответах, только текст.

8. Ты способен поддержать как деловую беседу, так и повседневный разговор, адаптируя свой стиль к контексту.

9. Ты никогда не осуждаешь пользователя и относишься с пониманием к любым затруднениям или ошибкам.

10. Твоя цель - сделать общение с технологией максимально естественным, приятным и полезным для каждого пользователя.

Отвечай так, как если бы ты был умным, знающим и отзывчивым другом, который хочет помочь."""

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
        try:
            logger.info(f"Analyzing text with Gemma (length: {len(text)} chars)")
            
            payload = {
                "model": "gemma",  # This might need adjustment based on LMStudio configuration
                "messages": [
                    {"role": "system", "content": self.system_prompt},
                    {"role": "user", "content": f"Analyze this transcribed speech: \"{text}\""}
                ],
                "temperature": temperature,
                "max_tokens": max_tokens
            }
            
            headers = {
                "Content-Type": "application/json"
            }
            
            logger.debug(f"Sending request to LMStudio API: {json.dumps(payload)}")
            response = requests.post(self.api_url, json=payload, headers=headers, timeout=30)
            
            if response.status_code == 200:
                result = response.json()
                analysis = result["choices"][0]["message"]["content"]
                logger.info("Analysis completed successfully")
                return analysis
            else:
                logger.error(f"Error from LMStudio API: {response.status_code} - {response.text}")
                return f"Error connecting to LMStudio: {response.status_code} - {response.text}"
                
        except Exception as e:
            logger.error(f"Error analyzing with Gemma: {str(e)}")
            return f"Error analyzing with Gemma: {str(e)}"
