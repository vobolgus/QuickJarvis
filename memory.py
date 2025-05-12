"""
Memory system for the Voice Assistant that implements Predictive Efficiently Indexed Feature Tracking (PEFT).
Provides persistent storage, feature extraction, indexing, and prediction for conversation history.
"""
import os
import sqlite3
import json
import time
import uuid
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional, Callable, Tuple
import re

from utils import logger


class PEFTMemory:
    """
    Predictive Efficiently Indexed Feature Tracking (PEFT) memory system.
    Provides persistent storage, feature extraction, indexing, and prediction
    for conversation history.
    """

    def __init__(self, storage_path: str = "memory.sqlite"):
        """
        Initialize the PEFT memory system.

        Args:
            storage_path: Path to the SQLite database file
        """
        self.storage_path = storage_path
        self.db_dir = os.path.dirname(os.path.abspath(storage_path))
        if self.db_dir and not os.path.exists(self.db_dir):
            os.makedirs(self.db_dir, exist_ok=True)

        self.init_database()
        self.feature_extractors = self.get_feature_extractors()
        self.current_conversation_id = None
        logger.debug(f"Initialized PEFT memory with storage at: {self.storage_path}")

    def init_database(self):
        """Initialize the SQLite database with the required schema."""
        with sqlite3.connect(self.storage_path) as conn:
            cursor = conn.cursor()

            # Create conversations table
            cursor.execute('''
                           CREATE TABLE IF NOT EXISTS conversations
                           (
                               id
                               TEXT
                               PRIMARY
                               KEY,
                               start_time
                               TIMESTAMP,
                               last_activity
                               TIMESTAMP,
                               metadata
                               TEXT
                           )
                           ''')

            # Create messages table
            cursor.execute('''
                           CREATE TABLE IF NOT EXISTS messages
                           (
                               id
                               TEXT
                               PRIMARY
                               KEY,
                               conversation_id
                               TEXT,
                               timestamp
                               TIMESTAMP,
                               role
                               TEXT,
                               content
                               TEXT,
                               FOREIGN
                               KEY
                           (
                               conversation_id
                           ) REFERENCES conversations
                           (
                               id
                           )
                               )
                           ''')

            # Create features table
            cursor.execute('''
                           CREATE TABLE IF NOT EXISTS features
                           (
                               id
                               TEXT
                               PRIMARY
                               KEY,
                               message_id
                               TEXT,
                               feature_type
                               TEXT,
                               feature_value
                               TEXT,
                               feature_score
                               REAL,
                               FOREIGN
                               KEY
                           (
                               message_id
                           ) REFERENCES messages
                           (
                               id
                           )
                               )
                           ''')

            # Create indices for performance
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_messages_conversation ON messages (conversation_id)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_features_message ON features (message_id)')
            cursor.execute(
                'CREATE INDEX IF NOT EXISTS idx_features_type_value ON features (feature_type, feature_value)')

            conn.commit()
            logger.debug("PEFT database initialized or confirmed")

    def get_feature_extractors(self) -> Dict[str, Callable]:
        """
        Return a dictionary of feature extraction functions.

        Returns:
            Dictionary mapping feature types to extraction functions
        """
        return {
            "keywords": self._extract_keywords,
            "entities": self._extract_entities,
            "intents": self._extract_intents,
            "sentiment": self._extract_sentiment,
            "questions": self._extract_questions
        }

    def _extract_keywords(self, content: str) -> List[Dict[str, Any]]:
        """Extract important keywords from content."""
        # Simple keyword extraction based on word frequency and filtering
        if not content:
            return []

        # Tokenize and lowercase
        words = re.findall(r'\b[a-zA-Z]{3,}\b', content.lower())

        # Count word frequencies
        word_counts = {}
        for word in words:
            if word in word_counts:
                word_counts[word] += 1
            else:
                word_counts[word] = 1

        # Filter out common stopwords
        stopwords = {"the", "and", "you", "for", "with", "this", "that", "what", "have", "from",
                     "are", "is", "in", "to", "it", "its", "of", "that", "was", "were", "be",
                     "been", "being", "by", "but", "not", "they", "their", "them", "these", "those"}
        for stopword in stopwords:
            if stopword in word_counts:
                del word_counts[stopword]

        # Convert to feature format
        features = []
        for word, count in word_counts.items():
            if count > 0:  # Only include words that appear at least once
                features.append({
                    "type": "keywords",
                    "value": word,
                    "score": min(1.0, count / 10)  # Normalize score to max 1.0
                })

        return features

    def _extract_entities(self, content: str) -> List[Dict[str, Any]]:
        """Extract named entities from content."""
        # Simple entity extraction with regex patterns
        entities = []

        # Look for dates
        date_patterns = [
            r'\b\d{1,2}/\d{1,2}/\d{2,4}\b',  # MM/DD/YYYY
            r'\b\d{1,2}-\d{1,2}-\d{2,4}\b',  # MM-DD-YYYY
            r'\b(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]* \d{1,2}(?:st|nd|rd|th)?, \d{4}\b'
            # Month Day, Year
        ]
        for pattern in date_patterns:
            for match in re.finditer(pattern, content):
                entities.append({
                    "type": "entities",
                    "value": f"date:{match.group(0)}",
                    "score": 1.0
                })

        # Look for times
        time_patterns = [
            r'\b\d{1,2}:\d{2}\s*(?:AM|PM)?\b',  # HH:MM AM/PM
            r'\b\d{1,2}\s*(?:AM|PM)\b'  # HH AM/PM
        ]
        for pattern in time_patterns:
            for match in re.finditer(pattern, content, re.IGNORECASE):
                entities.append({
                    "type": "entities",
                    "value": f"time:{match.group(0)}",
                    "score": 1.0
                })

        # Look for emails
        email_pattern = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
        for match in re.finditer(email_pattern, content):
            entities.append({
                "type": "entities",
                "value": f"email:{match.group(0)}",
                "score": 1.0
            })

        # Look for phone numbers
        phone_pattern = r'\b(?:\+\d{1,2}\s)?\(?\d{3}\)?[\s.-]?\d{3}[\s.-]?\d{4}\b'
        for match in re.finditer(phone_pattern, content):
            entities.append({
                "type": "entities",
                "value": f"phone:{match.group(0)}",
                "score": 1.0
            })

        return entities

    def _extract_intents(self, content: str) -> List[Dict[str, Any]]:
        """Extract possible user intents from content."""
        intents = []

        # Check for common intents based on keywords and patterns
        intent_patterns = {
            "greeting": [r'\b(?:hello|hi|hey|greetings)\b'],
            "farewell": [r'\b(?:goodbye|bye|see\s+you|later)\b'],
            "gratitude": [r'\b(?:thanks|thank\s+you|appreciate)\b'],
            "help": [r'\b(?:help|assist|support)\b', r'how\s+(?:do|can|would|should)\s+i'],
            "confirm": [r'\b(?:yes|yeah|sure|certainly|absolutely)\b'],
            "deny": [r'\b(?:no|nope|not)\b'],
            "question": [r'\?$', r'^(?:who|what|when|where|why|how)'],
            "command": [r'^(?:please\s+)?\w+\s+(?:the|this|that|my)']
        }

        for intent, patterns in intent_patterns.items():
            for pattern in patterns:
                if re.search(pattern, content.lower()):
                    intents.append({
                        "type": "intents",
                        "value": intent,
                        "score": 0.8  # Confidence score (could be refined)
                    })
                    break  # Only add each intent once

        return intents

    def _extract_sentiment(self, content: str) -> List[Dict[str, Any]]:
        """Extract sentiment indicators from content."""
        # Simple sentiment analysis based on keyword presence
        positive_words = {"good", "great", "excellent", "wonderful", "happy", "like", "love", "best",
                          "better", "thanks", "thank", "nice", "amazing", "awesome", "fantastic"}
        negative_words = {"bad", "terrible", "awful", "horrible", "sad", "hate", "worst", "worse",
                          "problem", "issue", "sorry", "unfortunately", "disappointed", "unhappy"}

        words = re.findall(r'\b[a-zA-Z]+\b', content.lower())

        pos_count = sum(1 for word in words if word in positive_words)
        neg_count = sum(1 for word in words if word in negative_words)

        sentiments = []

        if pos_count > neg_count:
            sentiments.append({
                "type": "sentiment",
                "value": "positive",
                "score": min(1.0, pos_count / 5)  # Normalize
            })
        elif neg_count > pos_count:
            sentiments.append({
                "type": "sentiment",
                "value": "negative",
                "score": min(1.0, neg_count / 5)  # Normalize
            })
        else:
            sentiments.append({
                "type": "sentiment",
                "value": "neutral",
                "score": 0.5
            })

        return sentiments

    def _extract_questions(self, content: str) -> List[Dict[str, Any]]:
        """Extract questions from content."""
        questions = []

        # Check for question marks
        if '?' in content:
            # Find sentences ending with question marks
            question_sentences = re.findall(r'[^.!?]*\?', content)

            for question in question_sentences:
                questions.append({
                    "type": "questions",
                    "value": question.strip(),
                    "score": 1.0
                })

        # Check for question words without question marks
        question_starters = ["what", "who", "when", "where", "why", "how", "can", "could",
                             "would", "should", "is", "are", "do", "does"]
        sentences = re.split(r'[.!?]', content)

        for sentence in sentences:
            sentence = sentence.strip().lower()
            if sentence and not sentence.endswith('?'):
                words = sentence.split()
                if words and words[0] in question_starters:
                    questions.append({
                        "type": "questions",
                        "value": sentence,
                        "score": 0.7  # Lower confidence without question mark
                    })

        return questions

    def start_conversation(self) -> str:
        """
        Start a new conversation and return its ID.

        Returns:
            The ID of the new conversation
        """
        conversation_id = str(uuid.uuid4())
        now = datetime.now().isoformat()

        with sqlite3.connect(self.storage_path) as conn:
            cursor = conn.cursor()
            cursor.execute(
                "INSERT INTO conversations (id, start_time, last_activity, metadata) VALUES (?, ?, ?, ?)",
                (conversation_id, now, now, json.dumps({}))
            )
            conn.commit()

        self.current_conversation_id = conversation_id
        logger.debug(f"Started new conversation with ID: {conversation_id}")
        return conversation_id

    def add_message(self, role: str, content: str, conversation_id: Optional[str] = None) -> str:
        """
        Add a new message to the conversation history.

        Args:
            role: The role of the sender (user, assistant, system)
            content: The message content
            conversation_id: ID of the conversation (uses current if None)

        Returns:
            The ID of the added message
        """
        if conversation_id is None:
            if self.current_conversation_id is None:
                self.current_conversation_id = self.start_conversation()
            conversation_id = self.current_conversation_id

        message_id = str(uuid.uuid4())
        now = datetime.now().isoformat()

        # Update conversation last activity time
        with sqlite3.connect(self.storage_path) as conn:
            cursor = conn.cursor()

            # Add message
            cursor.execute(
                "INSERT INTO messages (id, conversation_id, timestamp, role, content) VALUES (?, ?, ?, ?, ?)",
                (message_id, conversation_id, now, role, content)
            )

            # Update conversation last activity
            cursor.execute(
                "UPDATE conversations SET last_activity = ? WHERE id = ?",
                (now, conversation_id)
            )

            conn.commit()

        # Extract and index features (skip for system messages as they're less relevant for retrieval)
        if role != "system":
            features = self.extract_features(content)
            self.index_features(message_id, features)

        logger.debug(f"Added {role} message to conversation {conversation_id}: {content[:50]}...")
        return message_id

    def extract_features(self, content: str) -> List[Dict[str, Any]]:
        """
        Extract features from message content.

        Args:
            content: The message content

        Returns:
            A list of extracted features
        """
        features = []

        for extractor_name, extractor_func in self.feature_extractors.items():
            try:
                extracted = extractor_func(content)
                features.extend(extracted)
            except Exception as e:
                logger.warning(f"Feature extractor {extractor_name} failed: {e}")

        return features

    def index_features(self, message_id: str, features: List[Dict[str, Any]]):
        """
        Index the extracted features for efficient retrieval.

        Args:
            message_id: The ID of the message
            features: The extracted features
        """
        if not features:
            return

        with sqlite3.connect(self.storage_path) as conn:
            cursor = conn.cursor()

            for feature in features:
                feature_id = str(uuid.uuid4())

                cursor.execute(
                    "INSERT INTO features (id, message_id, feature_type, feature_value, feature_score) VALUES (?, ?, ?, ?, ?)",
                    (feature_id, message_id, feature["type"], feature["value"], feature["score"])
                )

            conn.commit()

        logger.debug(f"Indexed {len(features)} features for message {message_id}")

    def get_relevant_history(self, query: str, limit: int = 5) -> List[Dict[str, Any]]:
        """
        Get relevant historical messages based on the query.

        Args:
            query: The query to match against indexed features
            limit: Maximum number of messages to return

        Returns:
            A list of relevant messages
        """
        # Extract features from the query to find similar messages
        query_features = self.extract_features(query)

        if not query_features:
            return []

        # Prepare SQL for matching messages with similar features
        feature_conditions = []
        params = []

        for feature in query_features:
            feature_conditions.append("(feature_type = ? AND feature_value = ?)")
            params.extend([feature["type"], feature["value"]])

        if not feature_conditions:
            return []

        with sqlite3.connect(self.storage_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()

            # Find messages with matching features, excluding current conversation
            sql = f"""
            SELECT m.id, m.conversation_id, m.timestamp, m.role, m.content, COUNT(*) as matching_features
            FROM messages m
            JOIN features f ON m.id = f.message_id
            WHERE ({" OR ".join(feature_conditions)})
            AND m.conversation_id != ?
            GROUP BY m.id
            ORDER BY matching_features DESC, m.timestamp DESC
            LIMIT ?
            """

            params.append(self.current_conversation_id if self.current_conversation_id else '')
            params.append(limit)

            cursor.execute(sql, params)
            rows = cursor.fetchall()

            results = []
            for row in rows:
                results.append({
                    "id": row["id"],
                    "conversation_id": row["conversation_id"],
                    "timestamp": row["timestamp"],
                    "role": row["role"],
                    "content": row["content"],
                    "relevance_score": row["matching_features"]
                })

        logger.debug(f"Found {len(results)} relevant historical messages for query")
        return results

    def get_conversation_context(self, query: str) -> Tuple[List[Dict[str, Any]], List[str]]:
        """
        Get context from previous conversations and response suggestions.

        Args:
            query: The current user query

        Returns:
            A tuple containing (relevant historical messages, suggested responses)
        """
        # Get relevant messages from history
        relevant_messages = self.get_relevant_history(query, limit=5)

        # Get response suggestions based on the query
        suggested_responses = self.predict_response(query)

        return relevant_messages, suggested_responses

    def predict_response(self, query: str) -> List[str]:
        """
        Predict potential responses based on similar historical conversations.

        Args:
            query: The current user query

        Returns:
            A list of potential response templates
        """
        # Find similar user messages in history
        similar_messages = self.get_relevant_history(query, limit=10)

        if not similar_messages:
            return []

        # For each similar message, find the assistant's response that followed
        response_templates = []

        with sqlite3.connect(self.storage_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()

            for message in similar_messages:
                if message["role"] != "user":
                    continue

                # Find assistant message that followed this user message
                cursor.execute("""
                               SELECT content
                               FROM messages
                               WHERE conversation_id = ?
                                 AND timestamp
                                   > ?
                                 AND role = 'assistant'
                               ORDER BY timestamp ASC
                                   LIMIT 1
                               """, (message["conversation_id"], message["timestamp"]))

                row = cursor.fetchone()
                if row and row["content"]:
                    # Only add unique responses
                    if row["content"] not in response_templates:
                        response_templates.append(row["content"])

        # Limit to a reasonable number of suggestions
        return response_templates[:3]

    def get_conversation_history(self, conversation_id: Optional[str] = None, include_system: bool = True) -> List[
        Dict[str, str]]:
        """
        Get the full history of a conversation.

        Args:
            conversation_id: ID of the conversation (uses current if None)
            include_system: Whether to include system messages

        Returns:
            A list of messages in the conversation
        """
        if conversation_id is None:
            if self.current_conversation_id is None:
                return []
            conversation_id = self.current_conversation_id

        with sqlite3.connect(self.storage_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()

            # Query to get messages
            sql = """
                  SELECT role, content \
                  FROM messages
                  WHERE conversation_id = ? \
                  """

            if not include_system:
                sql += " AND role != 'system'"

            sql += " ORDER BY timestamp ASC"

            cursor.execute(sql, (conversation_id,))
            rows = cursor.fetchall()

            history = []
            for row in rows:
                history.append({
                    "role": row["role"],
                    "content": row["content"]
                })

        return history

    def get_conversation_stats(self) -> Dict[str, Any]:
        """
        Get statistics about the conversation database.

        Returns:
            Dictionary of statistics
        """
        with sqlite3.connect(self.storage_path) as conn:
            cursor = conn.cursor()

            # Get counts
            cursor.execute("SELECT COUNT(*) FROM conversations")
            conversation_count = cursor.fetchone()[0]

            cursor.execute("SELECT COUNT(*) FROM messages")
            message_count = cursor.fetchone()[0]

            cursor.execute("SELECT COUNT(*) FROM features")
            feature_count = cursor.fetchone()[0]

            # Get most recent conversation time
            cursor.execute("SELECT MAX(last_activity) FROM conversations")
            last_activity = cursor.fetchone()[0]

            # Get top intents
            cursor.execute("""
                           SELECT feature_value, COUNT(*) as count
                           FROM features
                           WHERE feature_type = 'intents'
                           GROUP BY feature_value
                           ORDER BY count DESC
                               LIMIT 5
                           """)
            top_intents = {row[0]: row[1] for row in cursor.fetchall()}

            # Get top sentiments
            cursor.execute("""
                           SELECT feature_value, COUNT(*) as count
                           FROM features
                           WHERE feature_type = 'sentiment'
                           GROUP BY feature_value
                           ORDER BY count DESC
                           """)
            sentiment_distribution = {row[0]: row[1] for row in cursor.fetchall()}

        return {
            "conversation_count": conversation_count,
            "message_count": message_count,
            "feature_count": feature_count,
            "last_activity": last_activity,
            "top_intents": top_intents,
            "sentiment_distribution": sentiment_distribution
        }

    def prune_old_conversations(self, max_age_days: int = 30):
        """
        Remove conversations older than the specified age.

        Args:
            max_age_days: Maximum age of conversations to keep
        """
        cutoff_date = (datetime.now() - timedelta(days=max_age_days)).isoformat()

        with sqlite3.connect(self.storage_path) as conn:
            cursor = conn.cursor()

            # Get IDs of old conversations
            cursor.execute(
                "SELECT id FROM conversations WHERE last_activity < ?",
                (cutoff_date,)
            )
            old_conversation_ids = [row[0] for row in cursor.fetchall()]

            if not old_conversation_ids:
                logger.debug("No old conversations to prune")
                return

            for conv_id in old_conversation_ids:
                # Get message IDs for this conversation
                cursor.execute(
                    "SELECT id FROM messages WHERE conversation_id = ?",
                    (conv_id,)
                )
                message_ids = [row[0] for row in cursor.fetchall()]

                # Delete features for messages
                if message_ids:
                    placeholders = ','.join(['?'] * len(message_ids))
                    cursor.execute(
                        f"DELETE FROM features WHERE message_id IN ({placeholders})",
                        message_ids
                    )

                # Delete messages
                cursor.execute(
                    "DELETE FROM messages WHERE conversation_id = ?",
                    (conv_id,)
                )

                # Delete conversation
                cursor.execute(
                    "DELETE FROM conversations WHERE id = ?",
                    (conv_id,)
                )

            conn.commit()

            logger.debug(f"Pruned {len(old_conversation_ids)} old conversations")