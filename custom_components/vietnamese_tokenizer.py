import re
from typing import Any, Dict, List, Text, Optional
from rasa.engine.graph import GraphComponent, ExecutionContext
from rasa.engine.recipes.default_recipe import DefaultV1Recipe
from rasa.engine.storage.resource import Resource
from rasa.engine.storage.storage import ModelStorage
from rasa.nlu.tokenizers.tokenizer import Token, Tokenizer
from rasa.shared.nlu.training_data.message import Message
from rasa.shared.nlu.training_data.training_data import TrainingData
from rasa.shared.nlu.constants import TEXT
from underthesea import word_tokenize


@DefaultV1Recipe.register(
    [DefaultV1Recipe.ComponentType.MESSAGE_TOKENIZER], is_trainable=False
)
class VietnameseTokenizer(Tokenizer, GraphComponent):
    @classmethod
    def create(
            cls,
            config: Dict[Text, Any],
            model_storage: ModelStorage,
            resource: Resource,
            execution_context: ExecutionContext,
    ) -> GraphComponent:
        return cls(config)

    def __init__(self, config: Dict[Text, Any]) -> None:
        # Thêm các giá trị mặc định cho các tham số cấu hình
        defaults = {
            # từ Tokenizer
            "intent_tokenization_flag": False,
            "intent_split_symbol": "_",
            "token_pattern": None,
            # Có thể thêm các tham số cấu hình khác ở đây nếu cần
        }

        # Cập nhật config với các giá trị mặc định
        config = {**defaults, **config}

        super().__init__(config)

    def tokenize(self, message: Message, attribute: Text = TEXT) -> List[Token]:
        text = message.get(attribute)
        words = word_tokenize(text)
        return self._convert_words_to_tokens(words, text)

    def _convert_words_to_tokens(self, words: List[str], text: Text) -> List[Token]:
        """Convert the words returned from word_tokenize into Token objects."""
        running_offset = 0
        tokens = []

        for word in words:
            word_without_underscore = word.replace("_", " ")

            try:
                word_offset = text.index(word, running_offset)
                word_len = len(word)
            except ValueError:
                try:
                    # Try finding the word without underscores
                    word_offset = text.index(word_without_underscore, running_offset)
                    word_len = len(word_without_underscore)
                    word = word_without_underscore  # Use the version that was found
                except ValueError:
                    # If still not found, try with case insensitive match
                    remaining_text = text[running_offset:].lower()
                    search_word = word.lower()
                    search_word_no_underscore = word_without_underscore.lower()

                    try:
                        word_offset = text.lower().index(search_word, running_offset)
                        word_len = len(word)
                    except ValueError:
                        try:
                            word_offset = text.lower().index(search_word_no_underscore, running_offset)
                            word_len = len(word_without_underscore)
                            word = word_without_underscore  # Use the version that was found
                        except ValueError:
                            # Skip this word if it can't be found
                            continue

            tokens.append(Token(word, word_offset))
            running_offset = word_offset + word_len

        return tokens

    def train(self, training_data: TrainingData) -> Resource:
        pass

    def process(self, messages: List[Message]) -> List[Message]:
        for message in messages:
            for attribute in [TEXT, "response"]:
                if message.get(attribute):
                    message.set(f"{attribute}_tokens", self.tokenize(message, attribute))
        return messages

    def process_training_data(self, training_data: TrainingData) -> TrainingData:
        self.process(training_data.training_examples)
        return training_data

# tokenizer = VietnameseTokenizer({})
# message = Message(data={"text": "hôm qua tôi đi siêu thị mua 100k bò"})
# tokens = tokenizer.tokenize(message)
# print([token.text for token in tokens])
