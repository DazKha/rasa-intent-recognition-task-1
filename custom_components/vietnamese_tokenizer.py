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