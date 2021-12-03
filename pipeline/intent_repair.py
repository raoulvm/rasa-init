from typing import Any, Dict, List, Optional, Text, Type
from rasa.shared.nlu.training_data.message import Message
from rasa.nlu.model import Metadata
from rasa.nlu.tokenizers.tokenizer import Token
import logging
from rasa.shared.nlu.constants import (
    INTENT_NAME_KEY,
    PREDICTED_CONFIDENCE_KEY,
    TEXT,
    ENTITY_ATTRIBUTE_START,
    ENTITY_ATTRIBUTE_END,
    ENTITIES,
    INTENT,
    INTENT_RANKING_KEY,
)
from rasa.nlu.constants import TOKENS_NAMES
from rasa.nlu.extractors.extractor import EntityExtractor
from rasa.nlu.classifiers.classifier import IntentClassifier
from rasa.shared.nlu.training_data.training_data import TrainingData
from rasa.nlu.config import RasaNLUModelConfig
from rasa.nlu.components import Component


logger = logging.getLogger(__name__)


class EntityOnlyIntentClassifier(Component):
    """An INTENT classifier that can predict a fixed intent if 
    the user utterance consists only of ENTITIES.

    Additionally a configurable number of stopwords is allowed. Stopwords need to be defined as list.

    It does only work if all entity extractors report extraction positions in the entities.

    """

    # Defines the default configuration parameters of a component
    # these values can be overwritten in the pipeline configuration
    # of the model. The component should choose sensible defaults
    # and should be able to create reasonable results with the defaults.
    # entityfile: Single yaml file name or GLOB pattern, such as ./entity/**/*.yml
    defaults: dict = {
        "intent_confidence_threshold": 0,
        "intent_name": None,
        "always_replace_intent": "nlu_fallback",
        "stopwords": [
            "der",
            "die",
            "das",
            "ein",
            "eine",
            "eines",
            "einen",
            "einer",
            "en",
            "ne",
            "ene",
        ],
        "max_number_of_stopwords": 2,
    }

    # Defines what language(s) this component can handle.
    # This attribute is designed for instance method: `can_handle_language`.
    # Default value is None which means it can handle all languages.
    # This is an important feature for backwards compatibility of components.
    supported_language_list = None

    # Defines what language(s) this component can NOT handle.
    # This attribute is designed for instance method: `can_handle_language`.
    # Default value is None which means it can handle all languages.
    # This is an important feature for backwards compatibility of components.
    not_supported_language_list = None

    def __init__(
        self,
        component_config: Optional[Dict[Text, Any]] = None,
        entityhierarchy: Optional[Dict[Text, Any]] = None,
    ) -> None:
        super().__init__(component_config)
        if not component_config:
            component_config = self.defaults
        self.intent_name = component_config.get("intent_name")
        if not self.intent_name:
            logger.error(
                "EntityOnlyIntent fixer defined but no target intent given. Specify intent_name in config.yml"
            )
            raise ValueError("Undefined intent_name")
        self.threshold = component_config.get("intent_confidence_threshold", 1)
        self.max_stopwords = component_config.get("max_number_of_stopwords", 0)
        if not isinstance(component_config.get("stopwords", []), list):
            self.stop_words = []
        else:
            self.stop_words = [w.lower() for w in component_config.get("stopwords", []) if isinstance(w, str)]
        self.intent_name_always_replace = component_config.get("always_replace_intent", "")

    @classmethod
    def required_components(cls) -> List[Type[Component]]:
        """Specify which components need to be present in the pipeline."""

        return [EntityExtractor, IntentClassifier]

    def train(
        self,
        training_data: TrainingData,
        config: Optional[RasaNLUModelConfig] = None,
        **kwargs: Any,
    ) -> None:
        logger.debug("Rain check on training....")
        # TODO: Implement training if/when needed

    def process(self, message: Message, **kwargs: Any) -> None:
        intent = message.get(INTENT)
        if not isinstance(intent, dict) or (
            intent.get(PREDICTED_CONFIDENCE_KEY, 0) > self.threshold
            and self.intent_name_always_replace == intent.get(INTENT_NAME_KEY, "")
        ):
            return
        entities = message.get(ENTITIES, [])
        # unique list of start/end pairs
        pairs = set()
        for e in entities or []:
            if isinstance(e, str):
                return  # only happens during training, in inference runs it is a dict!
            start = e.get(ENTITY_ATTRIBUTE_START, None)  # 0 is a valid entry!!
            end = e.get(ENTITY_ATTRIBUTE_END, None)
            # needs valid start and end
            if start is not None and end is not None:
                pairs.add((start, end))
        tokens: List[Token] = message.get(TOKENS_NAMES[TEXT])
        logger.debug(f"{tokens}")
        copy_token_list = tokens[:]
        stopword_hits = 0
        for i, t in enumerate(copy_token_list):
            for start, end in pairs:
                if t.start >= start and t.end <= end:
                    # token is part of an entity
                    copy_token_list.pop(i)
                elif (t.text.lower() in self.stop_words) and (stopword_hits < self.max_stopwords):
                    stopword_hits += 1
                    copy_token_list.pop(i)
        # if none are left, all tokens belong to entities extracted
        tokens_left = len(copy_token_list)
        if tokens_left == 0:
            # change the intent
            message.set(INTENT, {INTENT_NAME_KEY: self.intent_name, PREDICTED_CONFIDENCE_KEY: 1.0})
            ranking = message.get(INTENT_RANKING_KEY)
            ranking.insert(0, {INTENT_NAME_KEY: self.intent_name, PREDICTED_CONFIDENCE_KEY: 1.0})
            message.set(INTENT_RANKING_KEY, ranking)
            logger.debug(f"changed intent to {self.intent_name}")

    def persist(self, file_name: Text, model_dir: Text) -> Optional[Dict[Text, Any]]:
        """Persist this component to disk for future loading."""
        return {"file": None}

    @classmethod
    def load(
        cls,
        meta: Dict[Text, Any],
        model_dir: Text,
        model_metadata: Optional["Metadata"] = None,
        cached_component: Optional["Component"] = None,
        **kwargs: Any,
    ) -> "Component":
        """Load this component from file."""

        # file_name = meta.get("file")
        return cls(meta, None)
