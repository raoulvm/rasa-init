import os
import typing
from typing import Any, Dict, List, Optional, Text, Type
from glob import glob
import rasa.shared.utils.io

# from fuzzywuzzy import process
from rasa.nlu.components import Component
from rasa.nlu.config import RasaNLUModelConfig
from rasa.nlu.extractors.extractor import EntityExtractor
from rasa.nlu.utils import write_json_to_file
from rasa.shared.nlu.constants import (
    ENTITY_ATTRIBUTE_CONFIDENCE,
    TEXT,
    ENTITY_ATTRIBUTE_TYPE,
    ENTITY_ATTRIBUTE_START,
    ENTITY_ATTRIBUTE_END,
    ENTITY_ATTRIBUTE_VALUE,
    ENTITIES,
)
from rasa.nlu.model import Metadata
from rasa.shared.nlu.training_data.message import Message
from rasa.shared.nlu.training_data.training_data import TrainingData
import logging
from rasa.shared.utils.io import read_yaml_file
from pipeline._parser import topdownparser, ANY_SOURCE_ENTITY_KEY

from pipeline._flashtext_mod import KeywordProcessor

if typing.TYPE_CHECKING:
    from rasa.nlu.model import Metadata

logger = logging.getLogger(f"{__name__}.entity_hierarchy")

ENTITY_ATTRIBUTE_PROCESSORS = "processors"


###############
# Ontology approach (as Angel named it :-)
#
# An Entity, however it was extracted from the raw text, can be part of multiple
# entity hierarchies.
#
# Each hierarchy can have multiple (named) levels
#
#
################

# subclass EntityExtractor to skip featurize_message() in rasa.nlu.model.Interpreter
class EntityHierarchy(EntityExtractor):
    """A new component"""

    # Which components are required by this component.
    # Listed components should appear before the component itself in the pipeline.
    @classmethod
    def required_components(cls) -> List[Type[Component]]:
        """Specify which components need to be present in the pipeline."""

        return [EntityExtractor]

    # Defines the default configuration parameters of a component
    # these values can be overwritten in the pipeline configuration
    # of the model. The component should choose sensible defaults
    # and should be able to create reasonable results with the defaults.
    # entityfile: Single yaml file name or GLOB pattern, such as ./entity/**/*.yml
    defaults: dict = {
        "entityfile": None,
        "case_sensitive": False,
        "include_repeated_entities": False,  # if true the same entity will only return its first occurrence
        "non_word_boundaries": "_öäüÖÄÜß-",
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
        self.keyword_processor = KeywordProcessor(
            case_sensitive=self.component_config["case_sensitive"]
        )
        for non_word_boundary in self.component_config["non_word_boundaries"]:
            self.keyword_processor.add_non_word_boundary(non_word_boundary)
        self._entityfile = component_config.get("entityfile", None)
        self.include_repeated_entities = component_config.get("include_repeated_entities", False)

        if entityhierarchy:
            logger.debug(f"restore entityhierarchy")
            self._entityhierarchy = entityhierarchy
            self._parse_prepared_hierarchies()
        else:
            self._entityhierarchy = {}

    def _parse_prepared_hierarchies(self):
        for keyword, ent_dict in self._entityhierarchy.get("entities", {}).items():
            # keyword is the full text to be found, the dict contains entity:value pairs to be set
            # as flashtext can store ANY python object to be returned, we'll use the full dict as
            # return value
            self.keyword_processor.add_keyword(keyword, ent_dict)

        lookups = self.keyword_processor.get_all_keywords()
        if len(lookups.keys()) == 0:
            rasa.shared.utils.io.raise_warning(
                "No entity hierarchies defined in the training data that have "
                "text examples to use for the extractor"
            )
        # populate the secondary alternatives dictionary too

        for keyword, clean_name in self._entityhierarchy.get("alternatives", {}).items():
            self.keyword_processor.add_keyword(keyword, clean_name)

    def train(
        self,
        training_data: TrainingData,
        config: Optional[RasaNLUModelConfig] = None,
        **kwargs: Any,
    ) -> None:
        """Train this component.

        This is the components chance to train itself provided
        with the training data. The component can rely on
        any context attribute to be present, that gets created
        by a call to :meth:`components.Component.pipeline_init`
        of ANY component and
        on any context attributes created by a call to
        :meth:`components.Component.train`
        of components previous to this one."""

        self._entityhierarchy = {}
        # read the YAML file(s)
        if not self._entityfile:
            rasa.shared.utils.io.raise_warning(
                "EntityHierarchy is in the pipeline but no entityfile name is defined in config."
            )
            return
        filelist = glob(self._entityfile, recursive=True)

        raw_hierarchy = {}
        if filelist:
            # read each file and merge the results
            for fn in filelist:
                logger.debug(f"reading file {fn}")
                filecontent = read_yaml_file(fn)
                if isinstance(filecontent, dict):
                    if [k for k in filecontent if k in raw_hierarchy]:
                        raise ValueError(
                            f"Duplicate key(s) {[k for k in filecontent if k in raw_hierarchy]} in file {fn}"
                        )
                    raw_hierarchy.update(filecontent)
                    logger.info(f"Processed file {fn}")
                else:
                    logger.warn(f"{fn} invalid file format: must be a dictionary in YAML")
        self._entityhierarchy = topdownparser(raw_hierarchy)

        self._parse_prepared_hierarchies()

    # process from flashE
    def process(self, message: Message, **kwargs: Any) -> None:
        extracted_entities = self._extract_entities(message)
        extracted_entities = self.add_extractor_name(extracted_entities)
        entities = self._extent_entities(
            original_entities=message.get(ENTITIES, []), new_entities=extracted_entities
        )

        message.set(ENTITIES, entities, add_to_output=True)

    def _extract_entities(self, message: Message) -> List[Dict[Text, Any]]:
        """Extract entities of the given type from the given user message."""
        if len(self.keyword_processor) == 0:
            return []
        matches_ = self.keyword_processor.extract_keywords(message.get(TEXT), span_info=True)
        # matches looks like
        # [
        # ({"festnetz": true,"internet": "wlan","wlan": "wlan","topic": "festnetz"}, 39, 54),
        # ({'festnetz': True}, 63, 72)},
        # ('somethingfixed',100,112)
        # ]
        # if match[0] is a string it was an alternative spelling hit
        #
        matches = []
        for match in matches_:
            match = list(match)  # convert tuple to list to make it mutable
            # do the lookup of alternative spellings first
            if isinstance(match[0], (str, int, float)):
                # look it up and replace it
                match[0] = self._entityhierarchy.get("entities", {}).get(match[0], {})
            matches.append(match)
        # if duplicates are to be ignored, sort the list and remove duplicates
        if not self.include_repeated_entities:
            matches.sort(key=lambda e: e[1])  # sort by first occurrence in the message text

        extracted_entities = []
        name_cache = []

        for match in matches:
            for entity_type, entity_value in match[0].items():
                if entity_type not in name_cache:
                    if not self.include_repeated_entities:
                        name_cache.append(entity_type)
                    extracted_entities += [
                        {
                            ENTITY_ATTRIBUTE_TYPE: entity_type,
                            ENTITY_ATTRIBUTE_START: match[1],
                            ENTITY_ATTRIBUTE_END: match[2],
                            ENTITY_ATTRIBUTE_VALUE: entity_value,
                            ENTITY_ATTRIBUTE_CONFIDENCE: 1.0,
                        }
                    ]
        return extracted_entities

    def _extent_entities(
        self, original_entities: List[Dict[Text, Any]], new_entities: List[Dict[Text, Any]]
    ) -> List[Dict[Text, Any]]:
        """Adds new_entities to original_entities and returns the complete list.
           Respects setting of self._ignore_repeated_entities

        Args:
            original_entities (List[Dict[Text, Any]]): The entities already contained in the message, to be altered in_place
            new_entities (List[Dict[Text, Any]]): The entities to add to the message
        """
        entities = original_entities[:]
        entity_keys = [e.get(ENTITY_ATTRIBUTE_TYPE) for e in entities]
        for ent in new_entities:
            if self.include_repeated_entities or ent.get(ENTITY_ATTRIBUTE_TYPE) not in entity_keys:
                self.add_extractor_name([ent])
                entities.append(ent)
            else:
                # change value of passed entity
                pos = entity_keys.index(ent.get(ENTITY_ATTRIBUTE_TYPE))
                entity = entities[pos]
                entity.update({ENTITY_ATTRIBUTE_VALUE: ent.get(ENTITY_ATTRIBUTE_VALUE)})
                self.add_processor_name(entity)
        # entities.extend(new_ents)
        return entities

    ############# SAFE and LOAD methods #######################
    #
    def persist(self, file_name: Text, model_dir: Text) -> Optional[Dict[Text, Any]]:
        """Persist this component to disk for future loading."""
        if self._entityhierarchy:
            file_name = file_name + ".json"
            entity_files = os.path.join(model_dir, file_name)
            write_json_to_file(entity_files, self._entityhierarchy)

            return {"file": file_name}
        else:
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

        file_name = meta.get("file")
        if not file_name:
            enthier = None
            return cls(meta, enthier)

        entities_file = os.path.join(model_dir, file_name)
        if os.path.isfile(entities_file):
            enthier = rasa.shared.utils.io.read_json_file(entities_file)
        else:
            enthier = None
        return cls(meta, enthier)

