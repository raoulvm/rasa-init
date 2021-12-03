from typing import Any
from rasa.nlu.featurizers.sparse_featurizer.lexical_syntactic_featurizer import LexicalSyntacticFeaturizer
from rasa.nlu.tokenizers.spacy_tokenizer import POS_TAG_KEY
from rasa.shared.nlu.training_data.message import Message

class Patched(LexicalSyntacticFeaturizer):
    """A patched version of the LexicalSyntacticFeaturizer

    """
    function_dict = {
                "low": lambda token: token.text.islower(),
                "title": lambda token: token.text.istitle(),
                "prefix5": lambda token: token.text[:5].lower(),
                "prefix2": lambda token: token.text[:2].lower(),
                "suffix5": lambda token: (token.text[-5:].lower() if not print(token.text[-5:].lower()) else 0 ),
                "suffix3": lambda token: token.text[-3:].lower(),
                "suffix2": lambda token: token.text[-2:].lower(),
                "suffix1": lambda token: token.text[-1:].lower(),
                "pos": lambda token: token.data.get(POS_TAG_KEY)
                if POS_TAG_KEY in token.data
                else None,
                "pos2": lambda token: token.data.get(POS_TAG_KEY, [])[:2]
                if "pos" in token.data
                else None,
                "upper": lambda token: token.text.isupper(),
                "digit": lambda token: token.text.isdigit(),
            }
    def process(self, message: Message, **kwargs: Any) -> None:
        print("start process")
        self._create_sparse_features(message)



