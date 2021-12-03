from rasa.nlu.tokenizers.whitespace_tokenizer import WhitespaceTokenizer
from rasa.nlu.featurizers.sparse_featurizer.lexical_syntactic_featurizer import LexicalSyntacticFeaturizer as LSF
from rasa.shared.nlu.constants import TEXT
from pipeline.lexical_syntactic_featurizer import Patched as LSF_Patched
from rasa.shared.nlu.training_data.message import Message
from rasa.shared.nlu.training_data.training_data import TrainingData
import numpy as np

training_tokens = "a A ha HA Ha"
text1 = "A HA"
text2 = "a ha"

params =         {
            "features": [
                ["BOS"],
                ["prefix2", "suffix5"],
                ["EOS"],
            ]
        }



for featurizer_class in (LSF_Patched, LSF):
    featurizer = featurizer_class(component_config=params)

    print(featurizer.unique_name)
    train_message = Message(data={TEXT: training_tokens})
    test1_message = Message(data={TEXT: text1})
    test2_message = Message(data={TEXT: text2})
    WhitespaceTokenizer().process(test1_message)
    WhitespaceTokenizer().process(test2_message)
    WhitespaceTokenizer().process(train_message)


    featurizer.train(TrainingData([train_message]))
    print(featurizer.number_of_features)
    featurizer.process(test1_message)
    featurizer.process(test2_message)

    seq_vec, sen_vec = test1_message.get_sparse_features(TEXT, [])
    if seq_vec:
        seq_vec1 = seq_vec.features
    if sen_vec:
        sen_vec1 = sen_vec.features
    assert sen_vec is None
    seq_vec, sen_vec = test2_message.get_sparse_features(TEXT, [])
    if seq_vec:
        seq_vec2 = seq_vec.features
    if sen_vec:
        sen_vec2 = sen_vec.features    
    assert sen_vec is None
    print("test1")
    print(seq_vec1)
    print("test2")
    print(seq_vec2)
    assert np.all(seq_vec1.toarray()==seq_vec2.toarray())
    print("Features are equal\n\n")


