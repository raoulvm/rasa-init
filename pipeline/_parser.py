from typing import Any, Dict, List, Text, Union
from rasa.shared.utils.io import read_yaml_file
from rasa.shared.utils.validation import validate_yaml_schema
import re
import logging


# CONSTANTS
TARGET_VALUE_KEY = "value"
EXAMPLES = "examples"
EXAMPLE_TEXT = "text"
EXAMPLE_REF = "ref"
EXAMPLE_COMPOSITE = "composite"
EXAMPLE_TEXT_ALT_SPELLING = "alternatives"

DONT_CREATE_ENTITY = "_NO_ENTITY_"

ANY_SOURCE_ENTITY_KEY = "_ANY_"

logger = logging.getLogger(__file__)


def _walkthrough(dict_of_lists: dict) -> list:
    """List generator for GridSearch-like parameter searches
    Given a dict of lists it returns a list of dicts.


    >>>walkthrough({'a':[1],'b':5, 'c':[3,6],'d':[1,2,3,4]})
    [1] Iterating 8 item combinations
        [{'a': 1, 'b': 5, 'c': 3, 'd': 1},
        {'a': 1, 'b': 5, 'c': 3, 'd': 2},
        {'a': 1, 'b': 5, 'c': 3, 'd': 3},
        {'a': 1, 'b': 5, 'c': 3, 'd': 4},
        {'a': 1, 'b': 5, 'c': 6, 'd': 1},
        {'a': 1, 'b': 5, 'c': 6, 'd': 2},
        {'a': 1, 'b': 5, 'c': 6, 'd': 3},
        {'a': 1, 'b': 5, 'c': 6, 'd': 4}]

    Args:
        dict_of_lists (dict): The dictionary with lists as values. (Scalars will be converted)

    Returns:
        list: A list of dictionaries. For each dict another combination from the lists of the original dict is used.
    """

    def crossmul(dic_of_lists: dict) -> int:
        res = 1
        for tup in dic_of_lists.items():
            if isinstance(tup[1], list):
                res *= len(tup[1])
        return res

    def __iterate(the_list: list, __cur=[]):
        __debug = False

        def _print(*args, **kwargs):
            if __debug:
                print(*args, **kwargs)

        par = the_list[0][0]
        _print(__cur)  # , end=',')
        pvm = []
        if not isinstance(the_list[0][1], list):  # avoid single items breaking th e tool
            the_list[0][1] = [the_list[0][1]]

        if len(the_list) > 1:  # there are more parameters to come
            for pv in the_list[0][1]:
                pvm.extend(__iterate(the_list[1:], __cur + [(par, pv)]))  # go through the others
        else:
            # lowest level
            for pv in the_list[0][1]:
                _print("cur= ", __cur)
                _print("appending", __cur + [(par, pv)])
                pvm.append(__cur + [(par, pv)])
        _print("returning", pvm)
        return pvm

    # logger.debug(f"Iterating {crossmul(dict_of_lists)} item combinations")
    tups = [[k, v] for (k, v) in dict_of_lists.items()]
    return [dict(a) for a in __iterate(tups)]


def topdownparser(data: Dict[str, list]) -> dict:
    """Parses a top-down type entity hierarchy dictionary.
    The format is 
    ```
    (target-entity):
      - value: (target-value)  [value key is optional]
        entities_only:         [entities_only key is optional]
        - entity-name1 ...
        examples:
        - text: (source-string) ...
        - ref: (another target entity) ...
        - composite: (text {target entity name} text)...
    ```

    Args:
        data (Dict[str,dict]): top down hierarchy loaded from file(s)

    Returns:
        dict: bottom up hierarchy for fast replacements of values
    """
    target_mapping = {}
    alternatives_mapping = {}

    def create_text_entry(
        ent_source_value: Text,
        ent_target: Union[str, None],
        val_target: Any = True,
        # ent_restriction: str = ANY_SOURCE_ENTITY_KEY,
    ) -> None:
        if not ent_target:
            return
        if not target_mapping.get(ent_source_value):
            target_mapping[ent_source_value] = {}
        target_mapping[ent_source_value][ent_target] = val_target

    def collect_composite(keyword: str) -> List[str]:
        """Returns example texts as list of strings, starting with the given keyword.
        Local target values are ignored.
        'ref' and 'composite' references are followed recursively.

        Args:
            keyword (str): The entity value to look for examples

        Returns:
            List[str]: A list of strings, containing each text example found under keyword.
        """
        # collect all strings from the examples as list
        # if example is ref, include those as well
        # if example is composite, recurse
        start: list = data[keyword]
        # ignore value
        # take only example keys
        results = []
        for entry in start:
            for empl in entry.get(EXAMPLES):
                text = empl.get(EXAMPLE_TEXT)
                ref = empl.get(EXAMPLE_REF)
                if text:
                    results.append(text)
                    alts = empl.get(EXAMPLE_TEXT_ALT_SPELLING)
                    if alts:
                        results.extend(alts)
                if ref:
                    results.extend(collect_composite(ref))
                comp = empl.get(EXAMPLE_COMPOSITE)
                if comp:
                    results.extend(process_composite(comp))
        return results

    def process_composite(composite_text: str) -> List[str]:
        """Searches for all valid combinations that are possible given a
        f-string-like text, such as "{handy}-vertrag". It will look for the 
        handy-keyword and return all the text examples below, plus following
        all the 'ref' examples, plus also recursively resolving all other
        'composite' examples themselves.

        Args:
            composite_text (str): f-string-like text, such as "static {reference}"

        Returns:
            List[str]: List of composed text strings with all placeholders replaced
        """
        # take one f-string and return a list of populated strings
        compounds = re.findall(r"{(.*?)}", composite_text)
        # logger.debug(compounds)
        replacers = {}
        # collect the lists of replacements
        for placeholder in compounds:
            replacers[placeholder] = collect_composite(placeholder)
        # iterate the combinations
        results = []
        for one_combination in _walkthrough(replacers):
            # logger.debug(composite_text.format(**one_combination))
            results.append(composite_text.format(**one_combination))
        return results

    def parse_one(
        target_entity: Union[str, None], e_data_lst: list, parent_target_value: Any = None
    ):
        # logger.debug(f"\nentity {target_entity}")
        # logger.debug(f"list {e_data_lst}")
        if DONT_CREATE_ENTITY in e_data_lst:
            target_entity = None
            e_data_lst.remove(DONT_CREATE_ENTITY)
        for e_dict in e_data_lst:
            # if it is a string ignor ethe entry
            # logger.debug(f"dict {e_dict}")
            # the dictionary is supposed to have up to three keys:
            # value (optional) - string
            # entities_only (optional) - list
            # examples - list

            targ_val = parent_target_value or e_dict.get(TARGET_VALUE_KEY)

            # for ent_restriction in ent_restriction_list:

            for example in e_dict.get(EXAMPLES) or []:
                # if text
                # logger.debug(f"example {example}")
                text = example.get(EXAMPLE_TEXT)
                ref = example.get(EXAMPLE_REF)
                composite: str = example.get(EXAMPLE_COMPOSITE)

                if text:
                    # logger.debug(
                    #     f"create_text_entry(ent_source_value={text},ent_target={target_entity},val_target={targ_val or text},)"
                    # )
                    create_text_entry(
                        ent_source_value=text, ent_target=target_entity, val_target=targ_val or text
                    )
                    alts = example.get(EXAMPLE_TEXT_ALT_SPELLING, [])  # alternative spellings
                    for alternative in alts:
                        alternatives_mapping.update({alternative: text})

                if ref:
                    parse_one(
                        target_entity=target_entity,
                        e_data_lst=data[ref],
                        parent_target_value=targ_val,
                    )

                if composite:
                    # restrictions: composites do only pull all
                    # texts (also recursive) from mentioned
                    # entity-values
                    cmp_list = process_composite(composite)
                    for word in cmp_list:
                        # create an entry per returned string
                        create_text_entry(
                            ent_source_value=word,
                            ent_target=target_entity,
                            val_target=targ_val or text or word,
                        )

    for target_entity, e_data_lst in data.items():
        parse_one(target_entity, e_data_lst)
    return {"entities": target_mapping, "alternatives": alternatives_mapping}

