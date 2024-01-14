import json
import os
from dataclasses import dataclass
from functools import partial
from itertools import chain


@dataclass
class Turn:
    speaker: str
    utterance: str


def get_conversation(data_point, first_prefix, second_prefix):
    """Extracts a conversation from a data point with speaker information.

    Args:
        data_point: A dictionary containing conversation data.
        first_prefix: Prefix for the first user
        second_prefix: Prefix for the second user

    Returns:
        str: The complete conversation as a string.
    """
    turns = [Turn(speaker=first_prefix if i % 2 == 0 else second_prefix, utterance=message["utterance"])
             for i, message in enumerate(data_point["turns"])]
    return "".join(turn.speaker + turn.utterance for turn in turns)


def load_json(file_path):
    with open(file_path) as in_file:
        return json.load(in_file)


def get_json_data(file_paths):
    data = []
    for path in file_paths:
        data.append(load_json(path))
        print(f'Read {path}')
    # flatten the data
    return list(chain.from_iterable(data))


def load_woz_dataset():
    os.chdir(os.path.dirname(__file__))
    parent_directory = os.path.abspath('../resources')
    root_train_path = os.path.join(parent_directory, 'data/MultiWoz2.2/train/')
    train_paths = [f'{root_train_path}dialogues_00{i}.json' if i < 10 else f'{root_train_path}dialogues_0{i}.json' for i
                   in
                   range(1, 18)]
    dev_paths = [os.path.join(parent_directory, 'data/MultiWoz2.2/dev/dialogues_001.json'),
                 os.path.join(parent_directory, 'data/MultiWoz2.2/dev/dialogues_002.json')]
    test_paths = [os.path.join(parent_directory, 'data/MultiWoz2.2/test/dialogues_001.json'),
                  os.path.join(parent_directory, 'data/MultiWoz2.2/test/dialogues_002.json')]

    train = get_json_data(train_paths)
    dev = get_json_data(dev_paths)
    test = get_json_data(test_paths)
    assert len(train) + len(dev) + len(test) == 10437, "Few datapoints not available"
    return train, dev, test


def get_woz_conversations(first_prefix=" Person 1: ", second_prefix=" Person 2: "):
    train, dev, test = load_woz_dataset()
    train_conv = get_all_conversation(train, first_prefix, second_prefix)
    dev_conv = get_all_conversation(dev, first_prefix, second_prefix)
    test_conv = get_all_conversation(test, first_prefix, second_prefix)

    return train_conv, dev_conv, test_conv


def get_all_conversation(data, first_prefix, second_prefix):
    fn_conversation = partial(get_conversation, first_prefix=first_prefix, second_prefix=second_prefix)
    return list(map(fn_conversation, data))


if __name__ == "__main__":
    train_conv, dev_conv, test_conv = get_woz_conversations()
    print(len(train_conv), len(dev_conv), len(test_conv))
    print('Sample Training example:\n', train_conv[0], '=' * 70)
    print('Sample Dev example:\n', dev_conv[0], '=' * 70)
    print('Sample test example:\n', test_conv[0], '=' * 70)
