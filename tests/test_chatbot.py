import unittest
import os

import trax.data

from util.trax_util import setup_data_pipelines
import random


class TestChatBot(unittest.TestCase):
    def test_setup_data_pipelines(self):
        random.seed(10)
        vocab_dir = os.path.abspath("../src/resources/data/vocabs")
        vocab_file = "en_32k.subword"
        train_gen, eval_gen = setup_data_pipelines(vocab_dir, vocab_file, 2048)

        inp, _, _ = next(train_gen)
        # print(inp.shape)
        self.assertEqual((4, 512), inp.shape, msg=f"Shape Mismatch, Found {inp.shape}")
        print(trax.data.detokenize(inp[0], vocab_dir=vocab_dir, vocab_file=vocab_file))


if __name__ == "__main__":
    unittest.main()
