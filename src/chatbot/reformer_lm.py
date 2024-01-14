import trax.models
from trax import layers as tl


class ReformerWrapper:
    """
    Wrapper class to use Trax's ReformerLM
    """

    def __init__(self, vocab_size: int, d_model: int, d_hidden: int, num_layers: int, mode: str,
                 attention_type: trax.layers = tl.SelfAttention):
        """

        :param vocab_size: size of the vocabulary
        :param d_model: depth of each half of two part features
        :param d_hidden: depth of feed forward layer
        :param num_layers: number of decoder layers
        :param mode: One of 'train', 'eval' or 'predict'
        :param attention_type: attention class to use
        """
        self._model = trax.models.reformer.ReformerLM(vocab_size=vocab_size, d_model=d_model, d_ff=d_hidden,
                                                      n_layers=num_layers, mode=mode, attention_type=attention_type)

    @property
    def model(self):
        return self._model
