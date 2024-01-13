import trax
from trax.supervised import training
from trax import layers as tl
import random
import os

from chatbot.read_woz_dataset import get_woz_conversations


def training_function(model, train_generator, eval_generator, **kwargs):
    warmup_steps = kwargs.get("warmup_steps", 1000)
    lr_schedule = trax.lr.warmup_and_rsqrt_decay(n_warmup_steps=warmup_steps, max_value=0.01)
    loss_layer = kwargs.get("loss", tl.CrossEntropyLoss())
    optimizer = kwargs.get("optimizer", trax.optimizers.Adam(0.01))
    checkpoint_steps = kwargs.get("checkpoint_steps", 10)
    train_task = training.TrainTask(
        labeled_data=train_generator,
        loss_layer=loss_layer,
        optimizer=optimizer,
        lr_schedule=lr_schedule,
        n_steps_per_checkpoint=checkpoint_steps
    )
    metrics = kwargs.get("metrics", [tl.CrossEntropyLoss(), tl.Accuracy()])
    eval_task = training.EvalTask(
        labeled_data=eval_generator,
        metrics=metrics
    )

    output_dir = kwargs.get("output_dir", os.path.abspath('./model_output/'))
    return training.Loop(model, train_task, eval_tasks=[eval_task], output_dir=output_dir)


def train_model(model, train_generator, eval_generator, epochs, **kwargs):
    loop = training_function(model, train_generator, eval_generator, **kwargs)
    loop.run(epochs)
    return loop


def generate_data_pipeline(vocab_dir, vocab_file, max_seq_length=2048):
    return trax.data.Serial(
        trax.data.Shuffle(),  # shuffle data to randomize the stream
        trax.data.Tokenize(
            vocab_dir=vocab_dir,
            vocab_file=vocab_file
        ),
        trax.data.FilterByLength(max_seq_length),  # filter out long sequences
        trax.data.BucketByLength(boundaries=[128, 256, 512, 1024],
                                 batch_sizes=[16, 8, 4, 2, 1]),
        trax.data.AddLossWeights(id_to_mask=0)  # add loss weights except padding tokens
    )


def get_streaming_data(data_pipeline, dataset):
    def stream(data):
        while True:
            d = random.choice(data)
            yield d, d

    return data_pipeline(stream(dataset))


def get_data_generator(data_pipeline):
    train, dev, test = get_woz_conversations()
    new_train = train + dev
    train_generator = get_streaming_data(data_pipeline, new_train)
    eval_generator = get_streaming_data(data_pipeline, test)
    return train_generator, eval_generator
