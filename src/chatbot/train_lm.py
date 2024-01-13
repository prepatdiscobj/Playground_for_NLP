from reformer_lm import ReformerWrapper
import click
import util.trax_util as trax_util
import sys


def setup_output_loggin():
    stdout_file = open("output.txt", "w")
    stderr_file = open("err.txt", "w")
    sys.stdout = stdout_file
    sys.stderr = stderr_file

    return stdout_file, stderr_file


@click.command()
@click.argument("vocab_dir", type=str)
@click.argument("vocab_file", type=str)
@click.option("--vocab_size", type=int, default=33000, help="The size of model vocabulary")
@click.option("--d_model", type=int, default=512, help="Depth of each half of two part features")
@click.option("--d_hidden", type=int, default=2048, help="Depth of feed forward layer", )
@click.option("--n_layers", type=int, default=3, help="Number of Reformer layers")
@click.option("--mode", type=str, default="train", help="Mode to train, evaluate ore predict")
@click.option("--training_steps", type=int, default=10, help="Number of training steps for model")
@click.option("--max_seq_length", type=int, default=2048, help="Maximum sequence length supported")
def main(vocab_dir, vocab_file, vocab_size, d_model, d_hidden, n_layers, mode, training_steps, max_seq_length):
    """
    VOCAB_DIR full path  to the vocabulary directory

    VOCAB_FILE file name of vocabulary
    """
    out, err = setup_output_loggin()  #
    data_pipeline = trax_util.generate_data_pipeline(vocab_dir, vocab_file, max_seq_length)
    train_generator, eval_generator = trax_util.get_data_generator(data_pipeline)
    model_object = ReformerWrapper(vocab_size=vocab_size,
                                   d_model=d_model,
                                   d_hidden=d_hidden,
                                   num_layers=n_layers,
                                   mode=mode)
    print(model_object.model)
    if mode == "train":
        loop = trax_util.train_model(model_object.model, train_generator, eval_generator, training_steps, warmup_steps=2)
    else:
        print('Only Training Code Added')

    out.close()
    err.close()


if __name__ == "__main__":
    main()
