import click
from transformers import T5ForConditionalGeneration, T5Tokenizer


def qa_session(finetuned_model, tokenizer, prefix):
    counter = 0
    while True:
        question = input("Enter the question:")
        if question in {"quit", "Quit", "exit", "Exit"} or counter > 10:
            print('Exiting the session!!!')
            break
        qa_input = f"{prefix} {question}"
        qa_input = tokenizer(qa_input, return_tensors="pt")
        model_output = finetuned_model.generate(**qa_input)
        answer = tokenizer.decode(model_output[0])
        print(answer)
        print("=" * 80)
        counter += 1


@click.command()
@click.argument("checkpoint_dir", type=str)
@click.option("--prefix", type=str, default="Please Answer the question:", help="Prefix used during fine-tuning")
def main(checkpoint_dir, prefix):
    """
    CHECKPOINT_DIR full path to directory containing model fine-tuned checkpoint

    """
    finetuned_model = T5ForConditionalGeneration.from_pretrained(checkpoint_dir)
    tokenizer = T5Tokenizer.from_pretrained(checkpoint_dir)

    qa_session(finetuned_model, tokenizer, prefix)


if __name__ == "__main__":
    main()
