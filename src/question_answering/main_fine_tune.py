import nltk
import evaluate
import numpy as np
from datasets import load_dataset
from transformers import T5Tokenizer, DataCollatorForSeq2Seq
from transformers import T5ForConditionalGeneration, Seq2SeqTrainingArguments, Seq2SeqTrainer

from dataclasses import dataclass
from util.python_util import get_latest_file_name, setup_output_loggin


@dataclass
class TrainingArgs:
    eval_batch_size: int = 4
    learning_rate: float = 1e-4,
    num_epochs: int = 3,
    output_dir: str = "./results"
    save_total_limit: int = 3,
    train_batch_size: int = 8,
    weight_decay: float = 1e-2

    def create_training_arguments(self):
        return Seq2SeqTrainingArguments(
            evaluation_strategy="epoch",
            learning_rate=self.learning_rate,
            num_train_epochs=self.num_epochs,
            output_dir=self.output_dir,
            per_device_eval_batch_size=self.eval_batch_size,
            per_device_train_batch_size=self.train_batch_size,
            predict_with_generate=True,
            push_to_hub=False,
            save_total_limit=self.save_total_limit,
            weight_decay=self.weight_decay,
        )


def load_model_with_tokenizer(model_name):
    tokenizer = T5Tokenizer.from_pretrained(model_name)
    model = T5ForConditionalGeneration.from_pretrained(model_name)
    return tokenizer, model


def get_qa_dataset(qa_dataset_name="squad"):
    squad = load_dataset(qa_dataset_name)
    squad_qa = squad["train"].train_test_split(test_size=0.25)
    return squad_qa, squad["test"]


def preprocess_entry(tokenizer, qa_entry, prefix, max_question_length=128, max_answer_length=512):
    inputs = [prefix + question_text for question_text in qa_entry["question"]]
    model_inputs = tokenizer(inputs, max_question_length, truncation=True)
    model_labels = tokenizer(text_target=qa_entry["answer"],
                             max_length=max_answer_length,
                             truncation=True)
    model_inputs["labels"] = model_labels["input_ids"]
    return model_inputs


def compute_metrics(eval_preds, metric_type="rouge"):
    metric = evaluate.load(metric_type)
    preds, labels = eval_preds

    # decode preds and labels
    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

    # rougeLSum expects newline after each sentence
    decoded_preds = ["\n".join(nltk.sent_tokenize(pred.strip())) for pred in decoded_preds]
    decoded_labels = ["\n".join(nltk.sent_tokenize(label.strip())) for label in decoded_labels]

    result = metric.compute(predictions=decoded_preds, references=decoded_labels, use_stemmer=True)

    return result


if __name__ == "__main__":
    # download punkt once only
    try:
        nltk.data.find('tokenizers/punkt')
    except LookupError:
        print('Downloading Punkt tokenizer')
        nltk.download("punkt", quiet=True)

    print('Setting up Log files for capturing output and error')
    log_filename = get_latest_file_name(__file__, "log")
    err_filename = get_latest_file_name(__file__, "error")
    setup_output_loggin(err_filename, log_filename)
    print('Loading model .....')
    tokenizer, model = load_model_with_tokenizer("google/flan-t5-base")
    data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model)
    print('Obtaining Dataset ....')
    qa_dataset = get_qa_dataset()
    tokenized_dataset = qa_dataset.map(preprocess_entry, batched=True)
    training_args = TrainingArgs.create_training_arguments()
    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset["train"],
        eval_dataset=tokenized_dataset["test"],
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics
    )
    print('Everything ready!!! Preparing to train')
    trainer.train()
