
from transformers import AutoTokenizer, AutoModelWithLMHead, Trainer, TrainingArguments, DataCollatorForLanguageModeling, TextDataset

tokenizer = AutoTokenizer.from_pretrained('gpt2')


def load_dataset(train_path='data/train/train_mod.txt', test_path='data/test/test_mod.txt', tokenizer=AutoTokenizer.from_pretrained('gpt2')) -> tuple[TextDataset, TextDataset, DataCollatorForLanguageModeling]:
    train_dataset = TextDataset(
        tokenizer=tokenizer,
        file_path=train_path,
        block_size=128)
    test_dataset = TextDataset(
        tokenizer=tokenizer,
        file_path=test_path,
        block_size=128)
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer, mlm=False,
    )
    return train_dataset, test_dataset, data_collator


def train_setup(
    output_dir="storage/gpt2-happy",
    overwrite_output_dir=True,
    num_train_epochs=10,
    per_gpu_train_batch_size=12,
    per_gpu_eval_batch_size=24,
    logging_steps=500,
    save_steps=500,
    warmup_steps=500,
) -> Trainer:
    model = AutoModelWithLMHead.from_pretrained('gpt2')

    train_dataset, test_dataset, data_collator = load_dataset()

    training_args = TrainingArguments(
        output_dir=output_dir,  # The output directory
        overwrite_output_dir=overwrite_output_dir,  # overwrite the content of the output directory
        num_train_epochs=num_train_epochs,  # number of training epochs
        per_gpu_train_batch_size=per_gpu_train_batch_size,  # batch size for training
        per_gpu_eval_batch_size=per_gpu_eval_batch_size,  # batch size for evaluation
        logging_steps=logging_steps,  # Number of update steps between two evaluations.
        save_steps=save_steps,  # after # steps model is saved
        warmup_steps=warmup_steps,  # number of warmup steps for learning rate scheduler
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        data_collator=data_collator,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,

    )

    return trainer
