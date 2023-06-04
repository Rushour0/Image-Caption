from gpt2_happy import train_setup, tokenizer


def main():
    trainer = train_setup(per_gpu_train_batch_size=24, per_gpu_eval_batch_size=48)

    trainer.train()
    trainer.save_model("/storage/gpt2-happy-v1")

    tokenizer.save_pretrained("/storage/gpt2-happy-v1")


if __name__ == "__main__":
    main()
