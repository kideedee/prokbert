import os

import pandas as pd
from datasets import Dataset
from prokbert import LCATokenizer

from env_config import config
from phg_cls_log import log


def tokenize_function(examples, tokenizer, max_length=512):
    return tokenizer(
        examples["sequence"],
        return_tensors="pt",
        padding="longest",
        truncation=True,
        max_length=max_length,
    )


if __name__ == '__main__':
    for i in range(5):
        fold = i + 1
        max_length_tokenizer = 512
        # if fold != 1:
        #     continue

        for j in range(4):
            if j == 0:
                min_size = 100
                max_size = 400
                # max_length_tokenizer = int(max_size * 0.25)
            elif j == 1:
                min_size = 400
                max_size = 800
                # max_length_tokenizer = int(max_size * 0.25)
            elif j == 2:
                min_size = 800
                max_size = 1200
                # max_length_tokenizer = int(max_size * 0.25)
            elif j == 3:
                min_size = 800
                max_size = 1200
                # max_length_tokenizer = 512
            elif j == 4:
                min_size = 50
                max_size = 100
                # max_length_tokenizer = int(max_size * 0.25)
            elif j == 5:
                min_size = 100
                max_size = 200
                # max_length_tokenizer = int(max_size * 0.25)
            elif j == 6:
                min_size = 200
                max_size = 300
                # max_length_tokenizer = int(max_size * 0.25)
            elif j == 7:
                min_size = 300
                max_size = 400
                # max_length_tokenizer = 512
            else:
                raise ValueError

            # if j != 4:
            #     continue

            group = f"{min_size}_{max_size}"
            train_file = f"{config.CONTIG_OUTPUT_DATA_DIR_FIX_BUG}/{group}/fold_{fold}/train/data.csv"
            valid_file = f"{config.CONTIG_OUTPUT_DATA_DIR_FIX_BUG}/{group}/fold_{fold}/test/data.csv"
            log.info(f"Fold: {fold}, group: {group}")
            log.info(f"train_file: {train_file}")
            log.info(f"valid_file: {valid_file}")

            output_dir = os.path.join(config.PROKBERT_OUTPUT_DIR, f"{group}/fold_{fold}")
            os.makedirs(output_dir, exist_ok=True)
            output_prepared_train_dataset = os.path.join(output_dir, f"processed_train_dataset")
            output_prepared_val_dataset = os.path.join(output_dir, f"processed_val_dataset")
            output_tokenizer = os.path.join(output_dir, "tokenizer")

            model_path = 'neuralbioinfo/PhaStyle-mini'
            tokenizer = LCATokenizer.from_pretrained(model_path, trust_remote_code=True)

            processed_train_df = pd.read_csv(train_file)[["sequence", "target"]]
            processed_val_df = pd.read_csv(valid_file)[["sequence", "target"]]
            # processed_train_df = processed_train_df.sample(10000)
            # processed_val_df = processed_val_df.sample(10000)

            # Convert dataframes to Hugging Face datasets
            train_dataset = Dataset.from_pandas(processed_train_df)
            valid_dataset = Dataset.from_pandas(processed_val_df)

            tokenized_train = train_dataset.map(
                lambda examples: tokenize_function(examples, tokenizer, max_length_tokenizer),
                batched=True
            )
            tokenized_valid = valid_dataset.map(
                lambda examples: tokenize_function(examples, tokenizer, max_length_tokenizer),
                batched=True
            )

            tokenized_train.set_format(type="torch", columns=["input_ids", "attention_mask", "target"])
            tokenized_valid.set_format(type="torch", columns=["input_ids", "attention_mask", "target"])

            tokenized_train = tokenized_train.rename_column("target", "labels")
            tokenized_valid = tokenized_valid.rename_column("target", "labels")

            tokenized_train.save_to_disk(output_prepared_train_dataset)
            tokenized_valid.save_to_disk(output_prepared_val_dataset)

            tokenizer.save_pretrained(output_tokenizer)

            log.info("Data preparation completed!")
