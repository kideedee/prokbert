from datasets import load_from_disk
from prokbert import LCATokenizer
from prokbert.models import ProkBertForSequenceClassification
from prokbert.training_utils import *
from sklearn.metrics import precision_recall_fscore_support
from transformers import TrainerCallback, TrainerState, TrainerControl, EarlyStoppingCallback, DataCollatorWithPadding

from env_config import config
from phg_cls_log import log


class CustomLoggingCallback(TrainerCallback):
    def on_log(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, logs=None, **kwargs):
        log.info(f"Step {state.global_step}: {logs}")


class MemoryEfficientTrainer(Trainer):
    def prediction_step(
            self, model, inputs, prediction_loss_only, ignore_keys=None
    ):
        """
        Custom prediction step to reduce memory usage during evaluation.
        """
        has_labels = all(inputs.get(k) is not None for k in self.label_names)

        # Handle different signature for _prepare_inputs in transformers 4.28.0
        for k, v in inputs.items():
            if isinstance(v, torch.Tensor):
                inputs[k] = v.to(self.args.device)

        with torch.no_grad():
            if has_labels:
                with self.autocast_smart_context_manager():
                    outputs = model(**inputs)
                if prediction_loss_only:
                    loss = outputs[0].mean().detach()
                    return (loss, None, None)
                else:
                    loss = outputs[0].mean().detach()
                    logits = outputs[1]
                    labels = tuple(inputs.get(name).detach().cpu() for name in self.label_names)
                    if len(labels) == 1:
                        labels = labels[0]
            else:
                with self.autocast_smart_context_manager():
                    outputs = model(**inputs)
                loss = None
                if self.args.past_index >= 0:
                    logits = outputs[0]
                else:
                    logits = outputs
                labels = None

        # Move logits to CPU to save GPU memory
        if not prediction_loss_only:
            logits = logits.detach().cpu()

        return (loss, logits, labels)


def compute_metrics(pred):
    """Compute evaluation metrics"""
    labels = pred.label_ids
    logits = pred.predictions

    # Process in batches if predictions are large
    if isinstance(logits, np.ndarray) and logits.size > 1e6:  # Threshold for batch processing
        batch_size = 1000
        preds = []
        for i in range(0, len(logits), batch_size):
            batch_preds = np.argmax(logits[i:i + batch_size], axis=-1)
            preds.extend(batch_preds)
        preds = np.array(preds)
    else:
        preds = np.argmax(logits, axis=-1)

    # Compute confusion matrix
    cm = confusion_matrix(labels, preds)
    log.info(f"Confusion Matrix:\n{cm}")

    # Log detailed stats from confusion matrix
    tn, fp, fn, tp = cm.ravel()
    log.info(f"True Negatives: {tn}, False Positives: {fp}")
    log.info(f"False Negatives: {fn}, True Positives: {tp}")

    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average='binary')
    acc = accuracy_score(labels, preds)
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
    roc_auc = roc_auc_score(labels, torch.softmax(torch.from_numpy(logits), dim=1)[:, 1])
    gmean = np.sqrt(sensitivity * specificity) if (sensitivity > 0 and specificity > 0) else 0

    # Safely compute ROC AUC with batching for large datasets
    # try:
    #     if isinstance(logits, np.ndarray) and logits.shape[0] > 1e6:
    #         # Calculate ROC AUC in batches
    #         batch_size = 1000
    #         y_scores = []
    #         for i in range(0, len(logits), batch_size):
    #             y_scores.extend(logits[i:i + batch_size, 1])
    #         roc_auc = roc_auc_score(labels, np.array(y_scores))
    #     else:
    #         roc_auc = roc_auc_score(labels, logits[:, 1])
    # except:
    #     roc_auc = 0

    # Explicitly free memory
    del logits, preds

    result = {
        'accuracy': acc,
        'f1': f1,
        'precision': precision,
        'recall': recall,
        'specificity': specificity,
        'sensitivity': sensitivity,
        'roc_auc': roc_auc,
        'g_mean': gmean,  # Thêm G-Mean metric
        'tn': float(tn),
        'fp': float(fp),
        'fn': float(fn),
        'tp': float(tp)
    }

    # df = pd.DataFrame([result])
    # df.to_csv(result_path, mode='a', header=False, index=False)

    return result


if __name__ == '__main__':
    # Check if CUDA (GPU support) is available
    if not torch.cuda.is_available():
        raise SystemError('GPU device not found')
    else:
        device_name = torch.cuda.get_device_name(0)
        print(f'Found GPU at: {device_name}')
    num_cores = os.cpu_count()
    print(f'Number of available CPU cores: {num_cores}')

    model_path = 'neuralbioinfo/prokbert-mini-phage'

    for i in range(4, 5):
        if i != 4:
            continue

        if i == 0:
            min_size = 100
            max_size = 400
            batch_size = 64
        elif i == 1:
            min_size = 400
            max_size = 800
            batch_size = 48
        elif i == 2:
            min_size = 800
            max_size = 1200
            batch_size = 32
        elif i == 3:
            min_size = 1200
            max_size = 1800
            batch_size = 32
        elif i == 4:
            min_size = 50
            max_size = 100
            batch_size = 32
        elif i == 5:
            min_size = 100
            max_size = 200
            batch_size = 128
        elif i == 6:
            min_size = 200
            max_size = 300
            batch_size = 64
        elif i == 7:
            min_size = 300
            max_size = 400
            batch_size = 64
        else:
            raise ValueError

        group = f"{min_size}_{max_size}"
        for j in range(5):
            fold = j + 1
            if fold != 1:
                continue

            data_dir = os.path.join(config.PROKBERT_OUTPUT_DIR_TEST, f"{group}/fold_{fold}")

            tokenized_train = load_from_disk(os.path.join(data_dir, "processed_train_dataset"))
            tokenized_val = load_from_disk(os.path.join(data_dir, "processed_val_dataset"))

            model = ProkBertForSequenceClassification.from_pretrained(model_path, trust_remote_code=True)
            tokenizer = LCATokenizer.from_pretrained(model_path, trust_remote_code=True)
            tmp_output = "./prokbert_inference_output"
            os.makedirs(tmp_output, exist_ok=True)

            training_args = TrainingArguments(
                output_dir=tmp_output,
                do_train=False,
                do_eval=False,
                per_device_eval_batch_size=32,
                fp16=True,
                remove_unused_columns=False,

                num_train_epochs=3,

                # Evaluation and saving
                evaluation_strategy="epoch",
                save_strategy="epoch",
                save_total_limit=2,  # Keep best and last
                load_best_model_at_end=True,
            )

            trainer = MemoryEfficientTrainer(
                model=model,
                args=training_args,
                train_dataset=tokenized_train,
                eval_dataset=tokenized_val,
                compute_metrics=compute_metrics,
                data_collator=DataCollatorWithPadding(tokenizer=tokenizer),
                callbacks=[EarlyStoppingCallback(early_stopping_patience=2), CustomLoggingCallback]
            )

            trainer.train()
