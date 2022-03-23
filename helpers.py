import numpy as np
import collections
from collections import defaultdict, OrderedDict
from transformers import Trainer, EvalPrediction, TrainerCallback
from transformers.trainer_utils import PredictionOutput
from typing import Tuple
from tqdm.auto import tqdm
from selection.selection_utils import log_training_dynamics
from torch.utils.data import DataLoader, Dataset, RandomSampler, SequentialSampler
import torch
from transformers.trainer_pt_utils import IterableDatasetShard

QA_MAX_ANSWER_LENGTH = 30


# This function preprocesses an NLI dataset, tokenizing premises and hypotheses.
def prepare_dataset_nli(examples, tokenizer, max_seq_length=None):
    max_seq_length = tokenizer.model_max_length if max_seq_length is None else max_seq_length

    tokenized_examples = tokenizer(
        examples['premise'],
        examples['hypothesis'],
        truncation=True,
        max_length=max_seq_length,
        padding='max_length'
    )

    tokenized_examples['label'] = examples['label']
    return tokenized_examples


# This function computes sentence-classification accuracy.
# Functions with signatures like this one work as the "compute_metrics" argument of transformers.Trainer.
def compute_accuracy(eval_preds: EvalPrediction):
    return {
        'accuracy': (np.argmax(
            eval_preds.predictions,
            axis=1) == eval_preds.label_ids).astype(
            np.float32).mean().item()
    }


# This function preprocesses a question answering dataset, tokenizing the question and context text
# and finding the right offsets for the answer spans in the tokenized context (to use as labels).
# Adapted from https://github.com/huggingface/transformers/blob/master/examples/pytorch/question-answering/run_qa.py
def prepare_train_dataset_qa(examples, tokenizer, max_seq_length=None):
    questions = [q.lstrip() for q in examples["question"]]
    max_seq_length = tokenizer.model_max_length
    # tokenize both questions and the corresponding context
    # if the context length is longer than max_length, we split it to several
    # chunks of max_length
    tokenized_examples = tokenizer(
        questions,
        examples["context"],
        truncation="only_second",
        max_length=max_seq_length,
        stride=min(max_seq_length // 2, 128),
        return_overflowing_tokens=True,
        return_offsets_mapping=True,
        padding="max_length"
    )

    # Since one example might give us several features if it has a long context,
    # we need a map from a feature to its corresponding example.
    sample_mapping = tokenized_examples.pop("overflow_to_sample_mapping")
    # The offset mappings will give us a map from token to character position
    # in the original context. This will help us compute the start_positions
    # and end_positions to get the final answer string.
    offset_mapping = tokenized_examples.pop("offset_mapping")

    tokenized_examples["start_positions"] = []
    tokenized_examples["end_positions"] = []
    tokenized_examples["example_id"] = []

    for i, offsets in enumerate(offset_mapping):
        input_ids = tokenized_examples["input_ids"][i]
        # We will label features not containing the answer the index of the CLS token.
        cls_index = input_ids.index(tokenizer.cls_token_id)
        sequence_ids = tokenized_examples.sequence_ids(i)
        # from the feature idx to sample idx
        sample_index = sample_mapping[i]
        # get the answer for a feature
        answers = examples["answers"][sample_index]

        tokenized_examples["example_id"].append(examples["id"][sample_index])

        if len(answers["answer_start"]) == 0:
            tokenized_examples["start_positions"].append(cls_index)
            tokenized_examples["end_positions"].append(cls_index)
        else:
            # Start/end character index of the answer in the text.
            start_char = answers["answer_start"][0]
            end_char = start_char + len(answers["text"][0])

            # Start token index of the current span in the text.
            token_start_index = 0
            while sequence_ids[token_start_index] != 1:
                token_start_index += 1

            # End token index of the current span in the text.
            token_end_index = len(input_ids) - 1
            while sequence_ids[token_end_index] != 1:
                token_end_index -= 1

            # Detect if the answer is out of the span (in which case this feature is labeled with the CLS index).
            if not (offsets[token_start_index][0] <= start_char and
                    offsets[token_end_index][1] >= end_char):
                tokenized_examples["start_positions"].append(cls_index)
                tokenized_examples["end_positions"].append(cls_index)
            else:
                # Otherwise move the token_start_index and token_end_index to the two ends of the answer.
                # Note: we could go after the last offset if the answer is the last word (edge case).
                while token_start_index < len(offsets) and \
                        offsets[token_start_index][0] <= start_char:
                    token_start_index += 1
                tokenized_examples["start_positions"].append(
                    token_start_index - 1)
                while offsets[token_end_index][1] >= end_char:
                    token_end_index -= 1
                tokenized_examples["end_positions"].append(token_end_index + 1)

    return tokenized_examples


def prepare_validation_dataset_qa(examples, tokenizer):
    questions = [q.lstrip() for q in examples["question"]]
    max_seq_length = tokenizer.model_max_length
    tokenized_examples = tokenizer(
        questions,
        examples["context"],
        truncation="only_second",
        max_length=max_seq_length,
        stride=min(max_seq_length // 2, 128),
        return_overflowing_tokens=True,
        return_offsets_mapping=True,
        padding="max_length"
    )

    # Since one example might give us several features if it has a long context, we need a map from a feature to
    # its corresponding example. This key gives us just that.
    sample_mapping = tokenized_examples.pop("overflow_to_sample_mapping")

    # For evaluation, we will need to convert our predictions to substrings of the context, so we keep the
    # corresponding example_id and we will store the offset mappings.
    tokenized_examples["example_id"] = []

    for i in range(len(tokenized_examples["input_ids"])):
        # Grab the sequence corresponding to that example (to know what is the context and what is the question).
        sequence_ids = tokenized_examples.sequence_ids(i)
        context_index = 1

        # One example can give several spans, this is the index of the example containing this span of text.
        sample_index = sample_mapping[i]
        tokenized_examples["example_id"].append(examples["id"][sample_index])

        # Set to None the offset_mapping that are not part of the context so it's easy to determine if a token
        # position is part of the context or not.
        tokenized_examples["offset_mapping"][i] = [
            (o if sequence_ids[k] == context_index else None)
            for k, o in enumerate(tokenized_examples["offset_mapping"][i])
        ]

    return tokenized_examples


# This function uses start and end position scores predicted by a question answering model to
# select and extract the predicted answer span from the context.
# Adapted from https://github.com/huggingface/transformers/blob/master/examples/pytorch/question-answering/utils_qa.py
def postprocess_qa_predictions(examples,
                               features,
                               predictions: Tuple[np.ndarray, np.ndarray],
                               n_best_size: int = 20):
    if len(predictions) != 2:
        raise ValueError(
            "`predictions` should be a tuple with two elements (start_logits, end_logits).")
    all_start_logits, all_end_logits = predictions

    if len(predictions[0]) != len(features):
        raise ValueError(
            f"Got {len(predictions[0])} predictions and {len(features)} features.")

    # Build a map example to its corresponding features.
    example_id_to_index = {k: i for i, k in enumerate(examples["id"])}
    features_per_example = collections.defaultdict(list)
    for i, feature in enumerate(features):
        features_per_example[
            example_id_to_index[feature["example_id"]]].append(i)

    # The dictionaries we have to fill.
    all_predictions = collections.OrderedDict()

    # Let's loop over all the examples!
    for example_index, example in enumerate(tqdm(examples)):
        # Those are the indices of the features associated to the current example.
        feature_indices = features_per_example[example_index]

        prelim_predictions = []

        # Looping through all the features associated to the current example.
        for feature_index in feature_indices:
            # We grab the predictions of the model for this feature.
            start_logits = all_start_logits[feature_index]
            end_logits = all_end_logits[feature_index]
            # This is what will allow us to map some the positions in our logits
            # to span of texts in the original context.
            offset_mapping = features[feature_index]["offset_mapping"]

            # Go through all possibilities for the `n_best_size` greater start and end logits.
            start_indexes = np.argsort(start_logits)[
                            -1: -n_best_size - 1: -1].tolist()
            end_indexes = np.argsort(end_logits)[
                          -1: -n_best_size - 1: -1].tolist()
            for start_index in start_indexes:
                for end_index in end_indexes:
                    # Don't consider out-of-scope answers, either because the indices are out of bounds or correspond
                    # to part of the input_ids that are not in the context.
                    if (
                            start_index >= len(offset_mapping)
                            or end_index >= len(offset_mapping)
                            or offset_mapping[start_index] is None
                            or offset_mapping[end_index] is None
                    ):
                        continue
                    # Don't consider answers with a length that is either < 0 or > max_answer_length.
                    if end_index < start_index or \
                            end_index - start_index + 1 > QA_MAX_ANSWER_LENGTH:
                        continue

                    prelim_predictions.append(
                        {
                            "offsets": (offset_mapping[start_index][0],
                                        offset_mapping[end_index][1]),
                            "score": start_logits[start_index] +
                                     end_logits[end_index],
                            "start_logit": start_logits[start_index],
                            "end_logit": end_logits[end_index],
                        }
                    )

        # Only keep the best `n_best_size` predictions.
        predictions = sorted(prelim_predictions, key=lambda x: x["score"],
                             reverse=True)[:n_best_size]

        # Use the offsets to gather the answer text in the original context.
        context = example["context"]
        for pred in predictions:
            offsets = pred.pop("offsets")
            pred["text"] = context[offsets[0]: offsets[1]]

        # In the very rare edge case we have not a single non-null prediction,
        # we create a fake prediction to avoid failure.
        if len(predictions) == 0 or (
                len(predictions) == 1 and predictions[0]["text"] == ""):
            predictions.insert(0, {"text": "empty", "start_logit": 0.0,
                                   "end_logit": 0.0, "score": 0.0})

        all_predictions[example["id"]] = predictions[0]["text"]
    return all_predictions


class MyCallback(TrainerCallback):
    def __init__(self, trainer_class):
        super().__init__()
        self.trainer_class = trainer_class
        self.args = trainer_class.args
        # print("\n\n*************** MyCallback init...")
        # print("\n\n*************** MyCallback args:", trainer_class.args)

    "A callback that prints a message at the beginning of training"

    def on_epoch_end(self, args, state, control, **kwargs):
        print("\n\n************ on_epoch_end:", state.epoch)

        start_dir = self.args.output_dir + "/start_pos"
        end_dir = self.args.output_dir + "/end_pos"
        # print("\n\n************ on_epoch_end start_dir:", start_dir)
        # print("\n\n************ on_epoch_end end_dir:", end_dir)
        self.trainer_class.train_dynamic(start_dir, int(state.epoch), self.trainer_class.cur_train_ids,
                                         self.trainer_class.cur_train_start_logits,
                                         self.trainer_class.cur_train_start_golds)
        self.trainer_class.train_dynamic(end_dir, int(state.epoch), self.trainer_class.cur_train_ids,
                                         self.trainer_class.cur_train_end_logits,
                                         self.trainer_class.cur_train_end_golds)
        self.trainer_class.cur_train_ids = []
        self.trainer_class.cur_train_start_logits = []
        self.trainer_class.cur_train_start_golds = []
        self.trainer_class.cur_train_end_logits = []
        self.trainer_class.cur_train_end_golds = []


# Adapted from https://github.com/huggingface/transformers/blob/master/examples/pytorch/question-answering/trainer_qa.py
class QuestionAnsweringTrainer(Trainer):
    def __init__(self, *args, eval_examples=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.eval_examples = eval_examples
        # self.args = args
        # self.cur_num_epoch = 0
        self.cur_train_ids = []
        self.cur_train_start_logits = []
        self.cur_train_start_golds = []
        self.cur_train_end_logits = []
        self.cur_train_end_golds = []

    def train_dynamic(self, output_dir, epoch, train_ids, train_logits, train_golds):
        log_training_dynamics(output_dir=output_dir,
                              epoch=epoch,
                              train_ids=list(train_ids),
                              train_logits=list(train_logits),
                              train_golds=list(train_golds))

    def compute_loss(self, model, inputs, return_outputs=False):
        """
        How the loss is computed by Trainer. By default, all models return the loss in the first element.

        Subclass and override for custom behavior.
        """
        # print("\n\n*********************** before inputs:", inputs, "\n\n")
        old_inputs = inputs.copy()
        del inputs['example_id']
        # print("\n\n*********************** after inputs:", inputs, "\n\n")

        if self.label_smoother is not None and "labels" in inputs:
            labels = inputs.pop("labels")
        else:
            labels = None

        outputs = model(**inputs)
        # Save past state if it exists
        # TODO: this needs to be fixed and made cleaner later.
        if self.args.past_index >= 0:
            self._past = outputs[self.args.past_index]

        if labels is not None:
            loss = self.label_smoother(outputs, labels)
        else:
            # We don't use .loss here since the model may return tuples instead of ModelOutput.
            loss = outputs["loss"] if isinstance(outputs, dict) else outputs[0]

        # track the info
        # print("\n\n*********************** inputs:", inputs.shape, ", outputs:", outputs.shape, ", labels:", labels, "\n\n")
        # print("\n\n*********************** labels:", labels, "\n\n")
        # print("\n\n*********************** after inputs:", inputs, "\n\n")
        # print("\n\n*********************** outputs:", outputs, "\n\n")
        # print("*********************** inputs['end_positions']:", inputs['end_positions'].shape)
        # print("*********************** inputs['start_positions']:", inputs['start_positions'].shape)
        # print("*********************** inputs['input_ids']:", inputs['input_ids'].shape)
        # print("*********************** inputs['token_type_ids']:", inputs['token_type_ids'].shape)
        # print("*********************** inputs['attention_mask']:", inputs['attention_mask'].shape)
        # print("*********************** outputs['start_logits']:", outputs['start_logits'].shape)
        # print("*********************** outputs['end_logits']:", outputs['end_logits'].shape)
        # old_inputs['end_positions']
        # old_inputs['start_positions']
        # old_inputs['input_ids']
        # old_inputs['token_type_ids']
        # old_inputs['attention_mask']
        # old_inputs['example_id']

        # outputs['start_logits']
        # outputs['end_logits']

        self.cur_train_ids.append(old_inputs['example_id'].detach().cpu().numpy())
        self.cur_train_start_logits.append(outputs['start_logits'].detach().cpu().numpy())
        self.cur_train_start_golds.append(old_inputs['start_positions'].detach().cpu().numpy())
        self.cur_train_end_logits.append(outputs['end_logits'].detach().cpu().numpy())
        self.cur_train_end_golds.append(old_inputs['end_positions'].detach().cpu().numpy())

        return (loss, outputs) if return_outputs else loss

    def get_train_dataloader(self) -> DataLoader:
        """
        Returns the training :class:`~torch.utils.data.DataLoader`.

        Will use no sampler if :obj:`self.train_dataset` does not implement :obj:`__len__`, a random sampler (adapted
        to distributed training if necessary) otherwise.

        Subclass and override this method if you want to inject some custom behavior.
        """
        if self.train_dataset is None:
            raise ValueError("Trainer: training requires a train_dataset.")

        train_dataset = self.train_dataset
        # print("\n\n\n***************** train_dataset 1:", train_dataset)
        # if is_datasets_available() and isinstance(train_dataset, datasets.Dataset):
        #     train_dataset = self._remove_unused_columns(train_dataset, description="training")

        if isinstance(train_dataset, torch.utils.data.IterableDataset):
            if self.args.world_size > 1:
                train_dataset = IterableDatasetShard(
                    train_dataset,
                    batch_size=self.args.train_batch_size,
                    drop_last=self.args.dataloader_drop_last,
                    num_processes=self.args.world_size,
                    process_index=self.args.process_index,
                )

            return DataLoader(
                train_dataset,
                batch_size=self.args.train_batch_size,
                collate_fn=self.data_collator,
                num_workers=self.args.dataloader_num_workers,
                pin_memory=self.args.dataloader_pin_memory,
            )

        train_sampler = self._get_train_sampler()

        return DataLoader(
            train_dataset,
            batch_size=self.args.train_batch_size,
            sampler=train_sampler,
            collate_fn=self.data_collator,
            drop_last=self.args.dataloader_drop_last,
            num_workers=self.args.dataloader_num_workers,
            pin_memory=self.args.dataloader_pin_memory,
        )

    def evaluate(self,
                 eval_dataset=None,  # denotes the dataset after mapping
                 eval_examples=None,  # denotes the raw dataset
                 ignore_keys=None,  # keys to be ignored in dataset
                 metric_key_prefix: str = "eval"
                 ):
        eval_dataset = self.eval_dataset if eval_dataset is None else eval_dataset
        eval_dataloader = self.get_eval_dataloader(eval_dataset)
        eval_examples = self.eval_examples if eval_examples is None else eval_examples

        # Temporarily disable metric computation, we will do it in the loop here.
        compute_metrics = self.compute_metrics
        self.compute_metrics = None
        try:
            # compute the raw predictions (start_logits and end_logits)
            output = self.evaluation_loop(
                eval_dataloader,
                description="Evaluation",
                # No point gathering the predictions if there are no metrics, otherwise we defer to
                # self.args.prediction_loss_only
                prediction_loss_only=True if compute_metrics is None else None,
                ignore_keys=ignore_keys,
            )
        finally:
            self.compute_metrics = compute_metrics

        if self.compute_metrics is not None:
            # post process the raw predictions to get the final prediction
            # (from start_logits, end_logits to an answer string)
            eval_preds = postprocess_qa_predictions(eval_examples,
                                                    eval_dataset,
                                                    output.predictions)
            formatted_predictions = [{"id": k, "prediction_text": v}
                                     for k, v in eval_preds.items()]
            references = [{"id": ex["id"], "answers": ex['answers']}
                          for ex in eval_examples]

            # compute the metrics according to the predictions and references
            metrics = self.compute_metrics(
                EvalPrediction(predictions=formatted_predictions,
                               label_ids=references)
            )

            # Prefix all keys with metric_key_prefix + '_'
            for key in list(metrics.keys()):
                if not key.startswith(f"{metric_key_prefix}_"):
                    metrics[f"{metric_key_prefix}_{key}"] = metrics.pop(key)

            self.log(metrics)
        else:
            metrics = {}

        self.control = self.callback_handler.on_evaluate(self.args, self.state,
                                                         self.control, metrics)
        return metrics
