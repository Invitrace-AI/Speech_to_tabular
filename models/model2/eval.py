#!/usr/bin/env python3
import argparse
import re
from typing import Dict

from datasets import Audio, Dataset, load_dataset, load_metric

from transformers import AutoFeatureExtractor, pipeline

from pythainlp.tokenize import word_tokenize, syllable_tokenize
from deepcut import tokenize as deepcut_word_tokenize
from functools import partial


def log_results(result: Dataset, args: Dict[str, str]):
    """DO NOT CHANGE. This function computes and logs the result metrics."""

    log_outputs = args.log_outputs
    dataset_id = "_".join(args.dataset.split("/") + [args.config, args.split])

    # load metric
    wer = load_metric("wer")
    cer = load_metric("cer")

    # compute metrics
    wer_result = wer.compute(references=result["target"], predictions=result["prediction"])
    cer_result = cer.compute(references=result["target"], predictions=result["prediction"])

    # print & log results
    result_str = f"WER: {wer_result}\n" f"CER: {cer_result}"
    print(result_str)

    with open(f"robust-speech-event/{dataset_id}_eval_results_{args.thai_tokenizer}.txt", "w") as f:
        f.write(result_str)

    # log all results in text file. Possibly interesting for analysis
    if log_outputs is not None:
        pred_file = f"robust-speech-event/log_{dataset_id}_predictions_{args.thai_tokenizer}.txt"
        target_file = f"robust-speech-event/log_{dataset_id}_targets_{args.thai_tokenizer}.txt"

        with open(pred_file, "w") as p, open(target_file, "w") as t:

            # mapping function to write output
            def write_to_file(batch, i):
                p.write(f"{i}" + "\n")
                p.write(batch["prediction"] + "\n")
                t.write(f"{i}" + "\n")
                t.write(batch["target"] + "\n")

            result.map(write_to_file, with_indices=True)


def normalize_text(text: str, tok_func) -> str:
    """DO ADAPT FOR YOUR USE CASE. this function normalizes the target text."""

    chars_to_ignore_regex = '[,?.!\-\;\:"“%‘”�—’…–]'  # noqa: W605 IMPORTANT: this should correspond to the chars that were ignored during training

    text = re.sub(chars_to_ignore_regex, "", text.lower())

    # In addition, we can normalize the target text, e.g. removing new lines characters etc...
    # note that order is important here!
    token_sequences_to_ignore = ["\n\n", "\n", "   ", "  "]

    for t in token_sequences_to_ignore:
        text = " ".join(text.split(t))

    #thai tokenize
    text = " ".join(tok_func(text))
    
    return text

def retokenize(text:str, tok_func) -> str:
    """tokenize and rejoin prediction outputs without cleaning"""
    return " ".join(tok_func("".join(text.split())))


def main(args):
    # load dataset
    dataset = load_dataset(args.dataset, args.config, split=args.split, use_auth_token=True)

    # for testing: only process the first two examples as a test
    dataset = dataset.select(range(10))

    # load processor
    feature_extractor = AutoFeatureExtractor.from_pretrained(args.model_id)
    sampling_rate = feature_extractor.sampling_rate

    # resample audio
    dataset = dataset.cast_column("audio", Audio(sampling_rate=sampling_rate))

    # load eval pipeline
    asr = pipeline("automatic-speech-recognition", model=args.model_id)

    #select tokenizer
    if args.thai_tokenizer=='deepcut':
        tok_func = deepcut_word_tokenize
    elif args.thai_tokenizer=='newmm':
        tok_func = word_tokenize
    elif args.thai_tokenizer=='syllable':
        tok_func = syllable_tokenize
    else:
        tok_func = lambda x: x.replace(' ','')
    
    # map function to decode audio
    def map_to_pred(batch, tok_func):
        prediction = asr(
            batch["audio"]["array"], chunk_length_s=args.chunk_length_s, stride_length_s=args.stride_length_s
        )

        batch["prediction"] = retokenize(prediction["text"], tok_func)
        batch["target"] = normalize_text(batch["sentence"], tok_func)
        return batch

    # run inference on all examples
    result = dataset.map(partial(map_to_pred, tok_func=tok_func), 
                         remove_columns=dataset.column_names)

    # compute and log_results
    # do not change function below
    log_results(result, args)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--model_id", type=str, required=True, help="Model identifier. Should be loadable with 🤗 Transformers"
    )
    parser.add_argument(
        "--thai_tokenizer", type=str, default="newmm",
        required=True, help="newmm, syllable, or deepcut; if not specified, remove all spaces (used for CER calculation)"
    )    
    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        help="Dataset name to evaluate the `model_id`. Should be loadable with 🤗 Datasets",
    )
    parser.add_argument(
        "--config", type=str, required=True, help="Config of the dataset. *E.g.* `'en'`  for Common Voice"
    )
    parser.add_argument("--split", type=str, required=True, help="Split of the dataset. *E.g.* `'test'`")
    parser.add_argument(
        "--chunk_length_s", type=float, default=None, help="Chunk length in seconds. Defaults to 5 seconds."
    )
    parser.add_argument(
        "--stride_length_s", type=float, default=None, help="Stride of the audio chunks. Defaults to 1 second."
    )
    parser.add_argument(
        "--log_outputs", action="store_true", help="If defined, write outputs to log file for analysis."
    )
    args = parser.parse_args()

    main(args)