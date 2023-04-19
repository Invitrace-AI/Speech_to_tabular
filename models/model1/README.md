---
language:
- th
tags:
- automatic-speech-recognition
license: apache-2.0
datasets:
- common_voice
metrics:
- wer
- cer
---

# Thai Wav2Vec2 with CommonVoice V8 (newmm tokenizer) + language model

This model trained with CommonVoice V8 dataset by increase data from CommonVoice V7 dataset that It was use in [airesearch/wav2vec2-large-xlsr-53-th](https://huggingface.co/airesearch/wav2vec2-large-xlsr-53-th). It was finetune [wav2vec2-large-xlsr-53](https://huggingface.co/facebook/wav2vec2-large-xlsr-53).

## Model description
- Technical report: [Thai Wav2Vec2.0 with CommonVoice V8](https://arxiv.org/abs/2208.04799)

## Datasets

It is increase new data from The Common Voice V8 dataset to Common Voice V7 dataset or remove all data in Common Voice V7 dataset before split Common Voice V8 then add CommonVoice V7 dataset back to dataset.

It use [ekapolc/Thai_commonvoice_split](https://github.com/ekapolc/Thai_commonvoice_split) script for split Common Voice dataset.

## Models

This model was finetune [wav2vec2-large-xlsr-53](https://huggingface.co/facebook/wav2vec2-large-xlsr-53) model with Thai Common Voice V8 dataset and It use pre-tokenize with `pythainlp.tokenize.word_tokenize`.

## Training

I used many code from [vistec-AI/wav2vec2-large-xlsr-53-th](https://github.com/vistec-AI/wav2vec2-large-xlsr-53-th) and I fixed bug training code in [vistec-AI/wav2vec2-large-xlsr-53-th#2](https://github.com/vistec-AI/wav2vec2-large-xlsr-53-th/pull/2)

## Evaluation

**Test with CommonVoice V8 Testset**

| Model                 | WER by newmm (%) | WER by deepcut (%) | CER      |
|-----------------------|------------------|--------------------|----------|
| AIResearch.in.th and PyThaiNLP                  | 17.414503        | 11.923089          | 3.854153 |
| wav2vec2 with deepcut | 16.354521        | 11.424476          | 3.684060 |
| wav2vec2 with newmm   | 16.698299        | 11.436941          | 3.737407 |
| wav2vec2 with deepcut + language model | 12.630260        | 9.613886           | 3.292073 |
| **wav2vec2 with newmm + language model**   | 12.583706        | 9.598305          | 3.276610 |

**Test with CommonVoice V7 Testset (same test by CV V7)**

| Model                 | WER by newmm (%) | WER by deepcut (%) | CER      |
|-----------------------|------------------|--------------------|----------|
| AIResearch.in.th and PyThaiNLP                  | 13.936698        | 9.347462           | 2.804787 |
| wav2vec2 with deepcut | 12.776381        | 8.773006           | 2.628882 |
| wav2vec2 with newmm   | 12.750596        | 8.672616           | 2.623341 |
| wav2vec2 with deepcut + language model | 9.940050        | 7.423313           | 2.344940 |
| **wav2vec2 with newmm + language model**   | 9.559724        | 7.339654          | 2.277071 |


This is use same testset from [https://huggingface.co/airesearch/wav2vec2-large-xlsr-53-th](https://huggingface.co/airesearch/wav2vec2-large-xlsr-53-th).


**Links:**
- GitHub Dataset: [https://github.com/wannaphong/thai_commonvoice_dataset](https://github.com/wannaphong/thai_commonvoice_dataset)
- Technical report: [Thai Wav2Vec2.0 with CommonVoice V8](https://arxiv.org/abs/2208.04799)

## BibTeX entry and citation info

```
@misc{phatthiyaphaibun2022thai,
      title={Thai Wav2Vec2.0 with CommonVoice V8}, 
      author={Wannaphong Phatthiyaphaibun and Chompakorn Chaksangchaichot and Peerat Limkonchotiwat and Ekapol Chuangsuwanich and Sarana Nutanong},
      year={2022},
      eprint={2208.04799},
      archivePrefix={arXiv},
      primaryClass={cs.CL}
}
```
