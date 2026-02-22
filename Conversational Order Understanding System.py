!pip install transformers accelerate datasets peft trl bitsandbytes wandb sacrebleu

import os
from dataclasses import dataclass, field
from typing import Optional
import re

import torch
import sys

from accelerate import Accelerator
from datasets import load_dataset, Dataset
from peft import AutoPeftModelForCausalLM, LoraConfig
from tqdm import tqdm
from transformers import (
    HfArgumentParser,
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
    TextStreamer,
    logging as hf_logging,
)
import logging
from trl import SFTTrainer, SFTConfig

from trl.trainer import ConstantLengthDataset
from sklearn.model_selection import train_test_split

import numpy as np
from sacrebleu import corpus_bleu

#Q1, meta-llama/Llama-3.2-1B-Instruct를 허깅페이스 허브에서 로드하여 LoRA 기법으로 미세 조정하고 학습 결과를 fine_tuned_lora에 저장하여 활용함
base_model_id = "meta-llama/Llama-3.2-1B-Instruct"
device_map="auto"
torch_dtype = torch.bfloat16
output_dir = "./fine_tuned_lora"
dataset_name = "./llm-modeling-lab.jsonl"
seq_length = 512

full_dataset = Dataset.from_json(path_or_paths=dataset_name)

#Q2, 전체 데이터셋을 훈련용(2800개)과 검증용(200개)으로 분할함
full_dataset_df = full_dataset.to_pandas()  # Dataset을 pandas DataFrame으로 변환
train_dataset_df, val_dataset_df = train_test_split(full_dataset_df, test_size=200, shuffle=True) #데이터셋을 train_test_split을 사용해 훈련용과 검증용으로 분리함

# 다시 Dataset 객체로 변환
train_dataset = Dataset.from_pandas(train_dataset_df)
val_dataset = Dataset.from_pandas(val_dataset_df)

tokenizer = AutoTokenizer.from_pretrained(base_model_id)
tokenizer.padding_side = "right"

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
)

base_model = AutoModelForCausalLM.from_pretrained(
    base_model_id,
    quantization_config=bnb_config,
    device_map="auto",
)

base_model.config.use_cache = False

if getattr(tokenizer, "pad_token", None) is None:
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.pad_token_id = tokenizer.eos_token_id
tokenizer.padding_side = "right"
if base_model.config.pad_token_id != tokenizer.pad_token_id:
    base_model.config.pad_token_id = tokenizer.pad_token_id

def chars_token_ratio(dataset, tokenizer, prepare_sample_text, nb_examples=2800):
    total_characters, total_tokens = 0, 0
    for _, example in tqdm(zip(range(nb_examples), iter(dataset)), total=nb_examples):
        text = prepare_sample_text(example)
        total_characters += len(text)
        if tokenizer.is_fast:
            total_tokens += len(tokenizer(text).tokens())
        else:
            total_tokens += len(tokenizer.tokenize(text))
    return total_characters / total_tokens

def function_prepare_sample_text(tokenizer, for_train=True):
    """A Closure"""

    def _prepare_sample_text(example):
        """Prepare the text from a sample of the dataset."""
        user_prompt="너는 사용자가 입력한 주문 문장을 분석하는 에이전트이다. 주문으로부터 이를 구성하는 음식명, 옵션명, 수량을 차례대로 추출해야 한다.\n### 주문 문장: "
        messages = [
            {"role": "user", "content": f"{user_prompt}{example['input']}"},
        ]
        if for_train:
            messages.append({"role": "assistant", "content": f"{example['output']}"})

        text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False if for_train else True)
        return text
    return _prepare_sample_text

def create_datasets(tokenizer, dataset, seq_length):
    prepare_sample_text = function_prepare_sample_text(tokenizer)

    chars_per_token = chars_token_ratio(dataset, tokenizer, prepare_sample_text, nb_examples=len(dataset))
    print(f"The character to token ratio of the dataset is: {chars_per_token:.2f}")

    cl_dataset = ConstantLengthDataset(
        tokenizer,
        dataset,
        formatting_func=prepare_sample_text,
        infinite=True,
        seq_length=seq_length,
        chars_per_token=chars_per_token,
    )
    return cl_dataset

#BLEU 점수 계산을 정의함
def compute_metrics(eval_preds):
    metric = evaluate.load("sacrebleu")
    preds, labels = eval_preds
    if isinstance(preds, tuple):
        preds = preds[0]
    decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
    preds, labels = [p.strip() for p in decoded_preds], [[l.strip()] for l in decoded_labels]
    result = metric.compute(predictions=preds, references=labels)
    return {"bleu": round(result["score"], 4)}

#Q2, 2800개 훈련 데이터와 200개 검증 데이터로 분할된 걸 확인할 수 있음
train_ds = create_datasets(tokenizer, train_dataset, seq_length)
val_ds = create_datasets(tokenizer, val_dataset, seq_length)

it_train = iter(train_ds)
tokenizer.decode(next(it_train)['input_ids'])

it_val = iter(val_ds)
tokenizer.decode(next(it_val)['input_ids'])

lora_config = LoraConfig(
            r=8,
            lora_alpha=16,
            lora_dropout=0.05,
            target_modules=[
                "q_proj",
                "k_proj",
                "v_proj",
                "o_proj",
                "down_proj",
                "up_proj",
                "gate_proj",
            ],
            bias="none",
            task_type="CAUSAL_LM",
        )

peft_config = lora_config

from google.colab import userdata
import wandb

wandb_api_key = userdata.get('WB_API_KEY')
if wandb_api_key:
    wandb.login(key=wandb_api_key)
    print("Successfully logged in to Weights & Biases")
else:
    print("WANDB_API_KEY not found in Colab secrets")

sft_config = SFTConfig(
    output_dir=output_dir,
    per_device_train_batch_size=2,
    gradient_accumulation_steps=1,
    gradient_checkpointing=False,
    learning_rate=1e-4,
    warmup_ratio=0.1,
    max_grad_norm=0.3,
    weight_decay=0.05,
    num_train_epochs=2,
    logging_steps=20,
    eval_strategy="no",
    save_strategy="steps",
    save_steps=50,
    save_total_limit=2,
    max_seq_length=seq_length,
    report_to="wandb",
    run_name="llama-3.2-fine-tuning"
)

#fine-tuning 과정을 정의
trainer = SFTTrainer(
    model=base_model,
    train_dataset=train_ds,
    eval_dataset=None,
    peft_config=lora_config,
    tokenizer=tokenizer,
    args=sft_config,
    compute_metrics=compute_metrics,
)

#2800개만 훈련됨
trainer.train()

#Q3, 미세 조정된 LoRA 어댑터를 로컬에 저장하고 push_to_hub를 사용해 허깅페이스 허브에 업로드함
trainer.model.save_pretrained(f"{output_dir}/fine-tuned-lora")
trainer.model.push_to_hub("fine-tuned-llama-3-2-1B-order-analysis")

from peft import AutoPeftModelForCausalLM

#Q4, AutoPeftModelForCausalLM.from_pretrained를 사용해 허깅페이스 허브에서 Hyewxn/fine-tuned-llama-3-2-1B-order-analysis 어댑터 모델을 로드함
trained_model = AutoPeftModelForCausalLM.from_pretrained(
    "Hyewxn/fine-tuned-llama-3-2-1B-order-analysis",
    device_map="auto",
    trust_remote_code=True
)

import re
from sacrebleu.metrics import BLEU
from transformers import TextStreamer

#Q5, 검증 데이터셋에서 생성된 출력과 정답 간의 BLEU 점수를 sacrebleu를 사용하여 계산하여 출력하고 또, 마지막에 평균 BLEU 점수를 출력함

bleu_metric = BLEU(effective_order=True)


def get_text_after_prompt(generated_output):

    cleaned_output = re.sub(r'<\|.*?\|>', '', generated_output).strip()
    analysis_results = re.findall(r"분석 결과\s*\d+:.*?(?=\n|$)", cleaned_output)
    return "\n".join(f"- {line.strip()}" for line in analysis_results).strip()


def wrapper_generate(tokenizer, model, input_prompt, do_stream=False):

    data = tokenizer(input_prompt, return_tensors="pt")
    streamer = TextStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)
    input_ids = data.input_ids[..., :-1]

    with torch.no_grad():
        pred = model.generate(
            input_ids=input_ids.cuda(),
            streamer=streamer if do_stream else None,
            use_cache=True,
            max_new_tokens=128,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )

    decoded_text = tokenizer.batch_decode(pred, skip_special_tokens=False)
    return get_text_after_prompt(decoded_text[0])


preprocessor = function_prepare_sample_text(tokenizer, for_train=False)
generated_outputs, true_outputs = [], []
bleu_scores = []

#검증 데이터셋의 각 샘플에 대해 입력 프롬프트를 생성한 후, wrapper_generate를 사용해 모델이 출력한 텍스트를 얻음.
#이를 실제 정답과 비교하여 sacrebleu의 sentence_score를 사용해 각 샘플의 BLEU 점수를 계산하고 저장했으며, 생성된 출력, 정답, BLEU 점수만을 출력하게 함
for example in val_dataset:
    input_prompt = preprocessor(example)
    generated_output = wrapper_generate(tokenizer, trained_model, input_prompt, do_stream=False)
    generated_outputs.append(generated_output)
    true_outputs.append(example["output"])

    # 현재 출력에 대한 BLEU 점수 계산 (sentence_score 사용)
    bleu = bleu_metric.sentence_score(generated_output, [example["output"]]).score
    bleu_scores.append(bleu)

    print(f"Generated: {generated_output}\nTrue: {example['output']}\nBLEU: {bleu:.2f}\n")

#평균 BLEU 점수 출력
avg_bleu = sum(bleu_scores) / len(bleu_scores)
print(f"Average BLEU score: {avg_bleu:.2f}")

preprocessor = function_prepare_sample_text(tokenizer, for_train=False)

preprocessor({'input':'아이스아메리카노 그란데 한잔 주세요'})

wrapper_generate(tokenizer=tokenizer, model=trained_model, input_prompt=preprocessor({'input':'아이스아메리카노 그란데 한잔 주세요. 그리고 베이글 두개요.'}), do_stream=True)
