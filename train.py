import time
import transformers
import torch
import argparse
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModelForSeq2SeqLM, AutoModelForMaskedLM, Seq2SeqTrainingArguments
from datasets import load_dataset, load_metric
import nltk
import string


import nltk
nltk.download('punkt')

model_name = "google-t5/t5-small"
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

sarc_data = load_dataset("csv", data_files="dataset/sarcasm_v2.zip")
datasets_train_test = sarc_data["train"].train_test_split(test_size=3000)
datasets_train_validation = datasets_train_test["train"].train_test_split(test_size=3000)

sarc_data["train"] = datasets_train_validation["train"]
sarc_data["validation"] = datasets_train_validation["test"]
sarc_data["test"] = datasets_train_test["test"]



prefix = "Is this sarcasm?: "
max_input_length = 512
max_target_length = 64


def clean_text(text):
    sentences = nltk.sent_tokenize(text.strip())
    sentences_cleaned = [s for sent in sentences for s in sent.split("\n")]
    sentences_cleaned_no_titles = [sent for sent in sentences_cleaned
                                    if len(sent) > 0 and
                                    sent[-1] in string.punctuation]
    text_cleaned = "\n".join(sentences_cleaned_no_titles)
    return text_cleaned

def preprocess_data(examples):
    texts_cleaned = [clean_text(text) for text in examples["text"]]
    inputs = [prefix + text for text in texts_cleaned]
    model_inputs = tokenizer(inputs, max_length=max_input_length, truncation=True)

    # Setup the tokenizer for targets
    with tokenizer.as_target_tokenizer():
        labels = tokenizer(examples["class"], max_length=max_target_length, 
                            truncation=True)

    model_inputs["labels"] = labels["input_ids"]
    return model_inputs

tokenized_datasets = sarc_data.map(preprocess_data, batched=True)


print("shit")
if model_name == "google-bert/bert-base-uncased":
    model = AutoModelForMaskedLM.from_pretrained(model_name)
else:
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)


batch_size = 8
model_name = "t5-small-tuned"
model_dir = f"models/{model_name}"

from transformers import AutoModelForSeq2SeqLM, DataCollatorForSeq2Seq, Seq2SeqTrainingArguments, Seq2SeqTrainer
args = Seq2SeqTrainingArguments(
    model_dir,
    evaluation_strategy="steps",
    # eval_steps=100,
    logging_strategy="steps",
    logging_steps=100,
    save_strategy="steps",
    save_steps=200,
    learning_rate=4e-5,
    per_device_train_batch_size=batch_size,
    per_device_eval_batch_size=batch_size,
    weight_decay=0.01,
    save_total_limit=3,
    num_train_epochs=5,
    predict_with_generate=True,
    # fp16=True,
    load_best_model_at_end=True,
    metric_for_best_model="rouge1",
    report_to="tensorboard"
)

data_collator = DataCollatorForSeq2Seq(tokenizer)
metric = load_metric("rouge")





def main(FLAGS):
    
    if model_name in ["google-t5/t5-small", "Falconsai/text_summarization"]:
        task = "text2text-generation" # https://github.com/huggingface/transformers/issues/27870
    else:
        task = "text-generation"
    
    generator = transformers.pipeline(
        task,
        model=model,
        tokenizer=tokenizer,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        device_map="auto",
    )

    user_input = "start"

    while user_input != "stop":

        user_input = input(f"Provide Input to parameter Falcon (not tuned): ")
        
        start = time.time()

        if user_input != "stop":
            sequences = generator( 
            f""" {user_input}""",
            max_length=FLAGS.max_length,
            do_sample=False,
            top_k=FLAGS.top_k,
            num_return_sequences=1,
            eos_token_id=tokenizer.eos_token_id,)

        inference_time = time.time() - start
        
        for seq in sequences:
            print(f"Result: {seq['generated_text']}")
         
        print(f'Total Inference Time: {inference_time} seconds')


import numpy as np

def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    decoded_preds = tokenizer.batch_decode(predictions, skip_special_tokens=True)
    
    # Replace -100 in the labels as we can't decode them.
    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
    
    # Rouge expects a newline after each sentence
    decoded_preds = ["\n".join(nltk.sent_tokenize(pred.strip()))
                      for pred in decoded_preds]
    decoded_labels = ["\n".join(nltk.sent_tokenize(label.strip())) 
                      for label in decoded_labels]
    print(decoded_preds)
    print(decoded_labels)
    
    # Compute ROUGE scores
    result = metric.compute(predictions=decoded_preds, references=decoded_labels,
                            use_stemmer=True)

    # Extract ROUGE f1 scores
    result = {key: value.mid.fmeasure * 100 for key, value in result.items()}
    
    # Add mean generated length to metrics
    prediction_lens = [np.count_nonzero(pred != tokenizer.pad_token_id)
                      for pred in predictions]
    result["gen_len"] = np.mean(prediction_lens)
    
    return {k: round(v, 4) for k, v in result.items()}



trainer = Seq2SeqTrainer(
    model=model,
    args=args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["validation"],
    data_collator=data_collator,
    tokenizer=tokenizer,
    compute_metrics=compute_metrics
)

trainer.train()
model.save_pretrained('dataset/models')


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    
    # parser.add_argument('-fv',
    #                     '--falcon_version',
    #                     type=str,
    #                     default="7b",
    #                     help="select 7b or 40b version of falcon")
    parser.add_argument('-ml',
                        '--max_length',
                        type=int,
                        default="25",
                        help="used to control the maximum length of the generated text in text generation tasks")
    parser.add_argument('-tk',
                        '--top_k',
                        type=int,
                        default="5",
                        help="specifies the number of highest probability tokens to consider at each step")
    
    FLAGS = parser.parse_args()
    main(FLAGS)