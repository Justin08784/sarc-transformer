import nltk
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModelForSeq2SeqLM, AutoModelForMaskedLM, Seq2SeqTrainingArguments
import pandas as pd
model_name = "models/model_final"
model_dir = model_name

tokenizer = AutoTokenizer.from_pretrained(model_dir)
model = AutoModelForSeq2SeqLM.from_pretrained(model_dir)

max_input_length = 512


df = pd.read_csv("dataset/sarcasm_v2/HYP-sarc-notsarc.csv")
df1 = df[df["class"] == "notsarc"].head(50) 
df2 = df[df["class"] == "sarc"].head(50) 

df3 = pd.concat([df1, df2])

print(df3)

text = "yes..but what is the difference between killing for fun and killing by mistake???? you drive your big suv to work every day...how many animals do you accidently kill? how many insects? oh wait, you judge a life by how big the life is...ignorant humans!"
inputs = [text]
inputs = tokenizer(inputs, max_length=max_input_length, truncation=True, return_tensors="pt")
output = model.generate(**inputs, num_beams=8, do_sample=True, min_length=10, max_length=64)
decoded_output = tokenizer.batch_decode(output, skip_special_tokens=True)[0]
predicted_title = nltk.sent_tokenize(decoded_output.strip())[0]

print(predicted_title)