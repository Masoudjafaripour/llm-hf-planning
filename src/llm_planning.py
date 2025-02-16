from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset
from trl import SFTTrainer

# Load model & tokenizer
model_name = "gpt2"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

# Load dataset
dataset = load_dataset("imdb", split="train[:1%]")

# Tokenize dataset
def tokenize_function(examples):
    return tokenizer(examples["text"], truncation=True, padding="max_length", max_length=128)

tokenized_dataset = dataset.map(tokenize_function, batched=True)

# Train model
trainer = SFTTrainer(
    model=model,
    train_dataset=tokenized_dataset,
    tokenizer=tokenizer,
    args={"per_device_train_batch_size": 2, "num_train_epochs": 1}
)

trainer.train()
