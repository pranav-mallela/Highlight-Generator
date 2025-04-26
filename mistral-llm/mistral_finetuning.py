import random
import wandb
import torch
import pandas as pd
import json
from datasets import Dataset, DatasetDict
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import LoraConfig
from trl import SFTConfig, SFTTrainer, DataCollatorForCompletionOnlyLM
from dotenv import load_dotenv

def set_seed(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

SEED = 42
set_seed(SEED)
device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")

load_dotenv() # huggingface and wandb credentials

wandb.login()

run = wandb.init(
    project='Soccer Highlights from Commentary',
    job_type="training",
    anonymous="allow"
)

# ordered in priority (earlier labels supercede later ones)
# labels and their key-phrases
# LABELS = {
#     "STOPPED": (["not moving", "stopped", "still", "sitting", "stop", "isn't moving", "not moving", "stops", "parked","stationary"]),
#     "REVERSE": (["reverse", "reverses", "reversing"]),
#     "ACCELERATE": (["accelerates", "resumes", "picks up", "speeding up", "speeds up", "accelerate", "accelerating"]),
#     "RIGHT": (["merge", "turn", "turns", "veer", "makes", "veers", "steers", "steering", "merges",  "moves", "moving", "shifting", "switching", "merging", "turning"],["right"]), # needs a word in BOTH to match
#     "LEFT": (["merge", "turn", "turns", "veer", "makes", "veers", "steers", "steering", "merges",  "moves", "moving", "shifting", "switching", "merging", "turning"],["left"]), # needs a word in BOTH to match
#     "MAINTAIN": (["travelling down", "going fast", "continues", "continue", "steady", "moves down", "moves forward", "moving down", "moving forward", "drives slowly", "stays", "inches", "drives forward", "steadily", "driving slowly", "driving", "drives", "driving forward", "maintains"]),
#     "SLOW": (["slows", "slowing", "brakes", "braking", "slow", "is stopping"]),
#     "OTHER": ([])
# }

# BUCKET_MAPPING = {
#     "MAINTAIN": "MAINTAIN",
#     "ACCELERATE": "MAINTAIN",
#     "STOPPED": "SLOW",
#     "SLOW": "SLOW",
#     "RIGHT": "TURN",
#     "LEFT": "TURN",
#     "REVERSE": "REVERSE",
#     "OTHER": "OTHER"
# }

# def check_label_match(full_string, label_key):
#     if label_key == "OTHER":
#         return True

#     if label_key == "LEFT" or label_key == "RIGHT":
#         match = True
#         for key_word_set in LABELS[label_key]:
#             set_match = False
#             for key_word in key_word_set:
#                 if key_word in full_string:
#                     set_match = True
#             match = (match and set_match)
#         if match:
#             return True

#     else:
#         for key_word in LABELS[label_key]:
#             if key_word in full_string:
#                 return True

#     return False

# def get_label(full_string):
#     for k in LABELS.keys():
#         if (check_label_match(full_string, k)):
#             return k

# def clean_reason(reason):
#     words_to_remove = ["because", "to", "since", "as", "due", "for"]
#     # Split the reason into words
#     words = reason.split()
#     # Check if the first word is one of the words to remove
#     if words and words[0] in words_to_remove:
#         # Remove the first word
#         words = words[1:]
#     if words and words[0] in words_to_remove:
#         # Remove the first word
#         words = words[1:]
#     # Join the words back into a string
#     return " ".join(words)

# def load_examples():

#     with open("LLM_training_samples.json", "r") as file:
#         data = json.load(file)
    
#     return data


    # df = pd.read_csv('dataset.csv')

    # df = df.iloc[:, 3:] # remove first 3 cols

    # examples = []

    # for index, row in df.iterrows():

    #     # Iterate through columns in steps of 2 (pairing columns)
    #     for i in range(0, len(row), 2):  # Start from 0, step by 2 to get pairs
    #         first_value = row[i]
    #         second_value = row[i + 1] if (i + 1) < len(row) else None  # Handle case where second value might be missing

    #         # Check if the first value is NaN
    #         if pd.isna(first_value):
    #             break  # Stop processing once the first value in the pair is NaN

    #         if second_value == None or pd.isna(second_value):
    #             break

    #         if not isinstance(first_value, (int, float)):
    #             # Append the pair to the processed list
    #             examples.append((first_value, second_value))

    # return examples

# def preprocess_examples(raw_examples):
#     examples = []
#     for e in raw_examples:
#         examples.append((get_label(e[0]), clean_reason(e[1])))
#     return examples

def load_data_and_collator(
    dataset_dict,
    split="train",
    tokenizer=None,
    response_template="#### Answer:",
):

    if split not in dataset_dict:
        raise ValueError(f"Invalid split '{split}'. Must be one of {list(dataset_dict.keys())}.")
    dataset = dataset_dict[split]
    collator = DataCollatorForCompletionOnlyLM(
        response_template=response_template,
        tokenizer=tokenizer,
        mlm=False
    )

    return dataset, collator

def response_template():
    return "\n #### Answer:"

def prompt_instruction(commentary, timestamps=None, event_name=None):
    examples_segment = """Here are some example prompts and answers in the format you are expect to follow: \n
    Prompt: Here is match commentary for a 1 min segment of a match <commentary>\n. Utilize the commentary and video clip of this segment to accurately find all the match times that Goals occur in this segment.\n #### Answer: A goal occurs at (2:15, 2:27).\n
    Prompt: Here is match commentary for a 1 min segment of a match <commentary>\n. Utilize the commentary and video clip of this segment to accurately find all the match times that Shots on target occur in this segment.\n #### Answer: A shot on target occurs at (32:15, 32:27).\n
    Prompt: Here is match commentary for a 1 min segment of a match <commentary>\n. Utilize the commentary and video clip of this segment to accurately find all the match times that Corners occur in this segment.\n #### Answer: A corner occurs at (20:10, 20:22).\n
    Prompt: Here is match commentary for a 1 min segment of a match <commentary>\n. Utilize the commentary and video clip of this segment to accurately find all the match times that Goals occur in this segment.\n #### Answer: A goal occurs at (1:05, 1:18). A goal occurs at (1:45, 1:55).\n
    Prompt: Here is match commentary for a 1 min segment of a match <commentary>\n. Utilize the commentary and video clip of this segment to accurately find all the match times that Fouls occur in this segment.\n #### Answer: A foul occurs at (2:16, 2:23). A foul occurs at (2:44, 2:52).\n"""
    system_message = """You are a Large Language Model specialized in identifying an action in a soccer match from a fixed set of action classes as it occurs in given soccer commentary. The action classes are corner, shots on target, goal, clearance, foul, free-kick, and substitution. Your task is to observe the input commentary carefully and respond to the prompt. Prompts will ask for the match times of an action. You are to respond with a series of sentences that describe the time (start, end), in real match time (minutes:seconds), when the action occurred. If the action does not occur in the input commentary, simply state that in your response. The commentary will come as a list of comments in the format [start_time, end_time, 'comment'], and time will be in the format minutes:seconds. Focus on delivering accurate timestamps based on the live game time in the commentary data provided. Absolutely avoid additional explanation.\n"""
    gpt_answer = ""
    if timestamps is not None:
        print(timestamps)
        for i in range(len(timestamps)):
            gpt_answer += f"A {event_name} occurs at ({timestamps[i][0][0]}, {timestamps[i][0][1]}). "
        gpt_answer = gpt_answer.strip()
        return f"#### {system_message} {examples_segment} {commentary} Utilize the commentary of this segment to accurately find all the match times that {event_name} occurs in this segment.\n #### Answer: {gpt_answer}\n"
    else:
        return f"#### {system_message} {examples_segment} {commentary} Utilize the commentary of this segment to accurately find all the match times that {event_name} occurs in this segment.\n"


def format_prompts(example):
    # prompts = []
    # print(f"example[timestamps] = {example['timestamps']}")
    # for i in range(len(example['comments'])):
    #   instruction = prompt_instruction(example['comments'][i], example['timestamps'][i], example['event'][i])
    #   prompts.append(instruction)
    prompt = prompt_instruction(example['comments'], example['timestamps'], example['event'])
    return [prompt]

def initialize_model_and_tokenizer():

    tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-v0.3") # download MISTRAL model
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    quant_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
    )

    model = AutoModelForCausalLM.from_pretrained(
        "mistralai/Mistral-7B-v0.3",
        quantization_config=quant_config,
        trust_remote_code=True,
        low_cpu_mem_usage=True
    )

    model.config.use_cache = False

    return model, tokenizer


def train(
    model,
    dataset,
    tokenizer,
    collator,
    format_prompts_function,
):

    ####### Tune any hyperparameters here ########

    lora_config = LoraConfig(
        r=8,
        lora_alpha=16,
        target_modules=["gate_proj", "down_proj", "up_proj", "q_proj", "v_proj", "k_proj", "o_proj"],
        lora_dropout=0.1
    )

    training_arguments = SFTConfig(
        run_name="soccer-highlights",
        output_dir="./results",
        num_train_epochs=4,
        per_device_train_batch_size=4,
        gradient_accumulation_steps=4,
        optim="paged_adamw_32bit",
        do_eval=False,
        logging_steps=1,
        learning_rate=4e-5,
        fp16=True,
        max_grad_norm=0.25,
        warmup_ratio=0.025,
        group_by_length=True,
        lr_scheduler_type="linear",
        seed=42,
        report_to="wandb",
        max_seq_length=512,
    )

    trainer = SFTTrainer(
        model=model,
        train_dataset=dataset,
        peft_config=lora_config,
        formatting_func=format_prompts_function,
        data_collator=collator,
        tokenizer=tokenizer,
        args=training_arguments,
    )

    for name, module in trainer.model.named_modules():
        if "norm" in name:
            module = module.to(torch.float32)

    trainer.train(resume_from_checkpoint=True)

    return trainer, model


def test(model, tokenizer, dataset, predictions_file='predictions.torch'):

    results = []

    for example in dataset:
        prompt = prompt_instruction(example['comments'], None, example['event'])

        inputs = tokenizer(prompt, return_tensors="pt")
        inputs = {k: v.to(model.device) for k, v in inputs.items()}  # Move values to the device

        input_ids = inputs["input_ids"]
        outputs = model.generate(**inputs, max_new_tokens=512, pad_token_id=tokenizer.eos_token_id)
        # output_ids = model.generate(input_ids)
        prediction = tokenizer.decode(outputs[:, input_ids.shape[1]:])

        def get_timestamps(full_prediction):
            # for label in LABELS.keys():
            #     if label in full_prediction[111:]:
            #         return label
            # return "OTHER"
            pass

        # prediction = get_timestamps(tokenizer.decode(output[0], skip_special_tokens=True).strip())

        # Store the prediction and ground truth
        results.append({
            "prediction": prediction,
            "ground_truth": example['timestamps'],
        })

    # def compute_accuracy(results):
    #     count = 0
    #     bucket_count = 0
    #     for result in results:
    #         # Check if prediction matches the label
    #         if result['prediction'] == result['ground_truth']: # first 111 chars are the original prompt & don't contain new output
    #             count += 1

    #         # Check if pred vs ground truth buckets match
    #         if result['prediction_bucket'] == result['ground_truth_bucket']:
    #             bucket_count += 1

    #     return ((count / len(results)),(bucket_count / len(results)))

    def time_to_seconds(t):
        mins, secs = map(int, t.strip().split(":"))
        return mins * 60 + secs

    def iou(interval1, interval2):
        start1, end1 = interval1
        start2, end2 = interval2
        inter_start = max(start1, start2)
        inter_end = min(end1, end2)
        intersection = max(0, inter_end - inter_start)
        union = max(end1, end2) - min(start1, start2)
        return intersection / union if union > 0 else 0

    def evaluate(gt_intervals, pred_intervals, threshold=0.3):
        gt_secs = [(time_to_seconds(s), time_to_seconds(e)) for s, e in gt_intervals]
        pred_secs = [(time_to_seconds(s), time_to_seconds(e)) for s, e in pred_intervals]
        
        matched = 0
        used_preds = set()
        
        # Match GT intervals to model predictions
        for gt in gt_secs:
            best_iou = 0
            best_idx = -1
            for idx, pred in enumerate(pred_secs):
                if idx in used_preds:
                    continue
                score = iou(gt, pred)
                if score > best_iou:
                    best_iou = score
                    best_idx = idx
            if best_iou >= threshold:
                matched += 1
                used_preds.add(best_idx)
        
        # Precision: How many model intervals matched
        precision = matched / len(pred_secs) if pred_secs else 0
        
        # Recall: How many GT intervals were matched
        recall = matched / len(gt_secs) if gt_secs else 0
        
        # F1 score: Harmonic mean of precision and recall
        f1 = 2 * recall * precision / (recall + precision) if recall + precision else 0
        
        # Return results
        return {
            "matched": matched,
            "total_gt": len(gt_secs),
            "total_pred": len(pred_secs),
            "recall": recall,
            "precision": precision,
            "f1": f1
        }
    
    print(results)

    # metrics = evaluate(results)

    # print("Evaluation Metrics:")
    # print(f"Matched: {metrics['matched']}")
    # print(f"Total GT: {metrics['total_gt']}")
    # print(f"Total Pred: {metrics['total_pred']}")
    # print(f"Recall: {metrics['recall']}")
    # print(f"Precision: {metrics['precision']}")
    # print(f"F1 Score: {metrics['f1']}")

    # Save results to a JSON file for later use (to compute additional metrics from this run)
    with open("results_finetuning.json", "w") as file:
        json.dump(results, file, indent=4)

    print("Results saved to results.json")


def main():

    # raw_examples = load_examples()
    # examples = preprocess_examples(raw_examples)
    data = []
    with open("LLM_training_samples.json", "r") as file:
        data = json.load(file)
    print(len(data))
    data = [(e['timestamps'], e['comments'], e['event']) for e in data]

    df = pd.DataFrame(data, columns=["timestamps", "comments", "event"])
    dataset = Dataset.from_pandas(df)
    shuffled_dataset = dataset.shuffle(seed=42) # shuffle to account for implicit order in existing scenario-reason pairs
    sample_size = int(0.50 * len(shuffled_dataset)) # choose 50% of the dataset for fine-tuning in under 2 hours on gpu_mig40
    sampled_dataset = shuffled_dataset.select(range(sample_size))
    train_test_split = sampled_dataset.train_test_split(test_size=0.3, seed=42)
    validation_test_split = train_test_split['test'].train_test_split(test_size=0.5, seed=42)
    dataset_dict = DatasetDict({
        'train': train_test_split['train'],
        'validation': validation_test_split['train'],
        'test': validation_test_split['test']
    })

    model, tokenizer = initialize_model_and_tokenizer()
    response_token = response_template()
    tokenizer.add_special_tokens({'additional_special_tokens': [response_token]})
    model.resize_token_embeddings(len(tokenizer))

    # def preprocess_function(example):
    #     # Generate full text (prompt + answer)
    #     full_text = prompt_instruction(example['comments'], example['timestamps'], example['event'])

    #     # Split prompt and completion for masking
    #     prompt_only = prompt_instruction(example['comments'], timestamps=None, event_name=example['event'])

    #     print("=== Full text ===")
    #     print(full_text)
    #     print("=== Prompt only ===")
    #     print(prompt_only)

    #     # Tokenize full prompt+completion
    #     tokenized = tokenizer(
    #         full_text,
    #         truncation=True,
    #         padding='max_length',
    #         max_length=512,
    #         return_tensors=None
    #     )

    #     # Mask prompt tokens in the labels
    #     input_ids = tokenized["input_ids"]
    #     labels = input_ids.copy()

    #     prompt_ids = tokenizer(prompt_only, add_special_tokens=False)["input_ids"]
    #     num_prompt_tokens = len(prompt_ids)

    #     labels[:num_prompt_tokens] = [-100] * num_prompt_tokens
    #     tokenized["labels"] = labels

    #     return tokenized

    # tokenized_dataset = dataset_dict.map(preprocess_function, batched=False)

    train_set, collator = load_data_and_collator(dataset_dict, split="train", 
    tokenizer=tokenizer, response_template=response_token)
    # Sanity check to ensure loss will actually be computed
    # sample = train_set[0]
    # print("=== Formatted Prompt ===")
    # print(prompt_instruction(sample))
    # keys_to_keep = {"input_ids", "attention_mask", "labels"}
    # tokenized_sample = {k: v for k, v in sample.items() if k in keys_to_keep}

    # collated = collator([tokenized_sample])

    # # collated = collator([sample])
    # print("=== Tokenized Input ===")
    # print(tokenizer.decode(collated["input_ids"][0]))
    # print("=== Labels (should not be all -100) ===")
    # print(collated["labels"][0])
    trainer, model = train(model, train_set, tokenizer, collator, format_prompts_function=format_prompts) # , format_prompts_function=format_prompts

    # Use validation set in test() for debugging & improving the model
    # validation_set, _ = load_data_and_collator(dataset_dict, split="validation", tokenizer=tokenizer, response_template=response_token)
    # test(model, tokenizer, validation_set)

    # Use test set in test() for reporting
    test_set, _ = load_data_and_collator(dataset_dict, split="test", tokenizer=tokenizer, response_template=response_token)
    test(model, tokenizer, test_set)

    return model


if __name__ == "__main__":
  model = main()