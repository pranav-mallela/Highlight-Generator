import pandas as pd
from datasets import Dataset, DatasetDict
from transformers import pipeline
import json
import os

system_prompt = """You are a Large Language Model specialized in identifying an action in a soccer match from a fixed set of action classes as it occurs in given soccer commentary. The action classes are corner, shots on target, goal, clearance, foul, free-kick, and substitution. Your task is to observe the input commentary carefully and respond to the prompt. Prompts will ask for the match times of an action. You are to respond with a series of sentences that describe the time (start, end), in real match time (minutes:seconds), when the action occurred. If the action does not occur in the input commentary, simply state that in your response. The commentary will come as a list of comments in the format [start_time, end_time, 'comment'], and time will be in the format minutes:seconds. Focus on delivering accurate timestamps based on the live game time in the commentary data provided. Absolutely avoid additional explanation."""

"""
Generate Prompt
"""
def get_prompt(human_prompt, system_prompt=system_prompt):
  examples_segment = """Here are some example prompts and answers in the format you are expect to follow: \n
Prompt: Here is match commentary for a 1 min segment of a match <commentary>\n. Utilize the commentary and video clip of this segment to accurately find all the match times that Goals occur in this segment.\n #### Answer: A goal occurs at (2:15, 2:27).\n
Prompt: Here is match commentary for a 1 min segment of a match <commentary>\n. Utilize the commentary and video clip of this segment to accurately find all the match times that Shots on target occur in this segment.\n #### Answer: A shot on target occurs at (32:15, 32:27).\n
Prompt: Here is match commentary for a 1 min segment of a match <commentary>\n. Utilize the commentary and video clip of this segment to accurately find all the match times that Corners occur in this segment.\n #### Answer: A corner occurs at (20:10, 20:22).\n
Prompt: Here is match commentary for a 1 min segment of a match <commentary>\n. Utilize the commentary and video clip of this segment to accurately find all the match times that Goals occur in this segment.\n #### Answer: A goal occurs at (1:05, 1:18). A goal occurs at (1:45, 1:55).\n
Prompt: Here is match commentary for a 1 min segment of a match <commentary>\n. Utilize the commentary and video clip of this segment to accurately find all the match times that Fouls occur in this segment.\n #### Answer: A foul occurs at (2:16, 2:23). A foul occurs at (2:44, 2:52).\n"""
  # user_prompt = f'''{commentary}\nUtilize the commentary of this segment to accurately find all the match times that {event_name} occurs in this segment.\n'''
  return [
    {"role": "system", "content": system_prompt},
    {"role": "user", "content": examples_segment + human_prompt},
  ]

def main():
    # data = []
    # with open("LLM_training_samples.json", "r") as file:
    #     data = json.load(file)
    # print(len(data))
    # data = [(e['timestamps'], e['comments'], e['event']) for e in data]

    # df = pd.DataFrame(data, columns=["timestamps", "comments", "event"])
    # dataset = Dataset.from_pandas(df)
    # shuffled_dataset = dataset.shuffle(seed=42) # shuffle to account for implicit order in existing scenario-reason pairs
    # sample_size = int(0.10 * len(shuffled_dataset)) # choose 50% of the dataset for fine-tuning in under 2 hours on gpu_mig40
    # sampled_dataset = shuffled_dataset.select(range(sample_size))
    # train_test_split = sampled_dataset.train_test_split(test_size=0.3, seed=42)
    # validation_test_split = train_test_split['test'].train_test_split(test_size=0.5, seed=42)
    # dataset_dict = DatasetDict({
    #     'train': train_test_split['train'],
    #     'validation': validation_test_split['train'],
    #     'test': validation_test_split['test']
    # })

    llm_generator = pipeline(
        "text-generation",
        model="mistralai/Mistral-7B-Instruct-v0.3",
        device_map="auto",
    )

    results = []

    TEST_PATH = "test_data_1_min.json"
    OUT_PATH = "results_outofthebox.json"

    with open(TEST_PATH, "r") as file:
        test_data = json.load(file)

    print(f"Loaded {len(test_data)} samples for testing.")

    for i, sample in enumerate(test_data[:200]):
        print(f"Running inference for sample {i:03}...")
        # video_path = os.path.join(DATA_ROOT, sample["video"])
        human_prompt = next(conv for conv in sample["conversations"] if conv["from"] == "human")["value"]
        ground_truth = next(conv for conv in sample["conversations"] if conv["from"] == "gpt")["value"]
        for index, conversation in enumerate(sample["conversations"]):
            prompt = get_prompt(human_prompt)
            full_output = llm_generator(prompt, max_new_tokens=200)[0]['generated_text']
            for elt in full_output:
              if elt['role'] == 'assistant':
                  final_answer = elt['content']
                #   final_answer = get_timestamps(final_answer)
                  results.append({
                    "prediction": final_answer,
                    "ground_truth": ground_truth,
                  })

    print(f"Evaluation complete.")

    with open(OUT_PATH, "w") as f:
        json.dump(results, f, indent=4)

    print(f"Wrote results to {OUT_PATH}")

    # llm_generator = pipeline(
    #     "text-generation",
    #     model="mistralai/Mistral-7B-Instruct-v0.3",
    #     device_map="auto",
    # )

    # data = dataset_dict['train']
    # results = []
    # predictions = []
    # for example in data:
    #    prompt = get_prompt(example['comments'], example['event'])
    #    full_output = llm_generator(prompt, max_new_tokens=200)[0]['generated_text']
    #    for elt in full_output:
    #       if elt['role'] == 'assistant':
    #           final_answer = elt['content']
    #         #   final_answer = get_timestamps(final_answer)
    #           results.append({
    #              "prediction": final_answer,
    #              "ground_truth": example['timestamps'],
    #           })
    #           predictions.append(final_answer)
    
    # with open("results_outofthebox.json", "w") as file:
    #     json.dump(results, file, indent=4)

    # print("Results saved to results_outofthebox.json")

if __name__ == "__main__":
    main()