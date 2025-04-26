import json
import logging
import os
import torch

from transformers import AutoProcessor
from awq import AutoAWQForCausalLM
from peft import PeftModel
from qwen_vl_utils import process_vision_info


TEST_PATH = "/scratch/eecs545w25_class_root/eecs545w25_class/highlights/test_data_1_min.json"
DATA_ROOT = "/scratch/eecs545w25_class_root/eecs545w25_class/highlights/SoccerNet"
OUT_PATH = "/scratch/eecs545w25_class_root/eecs545w25_class/highlights/qwen_results.json"

SYSTEM_MESSAGE = """
    You are a Vision Language Model specialized in identifying an action in a soccer match from a fixed set of action classes as it occurs in a given soccer video. The action classes are corner, shots on target, goal, clearance, foul, free-kick, and substitution. Your task is to observe the input video and commentary carefully and respond to the prompt. Prompts will ask for the match times of an action. You are to respond with a series of sentences that describe the time (start, end), in real match time (minutes:seconds), when the action occurred. If the action does not occur in the input video, simply state that in your response. The video contains broadcast footage from soccer games between players of two teams distinguished by their team uniform. The commentary will come as a list of comments in the format [start_time, end_time, 'comment'], and time will be in the format minutes:seconds. Focus on delivering accurate timestamps based on the live game time displayed in the video. Absolutely avoid additional explanation. Here are some example prompts and answers in the format you are expect to follow: \n
    Prompt: <video>\nHere is match commentary for a 1 min segment of a match <commentary>\n. Utilize the commentary and video clip of this segment to accurately find all the match times that Goals occur in this segment.\n Answer: A goal occurs at (2:15, 2:27).\n
    Prompt: <video>\nHere is match commentary for a 1 min segment of a match <commentary>\n. Utilize the commentary and video clip of this segment to accurately find all the match times that Shots on target occur in this segment.\n Answer: A shot on target occurs at (2:15, 2:27).\n
    Prompt: <video>\nHere is match commentary for a 1 min segment of a match <commentary>\n. Utilize the commentary and video clip of this segment to accurately find all the match times that Corners occur in this segment.\n Answer: A corner occurs at (0:10, 0:22).\n
    Prompt: <video>\nHere is match commentary for a 1 min segment of a match <commentary>\n. Utilize the commentary and video clip of this segment to accurately find all the match times that Goals occur in this segment.\n Answer: A goal occurs at (1:05, 1:18). A goal occurs at (1:45, 1:55).\n
    Prompt: <video>\nHere is match commentary for a 1 min segment of a match <commentary>\n. Utilize the commentary and video clip of this segment to accurately find all the match times that Fouls occur in this segment.\n Answer: A foul occurs at (2:16, 2:23). A foul occurs at (2:44, 2:52).\n
"""


def query_video(prompt, video_path, processor, model):
    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "video",
                    "video": f"file://{video_path}",
                    "max_pixels": 360 * 420,
                    "fps": 1.0,
                },
                {"type": "text", "text": prompt},
            ],
        }
    ]
    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    )
    inputs = inputs.to("cuda")

    # Inference
    with torch.no_grad():  # Use no_grad to save memory during inference
        generated_ids = model.generate(**inputs, max_new_tokens=1024)

    # Trim the generated output to remove the input prompt
    generated_ids_trimmed = [
        out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]

    # Decode the generated text
    output_text = processor.batch_decode(
        generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )
    torch.cuda.empty_cache()
    return output_text

if __name__ == "__main__":
    logging.info("Loading model...")
    model = AutoAWQForCausalLM.from_pretrained(
        "Qwen/Qwen2-VL-2B-Instruct-AWQ",
        torch_dtype=torch.float16,
        attn_implementation="flash_attention_2",
        device_map="auto"
    )
    adapter_path = "/scratch/eecs545w25_class_root/eecs545w25_class/highlights/model/checkpoint-1100"
    model = PeftModel.from_pretrained(model, adapter_path)
    model = model.merge_and_unload()
    model.eval()

    processor = AutoProcessor.from_pretrained("Qwen/Qwen2-VL-2B-Instruct-AWQ")

    results = []

    with open(TEST_PATH, "r") as file:
        test_data = json.load(file)

    logging.info(f"Loaded {len(test_data)} samples for testing.")

    for i, sample in enumerate(test_data[:200]):
        video_path = os.path.join(DATA_ROOT, sample["video"])
        logging.info(f"Running inference for sample {i:03}: {sample['id']}")
        human_prompt = next(conv for conv in sample["conversations"] if conv["from"] == "human")["value"]
        ground_truth = next(conv for conv in sample["conversations"] if conv["from"] == "gpt")["value"]
        for index, conversation in enumerate(sample["conversations"]):
            prompt = f"{SYSTEM_MESSAGE}Prompt: {human_prompt}\n Answer:"
            prediction = query_video(prompt, video_path, processor, model)
            sample_result = {
                "id": sample["id"],
                "ground_truth": ground_truth,
                "prediction": prediction,
            }
            results.append(sample_result)

    logging.info(f"Evaluation complete.")

    with open(OUT_PATH, "w") as f:
        json.dump(results, f, indent=4)

    logging.info(f"Wrote results to {OUT_PATH}")
