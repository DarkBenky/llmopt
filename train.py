import unsloth
from getFunc import find_functions, find_structs, _read_sources, show_context, replace_function, run_bench, restore_all
import os
import torch
import wandb
from datasets import Dataset
from trl import GRPOConfig, GRPOTrainer
from unsloth import FastLanguageModel

HF_MODEL = "unsloth/Qwen3-VL-4B-Instruct-bnb-4bit"

GENGIN = os.path.join(os.path.dirname(os.path.abspath(__file__)), "gengin")

BENCHMARK_STATS = {
    "frames": 359,
    "duration_s": 2.5,
    "avg_ms": 6.983,
    "median_ms": 6.826,
    "p99_ms": 11.053,
    "frame_hashes": ["0x2717c50e", "0x81745ff7", "0x81745ff7", "0x81745ff7", "0x81745ff7"]
}

SYSTEM_PROMPT = """\
You are an expert C performance engineer. Your sole task is to rewrite the body of a \
single target C function to be faster, while preserving correctness.

## Rules (never break these)
- Return ONLY the complete function — signature + body — with no other text, explanation, or markdown.
- Do NOT change the function signature (return type, name, or parameters).
- Do NOT change any struct definitions, global variables, or helper functions.
- Do NOT call any new external functions that are not already used in the provided context.
- The function must produce bit-identical output to the original (verified by the frame hashes below).

## Optimisation targets (in priority order)
1. Reduce average frame time (avg_ms) and median frame time (median_ms).
2. Reduce tail latency (p99_ms).
3. Prefer cache-friendly memory access, SIMD-friendly data layouts, and loop transforms \
(unrolling, tiling, hoisting invariants) over algorithmic changes when the algorithm is already optimal.

## Current benchmark baseline
"""


def calculate_reward(data):
    if data["frame_hashes"] != BENCHMARK_STATS["frame_hashes"]:
        return -1.0
    if data["avg_ms"] <= 0 or data["median_ms"] <= 0 or data["p99_ms"] <= 0:
        return -1.0
    avg_speedup    = BENCHMARK_STATS["avg_ms"]    / data["avg_ms"]
    p99_speedup    = BENCHMARK_STATS["p99_ms"]    / data["p99_ms"]
    median_speedup = BENCHMARK_STATS["median_ms"] / data["median_ms"]
    return (0.3 * avg_speedup + 0.2 * p99_speedup + 0.5 * median_speedup) - 1.0


def build_prompt(context_text):
    return SYSTEM_PROMPT + str(BENCHMARK_STATS) + "\n\n## Target function with context\n" + context_text


def generate_response(model, tokenizer, prompt, max_new_tokens=512):
    FastLanguageModel.for_inference(model)
    messages = [{"role": "user", "content": [{"type": "text", "text": prompt}]}]
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(text=text, return_tensors="pt").to(model.device)
    with torch.no_grad():
        output_ids = model.generate(**inputs, max_new_tokens=max_new_tokens)
    result = tokenizer.decode(output_ids[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
    FastLanguageModel.for_training(model)
    return result


def load_lora_model():
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=HF_MODEL,
        max_seq_length=4096,
        load_in_4bit=True,
    )
    model = FastLanguageModel.get_peft_model(
        model,
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"],
        use_gradient_checkpointing="unsloth",
    )
    return model, tokenizer


def make_grpo_reward_fn(prompt_map):
    def reward_fn(completions, prompt_id=None, **kwargs):
        rewards = []
        for i, completion in enumerate(completions):
            idx = prompt_id[i] if prompt_id is not None else i % len(prompt_map)
            func, sources, functions = prompt_map[idx]

            replace_function(func, completion, functions, sources)
            bench_data = run_bench()
            restore_all()

            if bench_data is None:
                r = -1.0
                wandb.log({"grpo/reward": r})
            else:
                r = calculate_reward(bench_data)
                wandb.log({
                    "grpo/reward": r,
                    "grpo/avg_ms": bench_data["avg_ms"],
                    "grpo/median_ms": bench_data["median_ms"],
                    "grpo/p99_ms": bench_data["p99_ms"],
                })
            rewards.append(r)
        return rewards
    return reward_fn


def save_best_model(model, tokenizer, reward, best_reward, path="./lora_checkpoints/best"):
    if reward <= best_reward:
        return best_reward
    print(f"New best reward {reward:.4f} (prev {best_reward:.4f}), saving model...")
    model.save_pretrained(path)
    tokenizer.save_pretrained(path)
    wandb.log({"best_reward": reward})
    wandb.run.summary["best_reward"] = reward
    wandb.run.summary["best_model_path"] = path
    return reward


def run_training_step(successful_rewrites, model, tokenizer, best_reward):
    if len(successful_rewrites) < 4:
        return best_reward

    sources   = _read_sources(GENGIN)
    functions = find_functions(sources)
    structs   = find_structs(sources)

    prompts    = []
    prompt_map = {}

    for i, func in enumerate(functions):
        text = show_context(func, functions, structs, returnString=True)
        if text is None:
            continue
        prompts.append({"prompt": build_prompt(text), "prompt_id": i})
        prompt_map[i] = (func, sources, functions)

    if not prompts:
        return best_reward

    dataset    = Dataset.from_list(prompts)
    reward_fn  = make_grpo_reward_fn(prompt_map)

    grpo_cfg = GRPOConfig(
        output_dir="./lora_checkpoints",
        learning_rate=5e-5,
        num_train_epochs=1,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=8,
        max_new_tokens=512,
        num_generations=4,
        save_steps=50,
        logging_steps=1,
        report_to="wandb",
    )

    trainer = GRPOTrainer(
        model=model,
        args=grpo_cfg,
        train_dataset=dataset,
        reward_funcs=reward_fn,
        processing_class=tokenizer,
    )

    print("Running GRPO training step...")
    trainer.train()
    trainer.save_model("./lora_checkpoints/latest")

    avg_reward_in_batch = sum(r["reward"] for r in successful_rewrites) / len(successful_rewrites)
    best_reward = save_best_model(model, tokenizer, avg_reward_in_batch, best_reward)
    return best_reward


RESPONSE_LOG = os.path.join(os.path.dirname(os.path.abspath(__file__)), "responses.log")

def log_response(step, func, response_text, reward):
    with open(RESPONSE_LOG, "a") as f:
        f.write(f"\n{'='*60}\n")
        f.write(f"step={step}  func={func}  reward={reward:.4f}\n")
        f.write(f"{'-'*60}\n")
        f.write(response_text)
        f.write("\n")
    wandb.log({
        "step": step,
        "response": wandb.Html(f"<pre>{response_text}</pre>"),
        "response_func": func,
        "response_reward": reward,
    })


if __name__ == "__main__":
    wandb.init(project="gengin-rl", config={
        "hf_model": HF_MODEL,
        "benchmark_baseline": BENCHMARK_STATS,
    })

    model, tokenizer = load_lora_model()

    successful_rewrites = []
    best_reward = 0.0
    step = 0

    while True:
        sources   = _read_sources(GENGIN)
        functions = find_functions(sources)
        structs   = find_structs(sources)

        iteration_rewards = []

        for func in functions:
            print("Selected function:", func)
            text = show_context(func, functions, structs, returnString=True)
            if text is None:
                print("Could not find context for the function.")
                continue

            prompt        = build_prompt(text)
            response_text = generate_response(model, tokenizer, prompt)

            print("Model response :\n", response_text)

            replace_function(func, response_text, functions, sources)
            bench_data = run_bench()
            restore_all()

            if bench_data is None:
                print("Benchmarking failed. Please check the code for errors.")
                wandb.log({"step": step, "reward": -1.0, "bench_failed": True})
                continue

            reward = calculate_reward(bench_data)
            iteration_rewards.append(reward)

            wandb.log({
                "step":       step,
                "reward":     reward,
                "avg_ms":     bench_data["avg_ms"],
                "median_ms":  bench_data["median_ms"],
                "p99_ms":     bench_data["p99_ms"],
                "is_correct": bench_data["frame_hashes"] == BENCHMARK_STATS["frame_hashes"],
            })

            log_response(step, func, response_text, reward)
            print(f"Step {step} | reward {reward:.4f} | avg {bench_data['avg_ms']:.3f}ms")

            if reward > 0.0:
                print(f"Successful rewrite with reward {reward:.4f}.")
                successful_rewrites.append({"response": response_text, "prompt": prompt, "reward": reward})

            step += 1

        if iteration_rewards:
            iter_avg = sum(iteration_rewards) / len(iteration_rewards)
            print(f"Iteration average reward: {iter_avg:.4f} over {len(iteration_rewards)} functions")
            wandb.log({"iteration_avg_reward": iter_avg})
            best_reward = save_best_model(model, tokenizer, iter_avg, best_reward)

        if len(successful_rewrites) >= 4 and step % 20 == 0:
            best_reward = run_training_step(successful_rewrites, model, tokenizer, best_reward)