LLM based optimalization of functions

Training Pipeline

on init:
    run benchmarks save results
    parse project to get each function

training:
    Feed llm function and function data
    LLM will provide optimized function
    Run test
    Calculate reward based on speed and if program fail


Used tech:
    LORA
    PPO
    Qwen3 1.7b