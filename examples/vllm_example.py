from vllm import LLM, SamplingParams
import torch

prompts = [
    "Hello, my name is",
    "The president of the United States is",
    "The capital of France is",
    "The future of AI is",
]
sampling_params = SamplingParams(temperature=0.8, top_p=0.95)

# LLM -> LLMEngine -> EngineCoreClient
llm = LLM(model="/data/zwt/model/models/Qwen/Qwen3-0.6B", enforce_eager=True)

# torch.cuda.memory._record_memory_history(max_entries=100000)
outputs = llm.generate(prompts, sampling_params)
# Dump memory snapshot history to a file and stop recording
# torch.cuda.memory._dump_snapshot("profile.pkl")
# torch.cuda.memory._record_memory_history(enabled=None)

# Print the outputs.
# 打印输出

for output in outputs:
    prompt = output.prompt
    generated_text = output.outputs[0].text
    print(f"Prompt: {prompt!r}, Generated text: {generated_text!r}")
