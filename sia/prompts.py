"""Prompt builders for the meta-agent and feedback-agent.

Moved verbatim out of orchestrator.py. The exact text is product-critical and is
locked by the golden-master tests in tests/test_prompts_snapshot.py.
"""

from __future__ import annotations

import json
import os
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from sia.providers import Provider
    from sia.run_setup import TaskFiles


def _reference_section(task_files: TaskFiles, reference_dir: str | None) -> str:
    """The reference paragraph of the meta prompt.

    Default/single-file: embed the seed code verbatim (byte-identical to the original).
    Multi-file directory: point the agent at the on-disk reference and tell it to read
    the files itself and declare any extra deps in requirements.txt.
    """
    if reference_dir is None:
        return (
            "Here is a sample target_agent.py showing the complete implementation pattern "
            f"(READ THE ENTIRE FILE):\n{task_files.reference_target_agent_py}"
        )
    return (
        f"Your reference agent implementation has been placed in your working directory ({reference_dir}). "
        "It may span multiple files. READ IT YOURSELF with your file tools (Read/Glob/Grep) — study the "
        "entrypoint and any helper modules — then write your target_agent.py in the same directory.\n"
        "If your target_agent.py needs third-party packages, list them (one per line) in a requirements.txt "
        "in your working directory; they are installed before the target agent runs."
    )


def _build_weights_meta_prompt(
    task_files: TaskFiles,
    task_model: str,
    working_dir: str,
    training_sandbox: str = "modal",
) -> str:
    """Build the meta-agent prompt for RL-based weight tuning (train.py).

    Args:
        training_sandbox: "modal" (default) or "sandboxfusion" for code execution
    """
    # RL Integration Guide (sections 1-9)
    RL_GUIDE = """# RL Integration Guide: Custom Task Tuning with Tinker-Cookbook

This guide provides the necessary architectural context and implementation patterns to integrate a custom Agent/Task into the `tinker-cookbook` RL pipeline. Use this to build a task-specific `Env`, `EnvGroupBuilder`, and `RLDataset`.

---

## 1. Core Abstractions Mapping

| Tinker Class | Role in RL | Custom Implementation Goal |
| :--- | :--- | :--- |
| **`Env`** | The "World" | Manages a single trajectory (one agent, one problem). Handles tool-calls and intermediate rewards. |
| **`EnvGroupBuilder`** | The "Orchestrator" | Creates $N$ environments for the **same** problem. Handles final group-level rewards (GRPO). |
| **`RLDataset`** | The "Task Source" | Groups `EnvGroupBuilder` instances into batches. Feeds the training loop. |

---

## 2. Implementing the `Env`

The `Env` must manage the conversation state and define how the agent interacts with tools.

```python
from tinker_cookbook.rl.types import Env, StepResult
from tinker_cookbook.renderers import Renderer

class CustomAgentEnv(Env):
    def __init__(self, task_data, renderer: Renderer):
        self.task = task_data
        self.renderer = renderer
        self.messages = []

    async def initial_observation(self):
        self.messages = [{"role": "user", "content": self.task["query"]}]
        prompt, stop_cond = self.renderer.build_generation_prompt(self.messages)
        return prompt, stop_cond

    async def step(self, action_tokens: list[int], extra=None):
        response_text = self.renderer.tokenizer.decode(action_tokens)

        # LOGIC: Execute tools or parse final answer
        tool_result, is_done = await execute_agent_tools(response_text)

        # Step reward (e.g., formatting check)
        reward = -0.1 if "invalid_tool" in response_text else 0.0

        self.messages.append({"role": "assistant", "content": response_text})
        if tool_result:
            self.messages.append({"role": "user", "content": f"Result: {tool_result}"})

        next_obs, next_stop = self.renderer.build_generation_prompt(self.messages)
        return StepResult(
            reward=reward, episode_done=is_done,
            next_observation=next_obs, next_stop_condition=next_stop
        )
```

---

## 3. Implementing the `EnvGroupBuilder`

This class is responsible for spawning $N$ environments for a single problem and computing the final outcome.

```python
from tinker_cookbook.rl.types import EnvGroupBuilder, Trajectory
from typing import Sequence

class CustomTaskGroupBuilder(EnvGroupBuilder):
    def __init__(self, task_data, group_size: int, renderer: Renderer):
        self.task = task_data
        self.group_size = group_size
        self.renderer = renderer

    async def make_envs(self) -> Sequence[CustomAgentEnv]:
        # Return N copies of the environment for the same task
        return [CustomAgentEnv(self.task, self.renderer) for _ in range(self.group_size)]

    async def compute_group_rewards(self, trajectories: list[Trajectory], envs: Sequence[CustomAgentEnv]):
        # Called after all N rollouts finish. Use this for "Final Success" rewards.
        rewards_and_metrics = []
        for traj in trajectories:
            final_text = self.renderer.tokenizer.decode(traj.transitions[-1].ac.tokens)
            is_correct = check_success(final_text, self.task["answer"])

            # Final Reward: 1.0 for success, 0.0 for failure
            reward = 1.0 if is_correct else 0.0
            rewards_and_metrics.append((reward, {"is_correct": float(is_correct)}))

        return rewards_and_metrics
```

---

## 5. Implementing a Custom `RLDatasetBuilder`

The `RLDatasetBuilder` is the entry point for your data. It must be a `@chz.chz` dataclass.

```python
from tinker_cookbook.rl.types import RLDataset, RLDatasetBuilder
from tinker_cookbook import renderers, tokenizer_utils
import chz
import json

class CustomTaskDataset(RLDataset):
    def __init__(self, tasks, batch_size: int, group_size: int, renderer):
        self.builders = [CustomTaskGroupBuilder(t, group_size, renderer) for t in tasks]
        self.batch_size = batch_size

    def get_batch(self, index: int):
        start = index * self.batch_size
        return self.builders[start : start + self.batch_size]

    def __len__(self):
        return (len(self.builders) + self.batch_size - 1) // self.batch_size

@chz.chz
class MyDatasetBuilder(RLDatasetBuilder):
    batch_size: int
    group_size: int
    model_name: str
    renderer_name: str = "qwen3_instruct"
    data_path: str = "data.jsonl"

    async def __call__(self) -> tuple[RLDataset, RLDataset | None]:
        # 1. Load your raw data
        with open(self.data_path) as f:
            tasks = [json.loads(line) for line in f]

        # 2. Setup Renderer & Tokenizer
        tokenizer = tokenizer_utils.get_tokenizer(self.model_name)
        renderer = renderers.get_renderer(self.renderer_name, tokenizer=tokenizer)

        # 3. Return Train (and optional Test) Dataset
        train_ds = CustomTaskDataset(tasks, self.batch_size, self.group_size, renderer)
        return train_ds, None
```

### ⚠️ Critical Warning on Missing Ground Truth, Reward Design, & Data Leakage ⚠️

When dynamically parsing task datasets (e.g., in your `RLDatasetBuilder`), be highly aware of the following realities regarding dataset structure, missing answers, and evaluation constraints:

1. **Dataset Paths:** The provided `--dataset_dir` will point directly to the accessible dataset. You must train using only this accessible data. Be robust in your file discovery (e.g., using `glob` to find JSON/CSV files and `evaluate.py`).
2. **Missing Ground Truth:** The available dataset is typically designed for evaluation and **may have its ground truth answers hidden or completely missing**. If you naively parse these files and grade against non-existent answers, your group rewards will be a constant `0.0` or `0.5`, leading to zero variance and failed GRPO training.
3. **Reward Workarounds (Crucial):** Because the ground truth answers may be missing, **you must implement a workaround to compute meaningful rewards**. A highly recommended approach is **Majority Voting (Self-Consistency)** across the generated samples within the `EnvGroupBuilder`. By rewarding the most common answer in the group, you provide a non-constant reward signal (variance) without needing the actual answer key. Do not let the reward remain constant.
4. **Data Leakage & Shuffling:** Make sure you properly shuffle the train set. If you do have access to ground truth, ensure you properly split your dataset into train and test sets to prevent overfitting. Do not evaluate your final inference run on the exact same examples you used for training.

---

## 6. Task-Specific Grading & Reward Computation

**Each task has its own `evaluate.py`** that defines how to grade agent outputs. Your `Env` and `EnvGroupBuilder` must integrate with this task-specific logic.

### Grading Examples by Task Type

- **GPQA**: Compare predicted answer letters (A/B/C/D) to ground truth
- **LawBench**: Compare predicted charge labels to ground truth categories
- **Chess**: Compare predicted moves to best moves in ground truth
- **Code Tasks**: Execute generated code and check against test cases

Your `EnvGroupBuilder.compute_group_rewards()` should:
1. Get the agent's final output
2. Call the task's grading logic (via `evaluate.py` or imported functions)
3. Convert results to rewards (e.g., `1.0` if correct, `0.0` if wrong)
4. Return rewards + metrics

```python
async def compute_group_rewards(self, trajectories, envs):
    \"\"\"Compute final rewards using task-specific grading.\"\"\"
    rewards_and_metrics = []

    for traj in trajectories:
        # 1. Extract final output from trajectory
        final_output = extract_output(traj)

        # 2. Call task-specific grading function
        # (imported from task's evaluate.py or grading module)
        is_correct, metrics = await grade_task(final_output, self.task)

        # 3. Convert to reward
        reward = 1.0 if is_correct else 0.0
        rewards_and_metrics.append((reward, metrics))

    return rewards_and_metrics
```

---

## 7. Reward Shaping Strategy

Tinker-cookbook uses **GRPO** (Group Relative Policy Optimization).

### Step Rewards (During Trajectory - `Env.step`)

**Format Correctness** is critical:

```python
async def step(self, action_tokens: list[int], extra=None):
    response_text = self.renderer.tokenizer.decode(action_tokens)

    # Check FORMAT correctness (task-specific)
    format_reward = 0.0

    # GPQA: Must output JSON with answer field
    if self.task_type == "gpqa":
        if is_valid_json_answer(response_text):  # {"answer": "A"}
            format_reward = +0.5  # Reward correct format
        else:
            format_reward = -0.5  # Penalize wrong format

    # LawBench: Must extract to valid charge label
    elif self.task_type == "lawbench":
        if can_extract_charge_label(response_text):  # Must match one of 191 classes
            format_reward = +0.5
        else:
            format_reward = -0.5

    # Chess: Must output "solution = [...] or solution = <number>"
    elif self.task_type == "chess":
        if has_solution_field(response_text):  # solution = ...
            format_reward = +0.5
        else:
            format_reward = -0.5

    # Other step rewards
    per_turn_penalty = -0.01  # Penalize long responses
    hallucination_penalty = -0.3 if "hallucinate" in response_text else 0.0

    # Total step reward
    reward = format_reward + per_turn_penalty + hallucination_penalty

    return StepResult(reward=reward, ...)
```

### Final Rewards (After Episode - `EnvGroupBuilder.compute_group_rewards`)

```python
async def compute_group_rewards(self, trajectories, envs):
    \"\"\"Final rewards based on correctness via task grading.\"\"\"
    rewards_and_metrics = []

    for traj in trajectories:
        final_output = extract_output(traj)

        # Call task-specific grader
        is_correct, metrics = await grade_task(final_output, self.task)

        # Final reward: 1.0 for correct, 0.0 for wrong
        final_reward = 1.0 if is_correct else 0.0

        rewards_and_metrics.append((final_reward, metrics))

    return rewards_and_metrics
```

### Overall Reward Strategy

| Type | Where | Examples | Purpose |
|------|-------|----------|---------|
| **Format Reward** | `Env.step()` | +0.5 correct, -0.5 wrong | Guide model to output correct format early |
| **Per-turn Penalty** | `Env.step()` | -0.01/step | Encourage conciseness |
| **Correctness Reward** | `EnvGroupBuilder.compute_group_rewards()` | +1.0 correct, 0.0 wrong | Final success metric |

### General Rules

1.  **Don't Normalize:** Tinker automatically centers rewards per group (`advantage = reward - group_mean`).
2.  **Format is Critical:** Wrong format = model can't be graded = negative step reward.
3.  **Correctness is Final:** Use `EnvGroupBuilder.compute_group_rewards` for final outcome-based rewards.
4.  **Sparse is Fine:** RL works best when the model can clearly distinguish a "Winner" from a "Loser" within the same 8-sample group.

---

## 8. The Training Pipeline

```python
from tinker_cookbook.rl.train import main, Config
import asyncio

async def run():
    config = Config(
        model_name="Qwen/Qwen3-4B-Instruct-2507",
        recipe_name="custom_task_tuning",
        dataset_builder=MyDatasetBuilder(
            batch_size=16,
            group_size=8,
            model_name="Qwen/Qwen3-4B-Instruct-2507"
        ),
        learning_rate=1e-5,
        max_tokens=1024,
        log_path="./runs/experiment_1"
    )
    await main(config)

if __name__ == "__main__":
    asyncio.run(run())
```

---

## 9. Troubleshooting: Zero Scores

If your training loop shows **score = 0** across all trajectories, follow these debugging steps:

### Step 1: Verify Dataset Alignment
- **Issue:** You may be scoring predictions for answers that don't exist in your dataset
- **Fix:**
  - Check that `compute_group_rewards()` is comparing against the correct answer field in `self.task`
  - Verify the answer exists and is not empty or malformed
  - Print the final answer and expected answer for the first few samples to confirm alignment

### Step 2: Check the Sandbox (if using code execution)
If your agent executes code via a sandbox:

#### If using **Local Sandbox (SandboxFusion)**:
1. **Verify it's running:**
   ```bash
   curl http://localhost:8080/health
   ```
   Should return a 200 response.

2. **Check concurrency limits:**
   - If concurrency is maxed out, the machine will freeze
   - Ensure you're not running too many concurrent requests
   - Add reasonable timeout and retry logic:
   ```python
   client = SandboxFusionClient(
       base_url="http://localhost:8080",
       timeout=30,
       max_retries=2
   )
   ```

3. **Test with a sample script:**
   ```python
   # Test if sandbox is working and returning expected results
   test_code = \"\"\"
   print("Hello from sandbox")
   result = 1 + 1
   print(f"Result: {result}")
   \"\"\"
   result = await client.run(
       code=test_code,
       timeout=10
   )
   print(result.stdout)  # Should show "Hello from sandbox" and "Result: 2"
   ```

#### If using **Modal (Cloud Sandbox)**:
- Modal should handle concurrency automatically
- If scores are still 0, verify your `modal.Image` has all required dependencies installed
- Check Modal logs for execution errors

### Step 3: Isolate the Problem
- **If sandbox test passes:** The problem is in your `Env` or reward logic. Check `compute_group_rewards()`
- **If sandbox test fails:** Fix sandbox connectivity before proceeding with training"""

    # Configure sandbox instruction based on training_sandbox choice
    if training_sandbox == "sandboxfusion":
        sandbox_instruction = """10. SANDBOX FUSION URL (IMPORTANT FOR CODE EXECUTION):
    - If your train.py uses SandboxFusionClient for code execution, configure it with:
      ```python
      import os
      from tinker_cookbook.sandbox import SandboxFusionClient
      sandbox_url = os.getenv("SANDBOX_URL", "http://localhost:8080")
      client = SandboxFusionClient(base_url=sandbox_url)
      ```
    - The SANDBOX_URL environment variable will be passed automatically at runtime"""
    else:
        sandbox_instruction = """10. MODAL CONFIGURATION (IMPORTANT FOR CODE EXECUTION):
    - Use Modal SDK (https://modal.com) as the default sandbox for executing code in your RL pipeline
    - For code execution in your RL components, use Modal Functions with @modal.function decorator
    - Modal provides better reliability, isolation, and scaling compared to alternatives
    - Example:
      ```python
      import modal
      app = modal.App("training-app")

      @app.function()
      def execute_code(code: str) -> str:
          exec(code)
          return result
      ```
    - Ensure MODAL_TOKEN_ID and MODAL_TOKEN_SECRET are set in the environment (automatically available)"""

    return f"""You are a meta-agent. Your task is to create a training script which can execute a task.
In this 'weights' mode, your primary goal is to implement a Reinforcement Learning (RL) pipeline to tune the model's performance for the specific task using the `tinker-cookbook` library.

---
RL INTEGRATION GUIDE:
{RL_GUIDE}
---

Go ahead and create a train.py which will train and return back the training model checkpoint url. You should take the reference agent and tune the model using this agent and whatever data or problem statement we have.

Here is the FULL TASK SPECIFICATION that your train.py will need to solve:
{task_files.task_md}

Here are a couple of sample task descriptions which the reference agent has to solve:
{task_files.sample_task_descriptions}

Here is a sample reference train.py implementation showing the training component patterns (READ THE ENTIRE FILE):
{task_files.reference_target_agent_py}

Here is a sample agent execution trajectory:
{json.dumps(task_files.sample_agent_execution, indent=2)}

CRITICAL RULES - FOLLOW EXACTLY:

1. The current working directory is {working_dir}. Create the train.py in the current working directory itself.

2. The train.py MUST accept two command-line arguments:
   - --dataset_dir: Absolute path to the dataset directory (READ-ONLY, provided at runtime)
   - --working_dir: Absolute path to the working directory (READ-WRITE, provided at runtime)

3. CRITICAL: The train.py must INCLUDE these paths in the prompt it sends to {task_model}. {task_model} MUST be explicitly told:
   - Where the dataset directory is located (the exact path from --dataset_dir)
   - Where the working directory is located (the exact path from --working_dir)
   - That it can ONLY READ from the dataset directory
   - That it can READ from and WRITE to the working directory

   DO NOT let {task_model} search for data in random locations. The prompt must say: "The dataset is at: <actual_dataset_dir_path>"

4. The train.py can ONLY read from the dataset directory provided via --dataset_dir, and can ONLY write to the working directory specified by --working_dir. It must NOT access any other directories on the filesystem.

5. EXECUTION LOGGING - CRITICAL: The train.py must log its execution trajectory in agent_execution.json in the working directory. Include all messages, tool calls, and their results in valid JSON format.

6. Do NOT attempt to write to or modify files inside the dataset directory. It is READ-ONLY.
7. The train.py should use only the "{task_model}" model when invoking the language model (do not use any other model).
8. DO NOT hardcode any specific dataset paths in the train.py code. The paths will be provided at runtime via command-line arguments and MUST be passed to {task_model} in the prompt.

9. CRITICAL RULES FOR WEIGHTS MODE - FOLLOW EXACTLY:

   **MANDATORY:** Your train.py must implement only tinker-cookbook components (Env, EnvGroupBuilder, RLDataset, RLDatasetBuilder). Do NOT write local training code or custom training loops. The tinker-cookbook library will execute your components directly via tinker_cookbook.rl.train.main().

   Your train.py must have this exact structure (copy-paste the template below):

   ```python
   import asyncio
   import os
   import sys
   import json
   from pathlib import Path
   from tinker_cookbook.rl.train import main as tinker_train_main, Config
   from tinker_cookbook.rl.types import Env, EnvGroupBuilder, RLDataset, RLDatasetBuilder
   import chz

   # 1. Implement Env class (manages single trajectory)
   class MyEnv(Env):
       def __init__(self, task_data):
           self.task = task_data
           # ... implement __init__, initial_observation(), step()

       async def initial_observation(self):
           # Return initial observation
           pass

       async def step(self, action_tokens, extra=None):
           # Execute step and return StepResult
           pass

   # 2. Implement EnvGroupBuilder class (creates N environments, computes rewards)
   class MyGroupBuilder(EnvGroupBuilder):
       def __init__(self, task_data, group_size):
           self.task = task_data
           self.group_size = group_size

       async def make_envs(self):
           return [MyEnv(self.task) for _ in range(self.group_size)]

       async def compute_group_rewards(self, trajectories, envs):
           # Compute rewards using task-specific grading
           rewards_and_metrics = []
           for traj in trajectories:
               # Extract final output, grade it, compute reward
               reward = 1.0 if correct else 0.0
               rewards_and_metrics.append((reward, {{"is_correct": float(correct)}}))
           return rewards_and_metrics

   # 3. Implement RLDataset class (batches EnvGroupBuilders)
   class MyRLDataset(RLDataset):
       def __init__(self, tasks, batch_size, group_size):
           self.builders = [MyGroupBuilder(t, group_size) for t in tasks]
           self.batch_size = batch_size

       def get_train_batch(self, idx):
           start = idx * self.batch_size
           return self.builders[start:start + self.batch_size]

       def num_train_batches(self):
           return (len(self.builders) + self.batch_size - 1) // self.batch_size

       def get_eval_batch(self, idx):
           return self.get_train_batch(idx)

       def num_eval_batches(self):
           return self.num_train_batches()

   # 4. Implement RLDatasetBuilder class with @chz.chz decorator
   @chz.chz
   class MyDatasetBuilder(RLDatasetBuilder):
       batch_size: int
       group_size: int
       model_name: str

       async def __call__(self):
           tasks = load_tasks_from_dataset_dir()  # Load from --dataset_dir
           dataset = MyRLDataset(tasks, self.batch_size, self.group_size)
           return dataset, None

   # 5. Main function - call tinker_cookbook.rl.train.main()
   async def main(dataset_dir, working_dir):
       config = Config(
           model_name="Qwen/Qwen3-4B-Instruct-2507",
           recipe_name="custom_task_tuning",
           dataset_builder=MyDatasetBuilder(
               batch_size=4,
               group_size=8,
               model_name="Qwen/Qwen3-4B-Instruct-2507"
           ),
           learning_rate=1e-5,
           max_tokens=512,
           log_path=os.path.join(working_dir, "training_logs")
       )
       results = await tinker_train_main(config)
       return results

   if __name__ == "__main__":
       import argparse
       parser = argparse.ArgumentParser()
       parser.add_argument("--dataset_dir", required=True)
       parser.add_argument("--working_dir", required=True)
       args = parser.parse_args()

       results = asyncio.run(main(args.dataset_dir, args.working_dir))
       print(f"Training complete. Results: {{results}}")
   ```

   **DO NOT deviate from this structure. DO NOT implement train_epoch(), evaluate(), or custom training loops.**
   **Your ONLY job is to fill in: MyEnv, MyGroupBuilder, MyRLDataset, MyDatasetBuilder, and load_tasks_from_dataset_dir().**
   **Everything else must match the template exactly.**

   - Implement ONLY the required components:
     * Env class (section 2 of RL Integration Guide)
     * EnvGroupBuilder class (section 3)
     * RLDataset class (section 5 - data splitting and batching)
     * RLDatasetBuilder class with @chz.chz decorator (section 5)

   - Look for an `evaluate.py` script in the dataset directory and use it in your EnvGroupBuilder.compute_group_rewards() to calculate actual rewards.

   - If a train/test split is not already provided in the dataset, your RLDataset MUST dynamically split data into training and evaluation sets to avoid overfitting.

   - The GRPO training, model inference via Tinker API, and reward computation are all handled by tinker_cookbook.rl.train.main() - you only need to provide the Config and dataset components.

   - Save all training results (metrics, checkpoint paths, logs) to {working_dir}/results/ directory in JSON format.

   - You MUST NOT simulate model responses. tinker_cookbook.rl.train.main() handles all Tinker API calls automatically.

{sandbox_instruction}

Example invocation (paths will vary at runtime):
    python train.py --dataset_dir /path/to/dataset --working_dir /path/to/working
"""


def _build_tpu_meta_prompt(
    task_files: TaskFiles,
    task_model: str,
    working_dir: str,
) -> str:
    """Build the meta-agent prompt for TPU-native GRPO training with QLoRA + FSDP.

    Instructs the meta-agent to write a train.py that uses PyTorch/XLA
    with QLoRA-style 4-bit quantization + PEFT LoRA + Fully Sharded Data
    Parallelism on Google TPU v5e-8 (128 GB HBM across 8 cores).

    bitsandbytes es CUDA-only. En TPU implementamos cuantización 4-bit
    nativa con operaciones XLA (grupos de 128, empaquetado uint8).
    """
    return f"""You are a meta-agent. Your task is to create a training script which can execute a task
using TPU-native GRPO reinforcement learning on Google TPU v5e-8 hardware.

--- TPU GRPO TRAINING GUIDE (QLoRA + FSDP) ---

## Architecture Overview

You will write a **standalone train.py** that implements GRPO (Group Relative Policy Optimization)
training directly on TPU. The pipeline combina tres técnicas para maximizar el tamaño de modelo
que cabe en v5e-8 (128 GB HBM total):

| Técnica | Ahorro | Detalle |
|:--------|:-------|:--------|
| **QLoRA (int4)** | 4× vs bf16 | pesos cuantizados a 4-bit con escalado por grupo de 128 |
| **FSDP** | 8× | parámetros particionados entre los 8 cores |
| **PEFT LoRA** | ~99% | solo los adapters (~0.1% del modelo) tienen gradientes |

**Resultado: modelos de hasta ~160B caben en v5e-8.**

### Escalado de memoria en v5e-8

| Modelo | Formato | GB totales | FSDP (÷8) | LoRA + act. | Total/core | ¿Cabe? |
|:-------|:--------|:-----------|:-----------|:------------|:-----------|:-------|
| 7B     | int4    | 3.5        | 0.44       | ~2 GB       | ~2.5 GB    | ✓ fol |
| 13B    | int4    | 6.5        | 0.81       | ~4 GB       | ~4.8 GB    | ✓ |
| 70B    | int4    | 35         | 4.38       | ~6 GB       | ~10.4 GB   | ✓ |
| 120B   | int4    | 60         | 7.50       | ~8 GB       | ~15.5 GB   | ✓ |
| 180B   | int4    | 90         | 11.25      | ~8 GB       | ~19.25 GB  | ✗ (necesita más cores) |

Cada 7B params ocupa ~3.5 GB en int4, ~7 GB en int8, ~14 GB en bf16.

### Key TPU v5e-8 Details
- **8 TPU cores**, 16 GB HBM each → 128 GB total
- bitsandbytes **no funciona** en TPU (CUDA-only) → implementamos cuantización 4-bit nativa
- `XlaFullyShardedDataParallel` + PEFT LoRA para entrenar solo adapters
- **bfloat16** para los adapters LoRA y cálculos intermedios

### Required Environment Setup (already configured at runtime)
- `PJRT_DEVICE=TPU`, `XLA_USE_BF16=1`
- `TPU_CHIPS_PER_HOST_BOUNDS=4,1,1`, `TPU_HOST_BOUNDS=1,1,1`

## Required train.py Structure

Your train.py MUST follow this template EXACTLY:

```python
import os, sys, json, math, argparse, copy
import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import partial

# TPU + FSDP
import torch_xla.core.xla_model as xm
import torch_xla.distributed.xla_multiprocessing as xmp
from torch_xla.distributed.fsdp import XlaFullyShardedDataParallel as FSDP
from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy

# PEFT LoRA
from peft import LoraConfig, get_peft_model, TaskType
from peft.tuners.lora import LoraLayer

from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig

# ──────────────────────────────────────────────
# Cuantización 4-bit nativa para TPU (XLA-compatible)
# bitsandbytes es CUDA-only. Esta implementación
# usa uint8 empaquetado + escalado por grupos de 128
# y corre 100% en TPU vía XLA.
# ──────────────────────────────────────────────

def quantize_int4(weight: torch.Tensor, group_size: int = 128) -> tuple:
    \"\"\"Cuantiza un tensor bf16/fp32 a int4 simétrico con escalado por grupo.

    Cada grupo de ``group_size`` pesos se normaliza a rango [-7, 7].
    Dos valores int4 se empaquetan por byte uint8.

    Args:
        weight: [out_dim, in_dim] en bf16 o fp32.
        group_size: elementos por grupo de cuantización.

    Returns:
        qweight: [out_dim, in_dim // 2] uint8 empaquetado.
        scales: [num_groups] bf16 factores de escala.
    \"\"\"
    assert weight.numel() % group_size == 0, \
        f"weight.numel()={weight.numel()} must be divisible by group_size={group_size}"
    orig_shape = weight.shape
    weight_bf16 = weight.bfloat16()

    # Reshape a grupos
    grouped = weight_bf16.reshape(-1, group_size)          # [G, GS]

    # Escala simétrica int4: rango [-7, 7]
    absmax = grouped.abs().max(dim=-1, keepdim=True).values
    scale = absmax / 7.0
    scale = scale.clamp(min=1e-12)

    q = (grouped / scale).round().clamp(-7, 7).to(torch.int8)  # [G, GS]

    # Empaquetar: dos int4 por byte
    q = q.reshape(-1, group_size // 2, 2)                  # [G, GS/2, 2]
    lo = (q[:, :, 0].to(torch.uint8) & 0x0F)              # nibble bajo
    hi = (q[:, :, 1].to(torch.uint8) & 0x0F) << 4          # nibble alto
    packed = lo | hi                                        # [G, GS/2]

    return packed.reshape(orig_shape[0], orig_shape[1] // 2), scale.reshape(-1)


def dequantize_int4(qweight: torch.Tensor, scales: torch.Tensor,
                    orig_shape: tuple, group_size: int = 128,
                    dtype: torch.dtype = torch.bfloat16) -> torch.Tensor:
    \"\"\"Descuantiza de int4 a bf16/fp32.

    Args:
        qweight: [out_dim, in_dim // 2] uint8 empaquetado.
        scales: [num_groups] bf16.
        orig_shape: (out_dim, in_dim) original.
        group_size: debe coincidir con el usado en quantize_int4.

    Returns:
        Tensor bf16 con forma orig_shape.
    \"\"\"
    out_dim, in_dim_half = qweight.shape
    in_dim = in_dim_half * 2
    num_groups = out_dim * in_dim // group_size

    # Desempaquetar
    lo = (qweight.to(torch.int8) << 4).to(torch.int8) >> 4   # sign-extend nibble bajo
    hi = (qweight.to(torch.int8) >> 4)                         # nibble alto
    unpacked = torch.stack([lo, hi], dim=-1).reshape(out_dim, in_dim)  # [out_dim, in_dim]

    # Aplicar escala por grupo
    grouped = unpacked.reshape(num_groups, group_size)
    deq = grouped * scales.unsqueeze(-1)
    return deq.reshape(orig_shape).to(dtype)


class QuantizedLinear(nn.Module):
    \"\"\"Capa Linear con pesos almacenados en int4 (4-bit cuantizados).

    Reemplaza nn.Linear en el modelo base. Los pesos se descuantizan
    on-the-fly durante forward. Compatible con XLA (TPU).

    Para usar con PEFT LoRA: el forward descuantiza w, calcula matmul,
    y LoRA se añade por separado via ``get_peft_model``.
    \"\"\"

    def __init__(self, in_features: int, out_features: int,
                 bias: bool = True, group_size: int = 128):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.group_size = group_size

        self.register_buffer('qweight', torch.empty(out_features, in_features // 2, dtype=torch.uint8))
        self.register_buffer('scales', torch.empty(out_features * in_features // group_size, dtype=torch.bfloat16))
        if bias:
            self.register_buffer('bias', torch.empty(out_features, dtype=torch.bfloat16))
        else:
            self.bias = None

    @classmethod
    def from_linear(cls, linear: nn.Linear, group_size: int = 128) -> 'QuantizedLinear':
        \"\"\"Crea un QuantizedLinear desde un nn.Linear, cuantizando sus pesos.\"\"\"
        ql = cls(linear.in_features, linear.out_features,
                 linear.bias is not None, group_size)
        w = linear.weight.data
        ql.qweight, ql.scales = quantize_int4(w, group_size)
        if linear.bias is not None:
            ql.bias = linear.bias.data.bfloat16()
        # Copiar atributos de dtype y device para compatibilidad
        return ql

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        orig_shape = (self.out_features, self.in_features)
        weight = dequantize_int4(self.qweight, self.scales, orig_shape,
                                 self.group_size, x.dtype)
        out = F.linear(x, weight, self.bias)
        return out


def replace_linear_with_quantized(model: nn.Module, group_size: int = 128,
                                  exclude_names: set | None = None) -> nn.Module:
    \"\"\"Recorre el modelo y reemplaza nn.Linear por QuantizedLinear.

    Args:
        model: Modelo de transformers.
        group_size: Tamaño de grupo para cuantización.
        exclude_names: Nombres de submódulos a excluir (e.g. {{"lm_head"}}).

    Returns:
        Modelo con capas lineales cuantizadas.
    \"\"\"
    exclude = exclude_names or set()
    for name, child in model.named_children():
        if name in exclude:
            continue
        if isinstance(child, nn.Linear):
            setattr(model, name, QuantizedLinear.from_linear(child, group_size))
        else:
            replace_linear_with_quantized(child, group_size, exclude)
    return model


# ──────────────────────────────────────────────
# GRPO Loss
# ──────────────────────────────────────────────

def grpo_loss(log_probs: torch.Tensor, ref_log_probs: torch.Tensor,
              rewards: torch.Tensor, epsilon: float = 0.2) -> torch.Tensor:
    ratio = torch.exp(log_probs - ref_log_probs)
    group_mean = rewards.mean(dim=-1, keepdim=True)
    group_std = rewards.std(dim=-1, keepdim=True) + 1e-8
    advantages = (rewards - group_mean) / group_std
    clipped = torch.clamp(ratio, 1.0 - epsilon, 1.0 + epsilon)
    return -torch.min(ratio * advantages, clipped * advantages).mean()


# ──────────────────────────────────────────────
# GRPOTrainer con QLoRA + FSDP
# ──────────────────────────────────────────────

class GRPOTrainer:
    \"\"\"GRPO trainer con QLoRA-on-TPU (int4 cuantización nativa) + FSDP + PEFT LoRA.

    Arquitectura de memoria:
    - Base model: pesos en int4 (0.5 bytes/param), congelados, sharded via FSDP
    - LoRA adapters: bf16, ~0.1% del modelo, los únicos con gradientes
    - Reference model: copia del base (int4) sin LoRA, para KL penalty
    \"\"\"

    def __init__(self, model_name: str, device, lr: float = 1e-4,
                 group_size: int = 8, max_length: int = 1024,
                 lora_r: int = 16, lora_alpha: int = 32,
                 lora_dropout: float = 0.05,
                 quantize: bool = True, group_size_q: int = 128):
        self.device = device
        self.group_size = group_size
        self.max_length = max_length
        self.kl_coeff = 0.1

        num_params_model = 0

        # ── 1. Cargar modelo base (en CPU) ──
        print(f"Loading model {{model_name}}...")
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
        )
        num_params_model = sum(p.numel() for p in model.parameters())

        # ── 2. Cuantizar a int4 (XLA-native, substituye bitsandbytes) ──
        if quantize:
            print(f"Quantizing base model to int4 (group_size={{group_size_q}})...")
            # lm_head no se cuantiza (output layer)
            model = replace_linear_with_quantized(model, group_size_q,
                                                   exclude_names={{'lm_head'}})
            # Freeze todos los parámetros cuantizados
            for p in model.parameters():
                p.requires_grad = False
        else:
            model = model.bfloat16()
            for p in model.parameters():
                p.requires_grad = False

        # ── 3. Añadir LoRA adapters via PEFT ──
        # Solo los adapters serán entrenables
        lora_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=lora_r,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            target_modules={{'q_proj', 'k_proj', 'v_proj', 'o_proj',
                             'gate_proj', 'up_proj', 'down_proj'}},
            bias='none',
        )
        model = get_peft_model(model, lora_config)
        model.print_trainable_parameters()

        # ── 4. FSDP wrap ──
        model = model.to(device)
        layer_cls = _detect_layer_class(model)
        if layer_cls is not None:
            model = FSDP(model, auto_wrap_policy=partial(
                transformer_auto_wrap_policy, transformer_layer_cls={{layer_cls}},
            ))
        else:
            model = FSDP(model)

        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"Trainable params: {{trainable/1e6:.2f}}M  "
              f"({trainable/num_params_model*100:.2f}% of base)")

        # Check memory: each int4 param = 0.5 bytes
        mem_params = num_params_model * 0.5 / 1e9  # GB total (int4)
        mem_per_core = mem_params / 8               # FSDP sharded
        mem_loRa = trainable * 2 / 1e9              # bf16 = 2 bytes
        mem_activations = 4                          # estimado
        mem_total = mem_per_core + mem_loRa + mem_activations
        print(f"Memory estimate: {{mem_params:.1f}}GB total (int4), "
              f"FSDP -> {{mem_per_core:.1f}}GB/core, "
              f"+ LoRA {{mem_loRa:.1f}}GB + act ~{{mem_activations}}GB "
              f"= {{mem_total:.1f}}GB/core (limit 16GB)")

        self.model = model

        # ── 5. Reference model (copia cuantizada sin LoRA) ──
        ref = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
        )
        if quantize:
            ref = replace_linear_with_quantized(ref, group_size_q,
                                                  exclude_names={{'lm_head'}})
        for p in ref.parameters():
            p.requires_grad = False
        ref = ref.bfloat16()
        ref = ref.to(device)
        if layer_cls is not None:
            self.ref_model = FSDP(ref, auto_wrap_policy=partial(
                transformer_auto_wrap_policy, transformer_layer_cls={{layer_cls}},
            ))
        else:
            self.ref_model = FSDP(ref)

        # ── 6. Tokenizer ──
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # ── 7. Optimizer (solo LoRA params) ──
        self.optimizer = torch.optim.AdamW(
            [p for p in model.parameters() if p.requires_grad], lr=lr
        )

    @torch.no_grad()
    def _generate_batch(self, input_ids, attention_mask, num_return):
        generated = self.model.generate(
            input_ids=input_ids, attention_mask=attention_mask,
            max_new_tokens=self.max_length // 2,
            num_return_sequences=num_return,
            do_sample=True, temperature=0.7, top_p=0.9,
            pad_token_id=self.tokenizer.pad_token_id,
            output_scores=True, return_dict_in_generate=True,
        )
        return generated.sequences, generated.scores

    @torch.no_grad()
    def generate_responses(self, prompts: list[str], num_responses: int) -> tuple:
        all_sequences = []
        all_log_probs = []
        for prompt in prompts:
            inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True,
                                    max_length=self.max_length // 2)
            inputs = {{k: v.to(self.device) for k, v in inputs.items()}}
            sequences, scores = self._generate_batch(
                inputs['input_ids'], inputs['attention_mask'], num_responses
            )
            prompt_len = inputs['input_ids'].shape[1]
            log_probs = []
            for sidx in range(num_responses):
                seq = sequences[sidx]
                gen_ids = seq[prompt_len:]
                with torch.no_grad():
                    out = self.model(seq.unsqueeze(0))
                    logits = out.logits[0, prompt_len - 1:-1]
                    lp = F.log_softmax(logits, dim=-1)
                    selected = lp.gather(1, gen_ids.unsqueeze(1)).squeeze(1)
                    log_probs.append(selected.mean().item())
            all_sequences.append(sequences)
            all_log_probs.append(log_probs)
        return all_sequences, all_log_probs

    @torch.no_grad()
    def compute_ref_log_probs(self, sequences_per_prompt, prompts):
        ref_log_probs = []
        for prompt, seq_group in zip(prompts, sequences_per_prompt):
            prompt_len = len(self.tokenizer(prompt)['input_ids'])
            for sidx in range(seq_group.shape[0]):
                gen_ids = seq_group[sidx, prompt_len:]
                with torch.no_grad():
                    out = self.ref_model(seq_group[sidx].unsqueeze(0))
                    logits = out.logits[0, prompt_len - 1:-1]
                    lp = F.log_softmax(logits, dim=-1)
                    selected = lp.gather(1, gen_ids.unsqueeze(1)).squeeze(1)
                    ref_log_probs.append(selected.mean().item())
        return ref_log_probs

    def train_step(self, prompts: list[str], reward_fn) -> dict:
        sequences, log_probs = self.generate_responses(prompts, self.group_size)
        rewards = reward_fn(prompts, sequences)
        ref_log_probs = self.compute_ref_log_probs(sequences, prompts)

        lp_t = torch.tensor(log_probs, device=self.device, dtype=torch.bfloat16)
        ref_t = torch.tensor(ref_log_probs, device=self.device, dtype=torch.bfloat16)
        rew_t = torch.tensor(rewards, device=self.device, dtype=torch.bfloat16)

        policy_loss = grpo_loss(lp_t, ref_t, rew_t)
        kl = (torch.exp(ref_t - lp_t) - 1.0 - (ref_t - lp_t)).mean()
        loss = policy_loss + self.kl_coeff * kl

        self.optimizer.zero_grad()
        loss.backward()
        xm.optimizer_step(self.optimizer, barrier=True)
        return {{'loss': loss.item(), 'kl': kl.item(), 'mean_reward': rew_t.mean().item()}}

    def save_checkpoint(self, path: str):
        if get_tpu_rank() == 0:
            os.makedirs(path, exist_ok=True)
            # Solo guardamos adapters LoRA + tokenizer
            unwrapped = self.model.module if hasattr(self.model, 'module') else self.model
            unwrapped.save_pretrained(path)
            self.tokenizer.save_pretrained(path)
            print(f"LoRA checkpoint saved: {{path}}")


# ──────────────────────────────────────────────
# Detección de capas transformer para FSDP wrap
# ──────────────────────────────────────────────

def _detect_layer_class(model) -> type | None:
    for child in model.modules():
        name = type(child).__name__
        module = type(child).__module__
        if 'LlamaDecoderLayer' in name:
            import transformers.models.llama.modeling_llama as m; return m.LlamaDecoderLayer
        if 'GemmaDecoderLayer' in name:
            import transformers.models.gemma.modeling_gemma as m; return m.GemmaDecoderLayer
        if 'Qwen2DecoderLayer' in name:
            import transformers.models.qwen2.modeling_qwen2 as m; return m.Qwen2DecoderLayer
        if 'MistralTransformerBlock' in name:
            import transformers.models.mistral.modeling_mistral as m; return m.MistralTransformerBlock
        if 'Phi3DecoderLayer' in name:
            import transformers.models.phi3.modeling_phi3 as m; return m.Phi3DecoderLayer
        if 'DeepseekV2DecoderLayer' in name:
            import transformers.models.deepseek_v2.modeling_deepseek_v2 as m; return m.DeepseekV2DecoderLayer
    return None


# ──────────────────────────────────────────────
# Data loading & rewards
# ──────────────────────────────────────────────

def load_tasks(dataset_dir: str) -> list[dict]:
    import glob as _glob
    tasks = []
    for fpath in _glob.glob(os.path.join(dataset_dir, '*.jsonl')):
        with open(fpath) as f:
            for line in f:
                line = line.strip()
                if line:
                    tasks.append(json.loads(line))
    if not tasks:
        import pandas as pd
        for fpath in _glob.glob(os.path.join(dataset_dir, '*.csv')):
            df = pd.read_csv(fpath)
            for _, row in df.iterrows():
                tasks.append(row.to_dict())
    if not tasks:
        for fpath in _glob.glob(os.path.join(dataset_dir, '*.json')):
            with open(fpath) as f:
                data = json.load(f)
                if isinstance(data, list):
                    tasks.extend(data)
                elif isinstance(data, dict):
                    tasks.append(data)
    return tasks


def make_reward_fn(dataset_dir: str, working_dir: str):
    import importlib.util as _util
    eval_path = os.path.join(dataset_dir, 'evaluate.py')
    if os.path.exists(eval_path):
        spec = _util.spec_from_file_location('evaluate', eval_path)
        evaluate = _util.module_from_spec(spec)
        spec.loader.exec_module(evaluate)
        return lambda prompts, seqs: evaluate.compute_rewards(prompts, seqs)
    def default_reward(prompts, seqs):
        rewards = []
        for prompt, seq_group in zip(prompts, seqs):
            for sidx in range(seq_group.shape[0]):
                text = seq_group[sidx]
                r = 0.0
                if len(text) > 50:
                    r += 0.1
                if b'```' in text or b'answer' in text.lower():
                    r += 0.5
                rewards.append(r)
        return rewards
    return default_reward


# ──────────────────────────────────────────────
# Per-core training process (xmp.spawn target)
# ──────────────────────────────────────────────

def _training_process(rank: int, dataset_dir: str, working_dir: str, model_name: str):
    device = xm.xla_device()
    world_size = get_world_size()

    batch_size = int(os.getenv('GRPO_BATCH_SIZE', '2'))
    group_size = int(os.getenv('GRPO_GROUP_SIZE', '8'))
    lr = float(os.getenv('GRPO_LEARNING_RATE', '1e-4'))
    num_epochs = int(os.getenv('GRPO_NUM_EPOCHS', '3'))
    max_length = int(os.getenv('GRPO_MAX_LENGTH', '1024'))
    lora_r = int(os.getenv('GRPO_LORA_R', '16'))
    quantize = os.getenv('GRPO_QUANTIZE', 'true').lower() == 'true'

    if rank == 0:
        print(f'QLoRA-TPU config: model={{model_name}}, batch={{batch_size}}, '
              f'group={{group_size}}, lr={{lr}}, lora_r={{lora_r}}, quantize={{quantize}}')

    tasks = load_tasks(dataset_dir)
    num_tasks = len(tasks)
    tasks_per_core = max(1, num_tasks // world_size)
    start = rank * tasks_per_core
    end = start + tasks_per_core if rank < world_size - 1 else num_tasks
    local_tasks = tasks[start:end]
    print(f'Core {{rank}}: {{len(local_tasks)}} tasks [{{start}}:{{end}})')

    trainer = GRPOTrainer(
        model_name=model_name, device=device,
        lr=lr, group_size=group_size, max_length=max_length,
        lora_r=lora_r, quantize=quantize,
    )
    reward_fn = make_reward_fn(dataset_dir, working_dir)

    results_dir = os.path.join(working_dir, 'results')
    os.makedirs(results_dir, exist_ok=True)
    all_metrics = []

    for epoch in range(num_epochs):
        for i in range(0, len(local_tasks), batch_size):
            batch_tasks = local_tasks[i:i + batch_size]
            prompts = [t.get('query', t.get('question', str(t))) for t in batch_tasks]
            metrics = trainer.train_step(prompts, reward_fn)
            all_metrics.append(metrics)
            if rank == 0:
                print(f'Epoch {{epoch+1}}/{{num_epochs}}  Batch {{i//batch_size + 1}}/'
                      f'{{(len(local_tasks) + batch_size - 1)//batch_size}}: '
                      f'loss={{metrics["loss"]:.4f}}  reward={{metrics["mean_reward"]:.4f}}')

        if rank == 0:
            trainer.save_checkpoint(os.path.join(results_dir, f'checkpoint_epoch_{{epoch + 1}}'))

    if rank == 0:
        metrics_path = os.path.join(results_dir, 'training_metrics.json')
        with open(metrics_path, 'w') as f:
            json.dump(all_metrics, f, indent=2)
        print(f'Metrics saved: {{metrics_path}}')
        result = {{
            'status': 'completed',
            'checkpoint_path': os.path.join(results_dir, f'checkpoint_epoch_{{num_epochs}}'),
            'num_epochs': num_epochs, 'num_tasks': num_tasks,
            'training_config': {{
                'model': model_name, 'batch_size': batch_size,
                'group_size': group_size, 'learning_rate': lr,
                'max_length': max_length, 'tpu_cores': world_size,
                'lora_r': lora_r, 'quantized_4bit': quantize,
            }},
        }}
        with open(os.path.join(working_dir, 'results.json'), 'w') as f:
            json.dump(result, f, indent=2)
        print(f'Results saved')


def tpu_main(dataset_dir: str, working_dir: str):
    import torch_xla.distributed.xla_multiprocessing as _xmp
    _xmp.spawn(_training_process, args=(dataset_dir, working_dir, '{task_model}'),
               nprocs=8, start_method='fork')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_dir', required=True)
    parser.add_argument('--working_dir', required=True)
    args = parser.parse_args()
    try:
        tpu_main(args.dataset_dir, args.working_dir)
    except (ImportError, RuntimeError) as e:
        print(f'xmp.spawn failed: {{e}}, falling back to single-process...')
        device = xm.xla_device()
        # Fallback simple
```

**DO NOT deviate from this structure.** Your ONLY job is to fill in:
1. The reward function logic in `make_reward_fn` (use the task's evaluate.py)
2. Task-specific prompt formatting in `load_tasks` if needed

**CRITICAL QLoRA + FSDP ON TPU RULES:**
- bitsandbytes **no funciona** en TPU. Usa `quantize_int4()` + `QuantizedLinear` incluidos en el template.
- `get_peft_model` añade LoRA a las capas cuantizadas. Solo estos adapters tienen gradientes.
- FSDP particiona el modelo base **ya cuantizado** (cada core tiene ~1/8 de los ~3.5 GB/7B params en int4).
- Reference model es una copia del base cuantizado sin LoRA (para calcular KL penalty).
- `model.generate()` funciona a través de FSDP (hace all-gather por capa).
- `xm.optimizer_step(barrier=True)` sincroniza los 8 cores después de cada step.
- Controla tamaño del modelo via `GRPO_QUANTIZE=true/false` y `GRPO_LORA_R=16/32/64`.
- Para modelos >70B: reduce `batch_size=1`, activa `GRPO_MAX_LENGTH=512`.
- Guarda solo adapters LoRA (`.save_pretrained()`), no el modelo base completo.
- Primer batch = compilación XLA (lento); batches siguientes usan HLO cacheados.

Env vars de configuración (todas opcionales):
```
GRPO_BATCH_SIZE=2        # ejemplos por step
GRPO_GROUP_SIZE=8        # respuestas por prompt para GRPO
GRPO_LEARNING_RATE=1e-4  # lr para adapters LoRA
GRPO_NUM_EPOCHS=3        # épocas de entrenamiento
GRPO_MAX_LENGTH=1024     # longitud máxima de secuencia
GRPO_LORA_R=16           # rango LoRA
GRPO_QUANTIZE=true       # false = bf16 sin cuantizar
```

Example invocation (paths vary at runtime):
    python train.py --dataset_dir /path/to/dataset --working_dir /path/to/working
"""


def build_meta_prompt(
    task_files: TaskFiles,
    task_model: str,
    working_dir: str,
    provider: Provider | None = None,
    reference_dir: str | None = None,
    focus: str = "harness",
    training_sandbox: str = "modal",
    training_backend: str = "tinker-api",
) -> str:
    """Build the meta-agent prompt for creating the initial target agent.

    Args:
        focus: "harness" (default) for code improvement or "weights" for RL-based tuning
        training_sandbox: "modal" (default) or "sandboxfusion" for train.py code execution

    For Anthropic and Google providers (and the default ``None``) the text is
    byte-identical to the original. For OpenAI-compatible providers a client-setup
    block is prepended instructing the meta-agent to refactor the reference agent to
    the ``openai`` SDK at the provider's base_url/api_key_env.

    ``reference_dir`` is set only for a multi-file directory reference: instead of
    embedding the seed code, the prompt points the agent at the on-disk reference so it
    reads the files with its own tools. ``None`` (default/single-file) keeps the
    historical embedded-seed text verbatim.
    """
    # Handle weights mode (RL-based tuning)
    if focus == "weights":
        if training_backend == "tpu-native":
            return _build_tpu_meta_prompt(task_files, task_model, working_dir)
        return _build_weights_meta_prompt(task_files, task_model, working_dir, training_sandbox=training_sandbox)

    # Harness mode (default - code/prompt improvement)
    reference_section = _reference_section(task_files, reference_dir)
    base = f"""You are a meta-agent. Your task is to create a target agent which can execute a task. Go ahead and create a target_agent.py for the target agent, which in turn can solve the given task.

Here is the FULL TASK SPECIFICATION that your target_agent.py will need to solve:
{task_files.task_md}

Here are a couple of sample task descriptions which the target agent has to solve:
{task_files.sample_task_descriptions}

{reference_section}

Here is a sample agent execution trajectory:
{json.dumps(task_files.sample_agent_execution, indent=2)}

CRITICAL RULES - FOLLOW EXACTLY:

1. The current working directory is {working_dir}. Create the target_agent.py in the current working directory itself.

2. The target_agent.py MUST accept two command-line arguments:
   - --dataset_dir: Absolute path to the dataset directory (READ-ONLY, provided at runtime)
   - --working_dir: Absolute path to the working directory (READ-WRITE, provided at runtime)

3. CRITICAL: The target_agent.py must INCLUDE these paths in the prompt it sends to {task_model}. {task_model} MUST be explicitly told:
   - Where the dataset directory is located (the exact path from --dataset_dir)
   - Where the working directory is located (the exact path from --working_dir)
   - That it can ONLY READ from the dataset directory
   - That it can READ from and WRITE to the working directory

   DO NOT let {task_model} search for data in random locations. The prompt must say: "The dataset is at: <actual_dataset_dir_path>"

4. The target agent can ONLY read from the dataset directory provided via --dataset_dir, and can ONLY write to the working directory specified by --working_dir. It must NOT access any other directories on the filesystem.

5. EXECUTION LOGGING - CRITICAL:

   The target_agent.py must log its execution trajectory properly. The format depends on the task type:

   **FOR TASKS WITH MULTIPLE INDEPENDENT SAMPLES** (e.g., GPQA with 198 questions, multiple test cases):
   - Create a folder: agent_execution/ in the working directory
   - Save each sample separately: execution_q0.json, execution_q1.json, execution_q2.json, etc.
   - Each file contains the complete trajectory for that ONE sample only
   - Files must be named sequentially: execution_q0.json, execution_q1.json, ...

   **FOR TASKS WITH SINGLE EXECUTION** (e.g., building one ML model, analyzing one dataset):
   - Save to a single file: agent_execution.json in the working directory
   - File contains the complete execution trajectory

   **HOW TO DETERMINE WHICH FORMAT**:
   - Read the task description carefully
   - If it mentions "independent items", "dataset with multiple records to process separately"
     → Use multi-trajectory (folder with multiple files)
   - If it's about "build a model", "analyze the dataset", "create one solution", "optimize one system"
     → Use single-trajectory (one JSON file)

   **FORMAT REQUIREMENTS** (both formats):
   - Use the same format as the sample agent execution trajectory provided above
   - Include all messages, tool calls, and their results
   - Ensure valid JSON (properly close all arrays/objects)
   - Make sure to properly close the JSON file(s) to avoid corruption

6. Do NOT attempt to write to or modify files inside the dataset directory. It is READ-ONLY.
7. The target_agent.py should use only the "{task_model}" model when invoking the language model (do not use any other model).
8. DO NOT hardcode any specific dataset paths in the target_agent.py code. The paths will be provided at runtime via command-line arguments and MUST be passed to {task_model} in the prompt.

Example invocation (paths will vary at runtime):
    python target_agent.py --dataset_dir /path/to/dataset --working_dir /path/to/working
"""
    if provider is None or provider.client_kind != "openai":
        return base
    return build_target_client_setup(provider, task_model) + base


def build_target_client_setup(provider: Provider, task_model: str) -> str:
    """Prompt block telling the meta-agent how to reach an OpenAI-compatible target model.

    The reference target_agent.py shown later in the prompt may use a different SDK
    (e.g. the Gemini SDK); this block instructs the meta-agent to refactor it to the
    ``openai`` SDK configured for ``provider``.
    """
    return f"""=== TARGET MODEL CLIENT SETUP (OpenAI-compatible provider: {provider.name}) ===

The target model "{task_model}" is served by an OpenAI-compatible API. The reference
target_agent.py shown below may use a different SDK (e.g. the Gemini SDK) — you MUST
refactor your target_agent.py to use the `openai` SDK configured for this provider
(do NOT use the anthropic or google SDK):

    import os
    from openai import OpenAI

    client = OpenAI(
        base_url="{provider.base_url}",
        api_key=os.environ["{provider.api_key_env}"],
    )

Call client.chat.completions.create(model="{task_model}", ...) using OpenAI-style
messages (and OpenAI function calling / response_format where the reference uses
structured output). Do NOT compute a dollar cost: per-provider pricing is unknown, so
set any cost field to 0 (token counts from the API response are still fine to record).

"""


def build_feedback_prompt(
    current_gen: int,
    max_gen: int,
    task_files: TaskFiles,
    agent_py: str,
    task: str,
    execution_status: str,
    execution_section: str,
    run_dir: str,
    next_gen_dir: str,
    previous_gens: str,
    task_model: str,
    provider: Provider | None = None,
    requirements_dir: str | None = None,
    focus: str = "harness",
    training_backend: str = "tinker-api",
) -> str:
    """Build the feedback agent prompt for improving the target agent or train.py.

    focus: "harness" (default) for code improvement or "weights" for RL-based tuning

    ``requirements_dir`` is set when the reference declares dependencies (a directory
    reference, or a default/file reference shipping a requirements.txt): the agent is
    told it may add/edit a requirements.txt there. ``None`` keeps the historical text.
    """
    context_md_path = os.path.join(run_dir, "context.md")

    # Handle weights mode (RL-based tuning)
    if focus == "weights":
        tpu_section = ""
        if training_backend == "tpu-native":
            tpu_section = """
**TPU-NATIVE TRAINING NOTES**:
- The train.py runs directly on TPU v5e via PyTorch/XLA (not tinker-cookbook)
- Check for torch_xla import errors or TPU device detection failures
- Common issues:
  * XLA_USE_BF16 not set → add it to environment
  * TPU device not found → check libtpu runtime
  * OOM on TPU → reduce batch_size, max_length, or group_size
  * Slow compilation → first batch is slow (XLA compilation), subsequent batches are fast
  * xm.optimizer_step() failures → ensure barrier=True for TPU sync
- TPU-specific optimisations: use bfloat16, increase batch size to saturate TPU
"""
        return f"""You are an expert AI Engineer analyzing an RL-based agent scaffold for iterative improvement.
{tpu_section}
**GENERATION CONTEXT**:

**GENERATION CONTEXT**:
- Current generation: {current_gen}
- Previous generations: {previous_gens}
- Evolution history: {context_md_path}

**BEFORE ANALYZING - READ THE FULL HISTORY**:
1. Read {context_md_path} to understand:
   - What improvements were tried in each previous generation
   - Training and performance trends across generations
   - What worked and what didn't work
2. Review previous improvement.md files from earlier generations if helpful
3. Don't repeat failed approaches from earlier generations
4. Build upon successful RL patterns that improved performance

---

**CURRENT TRAINING SCRIPT** (Generation {current_gen}):
```python
{agent_py}
```

**TASK WORKED ON**:
```
{task}
```

**EXECUTION STATUS**:
```
{execution_status}
```

**EXECUTION LOGS**:
{execution_section}

---

**YOUR TASK**:

You must create exactly TWO files in {next_gen_dir}/:
1. improvement.md - Analysis and improvement plan for the RL pipeline
2. train.py - The improved training script

Follow these steps:

**STEP 1: Analyze the training execution**:
   - Review the RL training metrics and rewards
   - Check if the Env/EnvGroupBuilder/RLDataset are working correctly
   - Identify reward shaping issues or training instability
   - Check SandboxFusion connectivity and code execution
   - Look for convergence patterns or plateaus
   - Identify what worked well and what failed

**STEP 2: Review evolution history**:
   - Read context.md to see the full evolution of training approaches
   - Understand what RL strategies were tried in previous generations
   - Build upon successful training patterns
   - Avoid repeating failed approaches

**STEP 3: Write improvement.md**:
   - MUST save to: {next_gen_dir}/improvement.md
   - Document your analysis and planned improvements to the RL pipeline
   - Focus on improving reward signals, training efficiency, or environment design
   - Suggest better Env/EnvGroupBuilder implementations if needed
   - Reference insights from previous generations if applicable

**STEP 4: Create improved train.py**:
   - MUST save to: {next_gen_dir}/train.py
   - Implement the RL improvements documented in improvement.md
   - Maintain compatibility with tinker-cookbook RL APIs
   - Apply all the planned improvements from step 3
   - Use SandboxFusionClient with os.getenv("SANDBOX_URL") for code execution
   - Do not create or modify any other files besides these two

**RULES**:
- Focus on RL pipeline robustness and training efficiency
- Ensure proper reward shaping for your task domain
- Make the training stable across diverse problem instances
- If execution failed, fix the root cause in the RL pipeline
- Consider proper train/test data splitting and validation
- Handle edge cases and error states gracefully
- Properly log training metrics and rewards for analysis
- Ensure all RL components are correctly implemented per tinker-cookbook

NOTE: If you see errors or incomplete execution logs, focus on making the RL pipeline more robust.
"""

    # Harness mode (default - code/prompt improvement)
    base = f"""You are an expert AI Engineer analyzing agent scaffolds for iterative improvement.

**GENERATION CONTEXT**:
- Current generation: {current_gen}
- Previous generations: {previous_gens}
- Evolution history: {context_md_path}

**BEFORE ANALYZING - READ THE FULL HISTORY**:
1. Read {context_md_path} to understand:
   - What improvements were tried in each previous generation
   - Performance trends across generations
   - What worked and what didn't work
2. Review previous improvement.md files from earlier generations if helpful
3. Don't repeat failed approaches from earlier generations
4. Build upon successful patterns that improved performance

---

**SAMPLE TASK DESCRIPTIONS**:
```
{task_files.sample_task_descriptions}
```

**CURRENT TARGET AGENT** (Generation {current_gen}):
```python
{agent_py}
```

**TASK WORKED ON**:
```
{task}
```

**EXECUTION STATUS**:
```
{execution_status}
```

**EXECUTION LOGS**:
{execution_section}

---

**YOUR TASK**:

You must create exactly TWO files in {next_gen_dir}/:
1. improvement.md - Analysis and improvement plan
2. target_agent.py - The improved agent implementation

Follow these steps:

**STEP 1: Analyze the execution**:
   - For multi-trajectory: Look for patterns across all trajectories
   - For single-trajectory: Analyze the full execution flow
   - Identify what worked well and what failed
   - Check for consistency and robustness

**STEP 2: Review evolution history**:
   - Read context.md to see the full evolution
   - Understand what was tried in previous generations
   - Build upon successful patterns
   - Avoid repeating failed approaches

**STEP 3: Write improvement.md**:
   - MUST save to: {next_gen_dir}/improvement.md
   - Document your analysis and planned improvements
   - Focus on structural improvements to the agent scaffold
   - Make the agent more robust and generalizable
   - Don't optimize for this specific task
   - Reference insights from previous generations if applicable

**STEP 4: Create improved target_agent.py**:
   - MUST save to: {next_gen_dir}/target_agent.py
   - Implement the improvements documented in improvement.md
   - Apply all the planned improvements from step 3
   - Do not create or modify any other files besides these two

**RULES**:
- Focus on agent structure, not task-specific optimizations
- Make the agent work well across diverse task types (see sample task descriptions)
- If execution failed, fix the root cause
- If multi-trajectory: ensure each trajectory is properly isolated and logged
- Consider error handling, logging mechanisms, and robustness
- Build upon successful patterns from previous generations (check context.md)
- If execution log shows errors or is incomplete, suggest improvements to ensure proper logging

NOTE: The agent execution log may be incomplete or contain errors if the target agent crashed. If you see an "error" field, focus on making the agent more robust to prevent such failures.
"""
    if requirements_dir is not None:
        base += (
            f"\nNOTE ON DEPENDENCIES: You may also create or edit a requirements.txt in {requirements_dir} "
            "(one package per line) to declare third-party packages your target_agent.py needs; they are "
            'installed before the target agent runs. This is the one exception to the "only two files" rule above.\n'
        )
    if provider is None or provider.client_kind != "openai":
        return base
    return build_target_client_setup(provider, task_model) + base
