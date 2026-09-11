# headlinegpt

> "Can we train an LLM to write titles that people actually want to read, without doing painful RLHF?"

A clean, hackable, end-to-end recipe for fine-tuning small language models on headline generation using **Reward-Weighted Supervised Fine-Tuning (SFT)**. 

Based on `Qwen/Qwen2.5-1.5B-Instruct`. Includes full Google Colab training code, PyTorch weights, and exports to both **GGUF** (for CPU/local inference) and **ONNX / WebGPU** (for running 100% inside your browser with zero servers).

---

### Links & Artifacts

* 🚀 **Model (Safetensors / PyTorch)**: [`csankalp21/headlinegpt`](https://huggingface.co/csankalp21/headlinegpt) (~124+ downloads)
* ⚡ **Browser Model (ONNX / WebGPU)**: [`csankalp21/headlinegpt-onnx`](https://huggingface.co/csankalp21/headlinegpt-onnx) (~48+ downloads)
* 📓 **Notebook (Train + Eval + Export)**: [`HeadlineGPT_Training_and_Export.ipynb`](HeadlineGPT_Training_and_Export.ipynb) [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/s-chudmunge/headlinegpt/blob/main/HeadlineGPT_Training_and_Export.ipynb)

Combined HF downloads: **~170+** across PyTorch and ONNX models.

---

## Why this exists: The Problem with Standard SFT

If you take a base instruction model and fine-tune it with standard Cross-Entropy Loss on a corpus of articles and titles, the model simply learns the *average* style across the entire dataset:

$$\mathcal{L}_{\text{SFT}} = - \frac{1}{T} \sum_{t=1}^{T} \log P(w_t \mid w_{<t}, x)$$

The problem? In any real-world content dataset, most titles are completely mediocre. Only a small fraction are genuinely punchy, high-engagement headlines that drive curiosity and clarity. 

You usually have three ways to tackle this:
1. **Hard filtering**: Throw away 80% of your data and keep only top percentiles. (Wastes data diversity, hurts general language competence).
2. **RLHF / PPO / DPO**: Train reward models, manage reference policies, handle stability issues. (Heavy engineering overhead, tricky on a single GPU).
3. **Reward-Weighted SFT (Our Approach)**: Keep all the data so the model maintains broad topic coverage, but modulate the token gradient by an empirical engagement score:

$$\mathcal{L}_{\text{weighted}} = (1.0 + R(x)) \cdot \mathcal{L}_{\text{CE}}$$

Where $R(x) \in [0.0, 1.0]$ is computed from real-world engagement metrics (CTR, claps, likes, shares). A great title that earned a $0.95$ score exerts almost double the gradient pull of a flat, unengaging title ($0.0$). It's dead simple, remarkably stable, and fits on a free Google Colab T4 GPU.

---

## Dataset: What the model learns from

We trained on paired content-title instances where each sample has associated empirical audience metrics:

```json
{
  "content": "Full article body or summary...",
  "title": "Published title...",
  "metrics": {
    "impressions": 14200,
    "clicks": 1820,
    "ctr": 0.128,
    "claps": 450,
    "shares": 85,
    "score": 92
  },
  "reward_score": 0.84
}
```

The data is mapped into Qwen's chat format:
* **System**: `"You are an expert at writing highly engaging titles."`
* **User**: `"Generate a high-engagement title for the following content:\n\n<content>"`
* **Assistant**: `"<title>"`

---

## The Training Setup

Everything was run inside a single Google Colab session on a modest **NVIDIA T4 (16GB)**:

* **Base Model**: `Qwen/Qwen2.5-1.5B-Instruct`
* **Quantization**: 4-bit (`bitsandbytes`, `nf4` compute in `fp16`)
* **PEFT / LoRA**:
  * $r = 16$, $\alpha = 32$, dropout $= 0.05$
  * Targets: `q_proj`, `k_proj`, `v_proj`, `o_proj`, `gate_proj`, `up_proj`, `down_proj`
  * Only **~1.2%** of model parameters are trained
* **Batching**: Per-device batch size `1`, `gradient_accumulation_steps = 16` (effective batch size 16)
* **Optimization**: `paged_adamw_8bit`, cosine LR schedule, peak LR `2e-4`, 50 warmup steps, 2 epochs on 25,000 sampled items.

### The Loss Function (Under the Hood)

Here is the exact PyTorch loss implementation used in the custom `WeightedTrainer`:

```python
class WeightedTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        # Extract reward scalar and compute per-sample multiplier
        reward = inputs.pop("reward_score").float()
        weights = 1.0 + reward  # shape: (batch_size,)

        labels = inputs["labels"]
        outputs = model(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            labels=labels,
        )

        logits = outputs.logits
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()

        # Unreduced cross entropy across tokens
        loss_fct = nn.CrossEntropyLoss(reduction="none")
        loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
        loss = loss.view(shift_labels.size())

        # Mask padding tokens and compute per-sequence loss
        mask = (shift_labels != -100)
        per_sample_loss = (loss * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)

        # Scale loss by engagement weight
        weighted_loss = (per_sample_loss * weights).mean()
        return (weighted_loss, outputs) if return_outputs else weighted_loss
```

---

## Quickstart

### 1. Python (`transformers`)

```bash
pip install transformers torch accelerate
```

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

model_id = "csankalp21/headlinegpt"

tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    torch_dtype=torch.float16,
    device_map="auto"
)

content = """
DeepMind researchers have trained a robotic hand to solve a Rubik's cube 
one-handed using domain randomization and meta-learning, showing unprecedented 
dexterity and physical robustness to perturbations.
"""

messages = [
    {"role": "system", "content": "You are an expert at writing highly engaging titles."},
    {"role": "user", "content": f"Generate a high-engagement title for the following content:\n\n{content}"}
]

prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

with torch.inference_mode():
    outputs = model.generate(
        **inputs,
        max_new_tokens=30,
        temperature=0.7,
        top_p=0.9,
        repetition_penalty=1.1,
        do_sample=True
    )

headline = tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
print("Headline:", headline.strip())
```

---

### 2. Run in the Browser with WebGPU (Zero Server Costs)

Because the weights were converted to ONNX FP16 and hosted on Hugging Face, you can run the entire model right in a user's browser using `@huggingface/transformers` (Transformers.js v3).

```bash
npm install @huggingface/transformers
```

```javascript
import { pipeline } from "@huggingface/transformers";

// Downloads and caches model in IndexedDB, runs on local GPU via WebGPU
const generator = await pipeline(
  "text-generation",
  "csankalp21/headlinegpt-onnx",
  { device: "webgpu", dtype: "fp16" }
);

const messages = [
  { role: "system", content: "You are an expert at writing highly engaging titles." },
  { role: "user", content: "Generate a high-engagement title for the following content:\n\nAnthropic releases Claude 3.7 Sonnet featuring hybrid fast and extended reasoning modes in a single architecture." }
];

const output = await generator(messages, { max_new_tokens: 30, temperature: 0.7 });
console.log(output[0].generated_text.at(-1).content);
```

---

### 3. Local C++ Inference with `llama.cpp`

The notebook includes full cells to merge the LoRA adapter back into base weights and convert to GGUF:

```bash
# Convert to GGUF F16 & Quantize to 4-bit
python convert_hf_to_gguf.py ./merged_model --outfile HeadlineGPT-F16.gguf --outtype f16
./llama-quantize HeadlineGPT-F16.gguf HeadlineGPT-Q4_K_M.gguf Q4_K_M

# Run instant CPU inference
./llama-cli -m HeadlineGPT-Q4_K_M.gguf \
  -p "<|im_start|>system\nYou are an expert at writing highly engaging titles.<|im_end|>\n<|im_start|>user\nGenerate a high-engagement title for the following content:\n\nJames Webb Space Telescope detects carbon-bearing molecules in the atmosphere of habitable-zone exoplanet K2-18b.<|im_end|>\n<|im_start|>assistant\n" \
  -n 30 --temp 0.7
```

---

## File Structure

```
headlinegpt/
├── HeadlineGPT_Training_and_Export.ipynb  # Self-contained notebook: data loading -> weighted SFT -> LoRA merge -> GGUF/ONNX
├── README.md                              # What you are reading
├── requirements.txt                       # Minimal pip dependencies
└── LICENSE                                # Apache 2.0
```

---

## What's Next / Interesting Extensions

* **Mechanistic Interpretability**: Neel Nanda-style logit lens and attention inspection: *Which attention heads specifically attend to high-salience trigger words in the body text when proposing a catchy title?*
* **DPO Comparison**: Benchmarking reward-weighted SFT against Direct Preference Optimization (DPO) on the same 25k split to measure compute-to-quality tradeoffs.
* **4-bit / 8-bit ONNX**: Quantizing the in-browser model from FP16 (~3.7GB) down to Int4/Int8 (~900MB) for instant mobile web loading.

---

## Acknowledgements & Citations

* The [Qwen Team](https://github.com/QwenLM/Qwen2.5) for `Qwen2.5-1.5B`, an absurdly good small base model.
* Hugging Face for `transformers`, `peft`, and `transformers.js` (WebGPU runtime).
* Gerganov and the `llama.cpp` community for making local LLM deployment joyful.

```bibtex
@misc{chudmunge2026headlinegpt,
  author = {Chudmunge, Sankalp},
  title = {HeadlineGPT: Reward-Weighted Fine-Tuned Title Generation Model},
  year = {2026},
  publisher = {GitHub},
  howpublished = {\url{https://github.com/s-chudmunge/headlinegpt}}
}
```
