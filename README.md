# HeadlineGPT 🚀

[![Hugging Face Model](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-HeadlineGPT-yellow)](https://huggingface.co/csankalp21/headlinegpt)
[![Hugging Face ONNX](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-HeadlineGPT--ONNX%20(WebGPU)-blue)](https://huggingface.co/csankalp21/headlinegpt-onnx)
[![HF Model Downloads](https://img.shields.io/badge/HF%20Downloads-170%2B-brightgreen)](https://huggingface.co/csankalp21/headlinegpt)
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/s-chudmunge/headlinegpt/blob/main/HeadlineGPT_Training_and_Export.ipynb)
[![License: Apache-2.0](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](LICENSE)

**HeadlineGPT** is a fine-tuned language model based on [Qwen/Qwen2.5-1.5B-Instruct](https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct), purpose-built for turning articles, social media posts, academic papers, and talks into concise, catchy, and high-engagement titles and headlines.

Available both as standard PyTorch/Safetensors weights on Hugging Face and as an optimized ONNX / WebGPU export for **100% private, zero-server in-browser inference** via [Transformers.js](https://huggingface.co/docs/transformers.js).

---

## 🌟 Quick Links & Hugging Face Hub

| Model | Format & Target | Downloads | Link |
|---|---|---|---|
| **HeadlineGPT (Base LoRA/Merged)** | PyTorch / Safetensors (1.5B) | **124+ downloads** | [csankalp21/headlinegpt](https://huggingface.co/csankalp21/headlinegpt) |
| **HeadlineGPT ONNX (WebGPU)** | ONNX FP16 / Transformers.js | **48+ downloads** | [csankalp21/headlinegpt-onnx](https://huggingface.co/csankalp21/headlinegpt-onnx) |
| **Full Training & Export Notebook** | Jupyter / Google Colab | — | [HeadlineGPT_Training_and_Export.ipynb](HeadlineGPT_Training_and_Export.ipynb) |

Total Hub Downloads: **170+ downloads** across Safetensors & ONNX packages.

---

## 💡 Key Highlights

- **Base Architecture**: `Qwen2.5-1.5B-Instruct` — lightweight, fast, and highly capable.
- **Reward-Weighted SFT**: Instead of treating every training headline equally, training loss is dynamically modulated by an engagement/reward score:
  $$\mathcal{L}_{\text{weighted}} = (1.0 + \text{reward}) \times \mathcal{L}_{\text{CE}}$$
- **Parameter-Efficient**: Trained using LoRA ($r=16, \alpha=32$) on 4-bit quantized base weights (`paged_adamw_8bit`), enabling training within consumer/Colab T4 GPU limits.
- **Edge & Browser Ready**:
  - Exported to **GGUF** (F16 and Q4_K_M) for local CLI CPU/GPU inference with `llama.cpp`.
  - Exported to **ONNX FP16** and compiled for **WebGPU/Wasm** for client-side execution directly inside the browser.

---

## 📊 Dataset & Schema

The model was trained on content–title pairs extracted from real-world published articles and posts, along with rich engagement and viral metrics.

### Raw Data Schema

Each sample in `master_training_dataset.jsonl` contains:
- `content`: Full text body or abstract.
- `title`: Actual published headline.
- `metrics`:
  - `impressions`, `clicks`, `ctr`
  - `claps`, `responses`, `score`
  - `comments`, `views`, `likes`, `shares`
  - `facebook`, `linkedin`
- `metadata`: Source, category, tags, and timestamps.
- `reward_score`: Scaled engagement metric ($[0.0, 1.0]$) computed from CTR, claps, likes, and share velocity.

### Chat Template Formatting

Each record was wrapped in Qwen's standard chat template:
```python
messages = [
    {
        "role": "system",
        "content": "You are an expert at writing highly engaging titles."
    },
    {
        "role": "user",
        "content": f"Generate a high-engagement title for the following content:\n\n{content}"
    },
    {
        "role": "assistant",
        "content": title
    }
]
```

---

## 🧠 Training Process

Standard Supervised Fine-Tuning (SFT) learns the *average* style across all training samples, regardless of whether a title went viral or underperformed. 

To teach the model to favor click-worthy, engaging titles, **Reward-Weighted SFT** was employed:

1. **Dataset Split & Sampling**:
   - 90% Train / 10% Validation split (`seed=42`).
   - Sampled subset of **25,000 examples** with sequence length capped at 512 tokens.
2. **Custom Weighted Trainer**:
   A custom PyTorch loss function scales cross-entropy token loss using each sample's `reward_score`:
   ```python
   class WeightedTrainer(Trainer):
       def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
           reward = inputs.pop("reward_score").float()
           weights = 1.0 + reward  # High-scoring samples receive higher weight

           labels = inputs["labels"]
           outputs = model(
               input_ids=inputs["input_ids"],
               attention_mask=inputs["attention_mask"],
               labels=labels,
           )

           shift_logits = outputs.logits[..., :-1, :].contiguous()
           shift_labels = labels[..., 1:].contiguous()

           loss_fct = nn.CrossEntropyLoss(reduction="none")
           loss = loss_fct(
               shift_logits.view(-1, shift_logits.size(-1)),
               shift_labels.view(-1)
           )
           loss = loss.view(shift_labels.size())

           mask = (shift_labels != -100)
           per_sample_loss = (loss * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)

           weighted_loss = (per_sample_loss * weights).mean()
           return (weighted_loss, outputs) if return_outputs else weighted_loss
   ```
3. **LoRA Hyperparameters**:
   - Rank ($r$): `16`
   - Alpha ($\alpha$): `32`
   - Dropout: `0.05`
   - Target Modules: `q_proj`, `k_proj`, `v_proj`, `o_proj`, `gate_proj`, `up_proj`, `down_proj`
4. **Optimization**:
   - Epochs: `2`
   - Batch Size: `1` per device with `gradient_accumulation_steps=16` (effective batch size = 16)
   - Optimizer: `paged_adamw_8bit`
   - Learning Rate: `2e-4` with cosine schedule and 50 warmup steps
   - Mixed Precision: `FP16`

---

## 💻 Usage

### 1. Python (`transformers` + PyTorch)

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
Researchers have developed a new solid-state battery architecture that increases 
energy density by 40% while eliminating fire hazards common in traditional lithium-ion cells. 
Automakers expect commercial deployment by 2028.
"""

messages = [
    {
        "role": "system",
        "content": "You are an expert at writing highly engaging titles."
    },
    {
        "role": "user",
        "content": f"Generate a high-engagement title for the following content:\n\n{content}"
    }
]

prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

outputs = model.generate(
    **inputs,
    max_new_tokens=40,
    temperature=0.7,
    do_sample=True,
    top_p=0.9,
    repetition_penalty=1.1
)

headline = tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
print("Generated Headline:", headline.strip())
```

---

### 2. In-Browser / WebGPU (`Transformers.js` / ONNX)

Run inference client-side with no backend servers:

```bash
npm install @huggingface/transformers
```

```javascript
import { pipeline } from "@huggingface/transformers";

const generator = await pipeline(
  "text-generation",
  "csankalp21/headlinegpt-onnx",
  {
    device: "webgpu",
    dtype: "fp16",
  }
);

const messages = [
  { role: "system", content: "You are an expert at writing highly engaging titles." },
  { role: "user", content: "Generate a high-engagement title for the following content:\n\nOpenAI announces new lightweight reasoning models designed for real-time mobile agents." }
];

const output = await generator(messages, {
  max_new_tokens: 35,
  temperature: 0.7,
  do_sample: true
});

console.log("Headline:", output[0].generated_text.at(-1).content);
```

---

### 3. Local CLI with `llama.cpp` (GGUF)

As detailed in the [notebook](HeadlineGPT_Training_and_Export.ipynb), the merged LoRA checkpoint can be converted directly into GGUF format:

```bash
# Convert to GGUF F16
python convert_hf_to_gguf.py ./merged_model --outfile HeadlineGPT-F16.gguf --outtype f16

# Quantize to 4-bit (Q4_K_M)
./llama-quantize HeadlineGPT-F16.gguf HeadlineGPT-Q4_K_M.gguf Q4_K_M

# Run inference
./llama-cli -m HeadlineGPT-Q4_K_M.gguf -p "<|im_start|>system\nYou are an expert at writing highly engaging titles.<|im_end|>\n<|im_start|>user\nGenerate a high-engagement title for the following content:\n\nSpaceX successfully catches Starship booster on first attempt.<|im_end|>\n<|im_start|>assistant\n" -n 40 --temp 0.7
```

---

## 📁 Repository Structure

```
headlinegpt/
├── HeadlineGPT_Training_and_Export.ipynb  # End-to-end training, eval, LoRA merge, GGUF & ONNX export
├── README.md                              # Model overview, benchmarks, HF hub links, and tutorials
└── LICENSE                                # Apache 2.0 License
```

---

## 📜 Citation & Acknowledgements

If you use HeadlineGPT in your research or application, please cite:

```bibtex
@misc{headlinegpt2026,
  author = {Sankalp Chudmunge},
  title = {HeadlineGPT: Reward-Weighted Fine-Tuned Title Generation Model},
  year = {2026},
  publisher = {Hugging Face},
  howpublished = {\url{https://huggingface.co/csankalp21/headlinegpt}}
}
```

Special thanks to the [Qwen Team](https://github.com/QwenLM/Qwen2.5) for the Qwen2.5 base model and the [Hugging Face Transformers.js](https://github.com/huggingface/transformers.js) team for WebGPU acceleration support.
