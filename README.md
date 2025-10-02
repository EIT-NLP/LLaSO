# LLaSO: A Foundational Framework for Reproducible Research in Large Language and Speech Model
*Fully open corpus + benchmark + reference model for compositional speech-language understanding.*

<p align="center">
  <!-- Badges: replace placeholders with real links -->
  <a href="https://huggingface.co/datasets/YirongSun/LLaSO-Align"><img src="https://img.shields.io/badge/HF%20Dataset-LLaSO--Align-16a085.svg" alt="HF Align"></a>
  <a href="https://huggingface.co/datasets/YirongSun/LLaSO-Instruct"><img src="https://img.shields.io/badge/HF%20Dataset-LLaSO--Instruct-1abc9c.svg" alt="HF Ins"></a>
  <a href="https://huggingface.co/datasets/YirongSun/LLaSO-Eval"><img src="https://img.shields.io/badge/HF%20Dataset-LLaSO--Eval-27ae60.svg" alt="HF Eval"></a>
  <br>
  <a href="https://arxiv.org/abs/2508.15418"><img src="https://img.shields.io/badge/arXiv-2508.15418-B31B1B.svg" alt="arXiv"></a>
  <a href="https://huggingface.co/YirongSun/LLaSO-Base-3.8B-Instruct"><img src="https://img.shields.io/badge/HuggingFace-Model-ffcc00.svg" alt="HF Model"></a>
  <a href="https://github.com/EIT-NLP/LLaSO"><img src="https://img.shields.io/github/stars/EIT-NLP/LLaSO?style=social" alt="GitHub Stars"></a>
  <a href="#citation"><img src="https://img.shields.io/badge/Cite-BibTeX-9cf.svg" alt="Cite"></a>
</p>

<p align="center">
  <a href="https://arxiv.org/abs/2508.15418">Paper</a> •
  <a href="https://github.com/EIT-NLP/LLaSO">Code</a> •
  <a href="https://huggingface.co/YirongSun/LLaSO-Base-3.8B-Instruct">Models</a> •
  <a href="https://huggingface.co/datasets/YirongSun/LLaSO-Instruct">Datasets</a> •
  <a href="https://huggingface.co/datasets/YirongSun/LLaSO-Eval">Benchmark</a> •
  <a href="#install">Install</a> •
  <a href="#quick-start">Quick Start</a>
</p>

> **TL;DR.** 25.5M training samples, 20 tasks, 3 modality configurations; 15,044-sample stratified benchmark; 3.8B open reference model - an open-source LSLM stack.

<p align="center">
  <img src="figures/radar.png" width="600" alt="LLaSO overall performance">
</p>
<p align="center"><i>
LLaSO-Base achieves the best normalized overall score on LLaSO-Eval across 20 tasks spanning linguistic, semantic, and paralinguistic categories.
</i></p>


<p align="center">
  If you find LLaSO useful, please ⭐ star this repo!
</p>

## 🔍 What is LLaSO?

**LLaSO is the first fully open, end-to-end stack for large-scale speech–language modeling, unifying data, evaluation, and modeling in one framework.**

- **LLaSO-Align (12.0M):** ASR-based alignment for grounding speech in textual semantic space.
- **LLaSO-Instruct (13.5M / 20 tasks / 3 modality configs):** Multi-task instruction tuning across linguistic, semantic, and paralinguistic objectives.
- **LLaSO-Eval (15,044):** Stratified benchmark for instruction-following and cross-modality generalization.
- **LLaSO-Base (3.8B):** Two-stage trained reference model, adapted from LLaVA-style architectures for robust compositional understanding.


<p align="center">
  <table>
    <tr>
      <td align="center">
        <img src="figures/donut_trim.png" width="340" alt="Corpus Overview (Figure 2)"><br>
        <i>Corpus and Task Coverage</i>
      </td>
      <td align="center">
        <img src="figures/architecture_trim.png" width="350" alt="Architecture & Two-Stage Training (Figure 6)"><br>
        <i>Architecture & Two-Stage Training</i>
      </td>
    </tr>
  </table>
</p>
<p align='center'> <i>
LLaSO stack: Data, benchmark, and reference model for compositional speech–language modeling.
</i></p>

## ✨ Key Features

- **Fully Open, End-to-End Stack:** Unified release of corpus, benchmark, and model-enabling open-source research and fair comparison in speech-language modeling.
- **25.5M Samples, 20 Tasks, 3 Modality Configurations:** Supports all major text ↔ audio combinations (text + audio, audio + text, pure audio), covering linguistic, semantic, and paralinguistic tasks.
- **Stratified Evaluation (15,044):** Cohesive design between training and test sets enables systematic assessment of instruction following, cross-modality generalization, abstention rate, and stability.
- **Robust Reference Model (3.8B):** Two-stage training (ASR alignment → instruction tuning), easily reproducible and extensible for further research.
- **Empirical Insights:** Broader task and modality coverage consistently leads to stronger overall performance, but unseen modality/task configurations (especially pure audio) remain challenging; interleaving and parallel decoding strategies can bridge some gaps.
<a id="install"></a>
## 🛠️ Install
```bash
git clone https://github.com/EIT-NLP/LLaSO.git
cd LLaSO
conda create -n llaso python=3.10 -y
conda activate llaso
pip install --upgrade pip  # enable PEP 660 support
pip install -e .  # See pyproject.toml for dependencies
pip install librosa==0.10.2.post1

# Install additional packages for training
pip install -e ".[train]"

#install FlashAttention for acceleration
MAX_JOBS=8 pip install -v flash-attn --no-build-isolation
```
> **Tips:**  
> If you encounter issues with FlashAttention installation (e.g., build errors or the process getting stuck), we recommend manually downloading the appropriate FlashAttention 2 wheel from the [official Dao-AILab releases](https://github.com/Dao-AILab/flash-attention/releases).  
> For example, for `python3.10 + cu12.2 + torch2.1`, download:  
> 
> ```
> https://github.com/Dao-AILab/flash-attention/releases/download/v2.4.3.post1/flash_attn-2.4.3.post1+cu122torch2.1cxx11abiFALSE-cp310-cp310-linux_x86_64.whl
> ```
> and then install it via:  
> 
> ```bash
> pip install /path/to/flash_attn-2.4.3.post1+cu122torch2.1cxx11abiFALSE-cp310-cp310-linux_x86_64.whl
> ```

<a id="prepare-weights"></a>
## 📥 Prepare Model Weights

LLaSO relies on the following backbone weights before training or inference:

- **Audio encoder**: [openai/whisper-large-v3](https://huggingface.co/openai/whisper-large-v3)  
  Used as `--audio_tower` in training/inference scripts.

- **Base LLM**: [meta-llama/Llama-3.2-3B-Instruct](https://huggingface.co/meta-llama/Llama-3.2-3B-Instruct)  
  Used as `--model_name_or_path` in training scripts.

- **LLaSO checkpoint**: [YirongSun/LLaSO-Base-3.8B-Instruct](https://huggingface.co/YirongSun/LLaSO-Base-3.8B-Instruct)  
  Used as `--model_path` in the inference example.

## 📂 Data Preparation

The training and alignment scripts expect a **single JSON file** as input.  
However, for flexibility, we release the instruction data in multiple subdirectories (e.g., `audio_text/`, `pure_audio/`, `text_audio/`).  

- This split format allows users to **explore or train with individual modality subsets**.  
- For full training, these subsets need to be **merged into one JSON**.  

We provide [`./llaso/data/data_merge.py`](./llaso/data/data_merge.py) for this purpose.  
Use it to combine the JSON files under each modality subdirectory into a single training file.  

> **Dataset Availability**  
> - ✅ **[LLaSO-Eval](https://huggingface.co/datasets/YirongSun/LLaSO-Eval)** is already available on Hugging Face.  
> - ✅ **[LLaSO-Instruct](https://huggingface.co/datasets/YirongSun/LLaSO-Instruct)** has been fully uploaded and is available now.  
> - ✅ **[LLaSO-Align](https://huggingface.co/datasets/YirongSun/LLaSO-Align)** has been fully uploaded and is available now.  

<a id="quick-start"></a>
## 🚀 Quick Start


### 4.1 Training
Train LLaSO-Base from scratch with:
```python
# Stage 1: Speech-Text Alignment (ASR)
bash llaso/scripts/align.sh 

# Stage 2: Multi-task Instruction Tuning
bash llaso/scripts/finetune.sh 
```

### 4.2 Inference
Run model inference on your own data or evaluation set:
```python
python llaso/evaluation/model_eval.py \
  --audio_tower ./whisper-large-v3 \
  --model_path ./LLaSO-Base-3.8B-Instruct \
  --data_path ./LLaSO-Eval/your_eval.json \
  --output_dir ./your_output_dir
```

### 4.3 Evaluation
LLaSO provides flexible evaluation metrics for all supported tasks.
See the llaso/evaluation/metrics/ directory for dedicated metric scripts per task type.
> **Tip:**
> Refer to our [paper](arxiv) for detailed mapping of tasks to metrics and best practices.

> **Note:**
> Training and inference configs are fully modular; see scripts and configs for options.
## 📦 Model Zoo

| Model               | #Params | Training Data                | Modality Configs        | Normalized Score | Checkpoint          |
|---------------------|--------:|------------------------------|------------------------|------------------|---------------------|
| **LLaSO-Base**      | 3.8B    | LLaSO-Align + LLaSO-Instruct (25.5M) | (t,a), (a,t), (a)      | 0.72             | 🤗 [HF link](https://huggingface.co/YirongSun/LLaSO-Base-3.8B-Instruct)        |
| *(Future)* TBD | ...    | ...                          | ...                    | ...              | (coming)            |




## 🗃️ Data Cards

### **LLaSO Corpus Overview**
- **Composition:** 25.5M samples (12.0M Align + 13.5M Instruct) covering 20 tasks across all major modality configurations (text instr. with audio input, pure audio, audio instr. with text input).
- **Overall Task Distribution:** 52% linguistic, 8% semantic, 40% paralinguistic.
- **Real vs. Synthetic:** 71% real-world audio, 29% synthetic speech.
- **Design Motivation:** 
  - *Linguistic (ASR)* remains foundational for speech–text alignment and generalization.
  - *Semantic* tasks are intentionally underweighted, as their challenge lies more in language modeling than in speech understanding.
  - *Paralinguistic* tasks (speaker, accent, emotion, pronunciation scoring) are prioritized to address their underrepresentation in open datasets.
- **Flexible Modality Roles:** Both audio and text serve as input/instruction, enabling rich compositional interaction patterns.

---

### **LLaSO-Align (12.0M)**
- **Goal:** ASR-based alignment; encoder & LLM frozen, projector trained for speech-to-text semantic grounding.
- **Domains:** Conversational, narrative, audiobook, accented speech.
- **Templates:** 18 instruction types for ASR; unified JSON format for integration.

### **LLaSO-Instruct (13.5M / 20 tasks)**
- **Purpose:** Multi-task instruction tuning for robust, compositional understanding.
- **Task Types:** Spans linguistic, semantic, and paralinguistic objectives with a mix of closed- and open-ended formats.
- **Modality Configurations:**  
  - Text instruction + Audio input: **X**<sub>query</sub><sup>(t,a)</sup>
  - Audio instruction + Text input: **X**<sub>query</sub><sup>(a,t)</sup>
  - Pure audio: **X**<sub>query</sub><sup>(a)</sup>
- **Label Granularity:** Multi-granularity (e.g., coarse→fine age, accent).

### **LLaSO-Eval (15,044)**
- **Benchmarking:** Strictly stratified; consistent with training data.
- **Coverage:** All tasks and modality combinations.
- **Metrics:** Supports abstention rate analysis and cross-modality generalization evaluation.

<p align="center">
  <img src="figures/data_pipeline_trim.png" width="980" alt="LLaSO data construction pipeline">
</p>
<p align="center"><i>
LLaSO data construction pipeline, from text QA and multi-source speech to two-stage training datasets.
</i></p>

<p align="center">
  <img src="figures/cases1_trim.png" width="900"><br>
  <img src="figures/cases2_trim.png" width="900">
</p>


<p align="center"><i>
Cases for input modality configurations and task prototypes illustrate LLaSO's compositional flexibility. Pure audio (top), text instruction with audio input (middle), and audio instruction with text input (bottom). Each figure presents distinct tasks under its respective format.
</i></p>

## 🔬 Key Empirical Findings

- **Broader task and modality coverage** consistently yields higher overall performance and lower abstention rates, though unseen configurations (especially pure audio) remain challenging.
- **Interleaving and parallel decoding strategies** can partially bridge cross-modality generalization gaps.
- **Unfreezing the audio encoder in Stage 2** improves semantic task performance but can reduce paralinguistic accuracy, highlighting a training-stage trade-off.
- **Speech-to-speech systems** with narrow task focus demonstrate stable cross-modality behavior but lag in overall capability.
- **LSLMs tend to favor content-related tasks**, while paralinguistic subtleties are generally more difficult.

<a id="citation"></a>
## 📑 How to Cite

If you use LLaSO in your research or applications, please cite our paper:

```bibtex
@misc{sun2025llaso,
      title={LLaSO: A Foundational Framework for Reproducible Research in Large Language and Speech Model}, 
      author={Yirong Sun and Yizhong Geng and Peidong Wei and Yanjun Chen and Jinghan Yang and Rongfei Chen and Wei Zhang and Xiaoyu Shen},
      year={2025},
      eprint={2508.15418},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2508.15418}, 
}
```

## 🙏 Acknowledgement

Our work builds upon and is inspired by several outstanding open-source projects and pretrained models, including:

- [LLaVA (Haotian Liu et al.)](https://github.com/haotian-liu/LLaVA/tree/main)
- [UMOE-Scaling (HITsz-TMG)](https://github.com/HITsz-TMG/UMOE-Scaling-Unified-Multimodal-LLMs)
- [Llama-3.2-3B-Instruct (Meta)](https://huggingface.co/meta-llama/Llama-3.2-3B-Instruct)
- [Whisper-large-v3 (OpenAI)](https://huggingface.co/openai/whisper-large-v3)

We gratefully acknowledge the authors and contributors for making these resources publicly available.

## 📬 Contact
```
Email: win1282467298@gmail.com
Organization: EIT-NLP Lab, Logic Intelligence Technology
```
