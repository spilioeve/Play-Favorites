# [Playing Favorites: A Statistical Method to Measure Self-Bias in LLM-as-a-Judge](https://arxiv.org/abs/2508.06709)

This repository contains code and human annotation data for **measuring and quantifying self-bias in LLM-as-a-judge** evaluations, as described in our paper.

📄 **Paper:** [https://arxiv.org/abs/2508.06709](https://arxiv.org/abs/2508.06709)

## 📄 Overview
Large language models (LLMs) can act as judges for evaluating other LLM outputs. However, they may systematically give overly favorable ratings to their own outputs — a phenomenon known as **self-bias** — which can distort performance assessments.

Our paper introduces a **statistical framework** that:
- Explicitly models the conditions under which self-bias can be detected and measured.
- Accounts for genuine quality differences of models and separates them from self-bias, using an independent third-party judge (e.g., humans).
- Accounts for consistent annotator differences (across all models), which are independent of self-bias.

Through an empirical study of over **5000 prompt–completion pairs** rated by both humans and nine different LLM judges, we show:
- Certain models (e.g., GPT-4o, Claude 3.5 Sonnet) exhibit **systematic self-bias**.
- Models also show **family-bias** — favoring outputs from models in the same family.
- We offer **practical guidance** to mitigate these biases in automated evaluation pipelines.
- Finally, we **release** the human annotations, to facilitate future research. 

## 📂 Repository Structure
- **`regression.R`** – Code to replicate our statistical analysis and results.  
- **`data/`** – Human annotation data used in the analysis (one file per evaluation dimension).  
- **`indexer_prompts.csv`** – List of prompts and model completions used in the study, sampled from:
  - CNN/DailyMail  
  - XSUM  
  - HELM-Instruct  
  - ChatbotArena  
  - MT-Bench  
  - Stanford Human Preferences  

## 📚 Citation
If you use this code or data, please cite our paper:

> **BibTeX**
> ```bibtex
> @misc{spiliopoulou2025playfavoritesstatisticalmethod,
>       title={Playing Favorites: A Statistical Method to Measure Self-Bias in LLM-as-a-Judge}, 
>       author={Evangelia Spiliopoulou and Riccardo Fogliato and Hanna Burnsky and Tamer Soliman and Jie Ma and Graham Horwood and Miguel Ballesteros},
>       year={2025},
>       eprint={2508.06709},
>       archivePrefix={arXiv},
>       primaryClass={cs.CL},
>       url={https://arxiv.org/abs/2508.06709}, 
> }
> ```

---
📄 **Paper:** [https://arxiv.org/abs/2508.06709](https://arxiv.org/abs/2508.06709)
