This repo contains code and human annotations for measuring self-bias in LLM-as-a-judge, as described in our paper **Play Favorites: A Statistical Method to Measure Self-Bias in LLM-as-a-Judge**

The *regression.R* script contains all code to replicate the results of our analysis. 
The *data* folder contains the human annotation data used in the analysis (one file per evaluation dimension). The prompts and model completions can be found in *prompt_ids.csv* and they consist of randomly sampled prompts from the following public datasets: 

- CNN/DailyMail
- XSUM
- HELM-Instruct
- ChatbotArena
- MT-Bench
- Stanford Human Preferences.

