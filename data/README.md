# README

## Data Files Description

This directory (`data`) contains CSV files with model evaluation results, separated by evaluation dimensions:

- `Completeness.csv`
- `Logical correctness.csv`
- `Helpfulness.csv`
- `Logical robustness.csv`
- `Faithfulness.csv` *(ratings only from `cnn` and `xsum` datasets)*
- `Conciseness.csv`
- `prompts_and_completions.csv`

## Columns

Each CSV on the ratings includes:

| Column      | Description                                             |
|-------------|---------------------------------------------------------|
| dimension   | Evaluation dimension (matches filename).                |
| judge       | Model used to rate generated outputs.                   |
| model       | Model evaluated (generator of outputs).                 |
| gt          | Ground-truth rating (average human rating).             |
| dataset     | Dataset from which the prompt originated.               |
| prompt_id   | Unique identifier for each prompt.                      |
| rating      | Rating assigned by the judge model.                     |
| pred_length | Length of the generated completion.                     |


The additional file include, besides the prompt_id: 

| Column      | Description                                             |
|-------------|---------------------------------------------------------|
| prompt      | Prompt.                                                 |
| completion  | Generated completion.                                   |
