rm(list = ls())

packages <- c("tidyverse", "tidymodels", "glue", "here", "clubSandwich", "sandwich", "lmtest", "ggthemes", "mgcv", "patchwork", "paletteer", "ggridges", "MASS", "broom", "DescTools")
new_packages <- packages[!(packages %in% installed.packages()[, "Package"])]
if (length(new_packages)) install.packages(new_packages)
lapply(packages, library, character.only = TRUE)

# -------------------------------
# Data Loading and Initial Processing
# -------------------------------
# Read CSV files for each dimension and bind into a single dataframe.


df <- c(
  'Completeness',
  'Logical correctness',
  'Helpfulness',
  'Logical robustness',
  'Faithfulness',
  'Conciseness'
) %>%
  map(~ read_csv(glue(here('data/{.x}.csv'))) %>%
        mutate(dimension = .x)) %>%
  bind_rows()

# 
# df <- c(
#   'Completeness', 
#   'Harmlessness', 
#   'Helpfulness', 
#   'Understandability'
# ) %>% map(~ read_csv(glue(here('data-helm-instruct/{.x}.csv'))) %>%
#           mutate(dimension = .x)) %>%
#   bind_rows() %>%
#   dplyr::select(dimension, judge, model, gt, dataset, prompt_id, rating, pred_length)



# Create binary versions of ratings based on the mean (for later analysis).
df <- df %>% 
  group_by(dimension) %>% 
  mutate(mean_val = mean(unique(c(gt, rating)), na.rm = TRUE)) %>%
  mutate(
    gt_orig = gt, rating_orig = rating,
    bin_rating = ifelse(rating > mean_val, 1, 0),
         bin_gt = ifelse(gt > mean_val, 1, 0), 
         gt = gt / max(c(gt, rating), na.rm = TRUE), 
         rating = rating / max(c(gt, rating), na.rm = TRUE),
         mbin_gt = case_when(
    gt > 0.8 ~ 1,
    gt < 0.2 ~ 0,
    TRUE ~ 0.5
  )) %>%
  ungroup

# -------------------------------
# Preprocessing Functions
# -------------------------------
# Function to preprocess the data: create 'same_judge', clean judge/model names, extract families.
preprocess <- function(df) {
  # Create same_judge indicator with consistent type.
  df <- df %>% 
    mutate(same_judge = ifelse(model == judge, model, "0"),
           # Clean judge and model strings.
           judge = str_remove(judge, '20241022-'),
           model = str_remove(model, '20241022-'),
           judge = str_remove(judge, '-20240229'),
           model = str_remove(model, '-20240229'),
           judge = str_remove(judge, '-instruct-v1:0'),
           model = str_remove(model, '-instruct-v1:0'),
           judge = str_remove(judge, '-v2:0'),
           model = str_remove(model, '-v2:0'),
           judge = str_remove(judge, '-v1:0'),
           model = str_remove(model, '-v1:0'),
           judge = str_remove(judge, '-2407-v1:0'),
           model = str_remove(model, '-2407-v1:0'),
           judge = str_remove(judge, 'Qwen/'),
           model = str_remove(model, 'Qwen/'),
           judge = str_remove(judge, '-instruct-v0:2'),
           model = str_remove(model, '-instruct-v0:2'),
           judge = str_remove(judge, ':1'),
           model = str_remove(model, ':1'),
           judge = str_replace(judge, '3-5', '3.5'),
           model = str_replace(model, '3-5', '3.5')
    )
  
  # Recode values for clarity.
  df$model <- recode(df$model,
                     "anthropic.claude-v2" = "Claude v2",
                     "meta.llama3-1-70b" = "Llama 3 70B",
                     "meta.llama3-1-8b" = "Llama 3 8B",
                     "mistral.mistral-large-2407" = "Mistral Large",
                     "mistral.mistral-7b" = "Mistral 7B",
                     "anthropic.claude-3.5-sonnet" = "Claude 3.5 Sonnet",
                     "anthropic.claude-3-sonnet" = "Claude 3 Sonnet",
                     "gpt-3.5-turbo" = "GPT-3.5 Turbo",
                     "gpt-4o" = "GPT-4o")
  
  df$judge <- recode(df$judge,
                     "anthropic.claude-v2" = "Claude v2",
                     "meta.llama3-1-70b" = "Llama 3 70B",
                     "meta.llama3-1-8b" = "Llama 3 8B",
                     "mistral.mistral-large-2407" = "Mistral Large",
                     "mistral.mistral-7b" = "Mistral 7B",
                     "anthropic.claude-3.5-sonnet" = "Claude 3.5 Sonnet",
                     "anthropic.claude-3-sonnet" = "Claude 3 Sonnet",
                     "gpt-3.5-turbo" = "GPT-3.5 Turbo",
                     "gpt-4o" = "GPT-4o")
  
  df = df %>% mutate(same_judge = ifelse(judge == model, judge, 0))
  
  
  # Create family indicators.
  df <- df %>% 
    mutate(model_family = case_when(
      grepl('Claude', model) ~ 'Claude',
      grepl('GPT', model) ~ 'GPT',
      grepl('Mistral', model) ~ 'Mistral',
      grepl('Llama', model) ~ 'Llama 3',
      grepl('Command', model) ~ 'Command', 
      TRUE ~ NA_character_
    ),
    judge_family = case_when(
      grepl('Claude', judge) ~ 'Claude',
      grepl('GPT', judge) ~ 'GPT',
      grepl('Mistral', judge) ~ 'Mistral',
      grepl('Llama', judge) ~ 'Llama 3',
      grepl('Command', judge) ~ 'Command', 
      TRUE ~ NA_character_
    )) 
  
  df = df %>% mutate(
    same_family = ifelse(model_family == judge_family & same_judge == "0", judge_family, "0")
    ) %>% drop_na()
  
  return(df)
}


df <- preprocess(df)

# -------------------------------
# Define Orders and Confidence Quantile
# -------------------------------
model_order <- c(
  "Claude v1.3",
  "Claude v2",
  "Claude 3 Sonnet",
  "Claude 3.5 Sonnet",
  "GPT-3.5 Turbo",
  "GPT-4", 
  "GPT-4o",
  "Llama 3 8B",
  "Llama 3 70B",
  "Mistral 7B",
  "Mistral Large",
  "Command-Xlarge-Beta"
)
family_order <- c("Claude", "GPT", "Llama 3", "Mistral", "Command")
z_90 <- qnorm(p = 0.95, lower.tail = TRUE)

# -------------------------------
# Define Missing Variables from Original Code
# -------------------------------
# Define 'families' used for joining in plots.
families = df %>% distinct(judge_family, judge) %>%
  rename(family = judge_family)

# Define 'dimension_order' for ordinal logit regression plots.
dimension_order <- c(
  "Conciseness", 
  "Completeness",
  "Faithfulness", 
  "Helpfulness", 
  "Logical robustness", 
  "Logical correctness",
  "Understandability", 
  "Harmlessness"
)

# -------------------------------
# Exploratory Data Analysis & Heatmaps
# -------------------------------
# Create heatmap comparing LLM ratings with human ratings.
dfp <- df %>% 
  group_by(judge, model) %>% 
  summarise(scores = mean(rating)) %>%
  group_by(judge) %>%
  mutate(normalized_scores = (scores - mean(scores)) / (max(scores) - min(scores))) %>% 
  bind_rows(
    df %>% filter(model %in% model_order, judge %in% model_order) %>%
      group_by(model) %>% 
      summarise(scores = mean(gt)) %>%
      mutate(judge = "Human") %>%
      mutate(normalized_scores = (scores - mean(scores)) / (max(scores) - min(scores)))
  )
p <- dfp %>%
  ggplot(aes(factor(model, levels = model_order), 
             factor(judge, levels = c("Human", model_order)), 
             fill = normalized_scores)) +
  geom_tile() + 
  theme_bw() +
  scale_fill_gradient(low = "red", high = "green") + 
  geom_tile(data = dfp %>% filter(judge == model) %>% 
              mutate(model = model,
                     judge = judge), 
            fill = NA, color = "black", linewidth = 0.5, linetype = "dashed") + 
  xlab("Model") +
  ylab("Judge") + 
  geom_text(aes(label = round(scores, 2))) + 
  guides(fill = "none") + 
  scale_x_discrete(labels = function(x) str_wrap(x, width = 10)) +
  theme(axis.text.x = element_text(angle = 90, vjust = 0.5, hjust = 1))
p
ggsave(here("plots", "heatmap_human_vs_llm_scores.pdf"), height = 5, width = 6)
ggsave(here("plots", "heatmap_human_vs_llm_scores_wide.pdf"), height = 3, width = 9)

# Create heatmap by dimension.
dfp <- df %>% 
  group_by(dimension, judge, model) %>% 
  summarise(scores = mean(rating)) %>% 
  group_by(dimension, judge) %>%
  mutate(normalized_scores = (scores - mean(scores)) / (max(scores) - min(scores))) %>% 
  bind_rows(
    df %>% 
      group_by(dimension, model) %>% 
      summarise(scores = mean(gt)) %>%
      mutate(judge = "Human") %>%
      group_by(dimension) %>% 
      mutate(normalized_scores = (scores - mean(scores)) / (max(scores) - min(scores)))
  )
dfp %>%
  ggplot(aes(factor(model, levels = model_order),
             factor(judge, levels = c("Human", model_order)),
             fill = normalized_scores)) +
  geom_tile() +
  theme_bw() +
  scale_fill_gradient(low = "red", high = "green") + 
  geom_tile(data = dfp %>% filter(judge == model), 
            fill = NA, color = "black", linewidth = 0.5, linetype = "dashed") + 
  theme(axis.text.x = element_text(angle = 90, vjust = 0.5, hjust = 1)) +
  xlab("Model") +
  ylab("Judge") + 
  geom_text(aes(label = round(scores, 2))) + 
  guides(fill = "none") + 
  scale_x_discrete(labels = function(x) str_wrap(x, width = 10)) +
  facet_wrap(~ dimension)
ggsave(here("plots", "heatmap_human_vs_llm_scores_by_dimension.pdf"), height = 12, width = 12)

# Compute correlations between rating and ground truth.
corrs <- df %>%
  group_by(dimension, judge, judge_family) %>%
  summarise(
    pearson = cor(rating, gt, method = "pearson", use = "complete.obs"),
    spearman = cor(rating, gt, method = "spearman", use = "complete.obs"),
    kendall = cor(rating, gt, method = "kendall", use = "complete.obs"),
    gamma = DescTools::GoodmanKruskalGamma(rating, gt),
    somersD = DescTools::SomersDelta(rating, gt)
  )

# Loop over chosen correlation methods to plot and save figures.
for(corr_name in c("pearson", "gamma", "spearman")) {
  p_corr <- corrs %>%
    mutate(scores = !!sym(corr_name)) %>%
    ggplot(aes(judge, scores, fill = factor(judge_family, family_order))) +
    geom_col(width = 0.5) +
    theme_bw() +
    theme(legend.position = "bottom") +
    facet_wrap(~ dimension) +
    paletteer::scale_fill_paletteer_d("lisa::OskarSchlemmer") +
    ylab("Correlation") +
    xlab("Judge") +
    coord_flip() +
    guides(fill = "none")
  
  ggsave(here("plots", glue::glue("correlation_{corr_name}.pdf")), plot = p_corr, height = 3, width = 7)
}

# -------------------------------
# Regression Analysis
# -------------------------------
# Fit a linear model predicting rating.
mod <- lm(rating ~ judge + gt:judge + same_judge + same_family + dimension - 1, data = df)
coef_est <- coef(mod)
vcov_mat <- sandwich(mod)  # Use sandwich estimator for robust covariance
res <- coeftest(mod, vcov = vcov_mat)[,] %>% as.data.frame() %>% tibble::rownames_to_column(var = "term")
colnames(res) <- c("term", "estimate", "std.error", "statistic", "p.value")
res

# Separate coefficients for judge effects, ground truth interactions, self-preference bias, and family-preference bias.
judge_est <- res %>% 
  filter(str_starts(term, "judge")) %>% 
  filter(!grepl("gt", term)) %>%
  mutate(term = str_remove(term, "judge"))
gt_est <- res %>% filter(grepl("gt", term)) %>%
  mutate(term = str_remove(term, ":gt")) %>% 
  mutate(term = str_remove(term, "judge"))
sb_est <- res %>% filter(grepl("same_judge", term)) %>% mutate(term = str_remove(term, "same_judge"))
sf_est <- res %>% filter(grepl("same_family", term)) %>% mutate(term = str_remove(term, "same_family"))

# -------------------------------
# Plot Self-Preference Bias (Self-Bias)
# -------------------------------
main_sb_est <- sb_est
p1 <- sb_est %>%
  left_join(families %>% mutate(term = judge)) %>%
  rename(Family = family) %>%
  filter(!is.na(judge)) %>%
  ggplot(aes(factor(term, levels = model_order), 
             estimate, fill = factor(Family, family_order))) + 
  geom_col() +
  theme_bw() +
  coord_flip() +
  paletteer::scale_fill_paletteer_d("lisa::OskarSchlemmer") +
  paletteer::scale_colour_paletteer_d("lisa::OskarSchlemmer") +
  geom_errorbar(aes(ymin = estimate - z_90 * std.error, ymax = estimate + z_90 * std.error), width = 0.2) +
  guides(fill = "none") +
  geom_hline(yintercept = 0, linetype = "dashed", col = "black", alpha = 0.3) +
  ylab("Estimate of self-bias") +
  xlab("Judge")
p1
ggsave(here("plots", "self_preference_bias.pdf"), height = 4, width = 5)

# -------------------------------
# Plot Family-Preference Bias (Family-Bias)
# -------------------------------
main_sf_est <- sf_est
p2 <- sf_est %>% 
  mutate(family = term) %>%
  rename(Family = family) %>%
  mutate(term = str_remove(term, "mistral.")) %>%
  mutate(Family = str_remove(Family, "mistral.")) %>%
  ggplot(aes(factor(term, levels = family_order), estimate, 
             fill = factor(Family, family_order))) +
  geom_col() +
  theme_bw() +
  paletteer::scale_fill_paletteer_d("lisa::OskarSchlemmer") +
  paletteer::scale_colour_paletteer_d("lisa::OskarSchlemmer") +
  geom_hline(yintercept = 0, linetype = "dashed", color = "black", alpha = 0.1) +
  geom_errorbar(aes(ymin = estimate - z_90 * std.error, ymax = estimate + z_90 * std.error), width = 0.2) +
  ylab("Estimate of family-bias") +
  xlab("Family") +
  guides(fill = "none") +
  geom_hline(yintercept = 0, linetype = "dashed", col = "black", alpha = 0.3) +
  coord_flip()
p2
ggsave(here("plots", "family_preference_bias.pdf"), height = 3, width = 6)

# Combine self- and family-bias plots.
wrap_plots(p1 + ylab('Estimate of self-bias'), 
           p2 + ylab('Estimate of family-bias') + xlab('\nFamily'),
           ncol = 2, guides = "collect")
ggsave(here("plots", "self_and_family_preference_bias.pdf"), height = 3.5, width = 8)

# -------------------------------
# Plot Bias by Dimension
# -------------------------------

# Create an empty tibble to store results
all_linear_res <- tibble()

# Loop through each dimension
for(chosen_dim in unique(df$dimension)) {
  # Subset data for the chosen dimension
  df_dim <- df %>% filter(dimension == chosen_dim)
  
  # Fit linear model for this dimension
  mod_dim <- lm(rating ~ judge + gt:judge + same_judge + same_family - 1, data = df_dim)
  
  # Get robust standard errors
  vcov_mat <- sandwich(mod_dim)
  res_dim <- coeftest(mod_dim, vcov = vcov_mat)[,] %>% 
    as.data.frame() %>% 
    tibble::rownames_to_column(var = "term")
  
  colnames(res_dim) <- c("term", "estimate", "std.error", "statistic", "p.value")
  
  # Add dimension information
  all_linear_res <- bind_rows(all_linear_res, res_dim %>% mutate(dimension = chosen_dim))
}

# Create the plot
p_dimension = all_linear_res %>%
  filter(grepl("same_judge", term)) %>%
  mutate(term = str_remove(term, "same_judge")) %>%
  inner_join(families %>% rename(term = judge)) %>%
  rename(Dimension = dimension) %>%
  ggplot(aes(factor(term, levels = model_order), estimate, 
             fill = family, col = factor(family, levels = family_order), 
             shape = factor(Dimension, levels = dimension_order))) +
  geom_point(position = position_dodge(0.7)) +
  theme_bw() +
  geom_errorbar(aes(ymin = estimate - z_90 * std.error,
                    ymax = estimate + z_90 * std.error),
                width = 0.2, position = position_dodge(0.7)) +
  geom_hline(yintercept = 0, linetype = "dashed", color = "black", alpha = 0.3) +
  coord_flip() +
  scale_shape_manual("Dimension", values = 2:7) +
  paletteer::scale_colour_paletteer_d("lisa::OskarSchlemmer") +
  guides(colour = "none", fill = "none") +
  theme(legend.position = "bottom") +
  ylab("Estimate of self-bias") +
  xlab("")

p_dimension
ggsave(here("plots", "self_preference_bias_linear_by_dimension.pdf"), height = 5, width = 6)

# -------------------------------
# Regression by Dataset
# -------------------------------
all_res <- tibble()
# Use original gt and rating values.
dfo <- df %>% mutate(gt = gt_orig, rating = rating_orig)
for(chosen_dataset in unique(df$dataset)) {
  # Use the already normalized data from df
  df_dataset <- df %>% filter(dataset == chosen_dataset)
  
  # Fit linear model using normalized ratings
  mod <- lm(rating ~ judge + gt:judge + same_judge + same_family + dimension - 1, data = df_dataset)
  
  # Use robust standard errors
  vcov_mat <- sandwich(mod)
  res <- coeftest(mod, vcov = vcov_mat)[,] %>% 
    as.data.frame() %>% 
    tibble::rownames_to_column(var = "term")
  colnames(res) <- c("term", "estimate", "std.error", "statistic", "p.value")
  
  all_res <- bind_rows(all_res, res %>% mutate(dataset = chosen_dataset))
}

all_res %>%
  filter(grepl("same_judge", term)) %>%
  mutate(term = str_remove(term, "same_judge")) %>%
  inner_join(families %>% rename(term = judge)) %>%
  rename(Dataset = dataset) %>%
  ggplot(aes(factor(term, levels = model_order), estimate, 
             fill = family, col = factor(family, levels = family_order), 
             shape = Dataset)) +
  geom_point(position = position_dodge(0.7)) +
  theme_bw() +
  geom_errorbar(aes(ymin = estimate - z_90 * std.error,
                    ymax = estimate + z_90 * std.error),
                width = 0.2, position = position_dodge(0.7)) +
  geom_hline(yintercept = 0, linetype = "dashed", color = "black", alpha = 0.3) +
  coord_flip() +
  #scale_shape_manual("Dimension", values = 2:7) +
  paletteer::scale_colour_paletteer_d("lisa::OskarSchlemmer") +
  guides(colour = "none", fill = "none") +
  theme(legend.position = "bottom") +
  ylab("Estimate of self-bias") +
  xlab("")
ggsave(here("plots", "self_preference_bias_by_dataset.pdf"), height = 5, width = 6)

# -------------------------------
# Regression Analysis by Dataset Group
# -------------------------------
# First, create a function to map datasets to groups
get_dataset_group <- function(dataset) {
  if (dataset %in% c('xsum', 'cnn')) {
    return('summarization')
  } else if (dataset %in% c('stanford', 'helm-instruct', 'mtbench', 'chatbotarena', 'helm-instruct')) {
    return('open-ended qa')
  } else { #'stanford', 
    return('other')  # just in case there are other datasets
  }
}

# Add dataset_group to the dataframe
df <- df %>%
  mutate(dataset_group = sapply(dataset, get_dataset_group))

# Run regression by dataset group
all_res <- tibble()
for(chosen_group in unique(df$dataset_group)) {
  # Use the already normalized data from df
  df_group <- df %>% filter(dataset_group == chosen_group)
  
  # Fit linear model using normalized ratings
  mod <- lm(rating ~ judge + gt:judge + same_judge + same_family + dimension - 1, data = df_group)
  
  # Use robust standard errors
  vcov_mat <- sandwich(mod)
  res <- coeftest(mod, vcov = vcov_mat)[,] %>% 
    as.data.frame() %>% 
    tibble::rownames_to_column(var = "term")
  colnames(res) <- c("term", "estimate", "std.error", "statistic", "p.value")
  
  all_res <- bind_rows(all_res, res %>% mutate(dataset_group = chosen_group))
}

# Plot results
p_tasktype = all_res %>%
  filter(grepl("same_judge", term)) %>%
  mutate(term = str_remove(term, "same_judge")) %>%
  inner_join(families %>% rename(term = judge)) %>%
  rename(Dataset_Group = dataset_group) %>%
  ggplot(aes(factor(term, levels = model_order), estimate, 
             fill = family, col = factor(family, levels = family_order), 
             shape = Dataset_Group)) +
  geom_point(position = position_dodge(0.7)) +
  theme_bw() +
  geom_errorbar(aes(ymin = estimate - z_90 * std.error,
                    ymax = estimate + z_90 * std.error),
                width = 0.2, position = position_dodge(0.7)) +
  geom_hline(yintercept = 0, linetype = "dashed", color = "black", alpha = 0.3) +
  coord_flip() +
  paletteer::scale_colour_paletteer_d("lisa::OskarSchlemmer") +
  guides(colour = "none", fill = "none") +
  theme(legend.position = "bottom") +
  ylab("Estimate of self-bias") +
  xlab("") + 
  scale_shape_discrete(name = "Task type")
p_tasktype
ggsave(here("plots", "self_preference_bias_by_dataset_task_group.pdf"), height = 5, width = 6)


wrap_plots(
  p_dimension + theme(legend.position = "bottom"),
  p_tasktype + theme(legend.position = "bottom"),
  ncol = 2
)
ggsave(here("plots", "self_preference_bias_by_dimension_and_task_type.pdf"), 
       height = 4, width = 8.7)


# -------------------------------
# Ordinal Logit Regression by Dimension
# -------------------------------
all_res <- tibble()
# Use original gt and rating values.
dfo <- df %>% mutate(gt = gt_orig, rating = rating_orig)
for(chosen_dim in unique(df$dimension)) {
  dfo_dim <- dfo %>% filter(dimension == chosen_dim)
  if(length(unique(dfo_dim$rating)) > 2) {
    modpol <- MASS::polr(factor(rating, levels = sort(unique(dfo_dim$rating))) ~ judge + gt:judge + same_judge + same_family, dfo_dim)
  } else {
    modpol <- glm(factor(rating, levels = sort(unique(dfo_dim$rating))) ~ judge + gt:judge + same_judge + same_family, dfo_dim, family = binomial())
  }
  res_mod <- tidy(modpol)
  all_res <- bind_rows(all_res, res_mod %>% mutate(dimension = chosen_dim))
}

all_res %>%
  filter(grepl("same_judge", term)) %>%
  mutate(term = str_remove(term, "same_judge")) %>%
  inner_join(families %>% rename(term = judge)) %>%
  rename(Dimension = dimension) %>%
  ggplot(aes(factor(term, levels = model_order), estimate, 
             fill = family, col = factor(family, levels = family_order), 
             shape = factor(Dimension, levels = dimension_order))) +
  geom_point(position = position_dodge(0.7)) +
  theme_bw() +
  geom_errorbar(aes(ymin = estimate - z_90 * std.error,
                    ymax = estimate + z_90 * std.error),
                width = 0.2, position = position_dodge(0.7)) +
  geom_hline(yintercept = 0, linetype = "dashed", color = "black", alpha = 0.3) +
  coord_flip() +
  scale_shape_manual("Dimension", values = 2:7) +
  paletteer::scale_colour_paletteer_d("lisa::OskarSchlemmer") +
  guides(colour = "none", fill = "none") +
  theme(legend.position = "bottom") +
  ylab("Estimate of self-bias") +
  xlab("")
ggsave(here("plots", "self_preference_bias_ordinal_logit_by_dimension.pdf"), height = 4.5, width = 7)

# -------------------------------
# GAM Regression
# -------------------------------
df$judge <- as.factor(df$judge)
mod_gam <- gam(rating ~ judge + s(gt, by = judge) + same_judge + same_family + dimension, data = df)
res_gam <- tibble(term = names(mod_gam$coefficients), estimate = mod_gam$coefficients)
sb_est_gam <- res_gam %>% filter(grepl("same_judge", term)) %>% mutate(term = str_remove(term, "same_judge"))
sf_est_gam <- res_gam %>% filter(grepl("same_family", term)) %>% mutate(term = str_remove(term, "same_family"))

# -------------------------------
# Regression Dropping One Model at a Time
# -------------------------------
all_models <- unique(df$model)
all_res_drop <- c()
for(model_chosen in all_models) {
  df_sub = df %>% filter(model != model_chosen, judge != model_chosen)
  if(length(unique(df_sub$same_family))){
    mod_drop <- lm(rating ~ gt + same_judge + dimension, 
                   data = df_sub)
  } else{
    mod_drop <- lm(rating ~ judge + gt:judge + same_judge + same_family + dimension - 1,
                   data = df_sub)
  }
  all_res_drop <- bind_rows(all_res_drop, tidy(mod_drop) %>% mutate(model_removed = model_chosen))
}
sb_est_drop <- all_res_drop %>% filter(grepl("same_judge", term)) %>% mutate(term = str_remove(term, "same_judge"))
sf_est_drop <- all_res_drop %>% filter(grepl("same_family", term)) %>% mutate(term = str_remove(term, "same_family"))

sb_est_drop %>%
  left_join(families %>% mutate(term = judge)) %>%
  rename(Family = family) %>%
  ggplot(aes(factor(term, levels = model_order), estimate, col = Family, group = model_removed)) +
  theme_bw() +
  geom_point(position = position_dodge(width = 0.7)) +
  geom_errorbar(aes(ymin = estimate - z_90 * std.error, ymax = estimate + z_90 * std.error),
                width = 0.1, position = position_dodge(width = 0.7)) +
  paletteer::scale_colour_paletteer_d("lisa::OskarSchlemmer") +
  paletteer::scale_fill_paletteer_d("lisa::OskarSchlemmer") +
  geom_hline(yintercept = 0, linetype = "dashed", col = "black", alpha = 0.3) +
  coord_flip() +
  ylab("Estimate of self-bias") +
  xlab("Judge") +
  guides(fill = "none", colour = "none", shape = guide_legend(title = "Family used ground truth")) +
  theme(legend.position = "bottom")
ggsave(here("plots", "self_preference_bias_one_model_removed_at_a_time.pdf"), height = 4.5, width = 7)

# Regression dropping small capacity models.
small_models <- c("Llama 3 8B", "Mistral 7B")
mod_small <- lm(rating ~ judge + gt:judge + same_judge + same_family + dimension - 1,
                data = df %>% filter(!(model %in% small_models), !(judge %in% small_models)))
res_small <- tidy(mod_small)
sb_est_small <- res_small %>% filter(grepl("same_judge", term)) %>% mutate(term = str_remove(term, "same_judge"))
sf_est_small <- res_small %>% filter(grepl("same_family", term)) %>% mutate(term = str_remove(term, "same_family"))

p1_small <- sb_est_small %>%
  mutate(type = "Small-capacity models removed") %>%
  bind_rows(main_sb_est %>% mutate(type = "All data") %>% filter(!(term %in% small_models))) %>%
  left_join(families %>% mutate(term = judge)) %>%
  rename(Family = family) %>%
  ggplot(aes(factor(term, levels = model_order), estimate, shape = type, col = factor(Family, levels = family_order))) +
  theme_bw() +
  geom_point(position = position_dodge(width = 0.7)) +
  geom_errorbar(aes(ymin = estimate - z_90 * std.error, ymax = estimate + z_90 * std.error),
                width = 0.1, position = position_dodge(width = 0.7)) +
  paletteer::scale_colour_paletteer_d("lisa::OskarSchlemmer") +
  paletteer::scale_fill_paletteer_d("lisa::OskarSchlemmer") +
  scale_shape_manual("", values = 1:2) +
  geom_hline(yintercept = 0, linetype = "dashed", col = "black", alpha = 0.3) +
  coord_flip() +
  ylab("Estimate of self-bias") +
  xlab("Judge") +
  guides(fill = "none", colour = "none") +
  theme(legend.position = "bottom")
p1_small
ggsave(here("plots", "self_preference_bias_small_capacity_models_removed.pdf"), height = 4.5, width = 7)

p2_small <- sf_est_small %>%
  mutate(type = "Small-capacity models removed") %>%
  bind_rows(main_sf_est %>% mutate(type = "All data") %>% filter(term != "Llama 3", term != "Mistral")) %>%
  ggplot(aes(factor(term, levels = family_order), estimate, shape = type, col = factor(term, levels = family_order))) +
  theme_bw() +
  geom_point(position = position_dodge(width = 0.7)) +
  geom_errorbar(aes(ymin = estimate - z_90 * std.error, ymax = estimate + z_90 * std.error),
                width = 0.1, position = position_dodge(width = 0.7)) +
  paletteer::scale_colour_paletteer_d("lisa::OskarSchlemmer") +
  paletteer::scale_fill_paletteer_d("lisa::OskarSchlemmer") +
  scale_shape_manual("", values = 1:2) +
  geom_hline(yintercept = 0, linetype = "dashed", col = "black", alpha = 0.3) +
  coord_flip() +
  ylab("Estimate of family-bias") +
  xlab("Family") +
  guides(fill = "none", colour = "none") +
  theme(legend.position = "bottom")
p2_small
ggsave(here("plots", "family_preference_bias_small_capacity_models_removed.pdf"), height = 4.5, width = 7)

(p1_small | p2_small + guides(shape = "none")) +
  plot_layout(guides = "collect") &
  theme(legend.position = "bottom", legend.justification = "center")
ggsave(here("plots", "bias_small_capacity_models_removed.pdf"), height = 4, width = 10)

# -------------------------------
# Regression with Alternative Reference Scores by Family
# -------------------------------
models_chosen <- list(
  Claude = c("Claude v1.3", "Claude v2", "Claude 3 Sonnet", "Claude 3.5 Sonnet"),
  `Llama 3` = c("Llama 3 70B", "Llama 3 8B"),
  GPT = c("GPT-3.5 Turbo", "GPT-4", "GPT-4o"),
  Mistral = c("Mistral Large", "Mistral 7B"), 
  Cohere = c("Command-Xlarge-Beta")
)

models_chosen <- models_chosen[names(models_chosen) %in% unique(families$family)]

all_res <- map_dfr(names(models_chosen), function(fam) {
  mod_list <- models_chosen[[fam]]
  df_temp <- df %>%
    mutate(chosen_judge = ifelse(judge %in% mod_list, 1, 0)) %>%
    group_by(dimension, model, prompt_id) %>%
    mutate(gt = mean(rating[chosen_judge == 1])) %>%
    ungroup() %>%
    filter(!(judge %in% mod_list)) %>%
    droplevels()
  
  # Build model terms dynamically only if factor has more than one level.
  terms <- c()
  if(length(unique(df_temp$judge)) > 1) {
    terms <- c(terms, "judge", "gt:judge")
  }
  if(length(unique(df_temp$same_judge)) > 1) {
    terms <- c(terms, "same_judge")
  }
  if(length(unique(df_temp$same_family)) > 1) {
    terms <- c(terms, "same_family")
  }
  if(length(unique(df_temp$dimension)) > 1) {
    terms <- c(terms, "dimension")
  }
  
  # Fallback if no predictors remain.
  formula_str <- if(length(terms) > 0) {
    paste("rating ~", paste(terms, collapse = " + "))
  } else {
    "rating ~ 1"
  }
  
  mod <- lm(as.formula(formula_str), data = df_temp)
  tidy(mod) %>% mutate(family_as_gt = fam)
})



sb_est_all <- all_res %>% filter(grepl("same_judge", term)) %>% mutate(term = str_remove(term, "same_judge"))
sf_est_all <- all_res %>% filter(grepl("same_family", term)) %>% mutate(term = str_remove(term, "same_family"))

combined_data <- sb_est_all %>%
  bind_rows(main_sb_est %>% mutate(family_as_gt = "Human")) %>%
  left_join(families %>% mutate(term = judge)) %>%
  rename(Family = family)
p1_all <- combined_data %>%
  ggplot(aes(factor(term, levels = c(model_order)), 
             estimate, 
             col = Family, 
             shape = factor(family_as_gt, levels = c("Human", names(models_chosen))))) +
  theme_bw() +
  geom_point(position = position_dodge(width = 0.7)) +
  geom_errorbar(aes(ymin = estimate - z_90 * std.error, ymax = estimate + z_90 * std.error),
                width = 0.1, position = position_dodge(width = 0.7)) +
  paletteer::scale_colour_paletteer_d("lisa::OskarSchlemmer") +
  paletteer::scale_fill_paletteer_d("lisa::OskarSchlemmer") +
  geom_hline(yintercept = 0, linetype = "dashed", col = "black", alpha = 0.3) +
  coord_flip() +
  ylab("Estimate of self-bias") +
  xlab("Judge") +
  guides(fill = "none", colour = "none", shape = guide_legend(title = "Reference scores")) +
  theme(legend.position = "bottom")
p1_all
ggsave(here("plots", "self_preference_bias_ground_truth_replaced.pdf"), height = 4.5, width = 7)

combined_data <- sf_est_all %>% 
  bind_rows(main_sf_est %>% mutate(family_as_gt = "Human")) %>%
  mutate(family = term) %>%
  rename(Family = family) %>%
  mutate(term = str_remove(term, "mistral.")) 
p2_all <- combined_data %>%
  ggplot(aes(factor(term, levels = family_order), 
             estimate, 
             shape = factor(family_as_gt, levels = c("Human", family_order)), col = term)) +
  theme_bw() +
  geom_point(position = position_dodge(width = 0.7)) +
  geom_errorbar(aes(ymin = estimate - z_90 * std.error, ymax = estimate + z_90 * std.error),
                width = 0.1, position = position_dodge(width = 0.7)) +
  paletteer::scale_fill_paletteer_d("lisa::OskarSchlemmer") +
  paletteer::scale_colour_paletteer_d("lisa::OskarSchlemmer") +
  ylab("Estimate of family-bias") +
  xlab("Family") +
  geom_hline(yintercept = 0, linetype = "dashed", col = "black", alpha = 0.3) +
  coord_flip() +
  guides(fill = "none", colour = "none", shape = guide_legend(title = "Reference scores")) +
  theme(legend.position = "bottom")
p2_all
ggsave(here("plots", "family_preference_bias_ground_truth_replaced.pdf"), height = 4.5, width = 7)

(p1_all | p2_all + guides(shape = "none")) +
  plot_layout(guides = "collect") &
  theme(legend.position = "bottom", legend.justification = "center")
ggsave(here("plots", "bias_ground_truth_replaced.pdf"), height = 4, width = 10)

# -------------------------------
# Length Control Analysis
# -------------------------------
# Compute length differences and features.
df <- df %>%
  group_by(prompt_id) %>%
  mutate(
    length_diff = pred_length - mean(pred_length),
    length_feature = tanh(length_diff / sd(pred_length)),
    length_bin = ifelse(pred_length > mean(pred_length), 1, 0)
  ) %>%
  ungroup() %>%
  filter(!is.na(length_feature))

# Quick plots to visualize length features.
df %>% sample_n(1000) %>% ggplot(aes(length_diff, length_feature)) + geom_point()
df %>% sample_n(1000) %>% ggplot(aes(length_feature)) + geom_density()
df %>%
  ggplot(aes(y = model, x = length_feature)) +
  geom_density_ridges() +
  theme_bw()

# Summarize correlations between length features and scores.
df %>%
  group_by(dimension) %>%
  summarise(
    corcoef_feature_llm = cor(length_feature, rating),
    corcoef_diff_llm = cor(length_diff, rating),
    corcoef_feature_gt = cor(length_feature, gt),
    corcoef_diff_gt = cor(length_diff, gt),
    corcoef_bin_llm = cor(length_bin, rating),
    corcoef_bin_gt = cor(length_bin, gt)
  )

# Fit regression model with length control (interaction of judge and length_feature).
mod <- lm(rating ~ judge + gt:judge + same_judge + same_family + dimension + judge:length_feature - 1, data = df)
coef_est <- coef(mod)
vcov_mat <- sandwich(mod)
res <- coeftest(mod, vcov = vcov_mat)[,] %>% as.data.frame() %>% tibble::rownames_to_column(var = "term")
colnames(res) <- c("term", "estimate", "std.error", "statistic", "p.value")
res

# Extract self- and family-preference bias coefficients.
sb_est <- res %>% filter(grepl("same_judge", term)) %>% mutate(term = str_remove(term, "same_judge"))
sf_est <- res %>% filter(grepl("same_family", term)) %>% mutate(term = str_remove(term, "same_family"))

# Compare length-controlled estimates with main estimates.
sb_est <- sb_est %>% mutate(type = "Length-controlled")
main_sb_est <- main_sb_est %>% mutate(type = "No length control")
combined_data <- bind_rows(sb_est, main_sb_est) %>%
  left_join(families %>% mutate(term = judge), by = "term") %>%
  rename(Family = family)
p1 <- ggplot(combined_data, aes(
  x = factor(term, levels = model_order), 
  y = estimate, 
  fill = factor(Family, levels = family_order)
)) +
  geom_col() +
  geom_errorbar(aes(ymin = estimate - z_90 * std.error, ymax = estimate + z_90 * std.error),
                width = 0.2, position = position_dodge2()) +
  geom_hline(yintercept = 0, linetype = "dashed", col = "black", alpha = 0.3) +
  coord_flip() +
  paletteer::scale_fill_paletteer_d("lisa::OskarSchlemmer") +
  paletteer::scale_colour_paletteer_d("lisa::OskarSchlemmer") +
  guides(fill = "none") +
  theme_bw() +
  ylab("Estimate of self-bias") +
  xlab("Judge") +
  facet_wrap(~ type)
p1
ggsave(here("plots", "self_preference_bias_length_controlled.pdf"), height = 4, width = 5)

sf_est <- sf_est %>% mutate(type = "Length-controlled")
main_sf_est <- main_sf_est %>% mutate(type = "No length control")
combined_data <- bind_rows(sf_est, main_sf_est) %>%
  mutate(family = term) %>%
  rename(Family = family) %>%
  mutate(term = str_remove(term, "mistral.")) %>%
  mutate(Family = str_remove(Family, "mistral."))
p2 <- combined_data %>%
  ggplot(aes(factor(term, levels = family_order), estimate,
             fill = factor(Family, family_order))) +
  geom_col() +
  theme_bw() +
  paletteer::scale_fill_paletteer_d("lisa::OskarSchlemmer") +
  paletteer::scale_colour_paletteer_d("lisa::OskarSchlemmer") +
  geom_hline(yintercept = 0, linetype = "dashed", color = "black", alpha = 0.1) +
  geom_errorbar(aes(ymin = estimate - z_90 * std.error, ymax = estimate + z_90 * std.error),
                width = 0.2) +
  ylab("Estimate of family-bias") +
  xlab("Family") +
  guides(fill = "none") +
  geom_hline(yintercept = 0, linetype = "dashed", col = "black", alpha = 0.1) +
  coord_flip() +
  facet_wrap(~ type)
p2
ggsave(here("plots", "family_preference_bias_length_controlled.pdf"), height = 3, width = 5)

(p1 | p2) +
  plot_layout(guides = "collect") &
  theme(legend.position = "bottom", legend.justification = "center")
ggsave(here("plots", "bias_length_control.pdf"), height = 4, width = 10)

