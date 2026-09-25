# Binomial mixed-effects models for instructional strategies and ICAP levels.
# Formula: outcome ~ group_type + (1 | uid), family = binomial

library(lme4)
library(dplyr)

args <- commandArgs(trailingOnly = FALSE)
file_arg <- grep("^--file=", args, value = TRUE)
if (length(file_arg) == 1) {
  root <- dirname(dirname(normalizePath(sub("^--file=", "", file_arg))))
} else {
  root <- getwd()
}

find_data_dir <- function(root) {
  env <- Sys.getenv("TEACHPEER_DATA_DIR", unset = "")
  candidates <- c(
    if (nzchar(env)) env else NULL,
    file.path(root, "data"),
    file.path(dirname(root), "TeachPeer_dataset"),
    file.path(dirname(root), "dataset")
  )
  for (path in candidates) {
    if (file.exists(file.path(path, "experiment_2_chat.csv"))) {
      return(path)
    }
  }
  stop(
    "Could not find the TeachPeer dataset. Keep TeachPeer_dataset next to ",
    "paper_submission_code, copy the CSV files into paper_submission_code/data/, or set TEACHPEER_DATA_DIR."
  )
}

data_dir <- find_data_dir(root)
data_path <- file.path(data_dir, "experiment_2_chat.csv")
result_dir <- file.path(root, "outputs", "results")
dir.create(result_dir, recursive = TRUE, showWarnings = FALSE)

df <- read.csv(data_path)
df <- df %>% filter(role == "user")
df_icap <- df %>% filter(icap %in% c("Constructive", "Active", "Passive"))
df$group_type <- factor(df$group_type, levels = c("low-uptake", "high-uptake", "mixed-uptake"))
df_icap$group_type <- factor(df_icap$group_type, levels = c("low-uptake", "high-uptake", "mixed-uptake"))

cat(sprintf("User utterances: %d\n", nrow(df)))
cat(sprintf("Valid ICAP utterances: %d\n", nrow(df_icap)))

# Praise is omitted from the GLMM because one condition has a zero cell.
strategies <- c(
  "Amplification", "Reformulate", "Accept", "Reject",
  "Exemplification", "Connection", "Metacognitive"
)
icap_levels <- c("Constructive", "Active", "Passive")

fit_binary_glmm <- function(data, outcome_col, label) {
  data[[outcome_col]] <- as.integer(data[[if (label %in% icap_levels) "icap" else "strategy"]] == label)
  n_positive <- sum(data[[outcome_col]])
  cat(sprintf("\n%s: %d / %d\n", label, n_positive, nrow(data)))
  if (n_positive < 5) {
    return(NULL)
  }

  model <- glmer(
    as.formula(paste0(outcome_col, " ~ group_type + (1 | uid)")),
    data = data,
    family = binomial(link = "logit"),
    control = glmerControl(optimizer = "bobyqa", optCtrl = list(maxfun = 100000))
  )

  coefs <- fixef(model)
  ses <- sqrt(diag(vcov(model)))
  ors <- exp(coefs)
  or_ci <- exp(confint(model, parm = "beta_", method = "Wald"))
  z_vals <- coefs / ses
  p_vals <- 2 * pnorm(-abs(z_vals))

  rows <- list()
  for (i in seq_along(coefs)[-1]) {
    param_name <- names(coefs)[i]
    if (grepl("high-uptake", param_name)) {
      comparison <- "high-uptake vs low-uptake"
    } else if (grepl("mixed-uptake", param_name)) {
      comparison <- "mixed-uptake vs low-uptake"
    } else {
      comparison <- param_name
    }
    cat(sprintf(
      "  %s: OR = %.3f, 95%% CI [%.3f, %.3f], p = %.4g\n",
      comparison, ors[i], or_ci[i, 1], or_ci[i, 2], p_vals[i]
    ))
    rows[[length(rows) + 1]] <- data.frame(
      model = label,
      comparison = comparison,
      beta = unname(coefs[i]),
      se = unname(ses[i]),
      z = unname(z_vals[i]),
      p_value = unname(p_vals[i]),
      OR = unname(ors[i]),
      CI_lower = unname(or_ci[i, 1]),
      CI_upper = unname(or_ci[i, 2])
    )
  }
  bind_rows(rows)
}

cat("\nInstructional strategies\n")
strategy_results <- bind_rows(lapply(strategies, function(label) {
  fit_binary_glmm(df, paste0("is_", label), label)
}))

cat("\nICAP engagement\n")
icap_results <- bind_rows(lapply(icap_levels, function(label) {
  fit_binary_glmm(df_icap, paste0("is_", label), label)
}))

write.csv(strategy_results, file.path(result_dir, "strategy_glmm_R_results.csv"), row.names = FALSE)
write.csv(icap_results, file.path(result_dir, "icap_glmm_R_results.csv"), row.names = FALSE)

expected_strategy_rows <- 14
expected_icap_rows <- 6
if (nrow(strategy_results) != expected_strategy_rows) {
  stop(sprintf("Strategy results incomplete (%d/%d rows).", nrow(strategy_results), expected_strategy_rows))
}
if (nrow(icap_results) != expected_icap_rows) {
  stop(sprintf("ICAP results incomplete (%d/%d rows).", nrow(icap_results), expected_icap_rows))
}

cat("\nWrote strategy_glmm_R_results.csv and icap_glmm_R_results.csv\n")
