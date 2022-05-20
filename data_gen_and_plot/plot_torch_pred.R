library(tidyverse)
set.seed(42)
df <- read_csv("simple_data.csv")

dfs <- df %>% mutate(
    x = (x - mean(x)) / sd(x),
    x2 = (x2 - mean(x2)) / sd(x2),
    x3 = (x3 - mean(x3)) / sd(x3)
)
dfs
broom::tidy(summary(lm(y ~ ., dfs)))
write_csv(broom::tidy(summary(lm(y ~ ., dfs))), "./data_gen_and_plot/regression_params.csv")

f <- function(x) 72.5605 + 48.3745 * x + 82.0545 * x^2 + 1552.5685 * x^3
lf <- function(x) 38.835095380543265 + 5.706411502517686 * x + 86.88057334797983 * x^2 + 1557.4790515840994 * x^3
f_upper <- function(x) (38.835095380543265 + 5.706411502517686 * x + 86.88057334797983 * x^2 + 1557.4790515840994 * x^3) + 1.96 * 2061.7646
f_lower <- function(x) (38.835095380543265 + 5.706411502517686 * x + 86.88057334797983 * x^2 + 1557.4790515840994 * x^3) - 1.96 * 2061.7646
f_upperi <- function(x) (38.835095380543265 + 5.706411502517686 * x + 86.88057334797983 * x^2 + 1557.4790515840994 * x^3) + 2061.7646
f_loweri <- function(x) (38.835095380543265 + 5.706411502517686 * x + 86.88057334797983 * x^2 + 1557.4790515840994 * x^3) - 2061.7646

p <- ggplot(dfs, aes(x, y)) +
    geom_point(size = 0.5) +
    # stat_function(fun=f, linetype="dashed", color = 'red')+
    stat_function(fun = lf, linetype = "solid", color = "green") +
    theme_minimal()
ggsave("./data_gen_and_plot/plots/regression_fit.jpg", p, bg = "white")

p <- ggplot(dfs, aes(x, y)) +
    geom_point(size = 0.5) +
    stat_function(fun = f, linetype = "dashed", color = "red") +
    stat_function(fun = lf, linetype = "solid", color = "green") +
    theme_minimal()
ggsave("./data_gen_and_plot/plots/py_torch_regression_fit.jpg", p, bg = "white")


p <- ggplot(dfs, aes(x, y)) +
    geom_point(size = 0.5) +
    stat_function(fun = f, linetype = "dashed", color = "red") +
    stat_function(fun = f_upper, linetype = "dashed", color = "grey") +
    stat_function(fun = f_lower, linetype = "dashed", color = "grey") +
    stat_function(fun = f_upperi, linetype = "dashed", color = "black") +
    stat_function(fun = f_loweri, linetype = "dashed", color = "black") +
    stat_function(fun = lf, linetype = "solid", color = "green") +
    theme_minimal()
ggsave("./data_gen_and_plot/plots/py_torch_regression_fit_ci.jpg", p, bg = "white")
