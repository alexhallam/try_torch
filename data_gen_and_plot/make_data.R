library(tidyverse)
set.seed(42)
x <- runif(1000, -10, 10)
e <- rnorm(1000, 0, 500)
x2 <- x^2
x3 <- x^3
y <- 1 + 2 * x + 3 * x2 + 4 * x3 + e
df <- tibble(y, x, x2, x3)

p <- df %>%
    ggplot(aes(y = y, x = x)) +
    geom_point(size = 0.5) +
    theme_minimal()
ggsave("data.jpg", p, bg = "white")
write_csv(df, "simple_data.csv")
