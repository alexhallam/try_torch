df <- read_csv("./simple_regression.csv")
p <- df %>%
    ggplot(aes(y = y, x = predicted), color = "black") +
    geom_abline() +
    geom_point()
ggsave("plot.jpg", p)

p <- df %>%
    ggplot(aes(y = y, x = sqft_living), color = "black") +
    geom_point()
ggsave("plot_.jpg", p)

df <- read_csv("./poly_regression.csv")
p <- df %>%
    ggplot(aes(y = y, x = predicted), color = "black") +
    geom_abline() +
    geom_point()
ggsave("plot2.jpg", p)

p <- df %>%
    ggplot(aes(y = y, x = sqft_living), color = "black") +
    geom_point()
ggsave("plot2_.jpg", p)
