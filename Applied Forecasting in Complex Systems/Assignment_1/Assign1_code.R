## Assignment 1
library(fpp3)

# 1.1 United States GDP
global_economy %>%
  filter(Country == "United States") %>%
  autoplot(GDP)+
  labs(title = "United States GDP over time")

# 1.2 Slaughter of Victorian “Bulls, bullocks and steers”
aus_livestock %>%
  filter(Animal == "Bulls, bullocks and steers") %>%
  filter(State == "Victoria") %>%
  autoplot(Count)+
  labs(title = "Slaughter of Victorian “Bulls, bullocks and steers” count over time")

# 1.3 Gas production
aus_production %>%
  autoplot((Gas))+
  labs(title = "Log gas production (petajoules) over time")

# 2.1
takeaway <- aus_retail %>%
  filter(Industry == "Takeaway food services") %>%
  summarise(Turnover = sum(Turnover))
takeaway

train_set <- takeaway %>%
  slice(1:(n() - 4*12))
train_set

test_set <- takeaway %>%
  slice((n() - 4*12) : n())
test_set

train_set %>% 
  autoplot(Turnover)+labs(title = "Train set - Australian takeaway food turnover ($Million AUD) over time")

# 2.2
fc <- bind_cols(
  train_set %>% model(seasonal_naive = SNAIVE(Turnover)),
  train_set %>% model(random_walk_drift = RW(Turnover ~ drift())),
  train_set %>% model(seasonal_naive_drift = SNAIVE(Turnover ~ drift()))) %>% 
  forecast(h = "4 years")

fc %>%
  autoplot(takeaway)

# 2.3
fc %>% accuracy(test_set)

fit <- train_set %>%
  model(seasonal_naive_drift = SNAIVE(Turnover ~ drift()))

fit %>% gg_tsresiduals()

# 3.1
olympic_running %>% 
  filter(!is.na(Time)) %>%
  summarise(Time = mean(Time)) %>%
  autoplot(Time) +labs(title = "Average winning time (s) against the years")

olympic_running %>% 
  filter(!is.na(Time)) %>%
  autoplot(Time) +labs(title = "Winning time per event (s) against the years")

olympic_running %>% 
  filter(!is.na(Time)) %>%
  filter(Time < 410) %>%
  autoplot(Time) +labs(title = "Short distance rinning time per event (s) against the years")

# 3.2

sprint <- olympic_running %>% 
  filter(!is.na(Time)) %>%
  filter(Time < 500)%>%
  as_tsibble(
    index = Year,
    key = Time)%>%
  summarise(Time = mean(Time))

middle <- olympic_running %>% 
  filter(!is.na(Time)) %>%
  filter(Time < 1000) %>%
  filter(Time > 500)%>%
  as_tsibble(
    index = Year,
    key = Time)%>%
  summarise(Time = mean(Time))

long <- olympic_running %>% 
  filter(!is.na(Time)) %>%
  filter(Time > 1000) %>%
  as_tsibble(
    index = Year,
    key = Time)%>%
  summarise(Time = mean(Time))

fit1 <- sprint %>%
  model(lm = TSLM(Time ~ Year))
fit1 %>% report() #-0.07215 

fit2 <- middle %>%
  model(lm = TSLM(Time ~ Year))
fit2 %>% report() #-0.52

fit3 <- long %>%
  model(lm = TSLM(Time ~ Year))
fit3 %>% report() #-1.56

  