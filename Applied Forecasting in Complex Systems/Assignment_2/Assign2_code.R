## Resit Assignment 2
library(fpp3)

# 1.1 
souvenirs %>%
  autoplot(Sales)+
  labs(title = "Souvenir sales over time")

souvenirs %>% model(STL(Sales))%>% components() %>% autoplot()

# 1.2 
souvenirs %>%
  autoplot(sqrt(Sales))+
  labs(title = "Sqrt souvenir sales over time")

souvenirs %>%
  autoplot((Sales)^(1/3))+
  labs(title = "Cube root souvenir sales over time")

souvenirs %>%
  autoplot(log(Sales))+
  labs(title = "Log souvenir sales over time")

# 1.3 regression model
souvenirs_ <- souvenirs %>%
  mutate(surf = as.integer(month(Month)==3 & year(Month)>1987))

fit <- souvenirs_ %>%
  model(reg = TSLM(log(Sales) ~ trend() + season() + surf))

report(fit)

fit %>% gg_tsresiduals(type = "response")+
  labs(title = "Residuals Diagnostics")

augment(fit) %>%
  ggplot(aes(x = .fitted, y = .resid)) +
  geom_point() + labs(x = "Fitted", y = "Residuals", title = "Residuals against fitted values")

ggplot(augment(fit),aes( x = month(Month), y = .resid, group = month(Month))) +  geom_boxplot() + labs(x = "Month", y = "Residuals", title = "Boxplots of residuals per month")

# 1.4
augment(fit) %>% features(.innov, ljung_box, lag = 24, dof = 13)

# 1.5
test <- tsibble(
  Month = yearmonth("1994 Jan") + 0:35,
  Sales = rep(0.01),
  key = Sales
) %>% mutate(surf = as.integer(month(Month)==3 & year(Month)>1987))

test <- bind_rows(souvenirs_,test) %>% filter(year(Month)>1993)

fit %>% forecast(new_data=test) %>% autoplot(souvenirs_) +labs(title = "Sales forecasts")

glance(fit) %>%
  select(.model, r_squared, adj_r_squared, AICc, CV)
##
fit2 <- souvenirs_ %>%
  model(reg2 = TSLM(log(Sales) ~ trend + season() + surf))

report(fit2)

glance(fit2) %>%
  select(.model, r_squared, adj_r_squared, AICc, CV)

fit2 %>% forecast(new_data=test) %>% autoplot(souvenirs_) +labs(title = "Sales forecasts 2")


#### 2.1
data <- tourism %>% summarise(Trips = sum(Trips))
data %>% autoplot(Trips) + labs(title = "Trips over time")
  
dcmp <- data %>%
  model(stl = STL(Trips))
components(dcmp) %>% autoplot()

components(dcmp) %>%
  as_tsibble() %>%
  autoplot(Trips, colour = "gray") +
  geom_line(aes(y=season_adjust), colour = "#0072B2") +
  labs(title = "Seasonally adjusted trips vs original data")
  
# 2.2
season_adj <- components(dcmp) %>% select(season_adjust)

fit <-season_adj  %>%
  model(damped = ETS(season_adjust ~ error("A") + trend("Ad") + season("N"))) 

report(fit)
  
fit %>% forecast(h = 8) %>% autoplot(season_adj) + labs(title = "Seasonally adjusted trips forecasts with damped")

fit %>% accuracy ()

# 2.3
fit2 <-season_adj  %>%
  model(holt = ETS(season_adjust ~ error("A") + trend("A") + season("N")))

report(fit2)

fit2 %>% forecast(h = 8) %>% autoplot(season_adj) + labs(title = "Seasonally adjusted trips forecasted with Holt")

fit2 %>% accuracy ()


fit3 <-data  %>%
  model(seasonal = ETS(Trips))

report(fit3)

fit3 %>% forecast(h = 8) %>% autoplot(data) + labs(title = "Trips forecasted with ETS(A,A,A)")

fit3 %>% accuracy ()

# 2.4 
fit2%>% gg_tsresiduals()+ labs(title = "Residuals ETS(A,A,N)")
augment(fit2) %>% features(.innov, ljung_box, lag = 8, dof = 0)

#### 3.1
data <- aus_arrivals %>% filter(Origin == "Japan")
data <- data[-2]

data %>%autoplot(Arrivals)+
  labs(title = "Arrivals from Japan over time")

data %>%autoplot(log(Arrivals))+
  labs(title = "Log arrivals from Japan over time")

dcmp <- data %>%
  model(stl = STL(log(Arrivals)))
components(dcmp) %>% autoplot()

data %>%
  features(log(Arrivals), unitroot_nsdiffs)

data %>%autoplot(difference(log(Arrivals), 4))+
  labs(title = "Annual change in log arrivals")

data %>%
  features(difference(log(Arrivals), 4), unitroot_ndiffs)

data %>%autoplot(difference(difference(log(Arrivals), 4)))+
  labs(title = "Doubly differenced log arrivals")

data %>%
  features(difference(difference(log(Arrivals), 4)), unitroot_ndiffs)

diff_data <-data %>%
  mutate(doubly_diff = difference(difference(log(Arrivals), 4)))

diff_data %>% ACF(doubly_diff)%>% autoplot()+
  labs(title = "ACF Doubly differenced log arrivals")

diff_data %>% PACF(doubly_diff)%>% autoplot()+
  labs(title = "PACF Doubly differenced log arrivals")

fit <- diff_data %>%  model(arima1 = ARIMA(log(Arrivals) ~ pdq(0,1,4) + PDQ(0,1,1)),
                            auto = ARIMA(log(Arrivals)))

fit %>% select(auto) 
fit %>% select(arima1) 

glance(fit) %>% arrange(AICc) %>% select(.model, AIC, AICc, BIC)

fit %>% select(auto)  %>% gg_tsresiduals()+
  labs(title = "Residuals ARIMA(0,1,2)(0,1,2)[4]")

