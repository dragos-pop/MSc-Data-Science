setwd(file.path("/Users/dragos/Desktop/Statistics, Simulation and Optimization/Assignments/Assignment 1"))

# 1.3
data3 <- read.table("weather.txt", header = TRUE)
data3
# a) relevant summary of the data set, both graphically and numerically
summary(data3)
var(data3$humidity)
var(data3$temperature)
sd(data3$humidity)**2
sd(data3$temperature)**2
plot(data3$humidity, type='l',  main="Humidity trend", xlab="Day", ylab="Humidity")
plot(data3$temperature, type='l',  main="Temperature trend", xlab="Day", ylab="Temperature")
boxplot(data3, main="Humidity vs Temperature", horizontal = TRUE)


# b) Investigate the normality of the temperature graphically
hist(data3$temperature)
qqnorm(data3$temperature)

# c) 90% con- fidence interval for the mean temperature (assuming normality) 
# 1-α = 0.9 -> α = 0.1 -> 90% sure the population mean is in this interval
alpha = 0.1
# unknown standard deviation
n = length(data3$temperature)
m = mean(data3$temperature)
s = sd(data3$temperature)
t = qt(1-(alpha/2),df=n-1)
c(m-t*s/sqrt(n),m+t*s/sqrt(n))

# d) minimum n for a 95% CI  for the mean humidity st the CI <- 0.02
# 1-α = 0.95 -> α = 0.05 -> 90% sure the population mean is in this interval
E = 1
alpha = 0.05
s = sd(data3$humidity)
z = qnorm(1-(alpha/2))
z
1-(alpha/2)
n = (z^2*s^2)/(E^2)
n


####

# 1.4
data <- read.table("austen.txt", header =TRUE)
# α = 0.05
# a) homogenity because there are more than one sample (4)
# b) Test if Austen was consistent in her novels
# H0: The novels are homogenous
# H1: The novels are not homogenous
chisq.test(data[,-4])$expected # condition is satisfied (80% larger than 5)
chisq.test(data[,-4]) # p-value (0.16) > significance level -> do not reject H0 -> Austen was consistent in her novels
# c) Test if Admirer was succesful imitating Austen's style
# H0: The novels are homogenous
# H1: The novels are not homogenous
chisq.test(data)$expected # condition is satisfied (80% larger than 5)
chisq.test(data) # p-value (0.12) > significance level -> do not reject H0 -> the novels are homogenous -> Admirer was consistent with Austen
##############
