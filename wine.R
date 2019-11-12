# Installing packages
install.packages("tidyverse")
install.packages("janitor")
install.packages("purr")
install.packages("psych")
install.packages("tidytext")
install.packages("randomcoloR")
install.packages("broom")
install.packages("ggrepel")
install.packages("caret")
install.packages("e1071")
install.packages("randomForest")
install.packages("RColorBrewer")
install.packages("sentimentr")
install.packages("dplyr")
# Loading Libraries -------------------------------------------------------
library(tidyverse)
library(janitor)
library(purrr)
library(psych)
library(tidytext)
library(randomcoloR)
library(tidytext)
library(broom)
library(ggrepel)
library(caret)
library(e1071)
library(randomForest)
library(RColorBrewer)
library(sentimentr)
library(dplyr)
sentiments

#get_winements("afinn")

# Reading File -----------------------------------------------------------
wine <- read.csv("C:/Users/admin/Desktop/Research Project/wine.csv") %>% clean_names()


# Exploring Data ----------------------------------------------------------

glimpse(wine)
str(wine)

# Checking number of na
map_df(wine, ~ sum(is.na(.x)))

# Only 13695 NA values so omitting that
wine <- wine %>% na.omit()

# Checking distribution + correlation using pairs.panels
pairs.panels(wine[1:100, 2:5])
pairs.panels(wine[1:100, 6:11])


# Exploring & Visualising Data --------------------------------------------

# Distriution of Points Variable

ggplot(data = wine, aes(x = points, colour = I("black"), fill = I("#099DD9"))) +
  geom_histogram(binwidth = 1) +
  labs(x = "Points", y = "Frequency", title = "Distribution of points")
#--------------------------

# Distribution of Price variable
ggplot(data = wine, aes(x = price, colour = I("black"), fill = I("#099DD9"))) +
  geom_histogram() +
  labs(x = "Price", y = "Frequency", title = "Distribution of prices")

# Strongly right skewed, using log to plot wine prices
#--------------------------

ggplot(data = wine, aes(x = log(price), colour = I("black"), fill = I("#099DD9"))) +
  geom_histogram() +
  labs(x = "log(Price)", y = "Frequency", title = "Distribution of log(prices)")

# slightly right skewed
#--------------------------


# Checking the word count from reviews for each wine
wine$wordcount <- sapply(gregexpr("\\S+", wine$text), length)

ggplot(data = wine, aes(x = wordcount, colour = I("black"), fill = I("#099DD9"))) +
  geom_histogram(binwidth = 3) +
  labs(x = "Word Count", y = "Frequency", title = "Distribution of word count of text")

# slightly right skewed
#--------------------------

# Country with most costly wines

wine %>%
  group_by(country) %>%
  summarise(avg_price = mean(price)) %>%
  top_n(10) %>%
  ggplot(aes(x = reorder(country, -avg_price), avg_price, fill = country)) + geom_bar(stat = "identity") + labs(title = "Country with Highest wine Prices", x = "", y = "wine Price") + scale_fill_brewer(palette = "Paired") + theme_bw() + theme(legend.position = "none", plot.title = element_text(hjust = 0.5), axis.text.x = element_text(angle = 45, hjust = 1))
#--------------------------

# Score Vs Price

ggplot(wine, aes(x = price, y = points)) +
  geom_jitter(shape = 1) + geom_smooth() +
coord_cartesian(xlim = c(0, 4000), ylim = c(75, 100)) +
  labs(title = "Score vs Price", x = "Price", y = "Score")

ggplot(wine, aes(x = price, y = points)) +
  geom_jitter(shape = 1) +
  coord_cartesian(xlim = c(0, 40), ylim = c(75, 100)) +
  labs(title = "Score vs Price", x = "Price", y = "Score")

# It appears that the correlation between wine price and points given levels off around $40! Additinally, it is clear from the first larger plot that spending thousands of dollars on a bottle of wine is usually a poor decision and a comparable bottle can be had for much less

#--------------------------

# Explotation of Most Common Words

wine$text <- as.character(wine$text)
wine$sentiment <- as.numeric(wine$sentiment)

wine = wine %>%
  mutate(grade = ifelse(sentiment > "3","Good",ifelse(sentiment > "2","Average","Bad")))
#Creating Grade column based on points

wine$grade = as.factor(wine$grade)
wine$grade = factor(wine$grade,levels(wine$grade)[c(3,1,2)])

#Finding most common words
wine_Words = wine %>% dplyr::select(text, grade)%>%
  unnest_tokens(word, text)%>%
  anti_join(stop_words)%>%
  dplyr::filter(!str_detect(word, "[0-9]"))

colourCount = 15
getPalette = colorRampPalette(brewer.pal(9, "Set1"))


wine_Words %>%
  count(word, sort = TRUE) %>%
  top_n(15) %>%
  mutate(word = reorder(word, n)) %>%
  ggplot(aes(word, n, fill = word)) +
  geom_col() +
  labs(title = "Top 15 Significant Words From All Comments", x = "Total Number of Occurances", y = "Word") +
  coord_flip()+  scale_fill_manual(values = getPalette(colourCount))+
  theme_bw() + theme(legend.position = "none", plot.title = element_text(hjust = 0.5))

#--------------------------

comment_words = wine_Words %>%
  count(grade, word, sort = TRUE) %>%
  ungroup()

total_words = comment_words %>% 
  group_by(grade) %>% 
  summarize(total = sum(n))

grade_words = left_join(comment_words, total_words)

grade_words = grade_words %>%
  bind_tf_idf(word, grade, n)


#Words from Good review
grade_words %>% 
  arrange(desc(tf_idf)) %>%
  mutate(word = factor(word, levels = rev(unique(word)))) %>% 
  group_by(grade) %>% 
  top_n(10) %>% 
  ungroup %>%
  dplyr::filter(grade == "Good")%>%
  ggplot(aes(word, tf_idf)) +
  geom_col(show.legend = FALSE, fill = 'red') +
  labs(title ="Top 10 Significant Words from Good posts", x = NULL, y = "tf-idf") +
  coord_flip()+theme_bw()

#Words from Bad review
grade_words %>% 
  arrange(desc(tf_idf)) %>%
  mutate(word = factor(word, levels = rev(unique(word)))) %>% 
  group_by(grade) %>% 
  top_n(10) %>% 
  ungroup %>%
  dplyr::filter(grade == "Bad")%>%
  ggplot(aes(word, tf_idf)) +
  geom_col(show.legend = FALSE, fill = 'Blue') +
  labs(title ="Top 10 Significant Words from Bad reviews", x = NULL, y = "tf-idf") +
  coord_flip()


# Feature Extraction ------------------------------------------------------


#sentiment analysis using sentiment() function
score <- sentiment(wine$text, polarity_dt = lexicon::hash_sentiment_jockers_rinker, valence_shifters_dt = lexicon::hash_valence_shifters, hyphen = "")
score1 <- aggregate(sentiment~element_id, data=score, FUN=sum) 
wine$score <- score1$sentiment
sum(wine$score) # obtained sum of sentiment score as -19.99 / 3866 = -0.0052, so drop in the rating value

#Use this in program as well https://datascienceplus.com/automated-text-feature-engineering-using-textfeatures-in-r/
## This link shows that there is drop in share price of wine during december 2014 https://www.macrotrends.net/stocks/charts/AAPL/wine/stock-price-history
##AS our total sentiment score is negative it is proven that lower sentment score drops the value of company 

#Sentiment Analysis 

word_final <- inner_join(wine_Words, grade_words[,c("grade","word","tf")], by=c("grade","word"))

word_final <- inner_join(word_final, get_sentiments("afinn"))

sentiment_score <- word_final %>% group_by(grade) %>% summarise(sentiment_score=mean(value)) 

wine_final <- full_join(wine, sentiment_score)

wine_final %>% group_by(grade) %>% summarise(mean(sentiment_score,na.rm = TRUE))

temp <- wine_final %>% dplyr::filter(is.na(sentiment_score)) %>% 
  mutate(sentiment_score= ifelse(grade=="Good",1.76,ifelse(grade=="Average",1.54,1.13)))

wine_final <- wine_final %>% na.omit()

wine <- unique(rbind(wine_final,temp))

head(wine)


#change sentiment score
score1 <- sapply(score, FUN = function(x) {ifelse(x==1.74,1.74,ifelse(x==1.54,0,1.13))})
#Volume of missing value
wine$sentiment_score


#We have created three extra columns in our wine dataset, one for word count, second for grade (based on points) & third score (basis winement score from description of wine)


# Model Building ----------------------------------------------------------
#Creating Test & Train

wine <- wine %>% dplyr::select(points,price,winement_score,wordcount,country)
dmy <- dummyVars(" ~ .", data = wine,fullRank = T)
wine_transformed <- data.frame(predict(dmy, newdata = wine))

splitIndex <- createDataPartition (wine_transformed$points, p=.70, list=FALSE, times=1)

train <- wine_transformed[splitIndex, ]
test <- wine_transformed[-splitIndex, ]



# BUILD MODELS ------------------------------------------------------------

#KNN MODEL

RMSE = function(m, o){
  sqrt(mean((m - o)^2))
}


#trctrl <- trainControl(method = "repeatedcv", number = 5)
#set.seed(3333)
knn_fit <- train(points ~., data = train[1:2000,], method = "knn")

test$pred_knn <- predict(knn_fit, newdata = test)


#SVM

trctrl <- trainControl(method = "repeatedcv", number = 10, repeats = 3)

svm_Linear <- train(points ~., data = train[1:2000,], method = "svmLinear",trControl=trctrl)
                    
                  
test$pred_svm <- predict(svm_Linear, newdata = test)   
                    
#LINEAR REGRESSION
                    
linear_model <- lm(points ~., data = train[1:2000,])
test$pred_lm <- predict(linear_model, newdata = test)   
                    
             
                    
                    
#RmSE on Test from all algorithm

print("RMSE FOR KNN model is ");RMSE(test$pred_knn,test$points)
print("RMSE FOR SVM model is ");RMSE(test$pred_svm,test$points)
print("RMSE FOR LM model is ");RMSE(test$pred_lm,test$points)

result1 <- data.frame("Actual value" = test$points, "Predicted value" = test$pred_lm)
  result1
plot(result1, type= "o")




sentiment_by("My life has become terrible since I met you and lost money. But I still have got a little hope left in me") %>%
  highlight()
