#Chris Crown, Dylan Beers and COnnor Hill
#Final case comp R script


library(caret)
library(tidyverse)
library(glmnet)
library(e1071)
library(Matrix)
library(MASS) #watch out, this masks dplyr's select functions
library(dplyr)
library(ROCR)
library(skimr)
library(lubridate)
library(randomForest)
library(xgboost)
library(smotefamily)

#Turn off scientific notation as global setting
options(scipen=999)

#download the data for the case study
ghcase <- read.csv(file.choose(), header=T)

skim(ghcase) #1792 rows with 26 columns

#quick boxplot eda

boxplot(ghcase$NumWebPurchases~ghcase$Response)

boxplot(ghcase$Income~ghcase$Response)

boxplot(ghcase$Recency~ghcase$Response)

boxplot(ghcase$MntWines~ghcase$Response)

boxplot(ghcase$MntFruits~ghcase$Response)

boxplot(ghcase$MntMeatProducts~ghcase$Response)

boxplot(ghcase$MntGoldProds~ghcase$Response)

boxplot(ghcase$NumWebPurchases~ghcase$Response)

boxplot(ghcase$NumCatalogPurchases~ghcase$Response)

boxplot(ghcase$NumStorePurchases~ghcase$Response)

boxplot(ghcase$NumWebVisitsMonth~ghcase$Response)

boxplot(ghcase$MntFruits~ghcase$Response)

#duplicate row check -- 111 rows deleted
ghcase<-ghcase %>%
  distinct(.keep_all = TRUE)

skim(ghcase) #1681 rows with 26 columns now

#quick EDA
glimpse(ghcase)
str(ghcase)
skim(ghcase)
#looks like we have missing values only in the income column
#and some potentially erroneous values in the birth year column
#aka - people who couldn't possibly be that old
#we'll get to all these later tho

#lets check for negative values since everything in this data set should be above 0 in theory

negative_value_check <- sapply(ghcase, function(col) any(col < 0, na.rm = TRUE))
print(negative_value_check) #-- no negative values!

#a method we like to use to trigger a quick and dirty search for silly/erroneous values w/the summary function,
#as it quickly shows mins and maxs
#this can be used to put a pin in columns we would need to clean up later *IF* we choose to run analyses w/them
summary(ghcase)
#this shows us that:
#we have some unrealistically early birth years.
#no birth years that are too recent (assuming 2014 customer start date)
# we have some very low reported incomes and 15 MISSING VALUES in this column
#kidhome and teenhome are both COUNTS of kids/teens (up to 2) not binary
#none of the amount of luxury product $$ values seem ridiculously high
#and all Accepted Campaign vars, binary, min at 0 and max at 1 - perfect


#now, for something we did in Assignment 1 that we definitely want to repeat

#our justification is that, since this data seems to be collected from 2014
#we are assuming that , for the sake of this case, we're "analyzing" this data IN 2014
#given that assumption, we want to eliminate any ages that are over 95 years old
#upon the latest account creation data - 2014
#first let's see how many there are
too_old_rows <- ghcase |> filter(Year_Birth <= 1919) |> nrow()
print(too_old_rows) #-- there are only 3, so let's impute an average age for those

#this is assuming that these outliers were just input wrong
#this will replace them with the average we found when we skimmed
ghcase$Year_Birth <- ifelse(ghcase$Year_Birth < 1919, 1969, ghcase$Year_Birth)

skim(ghcase) # good, still the same amount of rows

#let's turn the Dt_Customer column into a date with lubridate

ghcase$Dt_Customer <- mdy(ghcase$Dt_Customer)

str(ghcase) # check to make sure that worked

#I create an object (and make sure it is actually a date and not a string) of this date
min_date <- as.Date("2012-07-31")

#then I create a new column that makes a "days since"-type number for each row's
#account creation date. This sets the earliest date as a "0" ref point and then
#simulates time passing since then. This way we can see if this new number has
#an effect on the model without having to create a gillion dummyvars for each
#character/date
ghcase <- ghcase %>% mutate(Dt_Cust_Num = as.numeric(Dt_Customer - min_date))

skim(ghcase) #check to make sure it took

#-------------------------------------------
#binning features - all repeat from assignment 2 part 2 submission, no changes
ghcase <- ghcase |> mutate(Marital_Level = case_when
                           (Marital_Status %in% c("Single", "Divorced", "Widow") ~ 0,
                             Marital_Status %in% c("Married", "Together") ~ 1))

#print to check
print(ghcase$Marital_Level)

#...and the education level into a binary
ghcase <- ghcase |> mutate(Ed_Level = case_when
                           (Education %in% c("Basic", "Graduation", "2n Cycle") ~ 0,
                             Education %in% c("PhD", "Master") ~ 1))

#print to check
print(ghcase$Ed_Level)

#...and the presence or absence of any children in the home as a binary
ghcase <- ghcase |> mutate(Dependents = case_when(
  Kidhome == 0 & Teenhome == 0 ~ 0,
  Kidhome > 0 | Teenhome > 0 ~ 1))

#print to check
print(ghcase$Dependents)

#we think this will sufficiently streamline our model, making it easier to interpret
#without impacting the data too much. See further justifications in our report
#ALSO - if we end up wanting to do SMOTE - my reading says that we will need
#factor (binary I think) variables for all. These changes will make sure
#they are all binary just in case we choose to do this (which I hope we can figure out)

#we also need to make sure that we delete the old columns so that they do not
#influence any future models we will build

#-------------------------------------------------------------

#I was having trouble with this but it seems the MASS package was impacting
#my ability to use dplyr, so I had to explicitly tell it the way I wanted it to work
ghcase <- ghcase |> dplyr::select(-Kidhome, -Teenhome, -Dt_Customer, -Marital_Status, -Education)

str(ghcase) #--here we go, this got 'em removed. Still 1681 rows

# change JUST the target variable and categorical vars into factors
ghcase <- ghcase %>% mutate_at(c(
                                 "Response"
                                 ), as.factor)

#do not rename it at this point, we need it to be num for SMOTE

# relevel response
ghcase$Response<- relevel(ghcase$Response, ref = "1")

#make sure levels are correct
levels(ghcase$Response)

#------------------------------------------------------------------------------

#one way to figure out if there are NA's - use apply function which does the search for
#every single var of the training set. Look at the columns and see, for any column, 
#search for null 
null_vars<-apply(ghcase, 2, function(x) any(grepl('NULL', x)))==T

null_vars[null_vars == TRUE] #doesn't appear any are NULL bc there are no true values

skim(ghcase)

#imputing missing vars
#impute the median for any continuous var using caret package

#i'm choosing to impute all 15 income and 5 marital status BEFORE data partitioning (gasp!)
#it is just smoother and easier and it is such a low % of our data that I don't
#think that this will cause a substantial "data leakage"
#basically by building a model - we are using median bc of its robustness to outliers
ghcase <- preProcess(ghcase$Income, method='medianImpute')

#use preprocessing like in class but make sure to remove the NA's
preprocess_model1 <- preProcess(ghcase, method = "medianImpute", na.remove = TRUE)

#ignore the warning here^
ghcase$Income <- predict(preprocess_model1, ghcase)$Income

skim(ghcase)# i have learned to check this SOB every single step of the way
#bc this is where the row mismatch got us last time

#---------------------------------------

#use a simple ifelse statement to replace with the mode (found via skim)
ghcase$Marital_Level <- ifelse(is.na(ghcase$Marital_Level),
                               1, ghcase$Marital_Level)

skim(ghcase)
str(ghcase)
#---------------------------------------------

#some more quick eda 


###############################
#patition data
#but bc we do this now, any changes we do to the dataset, after, will have to be replicated
#for both the training and testing set

set.seed(97)

# Split data into training and testing sets
index <- createDataPartition(ghcase$Response, p = 0.8, list = FALSE)
ghcase_train <- ghcase[index,]
ghcase_test <- ghcase[-index,]

#checkins
str(ghcase_train)
skim(ghcase_train)
summary(ghcase_train$Response) #i was having a helluva time with my response var getting
#all messed up once i tried to get fancy and do SMOTE, I will be checking in on this one alot

#a lot of it had to do with response being a factor or not
ghcase_train$Response <- as.factor(ghcase_train$Response)

#bc response is not the first var in the list of vars, make an object that includes
#all the PV's so we can call it easily
predictor_columns <- setdiff(names(ghcase_train), "Response")

#do SMOTE!!!!!
#my research has shown me that a K value of 5  and a dup size of 2 are the best
#parameters for this data imbalance
smote_result <- SMOTE(ghcase_train[, predictor_columns], ghcase_train$Response, K = 5, dup_size = 2)

#here, we extract the newly balanced dataset from the smote function
ghcase_train_bal <- smote_result$data

#assign the response column
colnames(ghcase_train_bal)[ncol(ghcase_train_bal)] <- "Response"

# make sure it's a factor just in-f****in case
ghcase_train_bal$Response <- as.factor(ghcase_train_bal$Response)

#look at our class distrubition now that we have balanced the dataset
#you can see we have tripled our original ghcase_train 201 respond yes's so now it's about 2:1
#for the majority and minority class instead of 1430 to 251 in the ghcase entire dataset
table(ghcase_train_bal$Response)

# look at the first few rows to make sure nothing is wonky
head(ghcase_train_bal)

skim(ghcase_train_bal) #check it this way
str(ghcase_train_bal) #check it that way

#next, a major issue I was having was just bc I did not understand what, exactly,
#smote does. It synthetically kinda imputes values BASED ON nearest neighbors of the
#class you are trying to oversample
#this is cool but potentialy problematic down the road bc it makes values like 0.46 for
#binary variables
#you can see if we print each column, some of them are not 1 or 0
print(ghcase_train_bal$AcceptedCmp3) #--has non binary values
print(ghcase_train_bal$AcceptedCmp4)
print(ghcase_train_bal$AcceptedCmp5)
print(ghcase_train_bal$AcceptedCmp1)
print(ghcase_train_bal$AcceptedCmp2)
print(ghcase_train_bal$Complain)
print(ghcase_train_bal$Marital_Level)
print(ghcase_train_bal$Ed_Level)
print(ghcase_train_bal$Dependents)#--- this and all above have synthetically non-binary values
print(ghcase_train_bal$Response) #- this one is fine bc we had it as a factor when we smoted

#so I go in and round each to the closest integer (0 or 1) to get them back to
#binary - do you guys have any issues with this method? Seems like a balanced
#way to get MORE RespondYEs but also treating their other values fairly
#but maybe I am overlooking something?
ghcase_train_bal$AcceptedCmp3 <- round(ghcase_train_bal$AcceptedCmp3)
ghcase_train_bal$AcceptedCmp4  <- round(ghcase_train_bal$AcceptedCmp4 )
ghcase_train_bal$AcceptedCmp5  <- round(ghcase_train_bal$AcceptedCmp5 )
ghcase_train_bal$AcceptedCmp1 <- round(ghcase_train_bal$AcceptedCmp1)
ghcase_train_bal$AcceptedCmp2 <- round(ghcase_train_bal$AcceptedCmp2)
ghcase_train_bal$Complain   <- round(ghcase_train_bal$Complain  )
ghcase_train_bal$Marital_Level <- round(ghcase_train_bal$Marital_Level)
ghcase_train_bal$Ed_Level <- round(ghcase_train_bal$Ed_Level)
ghcase_train_bal$Dependents  <- round(ghcase_train_bal$Dependents )

#now for a final check to make sure it is ONLY 1's or 0's up in here
print(ghcase_train_bal$AcceptedCmp3) 
print(ghcase_train_bal$AcceptedCmp4)
print(ghcase_train_bal$AcceptedCmp5)
print(ghcase_train_bal$AcceptedCmp1)
print(ghcase_train_bal$AcceptedCmp2)
print(ghcase_train_bal$Complain)
print(ghcase_train_bal$Marital_Level)
print(ghcase_train_bal$Ed_Level)
print(ghcase_train_bal$Dependents)
print(ghcase_train_bal$Response)
#hell yeah - we have successfully created synthetic values to fix our imbalance
#and then rounded all those values to their closest binary value
#let's make them factors now

#let's run some models babyyyyyyy
#gotta be a factor for this to run right
ghcase_train_bal$Response <- as.factor(ghcase_train_bal$Response)

# rename response 
ghcase_train_bal$Response<-fct_recode(ghcase_train_bal$Response, Respond = "1",notRespond = "0")

# relevel response
ghcase_train_bal$Response<- relevel(ghcase_train_bal$Response, ref = "Respond")

#make sure levels are correct
levels(ghcase_train_bal$Response)

str(ghcase_train_bal)

###################################################

#xgboosted model
set.seed(8)
gh_model_gbm <- train(Response ~ .,
                      data = ghcase_train_bal,
                      method = "xgbTree",
                      trControl = trainControl(method = "cv", 
                                               number = 5,
                                               classProbs = TRUE,
                                               summaryFunction = twoClassSummary),
                      # provide a grid of parameters
                      tuneGrid = expand.grid(
                        nrounds = 150,
                        eta = 0.5,
                        max_depth = 6,
                        gamma = 0,
                        colsample_bytree = 1,
                        min_child_weight = 1,
                        subsample = 1),
                      verbose = FALSE,
                      metric = "ROC")

#?make.names

#evaluate our model predictions
plot(gh_model_gbm)

gh_model_gbm$bestTune

#learn the most important vars in building the model
plot(varImp(gh_model_gbm))

gh_prob_gbm<- predict(gh_model_gbm, ghcase_test, type="prob")
gh_prob_gbm

#In label.ordering the negative class is first then the positive class
#it gives an error but this is explicitly how we were instructed to do it in class
pred_gbm = prediction(gh_prob_gbm$Respond, ghcase_test$Response,label.ordering =c("notRespond","Respond")) 
perf_gbm = performance(pred, "tpr", "fpr")
plot(perf_gbm, colorize=TRUE)

unlist(slot(performance(pred_gbm, "auc"), "y.values"))
#this is the best AUC we've gotten yet, 0.9072727!!!

###############################################

#let's get shapped 

#install.packages("SHAPforxgboost")
library(SHAPforxgboost)

str(ghcase_train_bal)

#turn this into a matrix for shap to understand it
ghcase_train_bal <- as.matrix(dplyr::select(ghcase_train_bal, -Response))

# Crunch SHAP values
shap <- shap.prep(gh_model_gbm$finalModel, X_train = ghcase_train_bal)

# SHAP importance plot
shap.plot.summary(shap)

# Use 15 most important predictor variables
top15<-shap.importance(shap, names_only = TRUE)[1:15]

for (x in top15) {
  p <- shap.plot.dependence(
    shap, 
    x = x, 
    color_feature = "auto", 
    smooth = FALSE, 
    jitter_width = 0.01, 
    alpha = 0.6,
    size=2
  ) +
    ggtitle(x)
  print(p)
}

#look at a correlation we think is important!!
web_conversion_corr <- cor(ghcase$NumWebVisitsMonth, ghcase$NumWebPurchases)
print(web_conversion_corr)

fish_store_corr <- cor(ghcase$MntFishProducts, ghcase$NumStorePurchases)
print(fish_store_corr)

meat_store_corr<- cor(ghcase$MntMeatProducts, ghcase$NumStorePurchases)
print(meat_store_corr)

