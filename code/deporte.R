# Including needed libraries
#install.packages("qdap")
#install.packages("XML")
#install.packages("tm")
#install.packages("splitstackshape")
#install.packages("caret")
#install.packages("e1071")
library(qdap)
library(XML)
library(tm)
library(splitstackshape)
library(caret)
library(e1071)

start.time <- Sys.time()

# Preparing parameters
n <- 10
lang <- "es"
path_training <- "./training"    # Your training path
path_test <- "./test"							# Your test path
k <- 3
r <- 1

# Auxiliar functions
# * GenerateVocabulary: Given a corpus (training set), obtains the n most frequent words
# * GenerateBoW: Given a corpus (training or test), and a vocabulary, obtains the bow representation

# GenerateVocabulary: Given a corpus (training set), obtains the n most frequent words
GenerateVocabulary <- function(path, n = 1000, lowcase = TRUE, punctuations = TRUE, numbers = TRUE, whitespaces = TRUE, swlang = "", swlist = "", verbose = TRUE) {
  setwd(path)
  
  # Reading corpus list of files
  files = list.files(pattern="*.xml")
  
  # Reading files contents and concatenating into the corpus.raw variable
  corpus.raw <- NULL
  i <- 0
  for (file in files) {
    xmlfile <- xmlTreeParse(file, useInternalNodes = TRUE)
    corpus.raw <- c(corpus.raw, xpathApply(xmlfile, "//document", function(x) xmlValue(x)))
    i <- i + 1
    if (verbose) print(paste(i, " ", file))
  }
  
  # Preprocessing the corpus
  corpus.preprocessed <- corpus.raw
  
  if (lowcase) {
    if (verbose) print("Tolower...")
    corpus.preprocessed <- tolower(corpus.preprocessed)
  }
  
  if (punctuations) {
    if (verbose) print("Removing punctuations...")
    corpus.preprocessed <- removePunctuation(corpus.preprocessed)
  }
  
  if (numbers) {
    if (verbose) print("Removing numbers...")
    corpus.preprocessed <- removeNumbers(corpus.preprocessed)
  }
  
  if (whitespaces) {
    if (verbose) print("Stripping whitestpaces...")
    corpus.preprocessed <- stripWhitespace(corpus.preprocessed)
  }
  
  if (swlang!="")	{
    if (verbose) print(paste("Removing stopwords for language ", swlang , "..."))
    corpus.preprocessed <- removeWords(corpus.preprocessed, stopwords(swlang))
  }
  
  if (swlist!="") {
    if (verbose) print("Removing provided stopwords...")
    corpus.preprocessed <- removeWords(corpus.preprocessed, swlist)
  }
  
  # Generating the vocabulary as the n most frequent terms
  if (verbose) print("Generating frequency terms")
  corpus.frequentterms <- freq_terms(corpus.preprocessed, n)
  if (verbose) plot(corpus.frequentterms)
  
  return (corpus.frequentterms)
}


# GenerateBoW: Given a corpus (training or test), and a vocabulary, obtains the bow representation
GenerateBoW <- function(path, vocabulary, n = 100000, lowcase = TRUE, punctuations = TRUE, numbers = TRUE, whitespaces = TRUE, swlang = "", swlist = "", class="variety", verbose = TRUE) {
  setwd(path)
  
  # Reading the truth file
  truth <- read.csv("truth.txt", sep=":", header=FALSE)
  truth <- truth[,c(1,4,7)]
  colnames(truth) <- c("author", "gender", "variety")
  
  i <- 0
  bow <- NULL
  # Reading the list of files in the corpus
  files = list.files(pattern="*.xml")
  for (file in files) {
    # Obtaining truth information for the current author
    author <- gsub(".xml", "", file)
    variety <- truth[truth$author==author,"variety"]
    gender <- truth[truth$author==author,"gender"]
    
    # Reading contents for the current author
    xmlfile <- xmlTreeParse(file, useInternalNodes = TRUE)
    txtdata <- xpathApply(xmlfile, "//document", function(x) xmlValue(x))
    
    # Preprocessing the text
    if (lowcase) {
      txtdata <- tolower(txtdata)
    }
    
    if (punctuations) {
      txtdata <- removePunctuation(txtdata)
    }
    
    if (numbers) {
      txtdata <- removeNumbers(txtdata)
    }
    
    if (whitespaces) {
      txtdata <- stripWhitespace(txtdata)
    }
    
    # Building the vector space model. For each word in the vocabulary, it obtains the frequency of occurrence in the current author.
    line <- author
    freq <- freq_terms(txtdata, n)
    for (word in vocabulary$WORD) {
      thefreq <- 0
      if (length(freq[freq$WORD==word,"FREQ"])>0) {
        thefreq <- freq[freq$WORD==word,"FREQ"]
      }
      line <- paste(line, ",", thefreq, sep="")
    }
    
    # Concatenating the corresponding class: variety or gender
    if (class=="variety") {
      line <- paste(variety, ",", line, sep="")
    } else {
      line <- paste(gender, ",", line, sep="")
    }
    
    # New row in the vector space model matrix
    bow <- rbind(bow, line)
    i <- i + 1
    
    if (verbose) {
      if (class=="variety") {
        print(paste(i, author, variety))
      } else {
        print(paste(i, author, gender))
      }
    }
  }
  
  return (bow)
}



# GENERATE VOCABULARY
#vocabulary <- GenerateVocabulary(path_training, n, swlang=lang)
vocabulary <- GenerateVocabulary(path_training, n)

deporte<-matrix(c("messi","ronaldo", "cristiano","madrid","barça","real","futbol","arbitro",
                  "gol","atletico","numancia","roja","expulsion",
                  "basket","canasta","partido","eurocopa","mundial","pique","champions",
                  "entrenador"))
frecuenciasDeporte<-matrix(rep(0,21))
deporte <-cbind(deporte,frecuenciasDeporte)
deporte<-as.data.frame(deporte)
names(deporte)<-c("WORD","FREQ")

vocabularyWithSports<-rbind(vocabulary,deporte)

# GENDER IDENTIFICATION DEPORTE
#######################
# GENERATING THE BOW FOR THE GENDER SUBTASK FOR THE TRAINING SET
bow_training_gender_Sport <- GenerateBoW(path_training, vocabularyWithSports, class="gender")

# PREPARING THE VECTOR SPACE MODEL FOR THE TRAINING SET
training_gender_Sport <- concat.split(bow_training_gender_Sport, "V1", ",")
training_gender_Sport <- cbind(training_gender_Sport[,2], training_gender_Sport[,4:ncol(training_gender_Sport)])
names(training_gender_Sport)[1] <- "theclass"

# Learning a SVM and evaluating it with k-fold cross-validation
train_control <- trainControl( method="repeatedcv", number = k , repeats = r)
model_SVM_gender_sport <- train( theclass~., data= training_gender_Sport, trControl = train_control, method = "svmLinear")
print(model_SVM_gender_sport)

# Learning a SVM with the whole training set and without evaluating it
#train_control <- trainControl(method="none")
#model_SVM_gender <- train( theclass~., data= training_gender, trControl = train_control, method = "svmLinear")

############DEPORTE
bow_test_gender_sport <- GenerateBoW(path_test, vocabularyWithSports, class="gender")

# Preparing the vector space model and truth for the test set
test_gender_sport <- concat.split(bow_test_gender_sport, "V1", ",")
truth_gender_sport <- unlist(test_gender_sport[,2])
test_gender_sport <- test_gender_sport[,4:ncol(test_gender_sport)]

# Predicting and evaluating the prediction
pred_SVM_gender_sport <- predict(model_SVM_gender_sport, test_gender_sport)
confusionMatrix(pred_SVM_gender_sport, truth_gender_sport)



# VARIETY IDENTIFICATION
########################
# GENERATING THE BOW FOR THE GENDER SUBTASK FOR THE TRAINING SET
bow_training_variety <- GenerateBoW(path_training, vocabulary, class="variety")

# PREPARING THE VECTOR SPACE MODEL FOR THE TRAINING SET
training_variety <- concat.split(bow_training_variety, "V1", ",")
training_variety <- cbind(training_variety[,2], training_variety[,4:ncol(training_variety)])
names(training_variety)[1] <- "theclass"

# Learning a SVM and evaluating it with k-fold cross-validation
train_control <- trainControl( method="repeatedcv", number = k , repeats = r)
model_SVM_variety <- train( theclass~., data= training_variety, trControl = train_control, method = "svmLinear")
print(model_SVM_variety)

# Learning a SVM with the whole training set and without evaluating it
#train_control <- trainControl(method="none")
#model_SVM_variety <- train( theclass~., data= training_variety, trControl = train_control, method = "svmLinear")

# GENERATING THE BOW FOR THE GENDER SUBTASK FOR THE TEST SET
bow_test_variety <- GenerateBoW(path_test, vocabulary, class="variety")

# Preparing the vector space model and truth for the test set
test_variety <- concat.split(bow_test_variety, "V1", ",")
truth_variety <- unlist(test_variety[,2])
test_variety <- test_variety[,4:ncol(test_variety)]

# Predicting and evaluating the prediction
pred_SVM_variety <- predict(model_SVM_variety, test_variety)
confusionMatrix(pred_SVM_variety, truth_variety)


# JOINT EVALUATION
##################
joint <- data.frame(pred_SVM_gender_sport, truth_gender_sport, pred_SVM_variety, truth_variety)
joint <- cbind(joint, ifelse(joint[,1]==joint[,2],1,0), ifelse(joint[,3]==joint[,4],1,0))
joint <- cbind(joint, joint[,5]*joint[,6])
colnames(joint) <- c("pgender", "tgender", "pvariety", "tvariety", "gender", "variety", "joint")

accgender <- sum(joint$gender)/nrow(joint)
accvariety <- sum(joint$variety)/nrow(joint)
accjoint <- sum(joint$joint)/nrow(joint)

end.time <- Sys.time()
time.taken <- end.time - start.time

print(paste(accgender, accvariety, accjoint, time.taken))



# N         GENDER  VARIETY JOINT   TIME
# 10        0.5875  0.2608  0.1442  3.62m
# 50        0.6850  0.3167  0.2142  4.32m      
# 100       0.7375  0.3383  0.2525  5.36m
# 500       0.7358  0.5717  0.4175  9.16m
# 1000      0.6983  0.6167  0.4325  12.11m      
# 5000      0.7550  0.7275  0.5517  51.81m     
# 10000     IMPOSSIBLE, RSTUDIO CRASHES