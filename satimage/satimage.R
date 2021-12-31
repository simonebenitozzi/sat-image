## ------------------------------------------------------------------------------------------------------------
library(ggplot2); # For some graphs
library(FactoMineR); # For PCA
library(factoextra); # For PCA
library(caret); # For feature plots
library(e1071); # For SVM
library(nnet); # For Neural nets
library(pROC);

library(cluster) # For k-means
library(fpc) # For k-means
library(seriation) # For k-means
library(ggpubr) # For clustering visualization
library(TSclust) # For clustering Evaluation


## ------------------------------------------------------------------------------------------------------------
# Create a vector containing all attribute names
attribute.names = c()
for(pixel.number in 1:9) {
  for(attribute.suffix in c("Red", "Green", "NIR1", "NIR2")) {
    attribute.name = paste("p", pixel.number, ".", attribute.suffix, sep="")
    attribute.names = c(attribute.names, attribute.name)
  }
}
attribute.names = c(attribute.names, "class")

# Create a vector containing all class names
class.names = c("red soil", "cotton crop", "grey soil", "damp grey soil", "soil with vegetation stubble", "very damp grey soil")

# Load the dataset from the csv file and cast the label to factor
dataset = data.frame(read.csv("dataset.csv", col.names = attribute.names, sep=" "))
dataset$class = factor(dataset$class, labels = class.names)

# Determine the input and the output space
dataset.input = dataset[, 1:ncol(dataset)-1]
dataset.output = dataset$class


## ------------------------------------------------------------------------------------------------------------
# Determine some statistical measures on the input attributes
dataset.mean = sapply(dataset.input, mean)
dataset.sd = sapply(dataset.input, sd)
dataset.var = sapply(dataset.input, var)
dataset.min = sapply(dataset.input, min)
dataset.max = sapply(dataset.input, max)
dataset.median = sapply(dataset.input, median)
dataset.range = sapply(dataset.input, range)
dataset.quantile = sapply(dataset.input, quantile)



## ------------------------------------------------------------------------------------------------------------
# Determine class frequencies
class.frequencies = data.frame(table(dataset.output))
colnames(class.frequencies) = c("class", "frequencies")
class.frequencies$percentages = paste(format((class.frequencies$frequencies / sum(class.frequencies$frequencies)) * 100, digits = 2), "%", sep = "")
ggplot(data = class.frequencies, aes(x = class, y = frequencies, fill = class)) + geom_bar(stat = "identity") + geom_text(aes(label = percentages), vjust = 1.5, colour = "white")
ggplot(data = class.frequencies, aes(x = "", y = frequencies, fill = class)) + geom_bar(stat = "identity", width = 1, color = "white") + coord_polar("y", start = 0)


## ------------------------------------------------------------------------------------------------------------
featurePlot(dataset.input[, c("p5.Red", "p5.Green", "p5.NIR1")], dataset.output, plot="pairs", scales=list((relation="free"), y=list(relation="free")), auto.key=list(columns=3))
featurePlot(dataset.input[, c("p5.Red", "p6.Red", "p4.Red")], dataset.output, plot="pairs", scales=list((relation="free"), y=list(relation="free")), auto.key=list(columns=3))
featurePlot(dataset.input[, c("p5.Green", "p6.Green", "p4.Green")], dataset.output, plot="pairs", scales=list((relation="free"), y=list(relation="free")), auto.key=list(columns=3))
featurePlot(dataset.input[, c("p5.Green", "p2.Green", "p8.Green")], dataset.output, plot="pairs", scales=list((relation="free"), y=list(relation="free")), auto.key=list(columns=3))
featurePlot(dataset.input[, c("p5.Red", "p2.Red", "p8.Red")], dataset.output, plot="pairs", scales=list((relation="free"), y=list(relation="free")), auto.key=list(columns=3))
featurePlot(dataset.input[, c("p5.NIR1", "p2.NIR1", "p8.NIR1")], dataset.output, plot="pairs", scales=list((relation="free"), y=list(relation="free")), auto.key=list(columns=3))


## ------------------------------------------------------------------------------------------------------------
# Execute the PCA. Extract only the first four principal components
pca.results <- PCA(dataset.input, scale.unit = TRUE, ncp = 4, graph = FALSE)

# Get the eigenvalues and plot them. The first four components explain about 92% of variance
pca.results.eig.val <- get_eigenvalue(pca.results)
fviz_eig(pca.results, addlabels = TRUE, ylim = c(0, 50))

# Show the correlation between the variables using the first two principal components
# Many variable are positively correlated
fviz_pca_var(pca.results, col.var = "black")



## ------------------------------------------------------------------------------------------------------------
# Extract the principal components
dataset.input.pca = data.frame(get_pca_ind(pca.results)$coord)
dataset.pca = cbind(class=dataset$class, dataset.input.pca)

# Draw some feature plot to highlight the level of difficulty to recognise the various classes
featurePlot(x=dataset.input.pca, y=dataset.output, plot="density", scales=list(x=list(relation="free"), y=list(relation="free")), auto.key=list(columns=3))
featurePlot(x=dataset.input.pca, y=dataset.output, plot="pairs", auto.key=list(columns=3))


## ------------------------------------------------------------------------------------------------------------
# Divide the dataset into trainset and testset
indexes = sample(2, nrow(dataset.pca), replace = TRUE, prob = c(0.7, 0.3))
temp.trainset = dataset.pca[indexes == 1,]
testset = dataset.pca[indexes == 2,]
testset.x = testset[, !names(testset) %in% c("class")]
testset.y = testset[,c("class")]



## ------------------------------------------------------------------------------------------------------------
# Divide the trainset to obtain a validationset
indexes = sample(2, nrow(temp.trainset), replace = TRUE, prob = c(0.9, 0.1))
trainset = temp.trainset[indexes == 1,]
validationset = temp.trainset[indexes == 2,]


## ------------------------------------------------------------------------------------------------------------
# Train a svm model
#svm.model = svm(class ~ Dim.1 + Dim.2 + Dim.3 + Dim.4, data = trainset, kernel = 'radial', cost = 1)

# Predict the labels and create the confusion matrix
#svm.prediction = predict(svm.model, testset)
#svm.table = table(svm.prediction, testset$class)
#svm.accuracy = sum(diag(svm.table)) / sum(svm.table)



## ------------------------------------------------------------------------------------------------------------
svm.tuning.control = tune.control(sampling = "fix")
svm.tuning = tune.svm(class ~ ., data = trainset, kernel = "radial", gamma = 10^(-3:0), cost = 2^(-2:2), validation.x = validationset[,!names(validationset) %in% c("class")], validation.y = validationset[,c("class")], tunecontrol = svm.tuning.control, probability = T)
plot(svm.tuning)


## ------------------------------------------------------------------------------------------------------------
svm.tuned = svm.tuning$best.model
svm.tuned.testset.prediction = predict(svm.tuned, testset.x)
svm.tuned.testset.confusion_matrix = table(svm.tuned.testset.prediction, testset$class)
svm.tuned.accuracy = sum(diag(svm.tuned.testset.confusion_matrix)) / sum(svm.tuned.testset.confusion_matrix)
sprintf("Tuned SVM accuracy is %2.1f%%", (1 - svm.tuning$best.performance) * 100)


## ------------------------------------------------------------------------------------------------------------
nnet.tuning.control = tune.control(sampling = 'fix', performances = T)
nnet.tuning = tune.nnet(class ~., data = trainset, size = c(2:4), validation.x = validationset[,!names(validationset) %in% c("class")], validation.y = validationset[,c("class")], decay = c(0, 2^-2:1))
nnet.tuned = nnet.tuning$best.model
plot(nnet.tuning)


## ------------------------------------------------------------------------------------------------------------
acc = 1 - nnet.tuning$best.performance
sprintf("Tuned neuralnet overall accuracy is %2.1f%%", acc * 100)


## ------------------------------------------------------------------------------------------------------------
predictions = predict(nnet.tuning$best.model, testset.x)
predictions.classes = class.names[max.col(predictions)]
errors = predictions.classes != testset.y
acc = length(which(errors))/length(errors)
nnet.tuned.testset.predictions = predictions.classes
nnet.tuned.testset.confusion_matrix = table(nnet.tuned.testset.predictions, testset.y)


## ------------------------------------------------------------------------------------------------------------
svm.tuned.testset.predictions = predict(svm.tuned, testset.x, probability = T)
svm.tuned.testset.predictions.probs = attr(svm.tuned.testset.predictions, "probabilities")
svm.roc = multiclass.roc(testset.y, svm.tuned.testset.predictions.probs)


## ------------------------------------------------------------------------------------------------------------
nnet.tuned.testset.predictions = predict(nnet.tuning$best.model, testset.x, probability = T)
nnet.roc = multiclass.roc(testset.y, nnet.tuned.testset.predictions)


## ------------------------------------------------------------------------------------------------------------
sprintf("SVM ROC AUC: %f", svm.roc$auc)
sprintf("NNET ROC AUC: %f", nnet.roc$auc)


## ------------------------------------------------------------------------------------------------------------
source("compute_f_scores.R")

weights = list()
for (cls in class.names){
  count = class.frequencies[class.frequencies$class == cls,"frequencies"]
  weights[[cls]] = count
}
total_instances_count = Reduce('+', weights)
weights = mapply('/', weights, total_instances_count)
fscores = compute_f_scores(class.names, svm.tuned.testset.confusion_matrix, weights)

svm.macro.fscore = Reduce('+', fscores$fscores) / length(class.names)
svm.micro.fscore = Reduce('+', fscores$fscores.weighted)

sprintf("SVM macro F-Score: %1.3f", svm.macro.fscore)
sprintf("SVM micro F-Score: %1.3f", svm.micro.fscore)


## ------------------------------------------------------------------------------------------------------------
fscores = compute_f_scores(class.names, nnet.tuned.testset.confusion_matrix, weights)
nnet.macro.fscore = Reduce('+', fscores$fscores) / length(class.names)
nnet.micro.fscore = Reduce('+', fscores$fscores.weighted)
sprintf("NNet macro F-Score: %1.3f", nnet.macro.fscore)
sprintf("Net micro F-Score: %1.3f", nnet.micro.fscore)



## ------------------------------------------------------------------------------------------------------------
trainset.clustering = dataset.input.pca
labels = dataset.pca[1]

# dataframe to vector conversion
labels.list <- c()
for(i in 1:nrow(labels)){
  labels.list <- append(labels.list, labels[i,1])
}

set.seed(1)
nk = 2:12
SW = sapply(nk, function(k){cluster.stats(dist(trainset.clustering), 
                              kmeans(trainset.clustering, centers=k, nstart=5)$cluster)$avg.silwidth})
#SW
plot(nk, SW, type="l", xlab="number of clusters", ylab="average silhouette")


## ------------------------------------------------------------------------------------------------------------
# Scelta dei parametri di seed e nstart

# Il valore ottimale per nstart e' stato individuato in 5, che garantisce un'ottima misura di stabilita' dell'algorimto, con una bassa influenza del seed iniziale rispetto alla suddivisione in clusterr del dataset

# Di seguito e' mostrato come i diversi valori di seed (scelti in maniera random), non influenzino la misura di silhouette
Z <- array(sample(1:100, 20, replace=T), c(20, 1))
silhouettes <- c()
for(i in 1:20) {
    set.seed(Z[i])
    fit = kmeans(trainset.clustering, 3, nstart=5)
    ss = silhouette(fit$cluster, dist(trainset.clustering))
    silhouettes = append(silhouettes, mean(ss[, 3]))
    # print(sprintf("Seed %2d: %f", Z[i], mean(ss[, 3])))
}
plot(Z, silhouettes, type="b", main = "Silhouette plot with different random seeds", xlab="seeds")
set.seed(1)



## ------------------------------------------------------------------------------------------------------------
fit = kmeans(trainset.clustering, 6, nstart=5)

#Silhouette
kms = silhouette(fit$cluster, dist(trainset.clustering))
plot(kms, col=1:6, border=NA, main = "Silhouette Plot for k=6")

# Dissimilarity Matrix
dissplot(dist(trainset.clustering), labels=fit$cluster, options=list(main="Kmeans clustering with k=6"))

# Evaluation
sprintf("Similarity with (k=6): %f", cluster.evaluation(labels.list, fit$cluster))


## ------------------------------------------------------------------------------------------------------------
# visualization
fviz_cluster(fit, trainset.clustering, ellipse.type = "norm", axes=c(1,2), main="Dimensions: 1 & 2", 
             geom="point")
fviz_cluster(fit, trainset.clustering, ellipse.type = "norm", axes=c(1,3), main="Dimensions: 1 & 3", 
             geom="point")
fviz_cluster(fit, trainset.clustering, ellipse.type = "norm", axes=c(1,4), main="Dimensions: 1 & 4", 
             geom="point")

fviz_cluster(fit, trainset.clustering, ellipse.type = "norm", axes=c(2,3), main="Dimensions: 2 & 3", 
             geom="point")
fviz_cluster(fit, trainset.clustering, ellipse.type = "norm", axes=c(2,4), main="Dimensions: 2 & 4", 
             geom="point")

fviz_cluster(fit, trainset.clustering, ellipse.type = "norm", axes=c(3,4), main="Dimensions: 3 & 4", 
             geom="point")


## ------------------------------------------------------------------------------------------------------------
fit = kmeans(trainset.clustering, 3, nstart=5)

kms = silhouette(fit$cluster, dist(trainset.clustering))
plot(kms, col=1:3, border=NA, main = "Silhouette Plot for k=3")

# Dissimilarity Matrix
dissplot(dist(trainset.clustering), labels=fit$cluster, options=list(main="Kmeans clustering with k=3"))

# Evaluation
sprintf("Similarity with (k=3): %f", cluster.evaluation(labels.list, fit$cluster))


## ------------------------------------------------------------------------------------------------------------
# visualization
fviz_cluster(fit, trainset.clustering, ellipse.type = "norm", axes=c(1,2), main="Dimensions: 1 & 2", 
             geom="point")
fviz_cluster(fit, trainset.clustering, ellipse.type = "norm", axes=c(1,3), main="Dimensions: 1 & 3", 
             geom="point")
fviz_cluster(fit, trainset.clustering, ellipse.type = "norm", axes=c(1,4), main="Dimensions: 1 & 4", 
             geom="point")

fviz_cluster(fit, trainset.clustering, ellipse.type = "norm", axes=c(2,3), main="Dimensions: 2 & 3", 
             geom="point")
fviz_cluster(fit, trainset.clustering, ellipse.type = "norm", axes=c(2,4), main="Dimensions: 2 & 4", 
             geom="point")

fviz_cluster(fit, trainset.clustering, ellipse.type = "norm", axes=c(3,4), main="Dimensions: 3 & 4", 
             geom="point")


## ------------------------------------------------------------------------------------------------------------
similarity = c()
for (k in 2:6) {
    fit = kmeans(trainset.clustering, k, nstart = 5)
    eval = cluster.evaluation(labels.list, fit$cluster)
    similarity = append(similarity, eval)
}
plot(2:6, similarity, type="b", main = "Similarity plot for ascending K", xlab="K")


## ------------------------------------------------------------------------------------------------------------


