## ----echo=F--------------------------------------------------------------------------------------------------
# Install all the required packages
install.packages('ggplot2', dependencies=TRUE)
install.packages('FactoMineR', dependencies=TRUE)
install.packages('factoextra', dependencies=TRUE)
install.packages('caret', dependencies=TRUE)
install.packages('e1071', dependencies=TRUE)
install.packages('nnet', dependencies=TRUE)
install.packages('pROC', dependencies=TRUE)
install.packages('multiROC', dependencies=TRUE)
install.packages('cluster', dependencies=TRUE)
install.packages('clustertend', dependencies=TRUE)
install.packages('fpc', dependencies=TRUE)
install.packages('seriation', dependencies=TRUE)
install.packages('ggpubr', dependencies=TRUE)
install.packages('TSclust', dependencies=TRUE)
install.packages('knitr', dependencies=TRUE)

# Load all the required packages
library(ggplot2) # For some graphs

library(FactoMineR) # For PCA
library(factoextra) # For PCA
library(caret) # For feature plots
library(e1071) # For SVM
library(nnet) # For Neural nets
library(pROC) # For ROC
library(multiROC) # For multiclass ROC

library(cluster) # For k-means
library(clustertend) # For Hopkins Statistic
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
ggplot(data = class.frequencies, aes(x = class, y = frequencies, fill = class)) + geom_bar(stat = "identity") + geom_text(aes(label = percentages), vjust = 1.5, colour = "white") + theme(axis.text.x=element_blank())
ggplot(data = class.frequencies, aes(x = "", y = frequencies, fill = class)) + geom_bar(stat = "identity", width = 1, color = "white") + coord_polar("y", start = 0)


## ------------------------------------------------------------------------------------------------------------
featurePlot(dataset.input[, c("p5.Red", "p5.Green", "p5.NIR1")], dataset.output, plot="pairs", scales=list((relation="free"), y=list(relation="free")), auto.key=list(columns=3), main="Feature plot for p5.Red, p5.Green, p5.NIR1 pairs")
featurePlot(dataset.input[, c("p5.Red", "p6.Red", "p4.Red")], dataset.output, plot="pairs", scales=list((relation="free"), y=list(relation="free")), auto.key=list(columns=3), main="Feature plot for p5.Red, p6.Red, p4.Red pairs")
featurePlot(dataset.input[, c("p5.Green", "p6.Green", "p4.Green")], dataset.output, plot="pairs", scales=list((relation="free"), y=list(relation="free")), auto.key=list(columns=3), main="Feature plot for p5.Green, p6.Green, p4.Green pairs")
featurePlot(dataset.input[, c("p5.Green", "p2.Green", "p8.Green")], dataset.output, plot="pairs", scales=list((relation="free"), y=list(relation="free")), auto.key=list(columns=3), main="Feature plot for p5.Green, p2.Green, p8.Green pairs")
featurePlot(dataset.input[, c("p5.Red", "p2.Red", "p8.Red")], dataset.output, plot="pairs", scales=list((relation="free"), y=list(relation="free")), auto.key=list(columns=3), main="Feature plot for p5.Red, p2.Red, p8.Red pairs")
featurePlot(dataset.input[, c("p5.NIR1", "p2.NIR1", "p8.NIR1")], dataset.output, plot="pairs", scales=list((relation="free"), y=list(relation="free")), auto.key=list(columns=3), main="Feature plot for p5.NIR1, p2.NIR1, p8.NIR1 pairs")


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
featurePlot(x=dataset.input.pca, y=dataset.output, plot="density", scales=list(x=list(relation="free"), y=list(relation="free")), auto.key=list(columns=3), main="Density Feature Plot")
featurePlot(x=dataset.input.pca, y=dataset.output, plot="pairs", auto.key=list(columns=3), main="Pairs Feature Plot")


## ------------------------------------------------------------------------------------------------------------

# Divide the dataset into trainset and testset
indexes = createDataPartition(dataset.pca$class, p = .7, list = F, times = 1)
temp.trainset = dataset.pca[indexes,]
testset = dataset.pca[- indexes,]
testset.x = testset[, !names(testset) %in% c("class")]
testset.y = testset[, c("class")]

# Divide the trainset to obtain a validationset
indexes = createDataPartition(temp.trainset$class, p = .9, list = F, times = 1)
trainset = temp.trainset[indexes,]
validationset = temp.trainset[-indexes,]
validationset.x = validationset[, !names(validationset) %in% c("class")]
validationset.y =  validationset[, c("class")]


## ------------------------------------------------------------------------------------------------------------
# Train a SVM model and tune the hyperparameters (cost and gamma)
svm.start_time = Sys.time()

svm.tuning.control = tune.control(sampling = "fix") # To use validationset instead
svm.tuning = tune.svm(class ~ Dim.1 + Dim.2 + Dim.3 + Dim.4, data = trainset, kernel = "radial", gamma = 10^(-3:1), cost = 10^(-2:1), validation.x = validationset.x, validation.y = validationset.y, tunecontrol = svm.tuning.control, probability = TRUE)

svm.end_time = Sys.time()
sprintf("Time required for svm training:")
svm.end_time - svm.start_time



## ------------------------------------------------------------------------------------------------------------
plot(svm.tuning)


## ------------------------------------------------------------------------------------------------------------
# Determine some metrics for each class
svm.tuned = svm.tuning$best.model
svm.tuned.prediction = predict(svm.tuned, testset.x)
svm.tuned.result = confusionMatrix(svm.tuned.prediction, testset.y, mode = "prec_recall")

sprintf("Tuned SVM micro-average global accuracy is %2.1f%%", svm.tuned.result$overall["Accuracy"] * 100)
print(svm.tuned.result$table)


## ------------------------------------------------------------------------------------------------------------
# Train a NNET model and tune the hyperparameters (size and decay)
nnet.start_time = Sys.time()

nnet.tuning.control = tune.control(sampling = 'fix')
nnet.tuning = tune.nnet(class ~ Dim.1 + Dim.2 + Dim.3 + Dim.4, data = trainset, size = c(2:4), validation.x = validationset.x, validation.y = validationset.y, decay = c(0, 2^-2:1), probability = TRUE)

nnet.end_time = Sys.time()
sprintf("Time required for svm training:")
nnet.end_time - nnet.start_time

## ------------------------------------------------------------------------------------------------------------
plot(nnet.tuning)


## ------------------------------------------------------------------------------------------------------------
# Determine some metrics for each class
nnet.tuned = nnet.tuning$best.model
nnet.tuned.prediction = factor(predict(nnet.tuned, testset.x, type = "class"), levels = class.names)
nnet.tuned.result = confusionMatrix(nnet.tuned.prediction, testset.y, mode = "prec_recall")

sprintf("Tuned NNET micro-average global accuracy is %2.1f%%", nnet.tuned.result$overall["Accuracy"] * 100)
print(nnet.tuned.result$table)


## ------------------------------------------------------------------------------------------------------------
metrics_of_interest = c("Precision", "Specificity", "Recall", "F1")


## ------------------------------------------------------------------------------------------------------------
svm.macro.fscore = Reduce('+', svm.tuned.result$byClass[,"F1"]) / length(class.names)
svm.micro.fscore = Reduce('+', svm.tuned.result$byClass[,"F1"] * svm.tuned.result$byClass[,"Prevalence"])

sprintf("SVM macro F-Score: %1.3f", svm.macro.fscore)
sprintf("SVM micro F-Score: %1.3f", svm.micro.fscore)

print(svm.tuned.result$byClass[, metrics_of_interest])


## ------------------------------------------------------------------------------------------------------------
# Determine some metrics for each class
nnet.macro.fscore = Reduce('+', nnet.tuned.result$byClass[,"F1"]) / length(class.names)
nnet.micro.fscore = Reduce('+', nnet.tuned.result$byClass[,"F1"] * nnet.tuned.result$byClass[,"Prevalence"])
sprintf("NNET macro F-Score: %1.3f", nnet.macro.fscore)
sprintf("NNET micro F-Score: %1.3f", nnet.micro.fscore)
sprintf("Tuned NNET metrics by class:")
nnet.tuned.result$byClass[, metrics_of_interest]


## ------------------------------------------------------------------------------------------------------------
metrics.comparison = svm.tuned.result$byClass[, metrics_of_interest] - nnet.tuned.result$byClass[, metrics_of_interest]
print("Metrics compared, positive values favor SVM")
print(metrics.comparison)



## ------------------------------------------------------------------------------------------------------------
# Predicts the most probable class class for each instance
svm.tuned.predictions = predict(svm.tuned, testset.x, probability = TRUE)

# Extracts, for each class, the probability of every instance of being of that class 
svm.tuned.predictions.probs = attr(svm.tuned.predictions, "probabilities")

# Extracts, for each class, the probability of every instance of being of that class
nnet.tuned.predictions = predict(nnet.tuned, testset.x, probability = TRUE)

# Creates the dataframe to be used for multiroc
testset.multiroc = data.frame(matrix(ncol=18, nrow=0))

for (i in 1:length(testset.y)) {
    row = matrix(0, 1, 6)
    # sets 1 to the correct class 
    row[testset.y[i]] = 1
    for (j in 1:6)
        # concatenates the probabilities of the svm
        row = c(row, svm.tuned.predictions.probs[i, class.names[j]])
    # concatenates the probabilities of the neural net
    row = c(row, nnet.tuned.predictions[i,])
    testset.multiroc = rbind(testset.multiroc, row)
}

# Formats the dataframe to be suitable for the multi_roc function
colnames(testset.multiroc) = c('red_soil_true', 'cotton_crop_true', 'grey_soil_true', 'damp_grey_soil_true', 'vegetation_true', 'very_damp grey_soil_true', 'red_soil_pred_SVM', 'cotton_crop_pred_SVM', 'grey_soil_pred_SVM', 'damp_grey_soil_pred_SVM', 'vegetation_pred_SVM', 'very_damp grey_soil_pred_SVM', 'red_soil_pred_NN', 'cotton_crop_pred_NN', 'grey_soil_pred_NN', 'damp_grey_soil_pred_NN', 'vegetation_pred_NN', 'very_damp grey_soil_pred_NN')

svm.nnet.multiroc = multi_roc(testset.multiroc, force_diag=T)
print("Per class AUC values:")
print(unlist(svm.nnet.multiroc$AUC))


## ------------------------------------------------------------------------------------------------------------
# MultiROC plotting
n_method <- length(unique(svm.nnet.multiroc$Methods))
n_group <- length(unique(svm.nnet.multiroc$Groups))

# changes the format of results to a ggplot2 friendly format
for (i in 1:n_group) {
      res_df <- data.frame(Specificity= numeric(0), Sensitivity= numeric(0), Group = character(0), AUC = numeric(0), Method = character(0))
      for (j in 1:n_method) {
        temp_data_1 <- data.frame(Specificity=svm.nnet.multiroc$Specificity[[j]][i],
                                  Sensitivity=svm.nnet.multiroc$Sensitivity[[j]][i],
                                  Group=tools::toTitleCase(gsub("_", " ", unique(svm.nnet.multiroc$Groups)[i])),
                                  AUC=svm.nnet.multiroc$AUC[[j]][i],
                                  Method = unique(svm.nnet.multiroc$Methods)[j])
        colnames(temp_data_1) <- c("Specificity", "Sensitivity", "Group", "AUC", "Method")
        res_df <- rbind(res_df, temp_data_1)

      }
      
      plot(ggplot2::ggplot(res_df, ggplot2::aes(x = 1-Specificity, y=Sensitivity)) + ggplot2::geom_path(ggplot2::aes(color = Group, linetype=Method)) + ggplot2::geom_segment(ggplot2::aes(x = 0, y = 0, xend = 1, yend = 1), colour='grey', linetype = 'dotdash') + ggplot2::theme_bw() + ggplot2::theme(plot.title = ggplot2::element_text(hjust = 0.5), legend.justification=c(1, 0), legend.position=c(.95, .05), legend.title=ggplot2::element_blank(), legend.background = ggplot2::element_rect(fill=NULL, size=0.5, linetype="solid", colour ="black")))
    
}



## ------------------------------------------------------------------------------------------------------------
source("clustering_metrics.R")
dataset.clustering = dataset.input.pca
labels = dataset.pca[1]

# dataframe to vector conversion
labels.list <- c()
labels.list = match(as.factor(labels$class), class.names)


## ------------------------------------------------------------------------------------------------------------
# Calculate Hopkins statistic for CLustering tendency
set.seed(1)
hopkins(dataset.clustering, n = nrow(dataset.clustering) - 1)



## ------------------------------------------------------------------------------------------------------------
# Since the entire dataset is too big to be allocated for VAT result, only a portion of it has been given in input
indexes = createDataPartition(dataset.pca$class, p = .2, list = F, times = 1)
partition = dataset.pca[indexes,]
partition.vat = partition[, !names(testset) %in% c("class")]

# VAT - Visual Assessment of cluster Tendency
fviz_dist(dist(partition.vat), show_labels = FALSE)



## ------------------------------------------------------------------------------------------------------------
# Evaluating seed e nstart parameters
# Best nstart value is 5, which guarantees stability for the algorithm, with a low influence of the initial seed on the clustering result of the dataset

# Following it's shown how different values of seed, chosen randomly, do not influence silhouette values
Z <- array(sample(1:100, 20, replace=T), c(20, 1))
silhouettes <- c()
for(i in 1:20) {
    set.seed(Z[i])
    fit = kmeans(dataset.clustering, 3, nstart=5)
    ss = silhouette(fit$cluster, dist(dataset.clustering))
    silhouettes = append(silhouettes, mean(ss[, 3]))
    # print(sprintf("Seed %2d: %f", Z[i], mean(ss[, 3])))
}
plot(Z, silhouettes, type="b", main = "Silhouette plot with different random seeds", xlab="seeds")
set.seed(1)


## ------------------------------------------------------------------------------------------------------------
set.seed(1)
nk = 2:12

# calculates average silhouette values for an ascending value of k
SW = sapply(nk, function(k){cluster.stats(dist(dataset.clustering), 
                              kmeans(dataset.clustering, centers=k, nstart=5)$cluster)$avg.silwidth})
#SW
plot(nk, SW, type="l", xlab="number of clusters", ylab="average silhouette")


## ------------------------------------------------------------------------------------------------------------
similarity = c()

# Calculates the similarity between clusters and targets for an ascending value of k
for (k in 2:6) {
    fit = kmeans(dataset.clustering, k, nstart = 5)
    eval = cluster.evaluation(labels.list, fit$cluster)
    similarity = append(similarity, eval)
}
plot(2:6, similarity, type="b", main = "Similarity plot for ascending K", xlab="K")


## ------------------------------------------------------------------------------------------------------------
fit = kmeans(dataset.clustering, 6, nstart=5)

# calculates silhouette values for k=6
kms = silhouette(fit$cluster, dist(dataset.clustering))
plot(kms, col=1:6, border=NA, main = "Silhouette Plot for k=6")



## ------------------------------------------------------------------------------------------------------------
# Dissimilarity Matrix
dissplot(dist(dataset.clustering), labels=fit$cluster, options=list(main="Kmeans clustering with k=6"))


## ------------------------------------------------------------------------------------------------------------
# Evaluation
sprintf("Similarity with (k=6): %f", cluster.evaluation(labels.list, fit$cluster))


## ------------------------------------------------------------------------------------------------------------
# visualization
fviz_cluster(fit, dataset.clustering, ellipse.type = "norm", axes=c(1,2), main="Dimensions: 1 & 2", 
             geom="point")
fviz_cluster(fit, dataset.clustering, ellipse.type = "norm", axes=c(1,3), main="Dimensions: 1 & 3", 
             geom="point")
fviz_cluster(fit, dataset.clustering, ellipse.type = "norm", axes=c(1,4), main="Dimensions: 1 & 4", 
             geom="point")

fviz_cluster(fit, dataset.clustering, ellipse.type = "norm", axes=c(2,3), main="Dimensions: 2 & 3", 
             geom="point")
fviz_cluster(fit, dataset.clustering, ellipse.type = "norm", axes=c(2,4), main="Dimensions: 2 & 4", 
             geom="point")

fviz_cluster(fit, dataset.clustering, ellipse.type = "norm", axes=c(3,4), main="Dimensions: 3 & 4", 
             geom="point")


## ------------------------------------------------------------------------------------------------------------
# Confusion Matrix

confusion_matrix = clustering_confusion_matrix(fit$cluster, labels.list, 6, 6)
colnames(confusion_matrix) = class.names
rownames(confusion_matrix) = 1:6
confusion_matrix


## ------------------------------------------------------------------------------------------------------------
for (i in 1:6) {
    pie_title = sprintf("Cluster %d Distribution", i)
    pie(confusion_matrix[i,], main=pie_title, labels=class.names, col=rainbow(6))
}


## ------------------------------------------------------------------------------------------------------------
#
for (i in 1:6) {
    pie_title = sprintf("Class %s Distribution", class.names[i])
    pie(confusion_matrix[,i], main=pie_title, labels=1:6, col=rainbow(6))
}

## ------------------------------------------------------------------------------------------------------------
source("clustering_metrics.R")

precisions = c()
recalls = c()
fmeasures = c()

for (i in 1:6) {
    precision = clustering_precision(confusion_matrix, i)
    recall = clustering_recall(confusion_matrix, i)
    fmeasure = (2*precision*recall) / (precision+recall)
    
    precisions = rbind(precisions, precision)
    recalls = rbind(recalls, recall)
    fmeasures = rbind(fmeasures, fmeasure)
}

km_metrics = cbind(precisions, recalls, fmeasures)

rownames(km_metrics) = class.names
colnames(km_metrics) = c('precision','recall','fmeasure')

accuracy = clustering_accuracy(confusion_matrix)
sprintf("K-means (K=6) micro-average global accuracy is %2.1f%%", accuracy * 100)

sprintf("Per class metrics:")
km_metrics

