library(ggplot2) # For some graphs
library(FactoMineR) # For PCA
library(factoextra) # For PCA
library(caret) # For feature plots
library(e1071) # For SVM

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

# Determine some statistical measures on the input attributes
dataset.mean = sapply(dataset.input, mean)
dataset.sd = sapply(dataset.input, sd)
dataset.var = sapply(dataset.input, var)
dataset.min = sapply(dataset.input, min)
dataset.max = sapply(dataset.input, max)
dataset.median = sapply(dataset.input, median)
dataset.range = sapply(dataset.input, range)
dataset.quantile = sapply(dataset.input, quantile)

# Determine class frequencies
class.frequencies = data.frame(table(dataset.output))
colnames(class.frequencies) = c("class", "frequencies")
ggplot(data = class.frequencies, aes(x = class, y = frequencies, fill = class)) + geom_bar(stat = "identity")
ggplot(data = class.frequencies, aes(x = "", y = frequencies, fill = class)) + geom_bar(stat = "identity", width = 1, color = "white") + coord_polar("y", start = 0)

# Execute the PCA. Extract only the first four principal components
pca.results <- PCA(dataset.input, scale.unit = TRUE, ncp = 4, graph = FALSE)

# Get the eigenvalues and plot them. The first four components explain about 92% of variance
pca.results.eig.val <- get_eigenvalue(pca.results)
fviz_eig(pca.results, addlabels = TRUE, ylim = c(0, 50))

# Show the correlation between the variables using the first two principal components
# Many variable are positively correlated
fviz_pca_var(pca.results, col.var = "black")

# Extract the principal components
dataset.input.pca = data.frame(get_pca_ind(pca.results)$coord)
dataset.pca = cbind(class=dataset$class, dataset.input.pca)

# Draw some feature plot to highlight the level of difficulty to recognise the various classes
featurePlot(x=dataset.input.pca, y=dataset.output, plot="density", scales=list(x=list(relation="free"), y=list(relation="free")), auto.key=list(columns=3))
featurePlot(x=dataset.input.pca, y=dataset.output, plot="pairs", auto.key=list(columns=3))

# Divide the dataset into trainset and testset
indexes = sample(2, nrow(dataset.pca), replace = TRUE, prob = c(0.7, 0.3))
temp.trainset = dataset.pca[indexes == 1,]
testset = dataset.pca[indexes == 2,]

# Divide the trainset to obtain a validationset
indexes = sample(2, nrow(temp.trainset), replace = TRUE, prob = c(0.9, 0.1))
trainset = temp.trainset[indexes == 1,]
validationset = temp.trainset[indexes == 2,]

# Train a svm model
svm.model = svm(class ~ Dim.1 + Dim.2 + Dim.3 + Dim.4, data = trainset, kernel = 'radial', cost = 1)

# Predict the labels and create the confusion matrix
svm.prediction = predict(svm.model, testset)
svm.table = table(svm.prediction, testset$class)
svm.accuracy = sum(diag(svm.table)) / sum(svm.table)