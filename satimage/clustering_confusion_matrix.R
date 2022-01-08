clustering_confusion_matrix = function(cluster, labels, k, n) {
    
    matrix = matrix(0, k, n)
    for(i in 1:length(cluster))
        matrix[cluster[i], labels[i]] = matrix[cluster[i], labels[i]] + 1
        
    return(matrix)
}