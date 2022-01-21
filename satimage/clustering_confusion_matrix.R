clustering_confusion_matrix = function(cluster, labels, k, n) {
    
    matrix = matrix(0, k, n)
    for(i in 1:length(cluster))
        matrix[cluster[i], labels[i]] = matrix[cluster[i], labels[i]] + 1
        
    return(matrix)
}

normalize_confusion_matrix = function(matrix) {
    result = matrix(0, nrow(matrix), ncol(matrix))
    for (i in 1:nrow(matrix)) 
        for (j in 1:ncol(matrix)) 
            result[i, j] = matrix[i, j] / sum(matrix[i, ])

    return (result)
}

clustering_accuracy = function(matrix) {
    sum = 0
    for (i in 1:ncol(matrix)) {
        sum = sum + max(matrix[, i])
    }
    
    return (sum / sum(matrix))
}

clustering_precision = function(matrix, class) {
    class_cluster = which.max(matrix[,class])
    
    tp = matrix[class_cluster, class]
    tp_fp = sum(matrix[class_cluster, ])
    
    return(tp / tp_fp)
}

clustering_recall = function(matrix, class) {
    class_cluster = which.max(matrix[,class])
    
    tp = matrix[class_cluster, class]
    tp_fn = sum(matrix[, class])
    
    return(tp / tp_fn)
}