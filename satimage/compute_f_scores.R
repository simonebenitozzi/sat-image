compute_f_scores = function(labels, confusion_matrix, weights){
  fscores = list()
  fscores.weighted = list()
  for (cls in labels){
    TP = confusion_matrix[cls, cls]
    precision = TP/sum(confusion_matrix[cls,])
    recall = TP/sum(confusion_matrix[,cls])
    f_score = (2 * precision * recall) / (precision+recall)
    f_score.weighted = f_score * weights[cls]
    fscores[[cls]] = f_score
    fscores.weighted[[cls]] = f_score.weighted
  }
  ret_val = list("fscores" = fscores, "fscores.weighted" = fscores.weighted)
  return(ret_val)
}