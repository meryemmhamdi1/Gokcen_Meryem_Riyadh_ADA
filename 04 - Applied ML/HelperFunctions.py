import matplotlib.pyplot as plt
import numpy as np
from sklearn.cross_validation import cross_val_score
from sklearn.model_selection import cross_val_predict, learning_curve


##### Functıons used ın the fırst part #####



# Create Categories for Binary Classification
def binaryColor(x):
    if x < 0.5: 
        return 0
    else: 
        return 1
    
# Create Categories for Multi Classification
def multiColor(x):
    if x < 0.25: 
        return 0
    elif 0.25 <= x <0.5: 
        return 1
    elif 0.5 <= x < 0.75:
        return 2
    elif 0.75 <= x <= 1:
        return 3
    
def weight_sample(labels):
    weight_class = labels.value_counts()/len(labels)
    
    sample_weights = []
    for i in labels:
        sample_weights += [weight_class[i]]
        
    return np.array(sample_weights)

def plot_confusion_matrix(cm, classes, normalize=False, title='Confusion matrix', cmap=plt.cm.Blues):
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label') 
    

def plot_learning_curve(estimator,X,Y,cv=20):
    plt.figure()
    plt.xlabel("the sıze of the data")
    plt.ylabel("Score")
    train_sizes, train_scores, test_scores = learning_curve(estimator, X, Y, cv=cv, train_sizes=np.linspace(0.2,1,20))
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)
    plt.grid()

    plt.fill_between(train_sizes, train_scores_mean - train_scores_std, train_scores_mean + train_scores_std, alpha=0.1, color="b")
    plt.fill_between(train_sizes, test_scores_mean - test_scores_std, test_scores_mean + test_scores_std, alpha=0.1, color="r")
    plt.plot(train_sizes, train_scores_mean, 'o-', color="b", label="Training score")
    plt.plot(train_sizes, test_scores_mean, 'o-', color="r", label="Cross-validation score")
    plt.legend(loc="best")
    plt.show()




##### Functıons used ın the second part #####

def get_kmeans_result(dataframe, combination):
    # get_dummies generates neccessary columns to convert Categorical features to numerical.
    _df_aggregated_with_dummies = pd.get_dummies(dataframe[combination])
    
    labeled, score = sil_score(_df_aggregated_with_dummies)
    
    cluster_0 = df_skin[labeled == 0]
    cluster_1 = df_skin[labeled == 1]
    
    cluster_0_blacks = len(cluster_0[cluster_0['skin'] > 0.5])
    cluster_0_whites = len(cluster_0[cluster_0['skin'] <= 0.5])
    
    cluster_1_blacks = len(cluster_1[cluster_1['skin'] > 0.5])
    cluster_1_whites = len(cluster_1[cluster_1['skin'] <= 0.5])
    
    return {
        'sil_score': score,
        'features': ','.join(list(_df_aggregated_with_dummies.columns)),
        'cluster_0_whites': cluster_0_whites,
        'cluster_1_whites': cluster_1_whites,
        'cluster_0_blacks': cluster_0_blacks,
        'cluster_1_blacks': cluster_1_blacks,
        'cluster_0_total': cluster_0_whites + cluster_0_blacks,
        'cluster_1_total': cluster_1_whites + cluster_1_blacks
    }


# We will be using this function to test our feauture configurations.
def sil_score(df_aggr):
    '''
    sil_score applies KMeans on the given paremeter df_aggr (a pandas DataFrame) by using all
    its columns as features, and returns results of clustering and silhouette score.
    '''
    scale = StandardScaler()
    df_aggr_scaled = scale.fit_transform(df_aggr.as_matrix())
    kmeans_model = KMeans(n_clusters=2, random_state=0, init='k-means++', n_jobs=1).fit(df_aggr_scaled)
    labels = kmeans_model.labels_
    return labels, silhouette_score(df_aggr_scaled, labels)


def get_tuple(combination,result):
    return (
        ','.join(combination),
        result['cluster_0_blacks']/result['cluster_0_total'],
        result['cluster_1_blacks']/result['cluster_1_total'],
        result['sil_score'],
        result['cluster_0_total'],
        result['cluster_1_total']
    )