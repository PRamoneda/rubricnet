from IPython.display import display
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.naive_bayes import GaussianNB
import seaborn as sns
import numpy as np
from sklearn.model_selection import GridSearchCV
from scipy import stats
from scipy.cluster import hierarchy
from scipy.spatial.distance import pdist
import ipywidgets as widgets
from sklearn.feature_selection import mutual_info_classif, mutual_info_regression
from sklearn.model_selection import StratifiedKFold


def train_and_validate(df_train, predictive_columns, var_smoothing=1e-9, test_frames=None, n_splits=5, shuffle=False,
                       random_state=None, equal_prior=True):
    if test_frames is None:
        test_frames = {"cross_val": df_train}
    X = df_train.loc[:, predictive_columns].to_numpy()
    y = df_train.grade
    if equal_prior:
        uniq_n = len(set(df_train.grade))
        priors = np.ones(uniq_n) / float(uniq_n)
    else:
        priors = None
    model = GaussianNB(var_smoothing=var_smoothing, priors=priors)
    model.fit(X, y)
    y_pred = model.predict(X)
    train_accuracy = (y == y_pred).sum() / len(y_pred)

    skf = StratifiedKFold(n_splits=n_splits, shuffle=shuffle, random_state=random_state)
    scores = {k: [] for k in test_frames.keys()}
    for train_index, test_index in skf.split(X, y):
        X_train = X[train_index]
        y_train = y[train_index]
        model = GaussianNB(var_smoothing=var_smoothing)
        model.fit(X_train, y_train)
        for k, v in test_frames.items():
            X_t = v.loc[:, predictive_columns].to_numpy()
            X_test = X_t[test_index]
            y_test = y[test_index]
            y_test_pred = model.predict(X_test)
            scores[k].append((y_test == y_test_pred).sum() / len(y_test_pred))

    scores = {k: np.array(v) for k, v in scores.items()}
    return train_accuracy, scores


def estimate_var_smoothing(df, predictive_columns):
    x_train = df.loc[:, predictive_columns]
    y_train = df.grade
    nb_classifier = GaussianNB()
    params_NB = {'var_smoothing': np.logspace(0, -9, num=100)}
    gs_NB = GridSearchCV(estimator=nb_classifier,
                         param_grid=params_NB,
                         scoring='accuracy')
    gs_NB.fit(x_train, y_train)
    return gs_NB.best_params_['var_smoothing']


def display_grade_correlations(df, feature):
    print("Correlations with grade")
    print("tau-c", stats.kendalltau(df.grade.tolist(), df[feature].tolist(), variant='c'))
    print("tau-b", stats.kendalltau(df.grade.tolist(), df[feature].tolist(), variant='b'))
    print(stats.spearmanr(df.grade.tolist(), df[feature].tolist()))
    sns.catplot(x="grade", y=feature, data=df)


def information_norm_matrix(
        features_samples_matrix,
        dtypes,
        n_neighbors=3,
        discrete_features=False,
        entropy_norm="min"):
    m = features_samples_matrix.shape[0]
    res = np.zeros((m * (m - 1)) // 2)
    entropies = np.zeros(m)
    for i in range(0, m):
        uniq_n = len(set(features_samples_matrix[i]))
        # print(features_samples_matrix[i+1:m].shape, features_samples_matrix[i].shape)
        if dtypes[i].kind == 'i' and uniq_n <= 10:
            mi = mutual_info_classif(
                features_samples_matrix[i:m].transpose(),
                features_samples_matrix[i].transpose(),
                discrete_features=discrete_features,
                n_neighbors=n_neighbors)
        else:
            mi = mutual_info_regression(
                features_samples_matrix[i:m].transpose(),
                features_samples_matrix[i].transpose(),
                discrete_features=discrete_features,
                n_neighbors=n_neighbors)
        start = m * i - ((i + 2) * (i + 1)) // 2 + i + 1
        res[start:start + len(mi) - 1] = mi[1:]
        entropies[i] = mi[0]
    for i in range(0, m - 1):
        for j in range(i+1, m):
            pos = m * i + j - ((i + 2) * (i + 1)) // 2
            #print(res[pos], entropies[i], entropies[j])
            if entropy_norm == 'min':
                res[pos] /= min(entropies[i], entropies[j])
            elif entropy_norm == 'max':
                res[pos] /= max(entropies[i], entropies[j])
            else:
                raise Exception("Unkown function" + entropy_norm)
    return res



def correlation_dendrogram(df, columns, first_grade=None, last_grade=None, metric="correlation", method="complete",
                           grade_aggregation_method="mean", filename=None):
    if first_grade is None:
        first_grade = min(df.grade.to_numpy())
    if last_grade is None:
        last_grade = max(df.grade.to_numpy())

    zero_deviation_titles = set()
    for g in range(first_grade, last_grade + 1):
        a = df[df.grade == g][columns].std()
        zero_deviation_titles.update(a.index[a.values == 0])

    correlation_titles = [x for x in columns if x not in zero_deviation_titles]
    if len(zero_deviation_titles) > 0:
        print("zero deviation columns excluded from correlation analysis:", zero_deviation_titles)
    correlation_distances = []
    if metric == 'correlation':
        for g in range(first_grade, last_grade + 1):
            dfg = df[df.grade == g][correlation_titles]
            dists = pdist(dfg.values.transpose(), metric='correlation')
            print(dfg.values.transpose().shape, type(dfg.values.transpose()))
            print(dists.shape, type(dists))
            correlation_distances.append(dists)
    elif metric == 'information':
        for g in range(first_grade, last_grade + 1):
            dfg = df[df.grade == g][correlation_titles]
            if dfg.shape[0] <= 2:
                continue
            dists = information_norm_matrix(dfg.values.transpose(), dtypes=[dfg[x].dtype for x in columns])
            print(dfg.values.transpose().shape, type(dfg.values.transpose()))
            print(dists.shape, type(dists))
            correlation_distances.append(dists)
        correlation_distances = np.array(correlation_distances)
        # total_max = np.max(correlation_distances)
        correlation_distances = 1 - correlation_distances
        correlation_distances[correlation_distances < 0] = 0
    elif metric == 'tau-c':
        for g in range(first_grade, last_grade + 1):
            dfg = df[df.grade == g][correlation_titles]
            if dfg.shape[0] <= 2:
                continue
            correlation_distances.append(pdist(dfg.values.transpose(), \
                                               metric=lambda x, y: 1 - abs(
                                                   stats.kendalltau(x, y, variant='c').statistic)))
        correlation_distances = np.array(correlation_distances)
        correlation_distances[correlation_distances < 0] = 0
    else:
        raise Exception("Wrong distance")

    correlation_distances = np.array(correlation_distances)
    if grade_aggregation_method == "mean":
        d = correlation_distances.mean(axis=0)
    elif grade_aggregation_method == "min":
        d = correlation_distances.min(axis=0)
    elif grade_aggregation_method == "max":
        d = correlation_distances.max(axis=0)
    else:
        raise Exception("Wrong grade_aggregation_method: " + grade_aggregation_method)
    Z = hierarchy.linkage(d, method)
    # Z = hierarchy.linkage(dfg.values.transpose(), 'single', metric='correlation')

    dn = hierarchy.dendrogram(Z, leaf_label_func=lambda x: dfg.columns[x],
                              leaf_rotation=90)
    if filename is not None:
        plt.savefig(filename, bbox_inches='tight')
    plt.show()
    return Z


def decisive_columns(df, predictive_columns, n_neighbors=3, exclude_nans=False, target_column="grade", calculate_mi=True, discrete_features=False):
    troubled = df.loc[:, predictive_columns].isnull().values
    if (troubled.any()):
        troubled_columns_numbers = troubled.sum(axis=0)
        troubled_columns = np.array(predictive_columns)[troubled_columns_numbers.nonzero()[0]]
        indices_to_drop = set()
        for c in troubled_columns:
            # ids for each NaN
            print("Column: ", c)
            print(df['id'][df[c].isnull()])
            indices_to_drop.update(np.where(df[c].isnull())[0])
        if exclude_nans:
            print(indices_to_drop)
            df = df.drop(indices_to_drop)
        else:
            raise ValueError("NaN detected")
    X = df.loc[:, predictive_columns]
    y = df[target_column]
    importance = []
    if calculate_mi:
        mi = mutual_info_classif(X, y, discrete_features=discrete_features, n_neighbors=n_neighbors)
    else:
        mi = np.zeros(len(predictive_columns))
    for i in range(len(predictive_columns)):
        tauc = stats.kendalltau(y.tolist(), df[predictive_columns[i]].tolist(), variant='c')
        # taub = stats.kendalltau(df.grade.tolist(), df[feature].tolist(), variant='b')
        # spearmanr = stats.spearmanr(df.grade.tolist(), df[feature].tolist())
        entry = {
            "feature": predictive_columns[i],
            "tau_c": tauc.correlation
            # "tauc_pvalue":tauc.pvalue
            # "taub":taub.correlation,
            # "taub_pvalue":taub.pvalue,
            # "r":spearmanr.correlation,
            # "r_pvalue":spearmanr.pvalue
        }
        if calculate_mi:
            entry["mutual_information"] = mi[i]

        importance.append(entry)
    return pd.DataFrame(importance)


def gradually_fit_correlated_columns(
        df_train,
        all_predictive_columns,
        test_frames = None,
        n_splits=5, shuffle=False, random_state=None, equal_prior=True, const_smoothing=1e-9):
    smoothings = []
    results = []
    for i in range(len(all_predictive_columns)):
        predictive_columns = all_predictive_columns[0:i+1]
        var_smoothing = const_smoothing
        train_acc, val_accs = train_and_validate(
            df_train,
            predictive_columns=predictive_columns,
            var_smoothing=var_smoothing,
            test_frames=test_frames,
            n_splits=n_splits,
            shuffle=shuffle,
            random_state=random_state,
            equal_prior=equal_prior)
        id = all_predictive_columns[i]
        results.append({'id': id, 'type':'training', 'value':train_acc})
        for k in val_accs.keys():
            results.extend([{'id': id, 'type':k, 'value':x} for x in val_accs[k]])
        smoothings.append(var_smoothing)
    return results, smoothings


def find_max_cross_val_accuracy(accuracies):
    types = set(accuracies.type)
    res = {}
    for type in types:
        grouped=accuracies[accuracies.type == type].groupby(['id']).mean()
        max_index=grouped['value'].argmax()
        max_id=grouped.index[max_index]
        print(type, max_id)
        display(accuracies[accuracies.id == max_id].groupby(['type']).mean())
        res[type] = (max_id, grouped['value'].max())
    return res


def fit_columns(df_train, all_titles, first_grade=0, last_grade=5, keep_column_order=False, test_frames=None,
                n_splits=5, shuffle=False, random_state=None, equal_prior=True):
    correlations = []

    for feature in all_titles:
        tauc = stats.kendalltau(df_train.grade.tolist(), df_train[feature].tolist(), variant='c')
        entry = {
            "feature": feature,
            "tau_c": tauc.correlation,
        }
        correlations.append(entry)

    correlations = pd.DataFrame(correlations)
    if not keep_column_order:
        correlations = correlations.sort_values(by=['tau_c'], ascending=False)

    widget1 = widgets.Output()
    widget2 = widgets.Output()
    widget3 = widgets.Output()

    with widget1:
        display(correlations)

    zero_deviation_titles = set()
    for g in range(first_grade, last_grade + 1):
        a = df_train[df_train.grade == g][all_titles].std()
        zero_deviation_titles.update(a.index[a.values == 0])

    correlation_titles = [x for x in all_titles if x not in zero_deviation_titles]
    if len(zero_deviation_titles) > 0:
        print("zero deviation columns excluded from correlation analysis:", zero_deviation_titles)

    correlation_distances = []
    for g in range(first_grade, last_grade + 1):
        dfg = df_train[df_train.grade == g][correlation_titles]
        correlation_distances.append(pdist(dfg.values.transpose(), metric='correlation'))

    correlation_distances = np.array(correlation_distances)
    d = correlation_distances.max(axis=0)
    Z = hierarchy.linkage(d, 'single')
    # Z = hierarchy.linkage(dfg.values.transpose(), 'single', metric='correlation')

    with widget2:
        dn = hierarchy.dendrogram(Z, leaf_label_func=lambda x: dfg.columns[x],
                                  leaf_rotation=90)
        plt.show()

    results, smoothings = gradually_fit_correlated_columns(df_train, correlations.feature.values, test_frames,
                                                           n_splits=n_splits, shuffle=shuffle,
                                                           random_state=random_state, equal_prior=equal_prior)

    accuracies = pd.DataFrame(results)

    with widget3:
        fig, ax = plt.subplots()
        sns.pointplot(x="id", y="value", hue='type', data=accuracies, ax=ax, ci=None)
        ax.set_xticklabels(ax.get_xticklabels(), rotation='vertical');
        plt.show()

    print(
        "Features correlated with Grade.       'Independence' given Grade.             Prediction accuracy for models")
    print("                                                                               accumulating more features.")
    display(widgets.HBox([widget1, widget2, widget3]))
    return correlations.feature.values, results, smoothings, accuracies

