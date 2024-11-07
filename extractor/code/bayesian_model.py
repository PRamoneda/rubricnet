from collections import namedtuple
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats
from scipy.stats import gamma, norm, expon, poisson
import matplotlib.pyplot as plt
import math
from collections import Counter
from sklearn.metrics import confusion_matrix
from sklearn.mixture import GaussianMixture
from sklearn.model_selection import StratifiedKFold, KFold
from sklearn.preprocessing import KBinsDiscretizer
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis, LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
import itertools
import sys
import os
from scipy.special import logsumexp
from sklearn.metrics import balanced_accuracy_score

from performance_chain import MIDI_TO_NEAREST_WHITES, is_white


def to_pandas(k):
    if type(k) is str:
        return k
    elif type(k) is tuple:
        return list(k)
    else:
        raise Exception("Unexpected type")


def predict_bayes(X, logpdfs):
    logps = np.array([x(X) for x in logpdfs])
    return np.argmax(logps, axis=0)


def fit_bayes(df, estimator_map, grade_column='grade'):
    return {columns: v.fit(df[to_pandas(columns)].to_numpy(), df[grade_column].to_numpy()) for columns, v in
            estimator_map.items()}


def predict_bayes_df(df, columns_logpdf_dict, debug=False, vote=False, probabilities=False):
    lpdfs = list(columns_logpdf_dict.values())[0]
    dims = (len(lpdfs), len(df))
    sum = np.zeros(dims)
    for columns, logpdf in columns_logpdf_dict.items():
        res = np.array([x.logpdf(df[to_pandas(columns)].to_numpy()) for x in logpdf])
        part = res.reshape(dims)
        if vote:
            norm = logsumexp(part, axis=0)
            part = np.exp(part - norm)
            if debug:
                print(norm)
        if debug:
            print(part)
        sum = sum + part
    if probabilities:
        return np.exp(sum - logsumexp(sum))
    else:
        return np.argmax(sum, axis=0)


GaussianFit = namedtuple("GaussianFit", "mu std ks k2 p")

GammaFit = namedtuple("GammaFit", "a loc scale ks")

class LogPDF:
    def __init__(self, class_info):
        self.class_info = class_info

    def logpdf(self, X):
        pass

class Estimator:
    def __init__(self, number_of_classes=None):
        self.number_of_classes = number_of_classes

    def get_class_info(self, X, y):
        pass

    def logpdf_instance(self, ci):
        pass

    def fit(self, X, y):
        """
        returns array of logpdf
        """
        class_info = self.get_class_info(X, y)
        return [self.logpdf_instance(ci) for ci in class_info]


class EstimatorGUI:
    def vislualize_feasibility(X, y):
        pass


class GaussianLogPDF(LogPDF):
    def logpdf(self, X):
        return norm.logpdf(X, self.class_info.mu, self.class_info.std)

class GaussianEstimator(Estimator):
    def __init__(self, number_of_classes=None):
        super().__init__(number_of_classes)

    def logpdf_instance(self, ci):
        return GaussianLogPDF(ci)

    def get_class_info(self, X, y):
        class_info = []
        # Assuming continuous classes
        if self.number_of_classes is None:
            self.number_of_classes=np.max(y) + 1
        for gr in range(self.number_of_classes):
            c_X = X[y == gr]
            if len(c_X) == 0:
                class_info.append(GaussianFit(float('nan'), float('nan'), None, float('nan'), float('nan')))
                continue
            mu, std = norm.fit(c_X)
            if (len(c_X) >= 8):
                k2, p = stats.normaltest(c_X)
            else:
                k2 = float('nan')
                p = float('nan')
            # kolmogorov-smirnov
            ks = stats.kstest(c_X, lambda x: norm.cdf(x, mu, std))
            class_info.append(GaussianFit(mu, std, ks, k2, p))
        return class_info


class GaussianEstimatorGUI(EstimatorGUI):
    def __init__(self, number_of_classes):
        self.estimator = GaussianEstimator(number_of_classes)

    def plot_to_gaussian(self, X, mu, std):
        plt.hist(X, density=True, alpha=0.6, color='g', bins=20)
        xmin, xmax = plt.xlim()
        x = np.linspace(xmin, xmax, 100)
        p = norm.pdf(x, mu, std)
        plt.plot(x, p, 'k', linewidth=2)
        plt.show()

    def vislualize_feasibility(self, X, y):
        classes_info = self.estimator.get_class_info(X, y)
        for gr in range(self.estimator.number_of_classes):
            ci = classes_info[gr]
            if math.isnan(ci.mu):
                continue
            print("Mu: %f Std: %f" % (ci.mu, ci.std))
            print("k2: %f p: %f" % (ci.k2, ci.p))
            print(ci.ks)
            c_X = X[y == gr]
            self.plot_to_gaussian(c_X, ci.mu, ci.std)
        ndf = pd.DataFrame({
            'grade': range(self.estimator.number_of_classes),
            'mu': [x.mu for x in classes_info],
            'std': [x.std for x in classes_info]})
        ndf.mu.plot(yerr=ndf['std'])
        plt.show()

        y_pred = predict_bayes(X, self.estimator.fit(X, y))
        print("Accuracy %f" % (sum(y_pred == y) / len(y)))
        cm = confusion_matrix(y, y_pred, normalize="true")
        ax = sns.heatmap(cm, annot=True, cmap='viridis', fmt='.1%')
        ax.set_ylabel("Truth")
        ax.set_xlabel("Prediction")
        pd.DataFrame(y_pred).hist()

class QDALogPDF(LogPDF):
    def logpdf(self, X):
        return self.class_info[0].predict_log_proba(X)[:, self.class_info[1]]


class QDAEstimator(Estimator):
    def __init__(self, number_of_classes=None, reg_param=0.0):
        super().__init__(number_of_classes)
        self.reg_param=reg_param

    def logpdf_instance(self, ci):
        return QDALogPDF(ci)

    def get_class_info(self, X, y):
        # Assuming continuous classes
        if self.number_of_classes is None:
            self.number_of_classes = np.max(y) + 1

        qda = QuadraticDiscriminantAnalysis(priors=np.ones(self.number_of_classes)/self.number_of_classes, reg_param=self.reg_param)
        qda.fit(X, y)

        class_info = []

        return [(qda, c) for c in range(self.number_of_classes)]

    def fit(self, X, y):
        class_info = self.get_class_info(X, y)
        qda = class_info[0][0]
        # TODO: optimize in order not to recalulate it each time
        return [self.logpdf_instance(ci) for ci in class_info]


class LDALogPDF(LogPDF):
    def logpdf(self, X):
        return self.class_info[0].decision_function(X)[:, self.class_info[1]]


class LDAEstimator(Estimator):
    def __init__(self, number_of_classes=None, solver='svd', shrinkage=None):
        super().__init__(number_of_classes)
        self.solver=solver
        self.shrinkage=shrinkage

    def logpdf_instance(self, ci):
        return LDALogPDF(ci)

    def get_class_info(self, X, y):
        # Assuming continuous classes
        if self.number_of_classes is None:
            self.number_of_classes = np.max(y) + 1

        lda = LinearDiscriminantAnalysis(priors=np.ones(self.number_of_classes)/self.number_of_classes, solver=self.solver, shrinkage=self.shrinkage)
        lda.fit(X, y)
        return [(lda, c) for c in range(self.number_of_classes)]

    def fit(self, X, y):
        class_info = self.get_class_info(X, y)
        lda = class_info[0][0]
        # TODO: optimize in order not tio recalulate it each time
        return [self.logpdf_instance(ci) for ci in class_info]


class GaussianNBLogPDF(LogPDF):
    def logpdf(self, X):
        return self.class_info[0].predict_log_proba(X)[:, self.class_info[1]]


class GaussianNBEstimator(Estimator):
    def __init__(self, number_of_classes=None, equal_priors=True):
        super().__init__(number_of_classes)
        self.equal_priors = equal_priors

    def logpdf_instance(self, ci):
        return GaussianNBLogPDF(ci)

    def get_class_info(self, X, y):
        # Assuming continuous classes
        if self.number_of_classes is None:
            self.number_of_classes = np.max(y) + 1
        if self.equal_priors:
            lda = GaussianNB(priors=np.ones(self.number_of_classes)/self.number_of_classes)
        else:
            lda = GaussianNB()

        lda.fit(X, y)
        return [(lda, c) for c in range(self.number_of_classes)]

    def fit(self, X, y):
        class_info = self.get_class_info(X, y)
        lda = class_info[0][0]
        # TODO: optimize in order not tio recalulate it each time
        return [self.logpdf_instance(ci) for ci in class_info]


class MultinomialRankLogPDF(LogPDF):
    def logpdf(self, X):
        l = float(sum(self.class_info.values()))
        lpdf = {x[0]: math.log(x[1] / l) for x in self.class_info.most_common()}
        return np.vectorize(lpdf.get)(X)


class MultinomialRankEstimator(Estimator):
    def __init__(self, number_of_classes=None, monotonize=None):
        super().__init__(number_of_classes)
        self.monotonize = monotonize

    def get_class_info(self, X, y):
        # Assuming continuous classes
        if self.number_of_classes is None:
            self.number_of_classes=np.max(y) + 1

        maxrank = np.max(X)
        counters = []
        for gr in range(self.number_of_classes):
            c_X = X[y == gr]
            c = Counter(c_X)
            c.update(range(maxrank + 1))
            counters.append(c)

        # Enforce monotonicity
        if self.monotonize is not None:
            something_changed = True
            while something_changed:
                old_mu = -1
                something_changed = False
                for gr in range(self.number_of_classes):
                    c = counters[gr]
                    lc = list(c.elements())
                    mu = np.mean(lc)
                    if old_mu <= mu:
                        old_mu = mu
                    else:
                        if self.monotonize == "lower":
                            c = Counter(counters[gr-1])
                            lc = list(c.elements())
                            old_mu = np.mean(lc)
                            counters[gr] = c
                            something_changed = True
                        elif self.monotonize == "upper":
                            counters[gr-1] = Counter(c)
                            old_mu = mu
                            something_changed = True
                        elif self.monotonize == "average":
                            c.update(counters[gr-1])
                            counters[gr-1] = Counter(c)
                            lc = list(c.elements())
                            old_mu = np.mean(lc)
                            counters[gr] = c
                            something_changed = True
                        else:
                            raise Exception("No such monotonizing method: " + self.monotonize)

        return counters

    def logpdf_instance(self, c):
        return MultinomialRankLogPDF(c)


class MultinomialRankEstimatorGUI(EstimatorGUI):
    def __init__(self, number_of_classes, monotonize = None):
        self.estimator = MultinomialRankEstimator(number_of_classes, monotonize = monotonize)

    def vislualize_feasibility(self, X, y):
        counters = self.estimator.get_class_info(X, y)
        mus = []
        stds = []
        reg_data = []
        for gr in range(self.estimator.number_of_classes):
            c = counters[gr]
            lc = list(c.elements())
            mu = np.mean(lc)
            mus.append(mu)
            stds.append(np.std(lc))
            reg_data.extend([{"grade": gr, "time_signature_rank": r} for r in lc])
        df_reg = pd.DataFrame(reg_data)
        sns.histplot(
            data=df_reg,
            x="grade", hue="time_signature_rank",
            multiple="fill", stat="proportion",
            discrete=True, shrink=.8
        )
        plt.show()

        ndf = pd.DataFrame({'grade': range(self.estimator.number_of_classes), 'mu': mus, 'std': stds})
        ndf.mu.plot(yerr=ndf['std'])
        plt.show()

        y_pred = predict_bayes(X, self.estimator.fit(X, y))
        print("Accuracy %f" % (sum(y_pred == y) / len(y)))
        cm = confusion_matrix(y, y_pred, normalize="true")
        ax = sns.heatmap(cm, annot=True, cmap='viridis', fmt='.1%')
        ax.set_ylabel("Truth")
        ax.set_xlabel("Prediction")
        pd.DataFrame(y_pred).hist()


class MultinomialLogPDF(LogPDF):
    def __init__(self, class_info, alpha0, dirichlet_alphas, K):
        super().__init__(class_info)
        self.alpha0 = alpha0
        self.dirichlet_alphas = dirichlet_alphas
        self.K = K

    def logpdf(self, X):
        l = float(sum(self.class_info.values()))
        most_common = {x[0]: x[1] for x in self.class_info.most_common()}
        lpdf = {x: math.log((self.dirichlet_alphas[x] + most_common.get(x, 0)) / (self.alpha0 + l)) for x in
                range(self.K)}
        return np.vectorize(lpdf.get)(X)


class MultinomialEstimator(Estimator):
    def __init__(self, number_of_classes=None, dirichlet_alphas=None):
        super().__init__(number_of_classes)
        self.dirichlet_alphas = dirichlet_alphas

    def get_class_info(self, X, y):
        # Assuming continuous classes from zero to max
        if self.number_of_classes is None:
            self.number_of_classes = np.max(y) + 1

        # Assuming continuous classes from zero to max
        self.K = max(X) + 1
        if self.dirichlet_alphas is None:
            # Jeffry prior
            self.dirichlet_alphas = [0.5] * self.K
        self.alpha0 = sum(self.dirichlet_alphas)

        counters = []
        for gr in range(self.number_of_classes):
            c_X = X[y == gr]
            c = Counter(c_X)
            counters.append(c)
        return counters

    def logpdf_instance(self, c):
        return MultinomialLogPDF(c, self.alpha0, self.dirichlet_alphas, self.K)

class WordDiscretizer:
    def fit(self, all_words, word_array):
        self.word_indices = {x[1]:x[0] for x in zip(range(len(all_words)), all_words)}

    def K(self):
        return len(self.word_indices)

    def transform(self, word_array):
        return [self.word_indices[x] for x in word_array]


class OrderedWordDiscretizer(WordDiscretizer):
    def __init__(self, bins):
        self.bins = bins

    def fit(self, all_words, word_array):
        self.min = min(all_words)
        self.max = max(all_words)
        word_indices = {x[1]: x[0] for x in zip(range(len(all_words)), all_words)}
        discretizer = KBinsDiscretizer(n_bins=self.bins, encode='ordinal', strategy='uniform')
        discretizer.fit(np.array([word_indices[x] for x in word_array]).reshape(-1, 1))
        new_indices = discretizer.transform(np.arange(len(all_words)).reshape(-1, 1))
        self.word_indices = {x[1]:int(x[0][0]) for x in zip(new_indices, all_words)}

    def transform(self, word_array):
        return [self.word_indices[0] if x < self.min else  self.word_indices[-1] if x > self.max else self.word_indices[x] for x in word_array]


class WordListMultinoulliLogPDF(LogPDF):
    def __init__(self, class_info, word_discretizer):
        super().__init__(class_info)
        self.word_discretizer = word_discretizer

    def _logpdf(self, words):
        res = 0
        c = Counter(self.word_discretizer.transform(words))
        for i, v in c.most_common():
            res += math.log(self.class_info[i]) * v
        return res

    def logpdf(self, X):
        lpdf = lambda words: self._logpdf(words)
        return np.vectorize(lpdf)(X)


class WordListMultinoulliEstimator(Estimator):
    def __init__(self, all_words, number_of_classes=None, alpha = 1.0, word_discretizer=None):
        super().__init__(number_of_classes)
        self.all_words = all_words
        self.alpha = alpha
        self.word_discretizer = word_discretizer if word_discretizer is not None else WordDiscretizer()

    def get_class_info(self, X, y):
        # Assuming continuous classes from zero to max
        if self.number_of_classes is None:
            self.number_of_classes = np.max(y) + 1

        self.word_discretizer.fit(all_words=self.all_words, word_array=itertools.chain.from_iterable(X))
        self.dirichlet_alphas = self.alpha * np.ones(self.word_discretizer.K())
        self.alpha0 = sum(self.dirichlet_alphas)

        distributions = []
        for gr in range(self.number_of_classes):
            c = Counter(self.word_discretizer.transform(itertools.chain.from_iterable(X[y == gr])))
            d = np.zeros(self.word_discretizer.K())
            for k, v in c.most_common():
                d[k] = v
            d = (d + self.dirichlet_alphas) / (sum(d) + self.alpha0)
            distributions.append(d)
        return distributions

    def logpdf_instance(self, d):
        return MultinomialLogPDF(d, self.word_discretizer)


class WordListMultinoulliLogPDF2(LogPDF):
    def __init__(self, class_info, word_indices):
        super().__init__(class_info)
        self.word_indices = word_indices

    def _logpdf(self, words):
        res = 0
        c = Counter([self.word_indices[x] for x in words])
        l = len(words)
        for i, v in c.most_common():
            res += math.log(self.class_info[i]) * v
        return res

    def logpdf(self, X):
        lpdf = lambda words: self._logpdf(words)
        return np.vectorize(lpdf)(X)

class WordListMultinoulliEstimator2(Estimator):
    def __init__(self, all_words, number_of_classes=None, prior_sample_size = 1.0, prior_alpha=1e-30):
        super().__init__(number_of_classes)
        self.word_indices = {x[1]:x[0] for x in zip(range(len(all_words)), all_words)}
        self.K = len(all_words)
        self.prior_sample_size = prior_sample_size
        self.prior_alpha=prior_alpha

    def count_distribution(self, cells, priors):
        words = itertools.chain.from_iterable(cells)
        c = Counter([self.word_indices[x] for x in words])
        d = np.zeros(self.K)
        for k, v in c.most_common():
            d[k] = v
        return (d + priors) / (sum(d) + sum(priors))

    def get_class_info(self, X, y):
        # Assuming continuous classes from zero to max
        if self.number_of_classes is None:
            self.number_of_classes = np.max(y) + 1
        self.dirichlet_alphas = self.prior_sample_size * self.count_distribution(X, self.prior_alpha * np.ones(self.K))
        distributions = []
        for gr in range(self.number_of_classes):
            distributions.append(self.count_distribution(X[y == gr], self.dirichlet_alphas))
        return distributions

    def logpdf_instance(self, d):
        return WordListMultinoulliLogPDF2(d)


def renumerate_grades(df):
    grades = df.grade.to_numpy()
    s = sorted(set(grades))
    mapping = {x[0]:x[1] for x in zip(s, range(len(s)))}
    df.grade = [mapping[x] for x in grades]
    return len(mapping)


class GammaLogPDF(LogPDF):
    def logpdf(self, X):
        return gamma.logpdf(X, self.class_info.a, self.class_info.loc, self.class_info.scale)

class GammaEstimator(Estimator):
    def __init__(self, number_of_classes=None, a = None, loc = 0, default_addon = 0.1, monotonize=None):
        super().__init__(number_of_classes)
        self.a = a
        self.floc = loc
        self.monotonize = monotonize
        self.default_addon = default_addon

    def logpdf_instance(self, ci):
         return GammaLogPDF(ci)

    def get_class_info(self, X, y):
        # Assuming continuous classes
        if self.number_of_classes is None:
            self.number_of_classes=np.max(y) + 1

        class_info = []
        for gr in range(self.number_of_classes):
            c_X = X[y == gr]
            if len(c_X) == 0:
                class_info.append(GammaFit(float('nan'), float('nan'), float('nan'), None))
                continue
            elif len(c_X) == 1:
                c_X = list(c_X)
                c_X.append(self.default_addon)
            if self.floc is None:
                a, loc, scale = gamma.fit(c_X, f0=self.a)
            else:
                a, loc, scale = gamma.fit(c_X, f0=self.a, floc=self.floc)
            # kolmogorov-smirnov
            ks = stats.kstest(c_X, lambda x: gamma.cdf(x, a=a, loc=loc, scale=scale))
            class_info.append(GammaFit(a, loc, scale, ks))
        # Enforce monotonicity
        if self.monotonize is not None:
            something_changed = True
            while something_changed:
                old_mu = -1
                something_changed = False
                for gr in range(self.number_of_classes):
                    params = class_info[gr]
                    mu = params.loc + params.a * params.scale
                    print(gr, old_mu, mu)

                    if old_mu <= mu:
                        old_mu = mu
                    else:
                        if self.monotonize == "lower":
                            old_params=class_info[gr-1]
                            class_info[gr] = GammaFit(old_params.a, old_params.loc, old_params.scale, old_params.ks)
                            something_changed = True
                        elif self.monotonize == "upper":
                            class_info[gr-1] = GammaFit(params.a, params.loc, params.scale, params.ks)
                            old_mu = mu
                            something_changed = True
                        elif self.monotonize == "average":
                            c_X = X[(y == gr) | (y == gr - 1)]
                            a, loc, scale = gamma.fit(c_X, f0=self.a, floc=self.floc)
                            # kolmogorov-smirnov
                            ks = stats.kstest(c_X, lambda x: gamma.cdf(x, a=a, loc=loc, scale=scale))
                            class_info.append(GammaFit(a, loc, scale, ks))
                            old_mu = loc + a * scale
                            something_changed = True
                        else:
                            raise Exception("No such monotonizing method: " + self.monotonize)
        return class_info


class GammaEstimatorGUI(EstimatorGUI):
    def __init__(self, number_of_classes, a=None, loc=0, default_addon = 0.1, monotonize=None):
        self.estimator = GammaEstimator(number_of_classes, a=a, loc=loc, default_addon=default_addon, monotonize=monotonize)

    def plot_to_gamma(self, X, a, loc, scale):
        plt.hist(X, density=True, alpha=0.6, color='g', bins=20)
        xmin, xmax = plt.xlim()
        x = np.linspace(xmin, xmax, 100)
        p = gamma.pdf(x, a, loc, scale)
        plt.plot(x, p, 'k', linewidth=2)
        plt.show()

    def vislualize_feasibility(self, X, y):
        classes_info = self.estimator.get_class_info(X, y)
        for gr in range(self.estimator.number_of_classes):
            ci = classes_info[gr]
            if math.isnan(ci.a):
                continue
            print("a: %f Loc: %f Scale: %f" % (ci.a, ci.loc, ci.scale))
            print(ci.ks)
            c_X = X[y == gr]
            self.plot_to_gamma(c_X, ci.a, ci.loc, ci.scale)
        ndf = pd.DataFrame({
            'grade': range(self.estimator.number_of_classes),
            'mu': [x.loc + x.a * x.scale for x in classes_info],
            'std': [x.a * (x.scale ** 2) for x in classes_info]})
        ndf.mu.plot(yerr=ndf['std'])
        plt.show()

        y_pred = predict_bayes(X, self.estimator.fit(X, y))
        print("Accuracy %f" % (sum(y_pred == y) / len(y)))
        cm = confusion_matrix(y, y_pred, normalize="true")
        ax = sns.heatmap(cm, annot=True, cmap='viridis', fmt='.1%')
        ax.set_ylabel("Truth")
        ax.set_xlabel("Prediction")
        pd.DataFrame(y_pred).hist()


class DummyLogPDF(LogPDF):
    def dummy(x):
        return 0

    def logpdf(self, X):
        return np.vectorize(self.dummy)(X)


class DummyEstimator(Estimator):
    def __init__(self):
        super().__init__()

    def get_class_info(self, X, y):
        return ["dummy"] * self.number_of_classes

    def logpdf_instance(self, ci):
        return DummyLogPDF(ci)


class ZeroOrSomethingLogPDF(LogPDF):
    def __init__(self, class_info, non_zero_estimator):
        super().__init__(class_info)
        self.non_zero_estimator = non_zero_estimator

    def logpdf(self, X):
        non_zero_logpdf = self.non_zero_estimator.logpdf_instance(self.class_info[1])
        nonvectorized = lambda y, nz_lpdf=non_zero_logpdf, p=self.class_info[0]: math.log(1-p) if y == 0 else (math.log(p) + non_zero_logpdf.logpdf(y))
        return np.vectorize(nonvectorized)(X)


class ZeroOrSomethingEstimator(Estimator):
    def __init__(self, number_of_classes=None, non_zero_estimator=DummyEstimator(), monotonize=None):
        super().__init__(number_of_classes)
        self.non_zero_estimator = non_zero_estimator
        self.monotonize = monotonize

    def compact_mapping(self, y):
        s = sorted(set(y))
        return {x[0]: x[1] for x in zip(s, range(len(s)))}

    def get_class_info(self, X, y):
        # Assuming continuous classes
        if self.number_of_classes is None:
            self.number_of_classes=np.max(y) + 1

        probs = []
        for gr in range(self.number_of_classes):
            c_X = X[y == gr]
            n = sum(c_X > 0) + 1
            d = (len(c_X) - 1)
            if d == 0:
                d = 1
            probs.append(n / d)
        if self.monotonize is not None:
            if self.monotonize == "average":
                changed_something = True
                while changed_something:
                    changed_something = False
                    for i in range(1, 6):
                        if probs[i] < probs[i - 1]:
                            probs[i] = probs[i - 1] = (probs[i] + probs[i - 1]) / 2.0
                            changed_something = True
            else:
                raise Exception("Only average is supported")

        nz_X = X[X > 0]
        nz_y = y[X > 0]
        mapping = self.compact_mapping(nz_y)
        mapped_y = np.array([mapping[x] for x in nz_y])
        self.non_zero_estimator.number_of_classes = len(mapping)
        classes = self.non_zero_estimator.get_class_info(nz_X, mapped_y)
        # redistribute
        results = []
        for gr in range(self.number_of_classes):
            if gr in mapping:
                results.append((probs[gr], classes[mapping[gr]]))
            else:
                results.append((probs[gr], classes[0]))
        return results

    def logpdf_instance(self, ci):
        return ZeroOrSomethingLogPDF(ci, self.non_zero_estimator)


class GaussianEstimatorSharedStd(Estimator):
    def __init__(self, sigma0=1000000, number_of_classes=None, include_tests=False):
        super().__init__(number_of_classes)
        self.include_tests = include_tests
        self.sigma0 = sigma0

    def logpdf_instance(self, ci):
        return GaussianLogPDF(ci)

    def get_class_info(self, X, y):
        # Assuming continuous classes
        if self.number_of_classes is None:
            self.number_of_classes = np.max(y) + 1

        mu0 = np.mean(X)

        # calculate pooled empirical variance s
        s = 0
        mus = []
        for gr in range(self.number_of_classes):
            c_X = X[y == gr]
            if len(c_X) == 0:
                mus.append(mu0)
                continue
            mu = np.mean(c_X)
            s += np.sum(np.power((c_X - mu), 2))
            mus.append(mu)
        n = len(X)
        if n > self.number_of_classes:
            n -= self.number_of_classes
        s /= n
        s = math.sqrt(s)

        self.sigma = s

        _lambda = 1 / (self.sigma * self.sigma)
        _lambda0 = 1 / (self.sigma0 * self.sigma0)

        class_info = []
        for gr in range(self.number_of_classes):
            c_X = X[y == gr]
            n = len(c_X)
            if len(c_X) == 0:
                continue
            _lambda_n = _lambda0 + n * _lambda
            w = n * _lambda / _lambda_n
            mu_n = w * mus[gr] + (1 - w) * mu0
            if (self.include_tests):
                k2, p = stats.normaltest(c_X)
                ks = stats.kstest(c_X, lambda x: norm.cdf(x, mu, s))
            else:
                k2 = float('nan')
                p = float('nan')
                ks = float('nan')
            #class_info.append(GaussianFit(mu_n, math.sqrt(1 / _lambda_n), ks, k2, p))
            class_info.append(GaussianFit(mu_n, self.sigma, ks, k2, p))
        return class_info


class PoissonLogPDF(LogPDF):
    def __init__(self, class_info, loc):
        super().__init__(class_info)
        self.loc = loc

    def logpdf(self, X):
        return poisson.logpmf(X, self.class_info, self.loc)


class PoissonEstimator(Estimator):
    def __init__(self, number_of_classes=None, loc = 0, beta = 0):
        super().__init__(number_of_classes)
        self.loc = loc
        self.beta = beta

    def logpdf_instance(self, ci):
         return PoissonLogPDF(ci, self.loc)

    def get_class_info(self, X, y):
        # Assuming continuous classes
        if self.number_of_classes is None:
            self.number_of_classes=np.max(y) + 1

        self.alpha = self.beta * np.mean(X)
        class_info = []
        for gr in range(self.number_of_classes):
            c_X = X[y == gr]
            if len(c_X) == 0:
                class_info.append(1e-30)
                continue
            else:
                class_info.append((np.sum(c_X - self.loc) + self.alpha)/(len(c_X) + self.beta))
        return class_info


def metrics(y, y_pred):
    return {
        "accuracy": (np.mean(y_pred == y))*100,
        "MSE": np.mean(np.power(y_pred - y, 2)),
        "MAE": np.mean(np.abs(y_pred - y)),
        "b. acc": balanced_accuracy_score(y, y_pred) * 100
    }

class PredefinedSplitGenerator(object):
    def __init__(self, df, split_map, train_id="train", test_id="test"):
        self.df = df
        self.split_map_iterator = iter(split_map.values())
        self.test_id = test_id
        self.train_id = train_id

    def __iter__(self):
        return self

    def __next__(self):
        return self.next()

    def next(self):
        values = self.split_map_iterator.__next__()
        train_set = values[self.train_id].keys()
        test_set = values[self.test_id].keys()
        #print(len(test_set), self.df.id.isin(test_set), np.where(self.df[self.df.id.isin(test_set)])[0])
        return np.where(self.df.id.isin(train_set))[0], np.where(self.df.id.isin(test_set))[0]


def crossvalidate_df(df, estimators, split_generator = None, n_splits=5, cross_validator=None, random_state=None, shuffle=False, vote=False):
    if split_generator is None:
        if cross_validator is None:
            cross_validator = StratifiedKFold(n_splits=n_splits, random_state=random_state, shuffle=shuffle)
        split_generator = cross_validator.split(df, df.grade.to_numpy())
    accuracies = {}
    all_y_train = {}
    all_y_train_pred = {}
    all_y_test = {}
    all_y_test_pred = {}
    all_id_test = {}
    split_index = 0
    for train, test in split_generator:
        train_df = df.iloc[train]
        test_df = df.iloc[test]
        logpdfs_map = fit_bayes(
            train_df, estimators)

        y_pred = predict_bayes_df(
            train_df,
            logpdfs_map,
            vote=vote)
        y = train_df.grade.to_numpy()
        acc_entry = {"train_"+k:v for k, v in metrics(y, y_pred).items()}
        all_y_train[split_index] = y
        all_y_train_pred[split_index] = y_pred

        y_pred = predict_bayes_df(
            test_df, logpdfs_map,
            vote=vote)
        y = test_df.grade.to_numpy()
        all_y_test[split_index] = y
        all_id_test[split_index] = test_df.id.to_numpy()
        all_y_test_pred[split_index] = y_pred
        acc_entry.update({"test_"+k:v for k, v in metrics(y, y_pred).items()})
        accuracies[split_index] = acc_entry
        split_index += 1
    return accuracies, all_y_train, all_y_train_pred, all_y_test, all_y_test_pred, all_id_test

def visualize_crossvalidation(accuracies_dict, y_train_dict, y_train_pred_dict, y_test_dict, y_test_pred_dict, id_test_dict):
    accuracies = list(accuracies_dict.values())
    y_test = list(itertools.chain.from_iterable(y_test_dict.values()))
    y_test_pred = list(itertools.chain.from_iterable(y_test_pred_dict.values()))
    ttdf = pd.DataFrame(accuracies)
    f, (ax1, ax2, ax3, ax4) = plt.subplots(ncols=4, figsize=(12, 3))
    sns.pointplot(ttdf[["train_accuracy", "test_accuracy"]], ax=ax1)
    sns.pointplot(ttdf[["train_MSE", "test_MSE"]], ax=ax2)
    sns.pointplot(ttdf[["train_MAE", "test_MAE"]], ax=ax3)
    sns.pointplot(ttdf[["train_b. acc", "test_b. acc"]], ax=ax4)
    plt.show()
    print(ttdf.mean())
    cm = confusion_matrix(y_test, y_test_pred, normalize="true")
    ax = sns.heatmap(cm, annot=True, cmap='viridis', fmt='.1%')
    ax.set_ylabel("Truth")
    ax.set_xlabel("Prediction")
    pd.DataFrame(y_test_pred).hist()


def empty_pair_to_intervals():
    finger_pair_to_intervals = {}
    for i in range(1, 6):
        for j in range(1, 6):
            finger_pair_to_intervals[str(abs(i)) + str(abs(j))] = []
    return finger_pair_to_intervals


def keys_distance_colors(pitch1, pitch2):
    return MIDI_TO_NEAREST_WHITES[pitch2][0] + MIDI_TO_NEAREST_WHITES[pitch2][1] - \
           (MIDI_TO_NEAREST_WHITES[pitch1][0] + MIDI_TO_NEAREST_WHITES[pitch1][1]), int(is_white(pitch1)), int(
        is_white(pitch2))


class FingeringIntervalCalculator:
    def count_pairs(self, pair, pairs):
        for p in pairs:
            fingers = str(abs(pair[1])) + str(abs(p[1]))
            fingers_inverse = str(abs(p[1])) + str(abs(pair[1]))
            interval, w1, w2 = keys_distance_colors(pair[0], p[0])
            if interval is None:
                continue
            self.colors[w1] += 1
            self.color_transfers[w1][w2] += 1
            self.finger_pair_to_intervals[fingers].append(interval)
            self.finger_pair_to_intervals[fingers_inverse].append(-interval)

    def count_self_pairs(self, pairs):
        for i in range(len(pairs)):
            self.count_pairs(pairs[i], pairs[i + 1:])

    def count_cross_pairs(self, this_pairs, other_pairs):
        for pair in this_pairs:
            self.count_pairs(pair, other_pairs)

    def process_single_song(self, pitch_sets, finger_sets):
        last_pairs = None
        for pitch_set, finger_set in zip(pitch_sets, finger_sets):
            this_pairs = [(x[0], x[1]) for x in zip(pitch_set, finger_set)]
            if last_pairs is not None and last_pairs == this_pairs:
                self.same += 1
            else:
                self.different += 1
                self.count_self_pairs(this_pairs)
                if last_pairs is not None:
                    self.count_cross_pairs(last_pairs, this_pairs)
            last_pairs = this_pairs

    def count_finger_distribution(self, X):
        """

        Parameters
        ----------
        X - two-column array

        Returns
        -------

        """
        self.same = 0
        self.different = 0
        self.finger_pair_to_intervals = empty_pair_to_intervals()
        self.colors = np.zeros(2)
        self.color_transfers = np.zeros((2,2))

        for x in X:
            pitch_sets = x[0]
            finger_sets = x[1]
            self.process_single_song(pitch_sets, finger_sets)
        self.fingers_p = {k: len(v) for k, v in self.finger_pair_to_intervals.items()}


class FingeringIntervalTransformer:
    def fit(self, finger_pair_to_intervals):
        pass

    def transform(self, pair, intervals):
        if pair[0] == pair[1]:
            pair = "11"
        return pair, intervals

# TODO: proper discretization
class FingeringIntervalDiscretizeCutTransformer:
    def __init__(self, max_interval=40):
        self.max_interval = max_interval

    def fit(self, finger_pair_to_intervals):
        pass

    def transform(self, pair, intervals):
        if pair[0] == pair[1]:
            pair = "11"
        intervals = np.round(np.multiply(2, intervals) + self.max_interval).astype(int)
        max_val = 2*self.max_interval
        intervals[intervals > max_val] = max_val
        intervals[intervals < 0] = 0
        return pair, intervals


class FingeringIntervalDiscretizeTransformer:
    def __init__(self, max_value=160):
        self.max_value = max_value

    def fit(self, finger_pair_to_intervals):
        # Eliminate all same-finger combination except one
        refined_pair_to_intervals = empty_pair_to_intervals()
        for pair, intervals in finger_pair_to_intervals.items():
            if pair[0] == pair[1]:
                pair = "11"
            refined_pair_to_intervals[pair].extend(intervals)
        self.discretizers = {}
        for i in range(1, 6):
            for j in range(1, 6):
                pair = str(abs(i)) + str(abs(j))
                if pair == "11" or i != j:
                    self.discretizers[pair] = OrderedWordDiscretizer(bins=self.max_value+1)
                    self.discretizers[pair].fit(range(min(refined_pair_to_intervals[pair]), max(refined_pair_to_intervals[pair])+1), refined_pair_to_intervals[pair])

    def transform(self, pair, intervals):
        if pair[0] == pair[1]:
            pair = "11"
        return pair, self.discretizers[pair].transform(intervals)


class FingeringIntervalDiscretizeTiedTransformer:
    def __init__(self, max_value=160):
        self.max_value = max_value

    def fit(self, finger_pair_to_intervals):
        # Eliminate all same-finger combination except one
        same_intervals = []
        direct_intervals = []
        inverse_intervals = []
        for pair, intervals in finger_pair_to_intervals.items():
            if pair[0] == pair[1]:
                same_intervals.extend(intervals)
            elif int(pair[0]) > int(pair[1]):
                direct_intervals.extend(intervals)
            else:
                inverse_intervals.extend(intervals)

        self.same_discretizer = KBinsDiscretizer(n_bins=self.max_value+1, encode='ordinal', strategy='uniform')
        self.same_discretizer.fit(np.array(same_intervals).reshape(-1, 1))
        self.direct_discretizer = KBinsDiscretizer(n_bins=self.max_value+1, encode='ordinal', strategy='uniform')
        self.direct_discretizer.fit(np.array(direct_intervals).reshape(-1, 1))
        self.inverse_discretizer = KBinsDiscretizer(n_bins=self.max_value+1, encode='ordinal', strategy='uniform')
        self.inverse_discretizer.fit(np.array(inverse_intervals).reshape(-1, 1))

    def transform(self, pair, intervals):
        if pair[0] == pair[1]:
            pair = "11"
        if pair[0] == pair[1]:
            discretizer = self.same_discretizer
        elif int(pair[0]) > int(pair[1]):
            discretizer = self.direct_discretizer
        else:
            discretizer = self.inverse_discretizer

        return pair, discretizer.transform(np.array(intervals).reshape(-1, 1)).reshape(-1).astype(int)


class FingerPairDistribution:
    def __init__(self, prior_alpha=0.5):
        self.dirichlet_alphas = {}
        for i in range(1, 6):
            for j in range(1, 6):
                self.dirichlet_alphas[str(abs(i)) + str(abs(j))] = prior_alpha
        self.alpha0 = sum(self.dirichlet_alphas.values())

    def fit(self, total_fingers_p, fingers_p_per_grade):
        self.grade_log_p_pairs = []
        for gr in range(len(fingers_p_per_grade)):
        # transformed pair->intervals map
            total = sum(fingers_p_per_grade[gr].values())
            self.grade_log_p_pairs.append(
                {k: math.log((v + self.dirichlet_alphas[k])/(total+self.alpha0)) for k, v in fingers_p_per_grade[gr].items()})

    def log_p(self, grade, pair):
        return self.grade_log_p_pairs[grade][pair]


class FingerPairDistributionTailored:
    def __init__(self, prior_sample_size=1.0, prior_alpha = 0.5):
        self.prior_sample_size = prior_sample_size
        self.prior_alpha = prior_alpha

    def fit(self, total_fingers_p, fingers_p_per_grade):
        self.dirichlet_alphas = {}
        for i in range(1, 6):
            for j in range(1, 6):
                pair = str(abs(i)) + str(abs(j))
                if pair in total_fingers_p:
                    self.dirichlet_alphas[pair] = total_fingers_p[pair] * self.prior_sample_size
                else:
                    self.dirichlet_alphas[pair] = self.prior_alpha * self.prior_sample_size
        self.alpha0 = sum(self.dirichlet_alphas.values())

        self.grade_log_p_pairs = []
        for gr in range(len(fingers_p_per_grade)):
        # transformed pair->intervals map
            total = sum(fingers_p_per_grade[gr].values())
            self.grade_log_p_pairs.append(
                {k: math.log((v + self.dirichlet_alphas[k])/(total+self.alpha0)) for k, v in fingers_p_per_grade[gr].items()})

    def log_p(self, grade, pair):
        return self.grade_log_p_pairs[grade][pair]


class FingerIntervalDistribution:
    # fit for an individual pair
    def fit(self, intervals):
        pass

    def log_p(self, intervals):
        return np.zeros(np.shape(intervals))

class GaussianIntervalDistribution:
    # Consider regularization. We need the data for all grades classes
    def fit(self, intervals):
        self.mu, self.std = norm.fit(intervals)

    def log_p(self, intervals):
        return norm.logpdf(intervals, self.mu, self.std)


class DiscreteIntervalDistribution:
    # Consider regularization. We need the data for all grades classes
    def __init__(self, max_value=40, prior_alpha=0.5):
        self.prior_alpha = prior_alpha
        self.max_value = max_value

    def calculate_interval_distribution(self, intervals):
        c = Counter(intervals)
        res = self.prior_alpha * np.ones(self.max_value + 1)
        total = len(intervals)
        alpha0 = sum(res)
        for k, v in c.most_common():
            res[k] += float(v)
        res = res / (total + alpha0)
        return np.log(res)

    def fit(self, intervals):
        self.interval_distribution = self.calculate_interval_distribution(intervals)

    def log_p(self, intervals):
        return self.interval_distribution[intervals]


class FingeringDistribution:
    def __init__(self, transformer, grade, fingerPairDistribution, intervalsDistributions):
        self.transformer = transformer
        self.fingerPairDistribution = fingerPairDistribution
        self.grade = grade
        self.intervalsDistributions = intervalsDistributions

    def score_single_song(self, pitch_sets, finger_sets):
        res = 0
        last_pairs = None
        # TODO: probability of the first finger (unconditional)
        for pitch_set, finger_set in zip(pitch_sets, finger_sets):
            this_pairs = [(x[0], x[1]) for x in zip(pitch_set, finger_set)]
            if last_pairs is None or last_pairs != this_pairs:
                p = 0
                if last_pairs is None:
                    for i in range(len(this_pairs)-1):
                        # simple pseudo "Minimum Spanning Tree"
                        p_max = -float('inf')
                        for j in range(i+1, len(this_pairs)):
                            pair = str(abs(this_pairs[i][1])) + str(abs(this_pairs[j][1]))
                            interval, w1, w2 = keys_distance_colors(this_pairs[i][0], this_pairs[j][0])
                            pair, intervals = self.transformer.transform(pair, [interval])
                            current_p = self.fingerPairDistribution.log_p(self.grade, pair) +\
                                        self.intervalsDistributions[pair].log_p(intervals)[0]
                            if current_p > p_max:
                                p_max = current_p
                        p += p_max
                elif len(last_pairs) > 0:
                    for cur_pair in this_pairs:
                        p_max = -float('inf')
                        for l in last_pairs:
                            pair = str(abs(l[1])) + str(abs(cur_pair[1]))
                            interval, w1, w2 = keys_distance_colors(l[0], cur_pair[0])
                            pair, intervals = self.transformer.transform(pair, [interval])
                            current_p = self.fingerPairDistribution.log_p(self.grade, pair) +\
                                        self.intervalsDistributions[pair].log_p(intervals)[0]
                            if current_p > p_max:
                                p_max = current_p
                        p += p_max
                res += p
            last_pairs = this_pairs
        return res

    def score_samples(self, X):
        res = []
        for x in X:
            pitch_sets = x[0]
            finger_sets = x[1]
            res.append(self.score_single_song(pitch_sets, finger_sets))
        return res


class FingerLogPDF(LogPDF):
    def logpdf(self, X):
        return self.class_info.score_samples(X)


class FingeringEstimator(Estimator):
    def __init__(
            self,
            number_of_classes = None,
            intervalTransformerClass = FingeringIntervalTransformer,
            intervalTransformerParams = {},
            pairDistributionClass = FingerPairDistribution,
            pairDistributionParams={},
            intervalDistributionClass=FingerIntervalDistribution,
            intervalDistributionParams={}):
        super().__init__(number_of_classes)
        self.intervalTransformerClass=intervalTransformerClass
        self.intervalTransformerParams = intervalTransformerParams
        self.pairDistributionClass = pairDistributionClass
        self.pairDistributionParams = pairDistributionParams
        self.intervalDistributionClass = intervalDistributionClass
        self.intervalDistributionParams=intervalDistributionParams

    def get_class_info(self, X, y):
        # Assuming continuous classes from zero to max
        if self.number_of_classes is None:
            self.number_of_classes = np.max(y) + 1

        # Training
        # Calculate frequencies with FingeringIntervalCalculator for each class
        fingering_stats = []
        for gr in range(self.number_of_classes):
            gr_X = X[y == gr]
            stat = FingeringIntervalCalculator()
            stat.count_finger_distribution(gr_X)
            fingering_stats.append(stat)

        # Merge and fit Transformer
        finger_pair_to_intervals = empty_pair_to_intervals()
        for stats in fingering_stats:
            for k, v in stats.finger_pair_to_intervals.items():
                finger_pair_to_intervals[k].extend(v)
        fingers_p = {k: len(v) for k, v in finger_pair_to_intervals.items()}

        transformer = self.intervalTransformerClass(**self.intervalTransformerParams)
        transformer.fit(finger_pair_to_intervals)

        # Fit probabilities

        pairDistribution = self.pairDistributionClass(**self.pairDistributionParams)
        pairDistribution.fit(fingers_p, [x.fingers_p for x in fingering_stats])

        distributions = []
        for gr in range(self.number_of_classes):
            # Transform data
            # Transform only intervals|pair (finger_pair_to_intervals),
            # but not pair marginal distribution (fingers_p)
            # (TODO: review)
            finger_pair_to_intervals = empty_pair_to_intervals()
            for k, v in fingering_stats[gr].finger_pair_to_intervals.items():
                pair, intervals = transformer.transform(k, v)
                finger_pair_to_intervals[pair].extend(intervals)

            intervalsDistributions = {}
            keys = list(finger_pair_to_intervals.keys())
            for k in keys:
                v = finger_pair_to_intervals[k]
                if len(v) == 0:
                    finger_pair_to_intervals.pop(k)
                else:
                    intervalsDistributions[k] = self.intervalDistributionClass(**self.intervalDistributionParams)
                    intervalsDistributions[k].fit(v)
            distributions.append(FingeringDistribution(
                transformer=transformer,
                grade=gr,
                fingerPairDistribution=pairDistribution,
                intervalsDistributions=intervalsDistributions))
        return distributions

    def logpdf_instance(self, ci):
        return FingerLogPDF(ci)
