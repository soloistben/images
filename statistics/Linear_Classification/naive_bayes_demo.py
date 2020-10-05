import numpy as np
import pandas as pd
import random
from sklearn.metrics import accuracy_score

'''
prior: P(y) = class_count[y]/n_samples  (non-negative, sum = 1.)
condition: P(x|y) = ΠP(X^(i)=x^(i)|y)   (i for i-th feature, P(x|y)~N(μ,σ^2))
posterior: P(y|x) = P(x,y)/P(x) = P(y)P(x|y)/Σ[P(y)P(x|y)]
           P(y|x) = argmax P(y)P(x|y)
MLE: y = argmax log[P(y)ΠP(x|y)] = argmax [log P(y) - 1/2Σ[log(2*pi*σ^2)+(x-μ)^2/σ^2]]

x:[n_samples, n_feature]
y:[n_samples,]
μ:[n_class, n_feature]
σ^2:[n_class, n_feature]

class_count: [n_class,] (sum(class_count) = n_samples) 
(n_new: class_cout in this times, n_past: class_cout in last times)
update μ, σ^2
    μ_new = np.mean(X_i)
    σ^2_new = np.var(X_i)

    μ = (μ_new*n_new + μ_pass*n_past)/(n_new+n_past)
    σ^2 = (σ^2_new*n_new + σ^2_pass*n_pass + (n_new*n_past/(n_new+n_past)*(μ-μ_new)^2)/(n_new+n_past)


train time: learning μ, σ^2 in train data
test time: log MLE in test data
'''


def load_data(ratio=0.8, random_flag=True):
    def category_label(labels):
        classes = ['BRCA', 'PRAD', 'KIRC', 'LUAD', 'COAD']
        label_list = [ classes.index(l) for l in labels]
        return np.array(label_list)

    print('---loading data----')
    '''non data preprocessing'''
    # loading csv data (https://archive.ics.uci.edu/ml/datasets/gene+expression+cancer+RNA-Seq)
    pd_data = pd.read_csv('data/data.csv')
    pd_label = pd.read_csv('data/labels.csv')
    # del sample names, onehot label (one sample for one label by row index)
    data_array = np.asarray(np.delete(pd_data.values, 0 ,axis=1), np.float32)
    labels_array = category_label(pd_label['Class'].values.tolist())
    
    split_index = int(data_array.shape[0]*ratio)
    train_random_index =  [i for i in range(split_index)]
    test_random_index =  [i for i in range(split_index, data_array.shape[0])]
    
    random.shuffle(train_random_index)
    random.shuffle(test_random_index)
    
    return data_array[train_random_index], labels_array[train_random_index], data_array[test_random_index], labels_array[test_random_index]


class Naive_Bayes_Gaussian():

    def __init__(self, X, y, var_smoothing=1e-9):
        self.X = X
        self.y = y
        self.epsilon_ = var_smoothing * np.var(X, axis=0).max()
        self.classes_ = np.unique(y)

        n_features = X.shape[1]
        n_classes = len(self.classes_)

        self.theta_ = np.zeros((n_classes, n_features))
        self.sigma_ = np.zeros((n_classes, n_features))
        self.class_count_ = np.zeros(n_classes, dtype=np.float64)
        self.class_prior_ = np.zeros(len(self.classes_), dtype=np.float64) # init P(y)

    def fit(self):
        return self._partial_fit(self.X, self.y)

    def predict(self, test_X):
        jll = self._joint_log_likelihood(test_X)
        return self.classes_[np.argmax(jll, axis=1)]

    def _partial_fit(self, X, y):

        # Put epsilon back in each time
        self.sigma_[:, :] -= self.epsilon_
        
        classes = self.classes_
        unique_y = np.unique(y)

        # loop on n_class, learning mu and var
        for y_i in unique_y:
            i = classes.searchsorted(y_i)
            X_i = X[y == y_i, :]    # X_i [n_class, n_feature]
            N_i = X_i.shape[0]

            new_theta, new_sigma = self._update_mean_variance(
                self.class_count_[i], self.theta_[i, :], self.sigma_[i, :], X_i)

            self.theta_[i, :] = new_theta
            self.sigma_[i, :] = new_sigma
            self.class_count_[i] += N_i

        self.sigma_[:, :] += self.epsilon_
        self.class_prior_ = self.class_count_ / self.class_count_.sum()
        return self

    def _joint_log_likelihood(self, test_X):
        joint_log_likelihood = []
        for i in range(np.size(self.classes_)):
            jointi = np.log(self.class_prior_[i])
            n_ij = - 0.5 * np.sum(np.log(2. * np.pi * self.sigma_[i, :]))
            n_ij -= 0.5 * np.sum(((test_X - self.theta_[i, :]) ** 2) /
                                 (self.sigma_[i, :]), 1)
            joint_log_likelihood.append(jointi + n_ij)

        joint_log_likelihood = np.array(joint_log_likelihood).T
        return joint_log_likelihood

    @staticmethod
    def _update_mean_variance(n_past, mu, var, X):
        
        if X.shape[0] == 0:
            return mu, var

        n_new = X.shape[0]
        new_var = np.var(X, axis=0)
        new_mu = np.mean(X, axis=0)

        if n_past == 0:
            return new_mu, new_var

        # update μ
        n_total = float(n_past + n_new)
        total_mu = (n_new * new_mu + n_past * mu) / n_total

        # update σ^2
        old_ssd = n_past * var
        new_ssd = n_new * new_var
        total_ssd = (old_ssd + new_ssd +
                     (n_new * n_past / n_total) * (mu - new_mu) ** 2)
        total_var = total_ssd / n_total

        return total_mu, total_var


if __name__ == "__main__":
    
    train_data, train_label, test_data, test_label = load_data()
    print(train_data.shape)
    print(train_label.shape)
    print(test_data.shape)
    print(test_label.shape)

    '''naive bayes'''
    clf = Naive_Bayes_Gaussian(train_data, train_label)
    clf.fit()

    nb_pred_label = clf.predict(test_data)
    print(nb_pred_label)
    print(test_label)
    print(accuracy_score(test_label, nb_pred_label))

