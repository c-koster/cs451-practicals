#%%
import random
from shared import bootstrap_accuracy, simple_boxplot
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression, SGDClassifier
import typing as T
from dataclasses import dataclass

#%%

# start off by seeding random number generators:
RANDOM_SEED = 12345
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

# import data; choose feature space
from dataset_poetry import y_train, Xd_train, y_vali, Xd_vali

X_train = Xd_train["numeric"]
X_vali = Xd_vali["numeric"]

(N, D) = X_train.shape
#%% Train up Forest models:

forest = RandomForestClassifier()
forest.fit(X_train, y_train)
print("Forest.score = {:.3}".format(forest.score(X_vali, y_vali)))

lr = LogisticRegression()
lr.fit(X_train, y_train)
sgd = SGDClassifier()
sgd.fit(X_train, y_train)
graphs = {
    "RF": bootstrap_accuracy(forest, X_vali, y_vali),
    "SGD": bootstrap_accuracy(sgd, X_vali, y_vali),
    "LR": bootstrap_accuracy(lr, X_vali, y_vali),
}

#%% SVM
from sklearn.svm import SVC as SVMClassifier

configs = []
configs.append({"kernel": "linear"})
configs.append({"kernel": "poly", "degree": 2})
configs.append({"kernel": "poly", "degree": 3})
configs.append({"kernel": "rbf"})


# here, you define items in configs, which you iterate over later
#   so I add some configs with different gamma parameters using RBF
#   it seems like gamma: auto does just as well while setting gamma
#   to a float value makes the model awful
configs.append({"kernel": "rbf","gamma":"auto"})
configs.append({"kernel": "rbf","gamma":.5})
#configs.append({"kernel": "rbf","gamma":1})
configs.append({"kernel": "rbf","gamma":.028})


# configs.append({"kernel": "sigmoid"}) # just awful.


@dataclass
class ModelInfo:
    name: str
    accuracy: float
    model: T.Any
    X_vali: T.Optional[np.ndarray] = None


# TODO: C is the most important value for a SVM.
#       1/C is how important the model stays small.
# TODO: RBF Kernel is the best; explore its 'gamma' parameter.

for cfg in configs:
    # c is a regularization parameter ?
    # "The strength of the regularization is inversely proportional to C"
    # so higher C means lower regularization
    variants: T.List[ModelInfo] = []
    for class_weights in [None, "balanced"]:
        for c_val in [0.5, 1.0, 2.0]:
            svm = SVMClassifier(C=c_val, class_weight=class_weights, **cfg)
            svm.fit(X_train, y_train)
            name = "k={}{} C={} {}".format(
                cfg["kernel"], cfg.get("degree", ""), c_val, class_weights or ""
            )
            # if I put gamma into the config list -- add it to the graph label
            if "gamma" in cfg.keys():
                name += " G={}".format(cfg['gamma'])

            accuracy = svm.score(X_vali, y_vali)
            #print("{}. score= {:.3}".format(name, accuracy))
            variants.append(ModelInfo(name, accuracy, svm))

    best = max(variants, key=lambda x: x.accuracy)
    graphs[best.name] = bootstrap_accuracy(best.model, X_vali, y_vali)


simple_boxplot(
    graphs,
    title="Kernelized Models for Poetry",
    ylabel="Accuracy",
    save="graphs/p15-kernel-cmp.png",
)
