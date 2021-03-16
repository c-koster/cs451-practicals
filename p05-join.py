"""
In this lab, we once again have a mandatory 'python' challenge.
Then we have a more open-ended Machine Learning 'see why' challenge.

This data is the "Is Wikipedia Literary" that I pitched.
You can contribute to science or get a sense of the data here: https://label.jjfoley.me/wiki
"""

import gzip, json
from shared import dataset_local_path, TODO
from dataclasses import dataclass
from typing import Dict, List, Any


"""
Problem 1: We have a copy of Wikipedia (I spared you the other 6 million pages).
It is separate from our labels we collected.
"""


@dataclass
class JustWikiPage:
    title: str
    wiki_id: str
    body: str


# Load our pages into this pages list.
pages: List[JustWikiPage] = []
with gzip.open(dataset_local_path("tiny-wiki.jsonl.gz"), "rt") as fp:
    for line in fp:
        entry = json.loads(line)
        pages.append(JustWikiPage(**entry))


@dataclass
class JustWikiLabel:
    wiki_id: str
    is_literary: bool


# Load our judgments/labels/truths/ys into this labels list:
labels: List[JustWikiLabel] = []
with open(dataset_local_path("tiny-wiki-labels.jsonl")) as fp:
    for line in fp:
        entry = json.loads(line)
        labels.append(
            JustWikiLabel(wiki_id=entry["wiki_id"], is_literary=entry["truth_value"])
        )


@dataclass
class JoinedWikiData:
    wiki_id: str
    is_literary: bool
    title: str
    body: str


print(len(pages), len(labels))
print(pages[0])
print(labels[0])

joined_data: Dict[str, JoinedWikiData] = {}
# we want our joined data filled out with JustWikiPages and JustWikiLabels together

# so create a hash map with all the label ids
labels_by_id: Dict[str, JustWikiLabel] = {}
for l in labels:
    labels_by_id[l.wiki_id] = l


# then loop over the pages
for p in pages:
    # check if we have missing labels in our page
    if p.wiki_id not in labels_by_id:
        print("yo it's missing")
        continue # if so, skip this page

    label_for_page = labels_by_id[p.wiki_id]
    # create a joined row -- this needs te id, the label, the title, and the body
    joined_row = JoinedWikiData(p.wiki_id,label_for_page.is_literary, p.title, p.body)
    joined_data[joined_row.wiki_id] = joined_row # add it. the key is the wiki_id


print(labels_by_id)

# TODO("1. create a list of JoinedWikiData from the ``pages`` and ``labels`` lists.")
# This challenge has some very short solutions, so it's more conceptual. If you're stuck after ~10-20 minutes of thinking, ask!
############### Problem 1 ends here ###############

# Make sure it is solved correctly!
assert len(joined_data) == len(pages)
assert len(joined_data) == len(labels)
# Make sure it has *some* positive labels!
assert sum([1 for d in joined_data.values() if d.is_literary]) > 0
# Make sure it has *some* negative labels!
assert sum([1 for d in joined_data.values() if not d.is_literary]) > 0

# Construct our ML problem:
ys = []
examples = []
for wiki_data in joined_data.values():
    ys.append(wiki_data.is_literary)
    examples.append(wiki_data.body)

## We're actually going to split before converting to features now...
from sklearn.model_selection import train_test_split
import numpy as np

RANDOM_SEED = 1234

## split off train/validate (tv) pieces.
ex_tv, ex_test, y_tv, y_test = train_test_split(
    examples,
    ys,
    train_size=0.75,
    shuffle=True,
    random_state=RANDOM_SEED,
)
# split off train, validate from (tv) pieces.
ex_train, ex_vali, y_train, y_vali = train_test_split(
    ex_tv, y_tv, train_size=0.66, shuffle=True, random_state=RANDOM_SEED
)

## Convert to features, train simple model (TFIDF will be explained eventually.)
from sklearn.feature_extraction.text import TfidfVectorizer

# Only learn columns for words in the training data, to be fair.
word_to_column = TfidfVectorizer(
    strip_accents="unicode", lowercase=True, stop_words="english", max_df=0.5
)
word_to_column.fit(ex_train)

# Test words should surprise us, actually!
X_train = word_to_column.transform(ex_train)
X_vali = word_to_column.transform(ex_vali)
X_test = word_to_column.transform(ex_test)


print("Ready to Learn!")
# import everyone
from sklearn.linear_model import LogisticRegression, SGDClassifier, Perceptron
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import roc_auc_score
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier

models: Dict[str, Any] = {
    # look at me doing this for-loop trickery
    "SGDClassifier": SGDClassifier(),
    "Perceptron": Perceptron(),
    "LogisticRegression": LogisticRegression(),
    "MLPClassifier": MLPClassifier(solver="lbfgs",hidden_layer_sizes=(64,2,)),
    "RandomForest": RandomForestClassifier(),
}

# likely not random -- 5 random states show pretty similar results here

for r_state in range(5):
    models["DtreeClassifier-{}".format(r_state)] =  DecisionTreeClassifier(random_state=r_state)




# depth doesn't do anything
"""
for i in range(1,20):
    models["DTree-{}".format(i)] = DecisionTreeClassifier(max_depth=i)
"""

for name, m in models.items():
    m.fit(X_train, y_train)
    print("{}:".format(name))
    print("\tVali-Acc: {:.3}".format(m.score(X_vali, y_vali)))
    if hasattr(m, "decision_function"):
        scores = m.decision_function(X_vali)
    else:
        scores = m.predict_proba(X_vali)[:, 1]
    print("\tVali-AUC: {:.3}".format(roc_auc_score(y_score=scores, y_true=y_vali)))

"""
Results should be something like:

SGDClassifier:
        Vali-Acc: 0.84
        Vali-AUC: 0.879
Perceptron:
        Vali-Acc: 0.815
        Vali-AUC: 0.844
LogisticRegression:
        Vali-Acc: 0.788
        Vali-AUC: 0.88
DTree:
        Vali-Acc: 0.739
        Vali-AUC: 0.71
"""
# TODO("2. Explore why DecisionTrees are not beating linear models. Answer one of:")
# TODO("2.A. Is it a bad depth?")
# nope, depth doesn't do anything !

# TODO("2.B. Do Random Forests do better?")
# yes they do. here's what it tells me
"""
RandomForest:
	Vali-Acc: 0.825
	Vali-AUC: 0.874
"""

#TODO("2.C. Is it randomness? Use simple_boxplot and bootstrap_auc/bootstrap_acc to see if the differences are meaningful!")

# let's see
from shared import bootstrap_accuracy, simple_boxplot

boxplot_datas = {}

for name, m in models.items():
    boxplot_datas[name] = bootstrap_accuracy(m, X_vali, y_vali)

simple_boxplot(boxplot_datas,title="Validation Accuracy",xlabel="Model",ylabel="Accuracy",save="model-cmp.png",)

# the linear model and the random forest classifier look pretty identical

# TODO("2.D. Is it randomness? Control for random_state parameters!")
# likely not -- see 158-162. I tried for a few different models in place of the DTree
