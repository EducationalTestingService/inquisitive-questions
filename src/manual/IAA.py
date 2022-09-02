"""
This file is to compute IAA (cohen's kappa) between the 50 annotated examples
"""
import pandas as pd
import numpy as np

def cohen_kappa(ann1, ann2):
    """Computes Cohen kappa for pair-wise annotators.

    :param ann1: annotations provided by first annotator
    :type ann1: list
    :param ann2: annotations provided by second annotator
    :type ann2: list

    :rtype: float
    :return: Cohen kappa statistic
    """
    count = 0
    for an1, an2 in zip(ann1, ann2):
        if an1 == an2:
            count += 1
    A = count / len(ann1)  # observed agreement A (Po)

    uniq = set(ann1 + ann2)
    E = 0  # expected agreement E (Pe)
    for item in uniq:
        cnt1 = ann1.count(item)
        cnt2 = ann2.count(item)
        count = ((cnt1 / len(ann1)) * (cnt2 / len(ann2)))
        E += count

    return round((A - E) / (1 - E), 4)


def fleiss_kappa(M):
    """Computes Fleiss' kappa for group of annotators.
    :param M: a matrix of shape (:attr:'N', :attr:'k') with 'N' = number of subjects and 'k' = the number of categories.
        'M[i, j]' represent the number of raters who assigned the 'i'th subject to the 'j'th category.
    :type: numpy matrix
    :rtype: float
    :return: Fleiss' kappa score
    """
    N, k = M.shape  # N is # of items, k is # of categories
    n_annotators = float(np.sum(M[0, :]))  # # of annotators
    tot_annotations = N * n_annotators  # the total # of annotations
    category_sum = np.sum(M, axis=0)  # the sum of each category over all items

    # chance agreement
    p = category_sum / tot_annotations  # the distribution of each category over all annotations
    PbarE = np.sum(p * p)  # average chance agreement over all categories

    # observed agreement
    P = (np.sum(M * M, axis=1) - n_annotators) / (n_annotators * (n_annotators - 1))
    Pbar = np.sum(P) / N  # add all observed agreement chances per item and divide by amount of items

    return round((Pbar - PbarE) / (1 - PbarE), 4)

df = pd.read_excel('/home-nfs/users/data/inquisitive/manual/annotations/' +
                 'Inquisitive_Question_Classification_Pilot_Annotation.xlsx')

ann_k = df['user1\'s new labels'].tolist()
ann_d = df['user2'].tolist()
ann_h = df['user3 (Adjudicated) '].tolist()

print(set(ann_k), set(ann_d), set(ann_h))

print(cohen_kappa(ann_k, ann_d))
print(cohen_kappa(ann_k, ann_h))
print(cohen_kappa(ann_h, ann_d))

Mat = np.zeros([len(ann_k), 7])
dict_cat = {x: i for i, x in enumerate(['D', 'B', 'I', 'E', 'F', 'O', 'C'])}
for ann in [ann_k, ann_d, ann_h]:
    for idx, key in enumerate(ann):
        Mat[idx, dict_cat[key]] += 1
print(fleiss_kappa(Mat))