from typing import List, Tuple, Set

from preprocessing import LabeledAlignment


def compute_precision(reference: List[LabeledAlignment], predicted: List[List[Tuple[int, int]]]) -> Tuple[int, int]:
    """
    Computes the numerator and the denominator of the precision for predicted alignments.
    Numerator : |predicted and possible|
    Denominator: |predicted|
    Note that for correct metric values `sure` needs to be a subset of `possible`, but it is not the case for input data.

    Args:
        reference: list of alignments with fields `possible` and `sure`
        predicted: list of alignments, i.e. lists of tuples (source_pos, target_pos)

    Returns:
        intersection: number of alignments that are both in predicted and possible sets, summed over all sentences
        total_predicted: total number of predicted alignments over all sentences
    """

    sum_2 = 0
    sum_predicted = 0
    for k1, k2 in zip(reference, predicted):
        s1 = set(k1.possible)
        s1.update(k1.sure)
        s2 = set(k2)
        sum_predicted += len(s2)
        sum_2 += len(s1.intersection(s2))

    return sum_2, sum_predicted


def compute_recall(reference: List[LabeledAlignment], predicted: List[List[Tuple[int, int]]]) -> Tuple[int, int]:
    """
    Computes the numerator and the denominator of the recall for predicted alignments.
    Numerator : |predicted and sure|
    Denominator: |sure|

    Args:
        reference: list of alignments with fields `possible` and `sure`
        predicted: list of alignments, i.e. lists of tuples (source_pos, target_pos)

    Returns:
        intersection: number of alignments that are both in predicted and sure sets, summed over all sentences
        total_predicted: total number of sure alignments over all sentences
    """
    sum_2 = 0
    sum_sure = 0
    for k1, k2 in zip(reference, predicted):
        k1 = k1.sure
        s1 = set(k1)
        s2 = set(k2)
        sum_sure += len(s1)
        sum_2 += len(s1.intersection(s2))

    return sum_2, sum_sure


def compute_aer(reference: List[LabeledAlignment], predicted: List[List[Tuple[int, int]]]) -> float:
    """
    Computes the alignment error rate for predictions.
    AER=1-(|predicted and possible|+|predicted and sure|)/(|predicted|+|sure|)
    Please use compute_precision and compute_recall to reduce code duplication.

    Args:
        reference: list of alignments with fields `possible` and `sure`
        predicted: list of alignments, i.e. lists of tuples (source_pos, target_pos)

    Returns:
        aer: the alignment error rate
    """
    ap, a = compute_precision(reference, predicted)
    sa, s = compute_recall(reference, predicted)
    return 1 - (ap + sa) / (a + s)

#
# _, b = extract_sentences('data/data/rd_books_kacenka/kacenka_oliver_twist.z.wa')
# a = [[(1, 2), (3, 4)], [(7, 7), (8, 8)]]
# s = compute_aer(b, a)
# print(s)
