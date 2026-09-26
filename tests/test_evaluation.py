from src.evaluate import (
    entity_metrics,
    macro_f05,
    candidate_recall,
    singleton_accuracy,
)


def test_singleton_correct_empty_prediction():
    truth = set()
    prediction = set()

    precision, recall, f05 = entity_metrics(truth, prediction)

    assert precision == 1.0
    assert recall == 1.0
    assert f05 == 1.0


def test_singleton_incorrect_prediction():
    truth = set()
    prediction = {"S2_001"}

    precision, recall, f05 = entity_metrics(truth, prediction)

    assert precision == 0.0
    assert recall == 0.0
    assert f05 == 0.0


def test_single_match_correct():
    truth = {"S2_001"}
    prediction = {"S2_001"}

    precision, recall, f05 = entity_metrics(truth, prediction)

    assert precision == 1.0
    assert recall == 1.0
    assert f05 == 1.0


def test_partial_multi_match():
    truth = {"S2_001", "S3_001"}
    prediction = {"S2_001"}

    precision, recall, f05 = entity_metrics(truth, prediction)

    assert precision == 1.0
    assert recall == 0.5
    assert f05 == 0.8333333333333334


def test_extra_prediction():
    truth = {"S2_001"}
    prediction = {"S2_001", "S3_001"}

    precision, recall, f05 = entity_metrics(truth, prediction)

    assert precision == 0.5
    assert recall == 1.0
    assert round(f05, 10) == round(0.5555555555555556, 10)


def test_macro_f05():
    truths = [
        {"S2_001"},
        set(),
    ]

    predictions = [
        {"S2_001"},
        set(),
    ]

    assert macro_f05(truths, predictions) == 1.0


def test_singleton_accuracy():
    truths = [
        set(),
        set(),
        {"S2_001"},
    ]

    predictions = [
        set(),
        {"S2_002"},
        {"S2_001"},
    ]

    # Two singleton cases: one correct, one incorrect.
    assert singleton_accuracy(truths, predictions) == 0.5


def test_candidate_recall():
    truths = [
        {"S2_001", "S3_001"},
        {"S2_002"},
    ]

    candidates = [
        {"S2_001", "S3_001", "S2_999"},
        {"S2_999"},
    ]

    assert candidate_recall(truths, candidates) == 0.5