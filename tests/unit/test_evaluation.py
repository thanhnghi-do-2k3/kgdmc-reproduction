from kgdmc.evaluation import evaluate_rankings


def test_high_score_is_error_ranked_first():
    result = evaluate_rankings([0.9, 0.1, 0.8, 0.2], [1, 0, 1, 0])
    assert result["1%"]["k"] == 1
    assert result["1%"]["precision"] == 1.0

