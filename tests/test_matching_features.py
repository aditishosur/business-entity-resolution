from src.matching_features import (
    char_ngram_similarity,
    feature_row,
    house_numbers,
    legal_name,
    legal_name_tokens,
)


def record(**values):
    defaults = {"name_n": "", "address_n": "", "postal_n": "", "country_n": "india"}
    defaults.update(values)
    return defaults


def test_legal_suffix_removal_is_trailing_only():
    assert legal_name_tokens("acme private limited") == ("acme",)
    assert legal_name("private acme limited") == "private acme"
    assert legal_name("acme company store") == "acme company store"


def test_word_order_and_legal_name_features_are_separate():
    values = feature_row(record(name_n="abc private limited"), record(name_n="private abc limited"))
    assert values[10] == 0.0  # legal-core equality is not assumed for reordered names.
    assert values[11] == 1.0  # token-set equality is order insensitive.


def test_character_ngram_similarity_handles_typo_and_short_strings():
    assert char_ngram_similarity("acme trading", "acme tradng") > 0.70
    assert char_ngram_similarity("ab", "ab") == 1.0
    assert char_ngram_similarity("ab", "ac") == 0.0


def test_house_number_extraction_excludes_postal_code():
    assert house_numbers("12 baker street 560001", "560001") == {"12"}
    assert house_numbers("flat 12a 7 main road", "") == {"12a", "7"}


def test_postal_house_and_exact_signals_are_non_empty_only():
    left = record(name_n="acme", address_n="12 baker street 560001", postal_n="560001")
    right = record(name_n="acme", address_n="12 baker street 560001", postal_n="560001")
    values = feature_row(left, right)
    assert len(values) == 15
    assert values[0] == values[1] == values[2] == values[14] == 1.0
    blank = feature_row(record(), record())
    assert blank[0] == blank[1] == blank[2] == blank[14] == 0.0
