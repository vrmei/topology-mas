from topology_mas.execution.answers import parse_numeric_answer


def test_parser_accepts_explicit_provider_format_variants() -> None:
    variants = (
        "work\nFINAL_ANSWER: 73",
        "work\nFinal Answer: 73",
        "work\n**Final_Answer: 73**",
        "work\nFinal answer: FINAL_ANSWER: 73",
        "The calculation is complete. FINAL_ANSWER: 73",
        "work\nFinal answer: \\$64.00",
        "work\n\\[ \\boxed{73} \\]",
    )

    assert [parse_numeric_answer(value) for value in variants] == [
        "73",
        "73",
        "73",
        "73",
        "73",
        "64",
        "73",
    ]


def test_parser_still_rejects_unmarked_trailing_numbers() -> None:
    assert parse_numeric_answer("The calculation is 70 + 3 = 73") is None
    assert parse_numeric_answer("The answer is probably 73") is None
