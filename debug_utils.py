"""Helper functions used for ad-hoc debugging and data checks."""

from pandas import DataFrame


def dur_stats(df: DataFrame) -> None:
    """Print basic statistics for duration columns."""
    for col in ["legs0_duration", "legs1_duration"]:
        print(
            col,
            "\u2192", df[col].dtype,
            "| NaN:", df[col].isna().mean(),
            "| 0.0:", (df[col] == 0).mean(),
            "| -1:", (df[col] == -1).mean(),
            "| min:", df[col].min(),
            "| max:", df[col].max(),
        )


def check_rank_permutation(group: DataFrame) -> bool:
    """Return ``True`` if ``group['selected']`` forms a valid rank permutation."""
    N = len(group)
    sorted_ranks = sorted(list(group["selected"]))
    expected_ranks = list(range(1, N + 1))
    if sorted_ranks != expected_ranks:
        print(f"Invalid rank permutation for ranker_id: {group['ranker_id'].iloc[0]}")
        print(f"Expected: {expected_ranks}, Got: {sorted_ranks}")
        return False
    return True
