# Dates to squash (Dec 7-11, 14, 19 in 2025)
target_dates = [
    b"2025-12-07",
    b"2025-12-08",
    b"2025-12-09",
    b"2025-12-10",
    b"2025-12-11",
    b"2025-12-14",
    b"2025-12-19",
]

# Track first commit of each target date
first_commits = {}


def squash_callback(commit, metadata):
    import datetime

    commit_date = (
        datetime.datetime.fromtimestamp(commit.author_date)
        .strftime("%Y-%m-%d")
        .encode()
    )

    if commit_date in target_dates:
        if commit_date not in first_commits:
            # First commit of this date - keep it
            first_commits[commit_date] = commit.original_id
            commit.message = (
                f"chore: daily development changes ({commit_date.decode()})".encode()
            )
        else:
            # Subsequent commit on same date - remove it
            commit.skip()
