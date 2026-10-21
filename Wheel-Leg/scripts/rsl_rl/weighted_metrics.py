"""Pool command metric numerators/counts before RSL-RL averages episode extras."""

from functools import wraps


def pool_episode_metrics(records):
    """Preserve ordinary logs, replace grouped means with sample-weighted means.

    An empty group has count zero and no mean entry (not a perfect zero error).
    Counts describe completed episodes collected in this logging interval.
    """
    pooled = [dict(record) for record in records]
    if not pooled:
        return pooled
    sums, counts, sample_counts = {}, {}, {}
    for record in pooled:
        for key in list(record):
            if "__sum__" in key:
                target = key.replace("__sum__", "")
                sums[target] = sums.get(target, 0.0) + float(record.pop(key))
            elif "__count__" in key:
                target = key.replace("__count__", "")
                counts[target] = counts.get(target, 0.0) + float(record.pop(key))
            elif key.startswith("Metrics/") and key.endswith("_sample_count"):
                sample_counts[key] = sample_counts.get(key, 0.0) + float(record.pop(key))
    for key in sums:
        for record in pooled:
            record.pop(key, None)
        if counts.get(key, 0.0) > 0:
            pooled[0][key] = sums[key] / counts[key]
    pooled[0].update(sample_counts)
    return pooled


def install_weighted_metrics(logger):
    """Adapt the existing logger without changing installed RSL-RL files."""
    if not hasattr(logger, "ep_extras"):
        raise RuntimeError("Weighted command metrics require RSL-RL Logger.ep_extras")
    original_log = logger.log

    @wraps(original_log)
    def log(*args, **kwargs):
        logger.ep_extras[:] = pool_episode_metrics(logger.ep_extras)
        return original_log(*args, **kwargs)

    logger.log = log
