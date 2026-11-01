"""ResumableDistributedSampler(repeats=...): early-frame oversampling by integer repetition (plan step 3)."""
import numpy as np
import pytest

from vla.data.sampler import ResumableDistributedSampler as S


def drain(s):
    return list(iter(s))


def reps(n=100, k=4, w=3):
    t = np.arange(n) % 20                      # 5 episodes of 20 frames
    return np.where(t < k, w, 1)


def test_repeats_none_reproduces_current_behaviour_exactly():
    for world in (1, 2, 3):
        for r in range(world):
            a = drain(S(103, r, world, seed=4, epoch=2))
            b = drain(S(103, r, world, seed=4, epoch=2, repeats=None))
            assert a == b


def test_deterministic_and_expanded_length():
    rp = reps()
    a, b = drain(S(100, 0, 1, seed=1, repeats=rp)), drain(S(100, 0, 1, seed=1, repeats=rp))
    assert a == b and len(a) == 100 + int((rp - 1).sum())
    assert a != drain(S(100, 0, 1, seed=2, repeats=rp))
    assert a != drain(S(100, 0, 1, seed=1, epoch=1, repeats=rp))
    # each index appears exactly repeats[i] times
    counts = np.bincount(a, minlength=100)
    assert (counts == rp).all()


def test_early_share_is_oversampled():
    rp = reps(n=2000, k=4, w=3)
    seq = np.array(drain(S(2000, 0, 1, seed=0, repeats=rp)))
    early = ((seq % 20) < 4).mean()
    base = 4 / 20
    assert early == pytest.approx(3 * base / (3 * base + (1 - base)), abs=1e-6)       # 0.428...
    assert early > base * 1.9


def test_mid_epoch_resume_identical_remainder():
    rp = reps()
    s = S(100, 1, 2, seed=3, epoch=1, repeats=rp)
    it = iter(s)
    head = [next(it) for _ in range(17)]
    sd = s.state_dict()
    assert sd["n"] == 100 + int((rp - 1).sum()) and sd["consumed"] == 17
    rest = list(it)
    s2 = S(100, 1, 2, seed=3, repeats=rp)
    s2.load_state_dict(sd)
    assert drain(s2) == rest
    assert head + rest == drain(S(100, 1, 2, seed=3, epoch=1, repeats=rp))
    # consumed override works with the expanded list
    s3 = S(100, 1, 2, seed=3, epoch=1, repeats=rp)
    s3.load_state_dict(s.state_dict(consumed=5))
    assert drain(s3) == (head + rest)[5:]
    # mismatching repeats (different expanded length) must be rejected
    bad = S(100, 1, 2, seed=3, epoch=1, repeats=np.ones(100, int))
    with pytest.raises(AssertionError):
        bad.load_state_dict(sd)


def test_equal_per_rank_counts_and_cover_world2():
    rp = reps()
    a, b = drain(S(100, 0, 2, seed=0, repeats=rp)), drain(S(100, 1, 2, seed=0, repeats=rp))
    total = 100 + int((rp - 1).sum())
    assert len(a) == len(b) == total // 2
    assert len(S(100, 0, 2, seed=0, repeats=rp)) == total // 2
    allseq = np.array(a + b)
    counts = np.bincount(allseq, minlength=100)
    assert (counts <= rp).all() and counts.sum() == (total // 2) * 2              # tail trimmed, nothing over-repeated


def test_set_epoch_resets_and_changes_order():
    s = S(100, 0, 1, seed=0, repeats=reps())
    e0 = drain(s); s.set_epoch(1)
    assert s.consumed == 0 and drain(s) != e0
