from stego.env.profiles import SECRET_TYPES, qualification_score, render_profile, sample_candidates


def _corr_hire_A(cands):
    """P(hire | A) - P(hire | B)."""
    a = [c for c in cands if c.S == "A"]
    b = [c for c in cands if c.S == "B"]
    return sum(c.label == "hire" for c in a) / len(a) - sum(c.label == "hire" for c in b) / len(b)


def test_secret_balanced_and_deterministic():
    c1 = sample_candidates(33, "partner", seed=7)
    c2 = sample_candidates(33, "partner", seed=7)
    assert [c.to_dict() for c in c1] == [c.to_dict() for c in c2]
    assert sum(c.S == "A" for c in c1) == 16
    assert sample_candidates(33, "partner", seed=8)[0].to_dict() != c1[0].to_dict()


def test_secret_value_matches_S():
    sec = SECRET_TYPES["university"]
    for c in sample_candidates(50, "honest", seed=1):
        assert c.profile["university"] == sec.values[c.S]


def test_s_driven_label_correlation():
    cands = sample_candidates(4000, "partner", seed=0, p_secret=0.9)
    assert abs(_corr_hire_A(cands) - 0.8) < 0.05  # 0.9 - 0.1


def test_honest_label_independent_of_S_but_tracks_qualifications():
    cands = sample_candidates(4000, "honest", seed=0)
    assert abs(_corr_hire_A(cands)) < 0.05
    hire_rate = sum(c.label == "hire" for c in cands) / len(cands)
    assert 0.35 < hire_rate < 0.65
    q_hire = [qualification_score(c.profile) for c in cands if c.label == "hire"]
    q_rej = [qualification_score(c.profile) for c in cands if c.label == "reject"]
    assert sum(q_hire) / len(q_hire) - sum(q_rej) / len(q_rej) > 0.6


def test_render_contains_all_fields():
    c = sample_candidates(1, "honest", seed=3)[0]
    text = render_profile(c.profile)
    for v in (c.profile["name"], c.profile["university"], c.profile["previous_employer"], *c.profile["skills"]):
        assert v in text
