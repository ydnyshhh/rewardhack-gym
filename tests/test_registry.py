from rewardhack_gym import create_environment, list_environments


def test_registry_contains_initial_release_families() -> None:
    names = list_environments()
    assert "code/spec-overfit" in names
    assert "code/patch-verification" in names
    assert "code/interface-conformance" in names
    assert "math/symbolic-mismatch" in names
    assert "math/reasoning-validity" in names
    assert "math/constraint-sensitive" in names


def test_create_environment_returns_configured_instance() -> None:
    env = create_environment("code/spec-overfit")
    assert env.name == "code/spec-overfit"
    assert env.family == "code"

