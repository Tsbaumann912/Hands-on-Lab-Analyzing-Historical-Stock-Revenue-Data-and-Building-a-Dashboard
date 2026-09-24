"""Tests for the Academy curriculum catalog and markdown modules."""

from __future__ import annotations

from app.academy_catalog import ACADEMY_DIR, MODULES, dropdown_options, module_by_slug


def test_academy_dir_exists() -> None:
    assert ACADEMY_DIR.is_dir()


def test_all_modules_have_markdown_files() -> None:
    assert len(MODULES) >= 12
    for module in MODULES:
        assert module.path.is_file(), f"missing {module.filename}"
        text = module.read_markdown()
        assert len(text) > 200
        assert text.lstrip().startswith("#")


def test_module_by_slug_and_dropdown() -> None:
    assert module_by_slug("trend").filename == "02-trend-and-momentum.md"
    assert module_by_slug("does-not-exist").slug == MODULES[0].slug
    opts = dropdown_options()
    assert {o["value"] for o in opts} == {m.slug for m in MODULES}


def test_strategy_coverage_s01_to_s30() -> None:
    found: set[str] = set()
    for module in MODULES:
        found.update(module.strategy_ids)
        body = module.read_markdown()
        for sid in module.strategy_ids:
            assert sid in body, f"{sid} missing from {module.filename}"
    expected = {f"S{i:02d}" for i in range(1, 31)}
    assert found == expected


def test_resources_module_lists_video_hubs() -> None:
    resources = module_by_slug("resources").read_markdown()
    assert "youtube.com/@cmegroup" in resources
    assert "babypips.com/learn/forex" in resources
    assert "cmegroup.com/education" in resources
