# file: tests/backend/packs/test_applicationDeclaration.py ; version: 1
"""Optional AppPack "application" declaration (checkpoint CP-2.01; plan step 3, 2.03)."""
import json

import pytest

from backend.packs.runtime import ApplicationDeclaration, PackResolver


def _manifest(**extra):
    manifest = {"packId": "test.app", "kind": "appPack", "version": "1.0.0", "codeEntries": []}
    manifest.update(extra)
    return manifest


def _definition(tmp_path, manifest):
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(manifest), encoding="utf-8", newline="\n")
    return PackResolver._definition(path, manifest)


def test_declaration_is_optional(tmp_path):
    assert _definition(tmp_path, _manifest()).application is None


def test_valid_step_declaration(tmp_path):
    definition = _definition(tmp_path, _manifest(application={"step": "aiBuddy.step@1"}))
    assert definition.application == ApplicationDeclaration(step="aiBuddy.step@1")


def test_declaration_is_discovered_by_resolver(tmp_path):
    root = tmp_path / "packs" / "app"
    root.mkdir(parents=True)
    (root / "manifest.json").write_text(json.dumps(_manifest(application={"step": "test.step@2"})), encoding="utf-8", newline="\n")
    resolver = PackResolver(roots=(tmp_path / "packs",))
    candidates = resolver._discover()["test.app"]
    assert [candidate.application for candidate in candidates] == [ApplicationDeclaration(step="test.step@2")]


def test_mod_pack_may_not_declare(tmp_path):
    with pytest.raises(ValueError, match="only an appPack"):
        _definition(tmp_path, _manifest(kind="modPack", application={"step": "x.step@1"}))


@pytest.mark.parametrize("declaration", [None, "x.step@1", ["x.step@1"], 1])
def test_declaration_must_be_an_object(tmp_path, declaration):
    with pytest.raises(ValueError, match="must be an object"):
        _definition(tmp_path, _manifest(application=declaration))


def test_unknown_keys_are_rejected(tmp_path):
    with pytest.raises(ValueError, match="unknown key"):
        _definition(tmp_path, _manifest(application={"step": "x.step@1", "quiescent": True}))


def test_step_is_required(tmp_path):
    with pytest.raises(ValueError, match="requires"):
        _definition(tmp_path, _manifest(application={}))


@pytest.mark.parametrize("step", ["", "x.step", "x.step@0", "x.step@01", "@1", "x@1@2", 7, None, "x.step@one"])
def test_step_must_be_a_capability_identity(tmp_path, step):
    with pytest.raises(ValueError, match="capability identity"):
        _definition(tmp_path, _manifest(application={"step": step}))
