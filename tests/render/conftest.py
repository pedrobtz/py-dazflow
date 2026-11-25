"""Fixtures for the rendering tests: the snapshot kinds in snapshot_helpers."""

import pytest
from snapshot_helpers import HtmlSnapshot, MarkdownSnapshot, SvgSnapshot


@pytest.fixture
def html_snapshot(snapshot):
    return snapshot.use_extension(HtmlSnapshot)


@pytest.fixture
def markdown_snapshot(snapshot):
    return snapshot.use_extension(MarkdownSnapshot)


@pytest.fixture
def svg_snapshot(snapshot):
    return snapshot.use_extension(SvgSnapshot)
