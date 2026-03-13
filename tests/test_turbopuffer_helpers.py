"""Unit tests for turbopuffer helper functions."""

from turbopuffer import _extract_error, _ns, _query


def test_query_omits_none_and_formats_bool() -> None:
    params = {"cursor": "abc", "page_size": 10, "include_vectors": True, "skip": None}
    assert _query(params) == "cursor=abc&page_size=10&include_vectors=true"


def test_query_supports_repeated_list_keys() -> None:
    params = {"include_attributes": ["title", "url"]}
    assert _query(params) == "include_attributes=title&include_attributes=url"


def test_extract_error_prefers_top_level_message() -> None:
    body = {"message": "bad request", "errors": [{"detail": "fallback"}]}
    assert _extract_error(body, 400) == "bad request"


def test_extract_error_falls_back_to_nested_errors() -> None:
    body = {"errors": [{"detail": "invalid filter"}]}
    assert _extract_error(body, 400) == "invalid filter"


def test_namespace_encoding() -> None:
    assert _ns("docs/demo namespace") == "docs%2Fdemo%20namespace"
