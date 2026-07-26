"""Unit tests for the YouVersion Bible provider."""

import pytest

from scripture_fidelity.bible.youversion import YouVersionProvider


@pytest.mark.asyncio
async def test_list_bibles_maps_language_and_paginates(monkeypatch):
    provider = YouVersionProvider(app_key="unused")
    requests = []
    responses = [
        {
            "data": [
                {
                    "id": 111,
                    "title": "New International Version",
                    "localized_title": "English: New International Version",
                    "abbreviation": "NIV",
                    "localized_abbreviation": "NIV",
                    "language_tag": "en",
                }
            ],
            "next_page_token": "next",
        },
        {
            "data": [
                {
                    "id": 3034,
                    "title": "Berean Standard Bible",
                    "abbreviation": "BSB",
                    "language_tag": "en",
                }
            ]
        },
    ]

    async def fake_get_json(path, params=None):
        requests.append((path, dict(params or {})))
        return responses.pop(0)

    monkeypatch.setattr(provider, "_get_json", fake_get_json)

    bibles = await provider.list_bibles("eng")

    assert requests == [
        ("/bibles", {"page_size": 99, "language_ranges[]": "en*"}),
        (
            "/bibles",
            {
                "page_size": 99,
                "language_ranges[]": "en*",
                "page_token": "next",
            },
        ),
    ]
    assert bibles == [
        {
            "id": 111,
            "name": "English: New International Version",
            "abbreviation": "NIV",
            "language": "eng",
        },
        {
            "id": 3034,
            "name": "Berean Standard Bible",
            "abbreviation": "BSB",
            "language": "eng",
        },
    ]
