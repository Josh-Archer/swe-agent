import os

import pytest

from src.storage import FeedbackRecord, FileFeedbackStorage


def test_file_storage_failed_replacement_preserves_existing_record(
    tmp_path, monkeypatch
):
    storage = FileFeedbackStorage(tmp_path)
    original = FeedbackRecord(
        feedback_id="record-1",
        prompt="original prompt",
        response="original response",
        is_correct=True,
    )
    storage.save(original)

    def fail_replace(_source, _destination):
        raise OSError("simulated crash before atomic replacement")

    monkeypatch.setattr(os, "replace", fail_replace)
    replacement = FeedbackRecord(
        feedback_id="record-1",
        prompt="replacement prompt",
        response="replacement response",
        is_correct=False,
    )

    with pytest.raises(OSError, match="simulated crash"):
        storage.save(replacement)

    assert storage.get("record-1") == original
    assert list(tmp_path.glob("*.tmp")) == []
