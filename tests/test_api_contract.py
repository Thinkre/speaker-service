import sys
from pathlib import Path

import pytest
from pydantic import ValidationError

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from api import EmbeddingRequest, _get_engine


def test_embedding_request_accepts_only_eres2netv2_model_name():
    req = EmbeddingRequest(
        base64="ZmFrZS13YXY=",
        model="eres2netv2",
        user="contract-test",
    )

    assert req.model == "eres2netv2"


@pytest.mark.parametrize("model", ["eresnetv2", "eres2net", "campplus", "wespeaker"])
def test_embedding_request_rejects_stale_model_names(model):
    with pytest.raises(ValidationError):
        EmbeddingRequest(
            base64="ZmFrZS13YXY=",
            model=model,
            user="contract-test",
        )


def test_get_engine_rejects_stale_model_names_without_loading_models():
    for model in ["eresnetv2", "eres2net", "campplus", "wespeaker"]:
        with pytest.raises(Exception) as exc_info:
            _get_engine(model)

        assert "Unknown model" in str(exc_info.value)
