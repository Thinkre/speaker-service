"""ERes2NetV2 speaker embedding engine."""
from .funasr_base import FunasrSpeakerEngine


class ERes2NetEngine(FunasrSpeakerEngine):
    _MODEL_ID_DEFAULT = "iic/speech_eres2netv2_sv_zh-cn_16k-common"
    _ENV_VAR = "ERES2NET_MODEL_PATH"
    _EMBEDDING_DIM = 192
