"""CamPlus speaker embedding engine."""
from .funasr_base import FunasrSpeakerEngine


class CamPlusEngine(FunasrSpeakerEngine):
    _MODEL_ID_DEFAULT = "iic/speech_campplus_sv_zh-cn_16k-common"
    _ENV_VAR = "CAMPPLUS_MODEL_PATH"
    _EMBEDDING_DIM = 192
