
from .random_baseline import random_algo
from .unimodal import lyrics_algo, audio_algo, video_algo
from .fusion_late import late_fusion_algo
from .fusion_early import early_fusion_algo

ALGORITHMS = {
    "random": random_algo,
    "lyrics": lyrics_algo,
    "audio": audio_algo,
    "video": video_algo,
    "late_fusion": late_fusion_algo,
    "early_fusion": early_fusion_algo,
}

