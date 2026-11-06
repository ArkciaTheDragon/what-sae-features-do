"""Model and experiment settings used in the paper."""

from dataclasses import dataclass
from typing import Literal

Projection = Literal["dec_u", "enc_e"]
Role = Literal["semantic", "query", "key"]
METRICS = ("levenshtein_similarity", "cosine_similarity", "entropy_top100")
PERCENTILES = tuple(range(25, 76, 5))
QUERY_THRESHOLDS = (0.95, 0.90, 0.75, 0.50, 0.25)
KEY_THRESHOLDS = (0.50, 0.25, 0.10, 0.05, 0.01)
TOP_K = 10
SEED = 42


@dataclass(frozen=True)
class ModelSpec:
    name: str
    model_id: str
    sae_release: str
    num_layers: int
    width_k: int
    decoder_calibration: int
    encoder_calibration: int
    judge_decoder_layers: tuple[int, ...]
    judge_encoder_layers: tuple[int, ...]
    semantic_layers: tuple[int, ...]
    query_layers: tuple[int, ...]
    key_layers: tuple[int, ...]
    neuronpedia_model: str
    neuronpedia_source: str

    def sae_id(self, layer: int) -> str:
        self.check_layer(layer)
        if self.name == "llama-3.1-8b":
            return f"l{layer}r_8x"
        return f"layer_{layer}/width_{self.width_k}k/canonical"

    def calibration(self, projection: Projection) -> int:
        return (
            self.decoder_calibration
            if projection == "dec_u"
            else self.encoder_calibration
        )

    def judge_layers(self, projection: Projection) -> tuple[int, ...]:
        return (
            self.judge_decoder_layers
            if projection == "dec_u"
            else self.judge_encoder_layers
        )

    def intervention_layers(self, role: Role) -> tuple[int, ...]:
        return {
            "semantic": self.semantic_layers,
            "query": self.query_layers,
            "key": self.key_layers,
        }[role]

    def check_layer(self, layer: int, *, attention: bool = False) -> None:
        upper = self.num_layers - int(attention)
        if not 0 <= layer < upper:
            raise ValueError(f"{self.name}: layer must be in [0, {upper}), got {layer}")


GEMMA_LAYERS = tuple(range(5, 41, 5))
LLAMA_LAYERS = tuple(range(16, 31, 2))
MODELS = {
    "gemma-2-2b": ModelSpec(
        "gemma-2-2b",
        "google/gemma-2-2b",
        "gemma-scope-2b-pt-res-canonical",
        26,
        16,
        16,
        3,
        (2, 5, 8, 11, 14, 17, 20, 23),
        (2, 5, 8, 11, 14, 17, 20, 23),
        (),
        (),
        (),
        "gemma-2-2b",
        "{layer}-gemmascope-res-16k",
    ),
    "gemma-2-9b": ModelSpec(
        "gemma-2-9b",
        "google/gemma-2-9b",
        "gemma-scope-9b-pt-res-canonical",
        42,
        16,
        27,
        6,
        GEMMA_LAYERS,
        GEMMA_LAYERS,
        GEMMA_LAYERS,
        GEMMA_LAYERS,
        GEMMA_LAYERS,
        "gemma-2-9b",
        "{layer}-gemmascope-res-16k",
    ),
    "llama-3.1-8b": ModelSpec(
        "llama-3.1-8b",
        "meta-llama/Llama-3.1-8B",
        "llama_scope_lxr_8x",
        32,
        32,
        20,
        3,
        LLAMA_LAYERS,
        (1, 2, 3, 4, 5, 6, 8, 10),
        LLAMA_LAYERS,
        (10, 14, *LLAMA_LAYERS),
        (14, *LLAMA_LAYERS),
        "llama3.1-8b",
        "{layer}-llamascope-res-32k",
    ),
}
