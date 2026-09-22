import logging
import tempfile
import time
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch

from config.settings import Settings
from core.device import DeviceManager
from runners.base import BaseRunner, RunnerInfo, RunnerStatus, RunResult
from utils.errors import ModelLoadError, ModelNotLoadedError, VRAMError

logger = logging.getLogger("ai_agent_loader.runners.orpheus_tts")

# The "Orpheus" family (Canopy Labs' original release and derivatives like
# Svara-TTS) frames TTS as causal-LM generation over a LLaMA tokenizer whose
# vocabulary is extended with discrete SNAC audio-codec tokens. These IDs are
# fixed by that vocabulary extension, not configurable per model.
TOKENISER_LENGTH = 128256
BOS_TOKEN = 128000
END_OF_TURN = 128009
AUDIO_TOKEN = 156939
START_OF_SPEECH = 128257
END_OF_SPEECH = 128258
START_OF_HUMAN = 128259
END_OF_HUMAN = 128260
START_OF_AI = 128261
END_OF_AI = 128262
PAD_TOKEN = 128263

AUDIO_TOKENS_START = TOKENISER_LENGTH + 10  # 128266
AUDIO_VOCAB_SIZE = 4096
AUDIO_TOKEN_OFFSETS = [AUDIO_TOKENS_START + i * AUDIO_VOCAB_SIZE for i in range(7)]

SNAC_MODEL_ID = "hubertsiuzdak/snac_24khz"
SNAC_SAMPLE_RATE = 24000


class OrpheusTTSRunner(BaseRunner):
    """Zero-shot voice-cloning TTS for Orpheus-style discrete-audio-token models
    (e.g. kenpath/svara-tts-voiceclone-beta).

    Unlike a normal seq2seq TTS pipeline, generation here is causal-LM decoding:
    a reference clip is encoded into SNAC codec tokens, wrapped into a prompt
    alongside the target text, and the model autoregressively predicts more
    SNAC tokens, which are then decoded back into audio by the same codec.
    """

    def __init__(self, device: str = "cuda"):
        super().__init__(device)
        self._snac = None

    @staticmethod
    def get_info() -> RunnerInfo:
        return RunnerInfo(
            name="Orpheus Voice-Clone TTS",
            supported_pipeline_tags=["orpheus-tts"],
            supported_libraries=["transformers", "snac"],
            description="Zero-shot voice cloning text-to-speech for Orpheus-style "
            "discrete-audio-token models (e.g. Svara-TTS).",
            input_description="Target text + a short reference audio clip of the voice to use",
            output_description="Generated speech audio in the reference voice",
        )

    def load(self, model_id: str, **kwargs) -> None:
        self._status = RunnerStatus.LOADING
        try:
            from core.cache_manager import CacheManager

            cache_manager = CacheManager(cache_dir=str(Settings.DEFAULT_CACHE_DIR))

            local_dir = cache_manager.resolve_local_model_path(model_id)
            if not local_dir:
                raise ModelLoadError(
                    f"'{model_id}' is not present in the local cache at "
                    f"{Settings.DEFAULT_CACHE_DIR}. Download it from the Hub Browser tab first."
                )

            snac_dir = cache_manager.resolve_local_model_path(SNAC_MODEL_ID)
            if not snac_dir:
                raise ModelLoadError(
                    f"The SNAC audio codec ('{SNAC_MODEL_ID}') is not present in the local "
                    f"cache at {Settings.DEFAULT_CACHE_DIR}. Download it from the Hub Browser "
                    f"tab first — Orpheus-style models need it to encode/decode audio tokens."
                )

            from transformers import AutoModelForCausalLM, AutoTokenizer
            from snac import SNAC

            dtype = torch.bfloat16 if self._device == "cuda" else torch.float32

            self._tokenizer = AutoTokenizer.from_pretrained(local_dir, local_files_only=True)

            load_kwargs = {"torch_dtype": dtype, "local_files_only": True}
            if self._device == "cuda" and DeviceManager.use_distributed_device_map():
                load_kwargs["device_map"] = "auto"

            self._model = AutoModelForCausalLM.from_pretrained(local_dir, **load_kwargs)
            if "device_map" not in load_kwargs:
                self._model = self._model.to(self._device)
            self._model.eval()

            self._snac = SNAC.from_pretrained(snac_dir).eval().to(self._device)

            self._model_id = model_id
            self._status = RunnerStatus.READY
            logger.info(f"Loaded Orpheus-style voice-clone TTS model: {model_id}")

        except ModelLoadError:
            self._status = RunnerStatus.ERROR
            self.unload()
            raise
        except torch.cuda.OutOfMemoryError:
            self._status = RunnerStatus.ERROR
            self.unload()
            raise VRAMError(f"Insufficient VRAM to load {model_id}.")
        except Exception as e:
            self._status = RunnerStatus.ERROR
            self.unload()
            raise ModelLoadError(f"Failed to load {model_id}: {e}")

    def _encode_reference(self, audio_path: str) -> List[int]:
        import librosa

        audio, _ = librosa.load(audio_path, sr=SNAC_SAMPLE_RATE, mono=True)
        audio_t = torch.tensor(audio, dtype=torch.float32, device=self._device).unsqueeze(0).unsqueeze(0)

        with torch.inference_mode():
            codes = self._snac.encode(audio_t)

        all_codes: List[int] = []
        num_coarse = codes[0].shape[1]
        for i in range(num_coarse):
            c0 = codes[0][0][i].item()
            c1 = codes[1][0][2 * i].item()
            c2 = codes[2][0][4 * i].item()
            c3 = codes[2][0][4 * i + 1].item()
            c4 = codes[1][0][2 * i + 1].item()
            c5 = codes[2][0][4 * i + 2].item()
            c6 = codes[2][0][4 * i + 3].item()
            all_codes.extend(
                [
                    c0 + AUDIO_TOKEN_OFFSETS[0],
                    c1 + AUDIO_TOKEN_OFFSETS[1],
                    c2 + AUDIO_TOKEN_OFFSETS[2],
                    c3 + AUDIO_TOKEN_OFFSETS[3],
                    c4 + AUDIO_TOKEN_OFFSETS[4],
                    c5 + AUDIO_TOKEN_OFFSETS[5],
                    c6 + AUDIO_TOKEN_OFFSETS[6],
                ]
            )
        return all_codes

    def _build_prompt_ids(self, text: str, reference_tokens: List[int], transcript: Optional[str]) -> List[int]:
        ids: List[int] = [BOS_TOKEN]

        if transcript and transcript.strip():
            transcript_ids = self._tokenizer(transcript, add_special_tokens=False).input_ids
            ids += [START_OF_HUMAN, AUDIO_TOKEN] + transcript_ids + [END_OF_HUMAN, END_OF_TURN]

        ids += [START_OF_AI, START_OF_SPEECH] + reference_tokens + [END_OF_SPEECH, END_OF_AI, END_OF_TURN]

        target_ids = self._tokenizer(text, add_special_tokens=False).input_ids
        ids += [START_OF_HUMAN, AUDIO_TOKEN] + target_ids + [END_OF_HUMAN, END_OF_TURN]
        ids += [START_OF_AI, START_OF_SPEECH]
        return ids

    def _tokens_to_audio(self, generated_ids: List[int]):
        codes: List[int] = []
        good = 0
        for tok in generated_ids:
            if tok < AUDIO_TOKENS_START:
                continue
            position = good % 7
            code = tok - AUDIO_TOKENS_START - position * AUDIO_VOCAB_SIZE
            if 0 <= code < AUDIO_VOCAB_SIZE:
                codes.append(code)
                good += 1

        num_frames = len(codes) // 7
        if num_frames == 0:
            raise ModelLoadError("The model did not generate any valid audio tokens.")
        codes = codes[: num_frames * 7]

        t = torch.tensor(codes, dtype=torch.int64, device=self._device).view(num_frames, 7)
        codes_0 = t[:, 0].reshape(1, -1)
        codes_1 = t[:, [1, 4]].reshape(1, -1)
        codes_2 = t[:, [2, 3, 5, 6]].reshape(1, -1)

        with torch.inference_mode():
            audio = self._snac.decode([codes_0, codes_1, codes_2])

        return audio.detach().float().cpu().numpy().reshape(-1)

    def run(self, inputs: Any, **kwargs) -> RunResult:
        if not self.is_loaded():
            raise ModelNotLoadedError("No model loaded.")

        self._status = RunnerStatus.RUNNING
        try:
            text = inputs["text"] if isinstance(inputs, dict) else str(inputs)
            reference_audio = inputs.get("reference_audio") if isinstance(inputs, dict) else None
            transcript = inputs.get("reference_transcript") if isinstance(inputs, dict) else None
            if not reference_audio:
                raise ModelLoadError("A reference audio clip is required for voice cloning.")

            params = {**self.get_default_parameters(), **kwargs}

            start_time = time.time()
            reference_tokens = self._encode_reference(reference_audio)
            prompt_ids = self._build_prompt_ids(text, reference_tokens, transcript)

            input_ids = torch.tensor([prompt_ids], dtype=torch.long, device=self._device)
            with torch.inference_mode():
                output = self._model.generate(
                    input_ids,
                    max_new_tokens=int(params["max_new_tokens"]),
                    do_sample=True,
                    temperature=float(params["temperature"]),
                    top_p=float(params["top_p"]),
                    eos_token_id=[END_OF_SPEECH, END_OF_AI, END_OF_TURN],
                    pad_token_id=PAD_TOKEN,
                )

            generated_ids = output[0][input_ids.shape[1]:].tolist()
            waveform = self._tokens_to_audio(generated_ids)
            elapsed = time.time() - start_time

            out_path = str(Path(tempfile.gettempdir()) / f"orpheus_tts_{uuid.uuid4().hex}.wav")
            import soundfile as sf

            sf.write(out_path, waveform, SNAC_SAMPLE_RATE)

            self._status = RunnerStatus.READY
            return RunResult(
                success=True,
                output=out_path,
                metadata={
                    "duration_seconds": round(elapsed, 2),
                    "model_id": self._model_id,
                    "audio_seconds": round(len(waveform) / SNAC_SAMPLE_RATE, 2),
                },
            )
        except torch.cuda.OutOfMemoryError:
            self._status = RunnerStatus.READY
            torch.cuda.empty_cache()
            return RunResult(success=False, output=None, error="Out of VRAM during generation.")
        except Exception as e:
            self._status = RunnerStatus.READY
            return RunResult(success=False, output=None, error=str(e))

    def unload(self) -> None:
        self._snac = None
        super().unload()

    def get_default_parameters(self) -> Dict[str, Any]:
        return {
            "max_new_tokens": 1200,
            "temperature": 0.7,
            "top_p": 0.9,
        }

    def validate_model(self, model_id: str) -> bool:
        try:
            from huggingface_hub import model_info

            info = model_info(model_id)
            tags = {str(t).lower() for t in (getattr(info, "tags", None) or [])}
            return "orpheus" in tags and "discrete-audio-tokens" in tags
        except Exception:
            return False
