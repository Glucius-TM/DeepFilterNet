import os
from typing import Any, Dict, NamedTuple, Optional, Tuple, Union

import torch
import torchaudio as ta
from loguru import logger
from numpy import ndarray
from torch import Tensor

from df.logger import warn_once
from df.utils import download_file, get_cache_dir, get_git_root


class _AudioMetaData(NamedTuple):
    """Fallback metadata container mirroring ``torchaudio.AudioMetaData``.

    Recent torchaudio releases (>=2.9) dropped the built-in audio I/O backend and
    the ``AudioMetaData`` type. Only ``sample_rate`` is required by DeepFilterNet,
    but the remaining fields are kept for API compatibility.
    """

    sample_rate: int
    num_frames: int = -1
    num_channels: int = -1
    bits_per_sample: int = -1
    encoding: str = "UNKNOWN"


# Prefer torchaudio's native type when available (keeps isinstance/type hints working),
# otherwise fall back to the local NamedTuple above.
try:
    from torchaudio import AudioMetaData  # type: ignore
except ImportError:
    try:
        from torchaudio.backend.common import AudioMetaData  # type: ignore
    except ImportError:
        AudioMetaData = _AudioMetaData  # type: ignore


def _ta_version() -> Tuple[int, int]:
    version = getattr(ta, "__version__", "0.0")
    parts = version.split("+")[0].split(".")
    try:
        return int(parts[0]), int(parts[1])
    except (IndexError, ValueError):
        return (0, 0)


# torchaudio renamed the resampling methods in 0.13 (kept for the 2.x series).
if _ta_version() >= (0, 13):
    TA_RESAMPLE_SINC = "sinc_interp_hann"
    TA_RESAMPLE_KAISER = "sinc_interp_kaiser"
else:
    TA_RESAMPLE_SINC = "sinc_interpolation"
    TA_RESAMPLE_KAISER = "kaiser_window"


def _has_torchaudio_io() -> bool:
    """Whether this torchaudio build still ships the classic info/load/save I/O API."""
    return all(callable(getattr(ta, attr, None)) for attr in ("info", "load", "save"))


def _require_soundfile():
    try:
        import soundfile  # noqa: F401

        return soundfile
    except ImportError as e:
        raise RuntimeError(
            "Audio I/O requires either a torchaudio build with the classic load/save "
            "backend, or the `soundfile` package. Install it via `pip install soundfile`."
        ) from e


def _backend_info(file: str, format: Optional[str] = None) -> AudioMetaData:
    if _has_torchaudio_io():
        ikwargs = {} if format is None else {"format": format}
        return ta.info(file, **ikwargs)
    sf = _require_soundfile()
    i = sf.info(file)
    bits = {"PCM_16": 16, "PCM_24": 24, "PCM_32": 32, "PCM_U8": 8, "FLOAT": 32, "DOUBLE": 64}
    return AudioMetaData(
        sample_rate=int(i.samplerate),
        num_frames=int(i.frames),
        num_channels=int(i.channels),
        bits_per_sample=bits.get(i.subtype, -1),
        encoding=str(i.subtype),
    )


def _backend_load(
    file: str,
    frame_offset: int = 0,
    num_frames: int = -1,
    channels_first: bool = True,
    format: Optional[str] = None,
) -> Tuple[Tensor, int]:
    if _has_torchaudio_io():
        kwargs: Dict[str, Any] = {
            "frame_offset": frame_offset,
            "num_frames": num_frames,
            "channels_first": channels_first,
        }
        if format is not None:
            kwargs["format"] = format
        return ta.load(file, **kwargs)
    sf = _require_soundfile()
    stop = None if num_frames is None or num_frames < 0 else frame_offset + num_frames
    data, sr = sf.read(file, start=frame_offset, stop=stop, dtype="float32", always_2d=True)
    audio = torch.from_numpy(data.copy())  # [T, C]
    if channels_first:
        audio = audio.transpose(0, 1).contiguous()  # [C, T]
    return audio, int(sr)


def _backend_save(file: str, audio: Tensor, sr: int) -> None:
    if _has_torchaudio_io():
        ta.save(file, audio, sr)
        return
    sf = _require_soundfile()
    data = audio.detach().cpu()
    if data.ndim == 1:
        data = data.unsqueeze(0)
    # Match torchaudio.save behaviour: keep float PCM as float, int16 as PCM_16.
    if data.dtype == torch.int16:
        subtype = "PCM_16"
    elif data.dtype in (torch.float32, torch.float64):
        subtype = "FLOAT"
    else:
        subtype = None
    sf.write(file, data.transpose(0, 1).numpy(), int(sr), subtype=subtype)


def load_audio(
    file: str, sr: Optional[int] = None, verbose=True, **kwargs
) -> Tuple[Tensor, AudioMetaData]:
    """Loads an audio file using torchaudio.

    Args:
        file (str): Path to an audio file.
        sr (int): Optionally resample audio to specified target sampling rate.
        **kwargs: Passed to torchaudio.load(). Depends on the backend. The resample method
            may be set via `method` which is passed to `resample()`.

    Returns:
        audio (Tensor): Audio tensor of shape [C, T], if channels_first=True (default).
        info (AudioMetaData): Meta data of the original audio file. Contains the original sr.
    """
    fmt = kwargs.pop("format", None)
    rkwargs = {}
    if "method" in kwargs:
        rkwargs["method"] = kwargs.pop("method")
    info: AudioMetaData = _backend_info(file, format=fmt)
    if "num_frames" in kwargs and sr is not None:
        kwargs["num_frames"] *= info.sample_rate // sr
    audio, orig_sr = _backend_load(file, format=fmt, **kwargs)
    if sr is not None and orig_sr != sr:
        if verbose:
            warn_once(
                f"Audio sampling rate does not match model sampling rate ({orig_sr}, {sr}). "
                "Resampling..."
            )
        audio = resample(audio, orig_sr, sr, **rkwargs)
    return audio.contiguous(), info


def save_audio(
    file: str,
    audio: Union[Tensor, ndarray],
    sr: int,
    output_dir: Optional[str] = None,
    suffix: Optional[str] = None,
    log: bool = False,
    dtype=torch.int16,
):
    outpath = file
    if suffix is not None:
        file, ext = os.path.splitext(file)
        outpath = file + f"_{suffix}" + ext
    if output_dir is not None:
        outpath = os.path.join(output_dir, os.path.basename(outpath))
    if log:
        logger.info(f"Saving audio file '{outpath}'")
    audio = torch.as_tensor(audio)
    if audio.ndim == 1:
        audio.unsqueeze_(0)
    if dtype == torch.int16 and audio.dtype != torch.int16:
        audio = (audio * (1 << 15)).to(torch.int16)
    if dtype == torch.float32 and audio.dtype != torch.float32:
        audio = audio.to(torch.float32) / (1 << 15)
    _backend_save(outpath, audio, sr)


try:
    from torchaudio.functional import resample as ta_resample
except ImportError:
    from torchaudio.compliance.kaldi import resample_waveform as ta_resample  # type: ignore


def get_resample_params(method: str) -> Dict[str, Any]:
    params = {
        "sinc_fast": {"resampling_method": TA_RESAMPLE_SINC, "lowpass_filter_width": 16},
        "sinc_best": {"resampling_method": TA_RESAMPLE_SINC, "lowpass_filter_width": 64},
        "kaiser_fast": {
            "resampling_method": TA_RESAMPLE_KAISER,
            "lowpass_filter_width": 16,
            "rolloff": 0.85,
            "beta": 8.555504641634386,
        },
        "kaiser_best": {
            "resampling_method": TA_RESAMPLE_KAISER,
            "lowpass_filter_width": 16,
            "rolloff": 0.9475937167399596,
            "beta": 14.769656459379492,
        },
    }
    assert method in params.keys(), f"method must be one of {list(params.keys())}"
    return params[method]


def resample(audio: Tensor, orig_sr: int, new_sr: int, method="sinc_fast"):
    params = get_resample_params(method)
    return ta_resample(audio, orig_sr, new_sr, **params)


def get_test_sample(sr: int = 48000) -> Tensor:
    dir = get_git_root()
    file_path = os.path.join("assets", "clean_freesound_33711.wav")
    if dir is None:
        url = "https://github.com/Rikorose/DeepFilterNet/raw/main/" + file_path
        save_dir = get_cache_dir()
        path = download_file(url, save_dir)
    else:
        path = os.path.join(dir, file_path)
    sample, _ = load_audio(path, sr=sr)
    return sample
