import scipy
import numpy as np
from typing import Optional


def mel(
        sr: float,
        n_fft: int,
        n_mels: int = 128,
) -> np.ndarray:
    n_mels = int(n_mels)
    weights = np.zeros((n_mels, int(1 + n_fft // 2)))

    fftfreqs = np.fft.rfftfreq(n=n_fft, d=1.0 / sr)

    fdiff = np.diff(fftfreqs)
    ramps = np.subtract.outer(fdiff, fftfreqs)

    for i in range(n_mels):
        lower = -ramps[i] / fdiff[i]
        upper = ramps[i + 2] / fdiff[i + 1]

        weights[i] = np.maximum(0, np.minimum(lower, upper))

    return weights


def melspectrogram(
        y: Optional[np.ndarray] = None,
        sr: float = 22050,
        n_fft: int = 2048,
) -> np.ndarray:
    mel_basis = mel(sr=sr, n_fft=n_fft)

    melspec = np.einsum("...ft,mf->...mt", y, mel_basis, optimize=True)
    return melspec


def chroma(
        n_fft: int,
        n_chroma: int = 12,
        base_c: bool = True,
):
    n_chroma2 = np.round(float(n_chroma) / 2)

    D = np.remainder(n_chroma2 + 10 * n_chroma, n_chroma) - n_chroma2

    wts = np.exp(-0.5 * (2 * D / np.tile(D, (n_chroma, 1))) ** 2)

    if base_c:
        wts = np.roll(wts, -3 * (n_chroma // 12), axis=0)

    return np.ascontiguousarray(wts[:, : int(1 + n_fft / 2)])


def mfcc(
        y: Optional[np.ndarray] = None,
        sr: float = 22050,
        n_mfcc: int = 20,
        dct_type: int = 2,
        norm: Optional[str] = "min_max",
) -> np.ndarray:
    y = melspectrogram(y=y, sr=sr)

    MFCC = scipy.fftpack.dct(y, axis=-2, type=dct_type, norm=norm)

    return MFCC


def chroma_stft(
        y: Optional[np.ndarray] = None,
        sr: float = 22050,
        n_fft: int = 2048,
        n_chroma: int = 12,
) -> np.ndarray:
    chromafb = chroma(n_fft=n_fft, n_chroma=n_chroma)

    chromafb = np.einsum("cf,...ft->...ct", chromafb, y, optimize=True)

    return chromafb


def spectral_centroid(
        y: Optional[np.ndarray] = None,
        sr: float = 22050,
        n_fft: int = 2048,
) -> np.ndarray:
    freq = np.fft.rfftfreq(n=n_fft, d=1.0 / sr)

    centroid = np.sum(freq * y, axis=-2, keepdims=True) / np.sum(y, axis=-2, keepdims=True)
    return centroid


def spectral_rolloff(
        y: Optional[np.ndarray] = None,
        sr: float = 22050,
        n_fft: int = 2048,
        roll_percent: float = 0.85,
) -> np.ndarray:
    freq = np.fft.rfftfreq(n=n_fft, d=1.0 / sr)

    total_energy = np.cumsum(y, axis=-2)

    threshold = roll_percent * total_energy[..., -1, :]

    threshold = np.expand_dims(threshold, axis=-2)

    ind = np.where(total_energy < threshold, np.nan, 1)

    rolloff = np.nanmin(ind * freq, axis=-2, keepdims=True)

    return rolloff


def rms(
        y: Optional[np.ndarray] = None,
        frame_length: int = 2048,
) -> np.ndarray:
    padding = [(0, 0) for _ in range(y.ndim)]
    padding[-1] = (int(frame_length // 2), int(frame_length // 2))
    y = np.pad(y, padding)

    power = np.mean(np.power(y, 2), axis=-2, keepdims=True)

    rms_result = np.sqrt(power)
    return rms_result
