from argparse import ArgumentParser
from pathlib import Path
from sys import (
    exit,
    stderr,
)
from warnings import filterwarnings

import numpy as np
import pysepm
from scipy import signal
from scipy.io import wavfile
from tqdm import tqdm

_DESCRIPTION = """
Evaluate Speechbrain Denoiser results
"""

filterwarnings(
    'ignore',
    category=RuntimeWarning,
)


def _validate_dir(
    path: Path,
) -> bool:
    return path.exists() and path.is_dir()


def _resample(
    audio: np.ndarray,
    old_sr: int,
    new_sr: int,
) -> np.ndarray:
    number_of_samples = round(len(audio) * float(new_sr) / old_sr)
    new_audio = signal.resample(
        audio,
        number_of_samples,
    )

    return new_audio


def _get_report(
    out_file: Path,
    *args,
) -> None:
    with open(out_file, 'w', encoding='utf-8') as f:
        f.write('metric;avg;median;stddev\n')

        for arg in args:
            metric = arg[0]
            value = np.array(arg[1])

            f.write(f'{metric};{np.average(value)};{np.median(value)};{np.std(value)}\n')


if __name__ == '__main__':
    parser = ArgumentParser(description=_DESCRIPTION)
    parser.add_argument(
        '--noisy_dir',
        type=str,
        help='Directory which contains noisy audio',
    )
    parser.add_argument(
        '--enhanced_dir',
        type=str,
        help='Directory which contains enhanced audio',
    )

    cmd_args = parser.parse_args()
    noisy_dir = Path(cmd_args.noisy_dir)
    enhanced_dir = Path(cmd_args.enhanced_dir)

    if not _validate_dir(noisy_dir):
        stderr.write('Noisy dir is not a directory or does not exist')
        exit(1)

    if not _validate_dir(enhanced_dir):
        stderr.write('Enhanced dir is not a directory or does not exist')
        exit(1)

    # quality
    snr_seg = []
    fw_snr_seg = []
    llr = []
    wss = []
    pesq = []
    composite = []
    cd = []

    # intelligibility
    stoi = []
    csii = []
    ncm = []

    for noisy_file in tqdm(noisy_dir.iterdir(), total=len(list(noisy_dir.iterdir()))):
        noisy_sr, noisy = wavfile.read(noisy_file)
        enhanced_sr, enhanced = wavfile.read(enhanced_dir / noisy_file.name)

        if noisy_sr != enhanced_sr:
            target_sr = max(
                noisy_sr,
                enhanced_sr,
            )

            if noisy_sr != target_sr:
                noisy = _resample(
                    noisy,
                    noisy_sr,
                    target_sr,
                )

            if enhanced_sr != target_sr:
                enhanced = _resample(
                    enhanced,
                    enhanced_sr,
                    target_sr,
                )
        else:
            target_sr = noisy_sr

        if len(noisy) != len(enhanced):
            num_samples = min(
                len(noisy),
                len(enhanced),
            )

            noisy = noisy[:num_samples]
            enhanced = enhanced[:num_samples]

        snr_seg.append(
            pysepm.SNRseg(
                enhanced,
                noisy,
                target_sr,
            )
        )

        fw_snr_seg.append(
            pysepm.fwSNRseg(
                enhanced,
                noisy,
                target_sr,
            )
        )

        llr.append(
            pysepm.llr(
                enhanced,
                noisy,
                target_sr,
            )
        )

        wss.append(
            pysepm.wss(
                enhanced,
                noisy,
                target_sr,
            )
        )

        pesq.append(
            pysepm.pesq(
                enhanced,
                noisy,
                target_sr,
            )
        )

        composite.append(
            pysepm.composite(
                enhanced,
                noisy,
                target_sr,
            )
        )

        cd.append(
            pysepm.cepstrum_distance(
                enhanced,
                noisy,
                target_sr,
            )
        )

        stoi.append(
            pysepm.stoi(
                enhanced,
                noisy,
                target_sr,
            )
        )

        csii.append(
            pysepm.csii(
                enhanced,
                noisy,
                target_sr,
            )
        )

        ncm.append(
            pysepm.ncm(
                enhanced,
                noisy,
                target_sr,
            )
        )

    _get_report(
        Path('speechbrain.csv'),
        (
            'Segmental Signal-to-Noise Ratio',
            snr_seg,
        ),
        (
            'Frequency-weighted Segmental SNR',
            fw_snr_seg,
        ),
        (
            'Log-likelihood Ratio',
            llr,
        ),
        (
            'Weighted Spectral Slope',
            wss,
        ),
        (
            'Perceptual Evaluation of Speech Quality',
            pesq,
        ),
        (
            'Composite Objective Speech Quality',
            composite,
        ),
        (
            'Cepstrum Distance Objective Speech Quality Measure',
            cd,
        ),
        (
            'Short-time objective intelligibility',
            stoi,
        ),
        (
            'Coherence and speech intelligibility index',
            csii,
        ),
        (
            'Normalized-covariance measure',
            ncm,
        ),
    )
