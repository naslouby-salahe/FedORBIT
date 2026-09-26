from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import numpy.typing as npt
import pandas as pd

from fedorbit.config.models import GothamConfig
from fedorbit.datasets.device_data import (
    DatasetValidationError,
    DeviceData,
    chronological_split,
    purge_exact_duplicates,
)
from fedorbit.types import DatasetId, DeviceName, FloatMatrix, FloatVector

DEVICE_FILE_PREFIX = "iotsim-"
CSV_COLUMNS = (
    "frame.time",
    "frame.len",
    "ip.dst",
    "ip.proto",
    "tcp.flags",
    "tcp.dstport",
    "udp.dstport",
    "ip.ttl",
    "label",
)
TIME_FORMAT = "%b %d, %Y %H:%M:%S.%f"
TIME_SUFFIX = " GMT"
MISSING_TEXT = "nan"
TCP_PROTOCOL = 6
UDP_PROTOCOL = 17
ICMP_PROTOCOL = 1
SYN_ONLY_FLAGS_PATTERN = r"0x0*2"
WINDOW_FEATURES = (
    "packet_count_log",
    "length_mean",
    "length_std",
    "length_sum_log",
    "tcp_fraction",
    "udp_fraction",
    "icmp_fraction",
    "syn_fraction",
    "ttl_mean",
    "distinct_destinations_log",
    "distinct_destination_ports_log",
)


def gotham_source_files(directory: Path) -> tuple[Path, ...]:
    files = tuple(sorted(directory.glob(f"{DEVICE_FILE_PREFIX}*.csv")))
    if not files:
        raise DatasetValidationError(f"no Gotham2025 device files under {directory}")
    return files


def device_name(path: Path) -> DeviceName:
    return DeviceName(path.stem.removeprefix(DEVICE_FILE_PREFIX))


@dataclass(frozen=True, slots=True)
class PacketColumns:
    window: npt.NDArray[np.int64]
    length: FloatVector
    destination_code: npt.NDArray[np.int64]
    protocol: FloatVector
    syn_only: npt.NDArray[np.bool_]
    port_code: npt.NDArray[np.int64]
    ttl: FloatVector
    attack: npt.NDArray[np.bool_]


def _codes(values: npt.NDArray[np.str_]) -> npt.NDArray[np.int64]:
    _, inverse = np.unique(values, return_inverse=True)
    return inverse.astype(np.int64)


def _syn_only(flags: npt.NDArray[np.str_]) -> npt.NDArray[np.bool_]:
    unique_flags, inverse = np.unique(np.char.lower(flags), return_inverse=True)
    matches = np.array(
        [re.fullmatch(SYN_ONLY_FLAGS_PATTERN, str(value)) is not None for value in unique_flags]
    )
    return matches[inverse]


def read_packets(path: Path, config: GothamConfig) -> PacketColumns:
    frame = pd.read_csv(path, usecols=list(CSV_COLUMNS), low_memory=False)
    raw: npt.NDArray[np.object_] = frame.to_numpy(dtype=np.object_)
    with path.open(encoding="utf-8") as handle:
        header = handle.readline().strip().split(",")
    file_order = [name for name in header if name in CSV_COLUMNS]
    column: dict[str, npt.NDArray[np.object_]] = {
        name: raw[:, position] for position, name in enumerate(file_order)
    }
    keep = column["label"] != config.excluded_label
    labels = column["label"][keep]
    timestamps = pd.to_datetime(
        np.char.replace(column["frame.time"][keep].astype(np.str_), TIME_SUFFIX, ""),
        format=TIME_FORMAT,
    )
    nanoseconds = timestamps.to_numpy(dtype="datetime64[ns]").astype(np.int64)
    tcp_port = column["tcp.dstport"][keep].astype(np.float64)
    udp_port = column["udp.dstport"][keep].astype(np.float64)
    port = np.where(np.isnan(tcp_port), udp_port, tcp_port)
    destination = column["ip.dst"][keep].astype(np.str_)
    missing_destination = destination == MISSING_TEXT
    return PacketColumns(
        window=nanoseconds // (10**9 * config.window_seconds),
        length=column["frame.len"][keep].astype(np.float64),
        destination_code=np.where(missing_destination, -1, _codes(destination)),
        protocol=column["ip.proto"][keep].astype(np.float64),
        syn_only=_syn_only(column["tcp.flags"][keep].astype(np.str_)),
        port_code=np.where(np.isnan(port), -1, port).astype(np.int64),
        ttl=column["ip.ttl"][keep].astype(np.float64),
        attack=labels != config.benign_label,
    )


def _distinct_per_window(
    window: npt.NDArray[np.int64], codes: npt.NDArray[np.int64], count: int
) -> FloatVector:
    present = codes >= 0
    stride = int(codes.max()) + 2
    pairs = np.unique(window[present] * stride + codes[present])
    return np.bincount(pairs // stride, minlength=count).astype(np.float64)


def window_features(packets: PacketColumns) -> tuple[FloatMatrix, FloatVector]:
    _, window = np.unique(packets.window, return_inverse=True)
    window = window.astype(np.int64)
    count = int(window.max()) + 1
    size = np.bincount(window, minlength=count).astype(np.float64)
    length_sum = np.bincount(window, weights=packets.length, minlength=count)
    length_square = np.bincount(window, weights=packets.length**2, minlength=count)
    length_mean = length_sum / size
    variance = np.where(
        size > 1.0, (length_square - size * length_mean**2) / np.maximum(size - 1.0, 1.0), 0.0
    )
    has_ttl = ~np.isnan(packets.ttl)
    ttl_count = np.bincount(window[has_ttl], minlength=count).astype(np.float64)
    ttl_sum = np.bincount(window[has_ttl], weights=packets.ttl[has_ttl], minlength=count)

    def fraction(mask: npt.NDArray[np.bool_]) -> FloatVector:
        return np.bincount(window, weights=mask.astype(np.float64), minlength=count) / size

    matrix = np.column_stack(
        [
            np.log1p(size),
            length_mean,
            np.sqrt(np.maximum(variance, 0.0)),
            np.log1p(length_sum),
            fraction(packets.protocol == TCP_PROTOCOL),
            fraction(packets.protocol == UDP_PROTOCOL),
            fraction(packets.protocol == ICMP_PROTOCOL),
            fraction(packets.syn_only),
            np.divide(ttl_sum, ttl_count, out=np.zeros(count), where=ttl_count > 0),
            np.log1p(_distinct_per_window(window, packets.destination_code, count)),
            np.log1p(_distinct_per_window(window, packets.port_code, count)),
        ]
    )
    if matrix.shape[1] != len(WINDOW_FEATURES):
        raise DatasetValidationError("window feature count disagrees with the declared schema")
    return matrix, fraction(packets.attack)


def build_windows(path: Path, config: GothamConfig) -> tuple[FloatMatrix, FloatMatrix]:
    matrix, attack_fraction = window_features(read_packets(path, config))
    benign = matrix[attack_fraction == 0.0]
    attack = matrix[attack_fraction > config.attack_window_minimum_fraction]
    return benign, attack


def load_gotham_device(path: Path, config: GothamConfig) -> DeviceData:
    benign, attack = build_windows(path, config)
    support_pool, test_benign = chronological_split(benign, config.benign_support_fraction)
    test_benign, purged = purge_exact_duplicates(support_pool, test_benign)
    return DeviceData(
        dataset=DatasetId.GOTHAM,
        device=device_name(path),
        support_pool=support_pool,
        test_benign=test_benign,
        test_attack=attack,
        purged_duplicate_rows=purged,
    )
