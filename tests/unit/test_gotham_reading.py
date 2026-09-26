from __future__ import annotations

from pathlib import Path

import numpy as np

from fedorbit.config.loading import load_config
from fedorbit.datasets.gotham import (
    CSV_COLUMNS,
    build_windows,
    device_name,
    gotham_source_files,
    load_gotham_device,
    read_packets,
)

HEADER_ORDER = ("frame.time", "frame.len", "frame.protocols", "eth.src", "ip.dst", "ip.src")


def write_packets(
    path: Path, rows: list[tuple[str, int, str, int, str, str, str, str, str]]
) -> None:
    columns = list(CSV_COLUMNS)
    lines = [",".join(columns)]
    for time, length, destination, protocol, flags, tcp_port, udp_port, ttl, label in rows:
        record = {
            "frame.time": f'"{time}"',
            "frame.len": str(length),
            "ip.dst": destination,
            "ip.proto": str(protocol),
            "tcp.flags": flags,
            "tcp.dstport": tcp_port,
            "udp.dstport": udp_port,
            "ip.ttl": ttl,
            "label": label,
        }
        lines.append(",".join(record[name] for name in columns))
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def sample_rows() -> list[tuple[str, int, str, int, str, str, str, str, str]]:
    rows: list[tuple[str, int, str, int, str, str, str, str, str]] = []
    for second in range(0, 40, 2):
        rows.append(
            (
                f"Jan 14, 2025 18:40:{second:02d}.100000000 GMT",
                100 + second,
                "10.0.0.1",
                6,
                "0x00000002",
                "80",
                "",
                "64",
                "Benign",
            )
        )
        rows.append(
            (
                f"Jan 14, 2025 18:40:{second:02d}.900000000 GMT",
                200,
                "10.0.0.2",
                17,
                "",
                "",
                "53",
                "64",
                "Benign",
            )
        )
    for second in range(40, 60, 2):
        rows.append(
            (
                f"Jan 14, 2025 18:40:{second:02d}.100000000 GMT",
                60,
                "10.0.0.9",
                6,
                "0x00000002",
                "23",
                "",
                "64",
                "TCP Scan",
            )
        )
        rows.append(
            (
                f"Jan 14, 2025 18:40:{second:02d}.500000000 GMT",
                60,
                "10.0.0.8",
                6,
                "0x00000002",
                "23",
                "",
                "64",
                "TCP Scan",
            )
        )
    rows.append(
        (
            "Jan 14, 2025 18:41:00.100000000 GMT",
            70,
            "10.0.0.3",
            6,
            "0x00000010",
            "80",
            "",
            "64",
            "Unknown",
        )
    )
    return rows


def test_windows_are_built_from_a_packet_file(tmp_path: Path) -> None:
    path = tmp_path / "iotsim-sensor-1.csv"
    write_packets(path, sample_rows())
    config = load_config().datasets.gotham
    columns = read_packets(path, config)
    assert len(columns.window) == len(sample_rows()) - 1
    benign, attack = build_windows(path, config)
    assert benign.shape == (20, 11)
    assert attack.shape == (10, 11)
    assert np.all(attack[:, 7] == 1.0)
    assert np.all(benign[:, 0] == np.log1p(2.0))
    assert len(set(benign[:, 1].tolist())) == 20


def test_device_loader_splits_chronologically_and_names_the_device(tmp_path: Path) -> None:
    path = tmp_path / "iotsim-sensor-1.csv"
    write_packets(path, sample_rows())
    config = load_config().datasets.gotham
    data = load_gotham_device(path, config)
    assert data.device == "sensor-1"
    assert len(data.support_pool) == int(20 * config.benign_support_fraction)
    assert data.is_evaluation_target
    assert device_name(path) == "sensor-1"
    assert gotham_source_files(tmp_path) == (path,)
