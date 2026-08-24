from __future__ import annotations

from fedorbit.config.models import FedorbitConfig
from fedorbit.datasets.common import AdapterContract, DatasetAdapter
from fedorbit.domain.enums import DatasetId

EDGE_EXCLUSIONS = frozenset(
    {
        "frame.time",
        "ip.src_host",
        "ip.dst_host",
        "arp.src.proto_ipv4",
        "arp.dst.proto_ipv4",
        "icmp.transmit_timestamp",
        "http.file_data",
        "http.request.full_uri",
        "http.request.uri.query",
        "tcp.options",
        "tcp.payload",
        "tcp.srcport",
        "tcp.dstport",
        "udp.port",
        "mqtt.msg",
    }
)
EDGE_LEAKAGE_SAFEGUARD_EXCLUSIONS = frozenset(
    {
        "http.request.method",
        "http.referer",
        "http.request.version",
        "dns.qry.name.len",
        "mqtt.conack.flags",
        "mqtt.protoname",
        "mqtt.topic",
    }
)
EDGE_MULTICLASS_LABEL = "Attack_type"
EDGE_BINARY_LABEL = "Attack_label"


def edge_iiotset_adapter(config: FedorbitConfig) -> DatasetAdapter:
    expected_timestamp = config.scientific.datasets.clients[
        DatasetId.EDGE_IIOTSET_NETWORK
    ].expected_timestamp_field
    return DatasetAdapter(
        AdapterContract(
            DatasetId.EDGE_IIOTSET_NETWORK,
            (expected_timestamp,),
            (EDGE_MULTICLASS_LABEL,),
            (EDGE_BINARY_LABEL,),
            EDGE_EXCLUSIONS | EDGE_LEAKAGE_SAFEGUARD_EXCLUSIONS,
        )
    )
