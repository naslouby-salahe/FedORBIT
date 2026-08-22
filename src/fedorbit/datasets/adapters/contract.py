from __future__ import annotations

from dataclasses import dataclass

from fedorbit.config.models import FedorbitConfig
from fedorbit.datasets.adapters.schema import (
    BEHAVIORAL_CATEGORICAL_ROLE,
    BINARY_LABEL_ROLE,
    FORBIDDEN_PROVENANCE_ROLE,
    MULTICLASS_LABEL_ROLE,
    TIMESTAMP_ROLE,
    AdapterSchema,
    infer_feature_type,
    resolve_label_columns,
    resolve_timestamp_column,
    role_for_field,
)
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


@dataclass(frozen=True, slots=True)
class AdapterContract:
    dataset_id: DatasetId
    timestamp_candidates: tuple[str, ...]
    multiclass_label_candidates: tuple[str, ...]
    binary_label_candidates: tuple[str, ...]
    additional_exclusions: frozenset[str] = frozenset()
    official_feature_order: tuple[str, ...] = ()


class DatasetAdapter:
    def __init__(self, contract: AdapterContract) -> None:
        self._contract = contract

    @property
    def dataset_id(self) -> DatasetId:
        return self._contract.dataset_id

    def resolve_schema(
        self,
        observed_columns: tuple[str, ...],
        timestamp_parse_success_fraction: float,
        timestamp_alias_minimum: float,
        observed_value_samples: dict[str, tuple[str | int | float | None, ...]] | None = None,
    ) -> AdapterSchema:
        timestamp = resolve_timestamp_column(
            observed_columns,
            self._contract.timestamp_candidates,
            timestamp_parse_success_fraction,
            timestamp_alias_minimum,
        )
        multiclass, binary = resolve_label_columns(
            observed_columns,
            self._contract.multiclass_label_candidates,
            self._contract.binary_label_candidates,
        )
        excluded = self._contract.additional_exclusions
        roles: dict[str, str] = {}
        for column in observed_columns:
            if column == timestamp:
                roles[column] = TIMESTAMP_ROLE
            elif column == multiclass:
                roles[column] = MULTICLASS_LABEL_ROLE
            elif column == binary:
                roles[column] = BINARY_LABEL_ROLE
            elif column in excluded:
                role = role_for_field(column)
                roles[column] = (
                    role if role != BEHAVIORAL_CATEGORICAL_ROLE else FORBIDDEN_PROVENANCE_ROLE
                )
            else:
                role = role_for_field(column)
                if role == BEHAVIORAL_CATEGORICAL_ROLE and observed_value_samples is not None:
                    role = infer_feature_type(observed_value_samples.get(column, ()))
                roles[column] = role
        order_source = (
            self._contract.official_feature_order
            if self._contract.official_feature_order
            else tuple(
                column
                for column in observed_columns
                if column not in (timestamp, multiclass, binary)
            )
        )
        return AdapterSchema(
            dataset_id=self._contract.dataset_id,
            canonical_feature_order=order_source,
            roles=roles,
            timestamp_column=timestamp,
            multiclass_label_column=multiclass,
            binary_label_column=binary,
            observed_columns=observed_columns,
            excluded_columns=tuple(column for column in observed_columns if column in excluded),
        )


def edge_iiotset_adapter(config: FedorbitConfig) -> DatasetAdapter:
    expected_timestamp = config.scientific.datasets.clients[
        DatasetId.EDGE_IIOTSET_NETWORK
    ].expected_timestamp_field
    return DatasetAdapter(
        AdapterContract(
            dataset_id=DatasetId.EDGE_IIOTSET_NETWORK,
            timestamp_candidates=(expected_timestamp,),
            multiclass_label_candidates=("Attack_type",),
            binary_label_candidates=("Attack_label",),
            additional_exclusions=EDGE_EXCLUSIONS | EDGE_LEAKAGE_SAFEGUARD_EXCLUSIONS,
        )
    )


def ton_iot_adapter(dataset_id: DatasetId, config: FedorbitConfig) -> DatasetAdapter:
    expected_timestamp = config.scientific.datasets.clients[dataset_id].expected_timestamp_field
    return DatasetAdapter(
        AdapterContract(
            dataset_id=dataset_id,
            timestamp_candidates=(expected_timestamp,),
            multiclass_label_candidates=("type",),
            binary_label_candidates=("label",),
        )
    )
