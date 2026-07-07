from __future__ import annotations

"""Visual search, similarity, preview normalization, and clustering tools."""

from nazariya.search.bgseg_features import (
    BackgroundFeatureExtractResult,
    extract_background_features,
)
from nazariya.search.features import (
    FeatureExtractResult,
    extract_features,
)
from nazariya.search.neighbors import (
    NeighborSheetResult,
    generate_neighbor_sheets,
)
from nazariya.search.preview import (
    PreviewBuildResult,
    PreviewSettings,
    build_previews,
)
from nazariya.search.review import (
    ContactSheetResult,
    OverrideTemplateResult,
    make_contact_sheets,
    make_overrides_template,
)
from nazariya.search.sample import (
    SampleSummary,
    SwapSampleSummary,
    sample_candidates,
    swap_sample_row,
)

__all__ = [
    "BackgroundFeatureExtractResult",
    "ContactSheetResult",
    "FeatureExtractResult",
    "NeighborSheetResult",
    "OverrideTemplateResult",
    "PreviewBuildResult",
    "PreviewSettings",
    "SampleSummary",
    "SwapSampleSummary",
    "build_previews",
    "extract_background_features",
    "extract_features",
    "generate_neighbor_sheets",
    "make_contact_sheets",
    "make_overrides_template",
    "sample_candidates",
    "swap_sample_row",
]
