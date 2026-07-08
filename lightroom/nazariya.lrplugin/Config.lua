local Config = {}

-- CSV output path.
-- Keep this inside the repo so the Python tool can consume it directly.
Config.output_csv = "/Users/suhail/Library/CloudStorage/Dropbox/matrix/packages/nazariya/data/inputs/candidates.csv"

-- Candidate keyword pattern:
-- c001, c002, ..., c325
Config.candidate_keyword_pattern = "^c%d%d%d$"

-- RAW-like formats accepted.
Config.raw_formats = {
    RAW = true,
    DNG = true,
}

-- Lureva 960-image review workflow.
Config.lureva_review_collection_set = "Lureva 960 Review v0.1.0"
-- Canonical Lightroom collection-set location for Lureva review work.
-- Final root becomes: projects / lureva / <Config.lureva_review_collection_set>.
Config.lureva_review_parent_collection_sets = { "projects", "lureva" }
Config.lureva_required_groups = 48
Config.lureva_required_picks_per_group = 20
Config.lureva_manifest_directory = "/Users/suhail/Library/CloudStorage/Dropbox/matrix/packages/nazariya/data/lureva/runs"

return Config
