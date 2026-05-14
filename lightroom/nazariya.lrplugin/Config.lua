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

return Config
