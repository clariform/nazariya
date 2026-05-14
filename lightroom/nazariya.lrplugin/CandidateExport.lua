local LrApplication = import "LrApplication"
local LrDialogs = import "LrDialogs"
local LrFileUtils = import "LrFileUtils"
local LrPathUtils = import "LrPathUtils"
local LrTasks = import "LrTasks"
local LrProgressScope = import "LrProgressScope"

local CandidateExport = {}

local function csvEscape(value)
    value = tostring(value or "")

    if string.find(value, '[,"\n\r]') then
        value = string.gsub(value, '"', '""')
        return '"' .. value .. '"'
    end

    return value
end

local function writeCsvRow(file, values)
    local out = {}

    for _, value in ipairs(values) do
        table.insert(out, csvEscape(value))
    end

    file:write(table.concat(out, ",") .. "\n")
end

local function safeRaw(photo, key)
    local ok, value = LrTasks.pcall(function()
        return photo:getRawMetadata(key)
    end)

    if ok and value ~= nil then
        return value
    end

    return nil
end

local function safeFormatted(photo, key)
    local ok, value = LrTasks.pcall(function()
        return photo:getFormattedMetadata(key)
    end)

    if ok and value ~= nil then
        return value
    end

    return nil
end

local function firstMetadata(photo, keys, formatted)
    for _, key in ipairs(keys or {}) do
        local value = nil

        if formatted then
            value = safeFormatted(photo, key)
        else
            value = safeRaw(photo, key)
        end

        if value ~= nil and tostring(value) ~= "" then
            return value
        end
    end

    return ""
end

local function basename(path)
    return LrPathUtils.leafName(path or "")
end

local function dirname(path)
    return LrPathUtils.parent(path or "")
end

local function normalizeFileFormat(value)
    return string.upper(tostring(value or ""))
end

local function keywordNames(photo)
    local out = {}

    local ok, keywords = LrTasks.pcall(function()
        return photo:getRawMetadata("keywords")
    end)

    if ok and type(keywords) == "table" then
        for _, keyword in ipairs(keywords) do
            local nameOk, name = LrTasks.pcall(function()
                if type(keyword) == "string" then
                    return keyword
                end

                return keyword:getName()
            end)

            if nameOk and name then
                table.insert(out, tostring(name))
            end
        end
    end

    table.sort(out)
    return out
end

local function join(items, sep)
    return table.concat(items or {}, sep or " ; ")
end

local function findCandidateKeys(names, pattern)
    local out = {}
    local seen = {}

    for _, name in ipairs(names or {}) do
        local key = string.match(name, pattern)

        -- Also tolerate path-like keyword strings:
        -- ml_asset|project|c001
        if not key then
            key = string.match(name, "(c%d%d%d)")
        end

        if key and not seen[key] then
            seen[key] = true
            table.insert(out, key)
        end
    end

    table.sort(out)
    return out
end

local function selectedPhotos(catalog)
    local photos = catalog:getTargetPhotos()

    if not photos or #photos == 0 then
        local target = catalog:getTargetPhoto()

        if target then
            return { target }
        end
    end

    return photos or {}
end

local function photoUuid(photo, sourcePath)
    return tostring(
        safeRaw(photo, "uuid") or
        safeRaw(photo, "localIdentifier") or
        sourcePath or
        ""
    )
end

local function locationField(photo, rawKeys, formattedKeys)
    local rawValue = firstMetadata(photo, rawKeys or {}, false)

    if rawValue ~= "" then
        return rawValue
    end

    return firstMetadata(photo, formattedKeys or rawKeys or {}, true)
end

local function updateProgress(progress, index, selectedCount, writtenCount)
    if index % 25 == 0 or index == 1 or index == selectedCount then
        progress:setPortionComplete(index, selectedCount)
        progress:setCaption(
            "Processed " ..
            tostring(index) ..
            " of " ..
            tostring(selectedCount) ..
            " photos. Rows written: " ..
            tostring(writtenCount)
        )

        -- Gives Lightroom a chance to redraw progress UI and respond to cancel.
        LrTasks.yield()
    end
end

function CandidateExport.run()
    local config = dofile(_PLUGIN.path .. "/Config.lua")
    local catalog = LrApplication.activeCatalog()

    local photos = selectedPhotos(catalog)

    if not photos or #photos == 0 then
        LrDialogs.message(
            "Nazariya",
            "Select one or more photos first, then run Export Candidate CSV.",
            "warning"
        )
        return
    end

    local outputCsv = config.output_csv
    local outputDir = dirname(outputCsv)

    LrFileUtils.createAllDirectories(outputDir)

    local f = io.open(outputCsv, "w")

    if not f then
        LrDialogs.message(
            "Nazariya",
            "Could not write CSV:\n" .. outputCsv,
            "critical"
        )
        return
    end

    writeCsvRow(f, {
        "source_path",
        "source_dir",
        "file_name",
        "file_stem",
        "file_extension",
        "file_format",
        "photo_uuid",

        "candidate_keys",
        "primary_candidate_key",
        "all_keywords",

        "rating",
        "label_color",
        "pick_status",
        "copy_name",

        "capture_time",
        "camera_make",
        "camera_model",
        "lens",
        "iso",
        "focal_length",
        "aperture",
        "shutter_speed",

        "width",
        "height",
        "orientation",

        "gps_latitude",
        "gps_longitude",
        "location",
        "sublocation",
        "city",
        "state",
        "country",
        "iso_country_code",

        "folder_name",
        "exported_at",
    })

    local selectedCount = #photos
    local writtenCount = 0
    local skippedNoCandidate = 0
    local skippedNotRaw = 0
    local skippedNoPath = 0
    local canceled = false

    local progress = LrProgressScope({
        title = "Exporting Nazariya Candidate CSV",
        caption = "Preparing export...",
        functionContext = nil,
    })

    progress:setCancelable(true)
    progress:setPortionComplete(0, selectedCount)

    local pattern = config.candidate_keyword_pattern or "^c%d%d%d$"
    local rawFormats = config.raw_formats or { RAW = true, DNG = true }
    local exportedAt = os.date("%Y-%m-%dT%H:%M:%S")

    for index, photo in ipairs(photos) do
        if progress:isCanceled() then
            canceled = true
            break
        end

        updateProgress(progress, index, selectedCount, writtenCount)

        local sourcePath = tostring(safeRaw(photo, "path") or "")

        if sourcePath == "" then
            skippedNoPath = skippedNoPath + 1
        else
            local fileFormat = normalizeFileFormat(safeRaw(photo, "fileFormat"))
            local isRaw = rawFormats[fileFormat] == true

            local names = keywordNames(photo)
            local candidateKeys = findCandidateKeys(names, pattern)
            local hasCandidate = #candidateKeys > 0

            if not isRaw then
                skippedNotRaw = skippedNotRaw + 1
            end

            if not hasCandidate then
                skippedNoCandidate = skippedNoCandidate + 1
            end

            if isRaw and hasCandidate then
                local fileName = basename(sourcePath)
                local sourceDir = dirname(sourcePath)
                local fileStem = LrPathUtils.removeExtension(fileName)
                local fileExtension = LrPathUtils.extension(fileName) or ""
                local primaryCandidateKey = candidateKeys[1] or ""

                local rating = safeRaw(photo, "rating") or safeFormatted(photo, "rating") or ""
                local labelColor = safeRaw(photo, "colorNameForLabel") or ""
                local pickStatus = safeRaw(photo, "pickStatus") or ""
                local copyName = safeRaw(photo, "copyName") or ""

                local captureTime =
                    safeRaw(photo, "dateTimeOriginal") or
                    safeFormatted(photo, "dateTimeOriginal") or
                    ""

                local cameraMake = firstMetadata(photo, { "cameraMake" }, false)
                local cameraModel = firstMetadata(photo, { "cameraModel" }, false)
                local lens = firstMetadata(photo, { "lens", "lensName" }, false)
                local iso = firstMetadata(photo, { "isoSpeedRating" }, false)
                local focalLength = firstMetadata(photo, { "focalLength" }, true)
                local aperture = firstMetadata(photo, { "aperture" }, true)
                local shutterSpeed = firstMetadata(photo, { "shutterSpeed" }, true)

                local width = firstMetadata(photo, { "croppedDimensions", "dimensions" }, true)
                local height = ""
                local orientation = firstMetadata(photo, { "orientation" }, false)

                local gpsLatitude = locationField(
                    photo,
                    { "gpsLatitude" },
                    { "gpsLatitude" }
                )

                local gpsLongitude = locationField(
                    photo,
                    { "gpsLongitude" },
                    { "gpsLongitude" }
                )

                -- These keys are intentionally defensive.
                -- Lightroom SDK metadata availability can vary by version/catalog.
                local location = locationField(
                    photo,
                    { "location" },
                    { "location" }
                )

                local sublocation = locationField(
                    photo,
                    { "sublocation", "iptcSublocation" },
                    { "sublocation", "iptcSublocation" }
                )

                local city = locationField(
                    photo,
                    { "city", "iptcCity" },
                    { "city", "iptcCity" }
                )

                local state = locationField(
                    photo,
                    { "state", "province", "iptcState" },
                    { "state", "province", "iptcState" }
                )

                local country = locationField(
                    photo,
                    { "country", "countryName", "iptcCountry" },
                    { "country", "countryName", "iptcCountry" }
                )

                local isoCountryCode = locationField(
                    photo,
                    { "isoCountryCode", "countryCode", "iptcIsoCountryCode" },
                    { "isoCountryCode", "countryCode", "iptcIsoCountryCode" }
                )

                writeCsvRow(f, {
                    sourcePath,
                    sourceDir,
                    fileName,
                    fileStem,
                    fileExtension,
                    fileFormat,
                    photoUuid(photo, sourcePath),

                    join(candidateKeys, " ; "),
                    primaryCandidateKey,
                    join(names, " ; "),

                    rating,
                    labelColor,
                    pickStatus,
                    copyName,

                    captureTime,
                    cameraMake,
                    cameraModel,
                    lens,
                    iso,
                    focalLength,
                    aperture,
                    shutterSpeed,

                    width,
                    height,
                    orientation,

                    gpsLatitude,
                    gpsLongitude,
                    location,
                    sublocation,
                    city,
                    state,
                    country,
                    isoCountryCode,

                    basename(sourceDir),
                    exportedAt,
                })

                writtenCount = writtenCount + 1
            end
        end
    end

    f:close()
    progress:done()

    if canceled then
        LrDialogs.message(
            "Nazariya",
            "Candidate CSV export canceled.\n\n" ..
                "Selected photos: " .. tostring(selectedCount) ..
                "\nRows written before cancel: " .. tostring(writtenCount) ..
                "\nSkipped no candidate key: " .. tostring(skippedNoCandidate) ..
                "\nSkipped not RAW/DNG: " .. tostring(skippedNotRaw) ..
                "\nSkipped no path: " .. tostring(skippedNoPath) ..
                "\n\nPartial file:\n" .. outputCsv,
            "warning"
        )
        return
    end

    LrDialogs.message(
        "Nazariya",
        "Candidate CSV exported.\n\n" ..
            "Selected photos: " .. tostring(selectedCount) ..
            "\nRows written: " .. tostring(writtenCount) ..
            "\nSkipped no candidate key: " .. tostring(skippedNoCandidate) ..
            "\nSkipped not RAW/DNG: " .. tostring(skippedNotRaw) ..
            "\nSkipped no path: " .. tostring(skippedNoPath) ..
            "\n\nFile:\n" .. outputCsv,
        "info"
    )
end

return CandidateExport
