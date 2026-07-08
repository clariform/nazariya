local LrApplication = import "LrApplication"
local LrDialogs = import "LrDialogs"
local LrFileUtils = import "LrFileUtils"
local LrPathUtils = import "LrPathUtils"
local LrProgressScope = import "LrProgressScope"
local LrTasks = import "LrTasks"

local LurevaReview = {}

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

local function parseCsvLine(line)
    local fields = {}
    local field = ""
    local quoted = false
    local index = 1
    while index <= #line do
        local char = string.sub(line, index, index)
        if quoted then
            if char == '"' then
                local nextChar = string.sub(line, index + 1, index + 1)
                if nextChar == '"' then
                    field = field .. '"'
                    index = index + 1
                else
                    quoted = false
                end
            else
                field = field .. char
            end
        elseif char == '"' then
            quoted = true
        elseif char == "," then
            table.insert(fields, field)
            field = ""
        else
            field = field .. char
        end
        index = index + 1
    end
    table.insert(fields, field)
    return fields
end

local function readCsv(path)
    local file = io.open(path, "r")
    if not file then
        return nil, "Could not open CSV: " .. tostring(path)
    end
    local headerLine = file:read("*l")
    if not headerLine then
        file:close()
        return nil, "CSV is empty: " .. tostring(path)
    end
    local headers = parseCsvLine(headerLine)
    local rows = {}
    for line in file:lines() do
        if line ~= "" then
            local values = parseCsvLine(line)
            local row = {}
            for index, header in ipairs(headers) do
                row[header] = values[index] or ""
            end
            table.insert(rows, row)
        end
    end
    file:close()
    return rows, nil
end

local function chooseManifest(config)
    local result = LrDialogs.runOpenPanel({
        title = "Choose Lureva Lightroom Review Manifest",
        canChooseFiles = true,
        canChooseDirectories = false,
        allowsMultipleSelection = false,
        fileTypes = { "csv" },
        initialDirectory = config.lureva_manifest_directory,
    })
    if result and result[1] then
        return result[1]
    end
    return nil
end

local function childSetByName(parent, name)
    local sets = parent and parent:getChildCollectionSets() or LrApplication.activeCatalog():getChildCollectionSets()
    for _, set in ipairs(sets or {}) do
        if set:getName() == name then
            return set
        end
    end
    return nil
end

local function childCollectionByName(set, name)
    for _, collection in ipairs(set:getChildCollections() or {}) do
        if collection:getName() == name then
            return collection
        end
    end
    return nil
end

local function ensureSet(catalog, parent, name)
    local set = childSetByName(parent, name)
    if set then
        return set
    end
    return catalog:createCollectionSet(name, parent, true)
end

local function ensureSetPath(catalog, names)
    local parent = nil
    for _, name in ipairs(names or {}) do
        if name and name ~= "" then
            parent = ensureSet(catalog, parent, name)
        end
    end
    return parent
end

local function ensureReviewSet(catalog, config)
    local parent = ensureSetPath(catalog, config.lureva_review_parent_collection_sets or {})
    return ensureSet(catalog, parent, config.lureva_review_collection_set)
end

local function findSetPath(names)
    local parent = nil
    for _, name in ipairs(names or {}) do
        parent = childSetByName(parent, name)
        if not parent then
            return nil
        end
    end
    return parent
end

local function findReviewSet(config)
    local parent = findSetPath(config.lureva_review_parent_collection_sets or {})
    if parent then
        local nested = childSetByName(parent, config.lureva_review_collection_set)
        if nested then return nested end
    end
    return childSetByName(nil, config.lureva_review_collection_set)
end

local function findGroupsSet(config)
    local reviewSet = findReviewSet(config)
    if not reviewSet then return nil end
    return childSetByName(reviewSet, "Groups") or reviewSet
end

local function setPhotoFlag(photo, status)
    photo:setRawMetadata("pickStatus", tonumber(status) or 0)
end

function LurevaReview.buildCollections()
    local config = dofile(_PLUGIN.path .. "/Config.lua")
    local manifestPath = chooseManifest(config)
    if not manifestPath then
        return
    end
    local rows, errorMessage = readCsv(manifestPath)
    if not rows then
        LrDialogs.message("Nazariya", errorMessage, "critical")
        return
    end

    local catalog = LrApplication.activeCatalog()
    local byGroup = {}
    local missing = {}
    for _, row in ipairs(rows) do
        local group = row.final_group or ""
        byGroup[group] = byGroup[group] or {}
        local photo = catalog:findPhotoByPath(row.source_path or "")
        if photo then
            table.insert(byGroup[group], { photo = photo, row = row })
        else
            table.insert(missing, row.source_path or row.file_name or "<unknown>")
        end
    end

    local progress = LrProgressScope({ title = "Build Lureva review collections" })
    local groupCount = 0
    for _ in pairs(byGroup) do groupCount = groupCount + 1 end
    local completed = 0

    catalog:withWriteAccessDo("Build Lureva review collections", function()
        local reviewSet = ensureReviewSet(catalog, config)
        local groupsSet = ensureSet(catalog, reviewSet, "Groups")
        for group, items in pairs(byGroup) do
            local collection = childCollectionByName(groupsSet, group)
            if not collection then
                collection = catalog:createCollection(group, groupsSet, true)
            else
                collection:removeAllPhotos()
            end
            local photos = {}
            for _, item in ipairs(items) do
                table.insert(photos, item.photo)
                setPhotoFlag(item.photo, item.row.initial_pick_status)
            end
            collection:addPhotos(photos)
            completed = completed + 1
            progress:setPortionComplete(completed, groupCount)
            progress:setCaption(group .. "  " .. tostring(#photos) .. " photos")
            LrTasks.yield()
        end
    end)
    progress:done()

    local message = "Created " .. tostring(groupCount) .. " review collections."
    if #missing > 0 then
        message = message .. "\n\nMissing source photos: " .. tostring(#missing) .. "\n" .. table.concat(missing, "\n", 1, math.min(#missing, 20))
    end
    LrDialogs.message("Nazariya", message, #missing > 0 and "warning" or "info")
end

function LurevaReview.validateSelection()
    local config = dofile(_PLUGIN.path .. "/Config.lua")
    local groupsSet = findGroupsSet(config)
    if not groupsSet then
        LrDialogs.message("Nazariya", "Review Groups collection set not found under: " .. config.lureva_review_collection_set, "warning")
        return
    end

    local collections = groupsSet:getChildCollections() or {}
    table.sort(collections, function(a, b) return a:getName() < b:getName() end)
    local lines = {}
    local total = 0
    local invalid = 0
    for _, collection in ipairs(collections) do
        local picked = 0
        for _, photo in ipairs(collection:getPhotos() or {}) do
            if tonumber(photo:getRawMetadata("pickStatus") or 0) == 1 then
                picked = picked + 1
            end
        end
        total = total + picked
        local valid = picked == config.lureva_required_picks_per_group
        if not valid then invalid = invalid + 1 end
        table.insert(lines, collection:getName() .. ": " .. tostring(picked) .. (valid and "  OK" or "  NEEDS REVIEW"))
    end

    local status = invalid == 0 and "info" or "warning"
    local summary = tostring(#collections) .. " groups, " .. tostring(total) .. " Picks."
    if invalid == 0 and #collections == config.lureva_required_groups then
        summary = summary .. "\nSelection is valid."
    else
        summary = summary .. "\nInvalid groups: " .. tostring(invalid)
    end
    LrDialogs.message("Lureva selection validation", summary .. "\n\n" .. table.concat(lines, "\n"), status)
end

function LurevaReview.exportFinalSelection()
    local config = dofile(_PLUGIN.path .. "/Config.lua")
    local groupsSet = findGroupsSet(config)
    if not groupsSet then
        LrDialogs.message("Nazariya", "Review Groups collection set not found.", "warning")
        return
    end
    local output = LrDialogs.runSavePanel({
        title = "Export Final Lureva 960 Manifest",
        requiredFileType = "csv",
        initialDirectory = config.lureva_manifest_directory,
        initialFileName = "lureva_final_960.csv",
    })
    if not output then return end

    local collections = groupsSet:getChildCollections() or {}
    table.sort(collections, function(a, b) return a:getName() < b:getName() end)
    local selected = {}
    local invalid = {}
    for _, collection in ipairs(collections) do
        local picked = {}
        for _, photo in ipairs(collection:getPhotos() or {}) do
            if tonumber(photo:getRawMetadata("pickStatus") or 0) == 1 then
                table.insert(picked, photo)
            end
        end
        if #picked ~= config.lureva_required_picks_per_group then
            table.insert(invalid, collection:getName() .. "=" .. tostring(#picked))
        end
        for _, photo in ipairs(picked) do
            table.insert(selected, { group = collection:getName(), photo = photo })
        end
    end

    if #invalid > 0 or #collections ~= config.lureva_required_groups then
        LrDialogs.message(
            "Nazariya",
            "Cannot export until every group has exactly " .. tostring(config.lureva_required_picks_per_group) .. " Picks.\n" .. table.concat(invalid, "\n"),
            "critical"
        )
        return
    end

    local file = io.open(output, "w")
    if not file then
        LrDialogs.message("Nazariya", "Could not write: " .. output, "critical")
        return
    end
    writeCsvRow(file, { "final_group", "photo_uuid", "source_path", "file_name", "pick_status" })
    for _, item in ipairs(selected) do
        local photo = item.photo
        local sourcePath = tostring(photo:getRawMetadata("path") or "")
        writeCsvRow(file, {
            item.group,
            tostring(photo:getRawMetadata("uuid") or photo:getRawMetadata("localIdentifier") or ""),
            sourcePath,
            LrPathUtils.leafName(sourcePath),
            "1",
        })
    end
    file:close()
    LrDialogs.message("Nazariya", "Exported " .. tostring(#selected) .. " selected photos to:\n" .. output, "info")
end

return LurevaReview
