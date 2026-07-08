local LrApplication = import "LrApplication"
local LrDialogs = import "LrDialogs"
local LrPathUtils = import "LrPathUtils"
local LrProgressScope = import "LrProgressScope"
local LrTasks = import "LrTasks"

local M = {}

local function parseCsvLine(line)
    local fields, field, quoted, index = {}, "", false, 1
    while index <= #line do
        local char = string.sub(line, index, index)
        if quoted then
            if char == '"' then
                local nextChar = string.sub(line, index + 1, index + 1)
                if nextChar == '"' then field = field .. '"'; index = index + 1 else quoted = false end
            else field = field .. char end
        elseif char == '"' then quoted = true
        elseif char == "," then table.insert(fields, field); field = ""
        else field = field .. char end
        index = index + 1
    end
    table.insert(fields, field)
    return fields
end

local function readCsv(path)
    local file = io.open(path, "r")
    if not file then return nil, "Could not open CSV: " .. tostring(path) end
    local headerLine = file:read("*l")
    if not headerLine then file:close(); return nil, "CSV is empty: " .. tostring(path) end
    local headers, rows = parseCsvLine(headerLine), {}
    for line in file:lines() do
        if line ~= "" then
            local values, row = parseCsvLine(line), {}
            for index, header in ipairs(headers) do row[header] = values[index] or "" end
            table.insert(rows, row)
        end
    end
    file:close()
    return rows, nil
end

local function chooseCsv(title, initialDirectory)
    local result = LrDialogs.runOpenPanel({
        title = title,
        canChooseFiles = true,
        canChooseDirectories = false,
        allowsMultipleSelection = false,
        fileTypes = { "csv" },
        initialDirectory = initialDirectory,
    })
    return result and result[1] or nil
end

local function childSetByName(parent, name)
    local catalog = LrApplication.activeCatalog()
    local sets = parent and parent:getChildCollectionSets() or catalog:getChildCollectionSets()
    for _, set in ipairs(sets or {}) do if set:getName() == name then return set end end
    return nil
end

local function childCollectionByName(parent, name)
    for _, collection in ipairs(parent:getChildCollections() or {}) do
        if collection:getName() == name then return collection end
    end
    return nil
end

local function createSet(catalog, parent, name)
    local created = nil
    catalog:withWriteAccessDo("Create collection set " .. name, function()
        created = catalog:createCollectionSet(name, parent, true)
    end)
    LrTasks.yield()
    return created or childSetByName(parent, name)
end

local function createCollection(catalog, parent, name)
    local created = nil
    catalog:withWriteAccessDo("Create collection " .. name, function()
        created = catalog:createCollection(name, parent, true)
    end)
    LrTasks.yield()
    return created or childCollectionByName(parent, name)
end

local function ensureSet(catalog, parent, name)
    return childSetByName(parent, name) or createSet(catalog, parent, name)
end

local function ensureCollection(catalog, parent, name)
    return childCollectionByName(parent, name) or createCollection(catalog, parent, name)
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

local function ensureReviewRoot(catalog, config, collectionSetName)
    local parent = ensureSetPath(catalog, config.lureva_review_parent_collection_sets or {})
    return ensureSet(catalog, parent, collectionSetName)
end

local function keywordChild(parent, name)
    local catalog = LrApplication.activeCatalog()
    local keywords = parent and parent:getChildren() or catalog:getKeywords()
    for _, keyword in ipairs(keywords or {}) do if keyword:getName() == name then return keyword end end
    return nil
end

local function createKeyword(catalog, name, parent)
    local created = nil
    catalog:withWriteAccessDo("Create keyword " .. name, function()
        created = catalog:createKeyword(name, {}, true, parent, true)
    end)
    LrTasks.yield()
    return created or keywordChild(parent, name)
end

local function ensureKeywordPath(catalog, path)
    local parent = nil
    for segment in string.gmatch(path or "", "[^/]+") do
        parent = keywordChild(parent, segment) or createKeyword(catalog, segment, parent)
    end
    return parent
end


local function trim(value)
    return tostring(value or ""):match("^%s*(.-)%s*$")
end

local function getenv(name)
    if os and type(os.getenv) == "function" then
        return os.getenv(name) or ""
    end
    return ""
end

local function expandEnvPath(path)
    local value = trim(path)
    if value == "" then return "" end
    value = value:gsub("%${([%w_]+)}", function(name) return getenv(name) end)
    value = value:gsub("%$([%w_]+)", function(name) return getenv(name) end)
    return value
end

local function archiveRelativePath(path)
    local value = trim(path):gsub("\\", "/")
    if value == "" then return "" end
    local lower = string.lower(value)
    for _, marker in ipairs({ "/pictures/images/", "/proetus/images/" }) do
        local startIndex, endIndex = string.find(lower, marker, 1, true)
        if startIndex then
            return string.sub(value, endIndex + 1)
        end
    end
    local matched = string.match(value, "(%d%d%d%d/%d%d%d%d%-%d%d/%d%d%d%d%-%d%d%-%d%d/[^/]+)$")
    return matched or ""
end

local function buildArchiveRelativeIndex(catalog)
    local index = {}
    for _, photo in ipairs(catalog:getAllPhotos() or {}) do
        local ok, path = pcall(function() return photo:getRawMetadata("path") end)
        if ok and path then
            local key = archiveRelativePath(path)
            if key ~= "" and not index[key] then
                index[key] = photo
            end
        end
    end
    return index
end

local function findPhotoForRow(catalog, row, relativeIndex)
    local candidates = {
        trim(row.resolved_source_path),
        expandEnvPath(row.source_path_env),
        trim(row.source_path),
    }
    local relative = trim(row.archive_relative_path)
    if relative == "" then relative = archiveRelativePath(row.source_path) end
    local rootEnv = trim(row.source_root_env)
    if rootEnv ~= "" and relative ~= "" then
        local root = getenv(rootEnv)
        if root ~= "" then table.insert(candidates, root .. "/" .. relative) end
    end
    for _, candidate in ipairs(candidates) do
        if candidate and candidate ~= "" then
            local photo = catalog:findPhotoByPath(candidate)
            if photo then return photo, candidate, "path" end
        end
    end
    if relative ~= "" and relativeIndex then
        local photo = relativeIndex[relative]
        if photo then return photo, relative, "archive_relative_path" end
    end
    return nil, relative ~= "" and relative or (row.source_path or row.file_name or "<unknown>"), "missing"
end

local function countMap(map)
    local count = 0
    for _ in pairs(map) do count = count + 1 end
    return count
end

function M.buildStructure()
    local config = dofile(_PLUGIN.path .. "/Config.lua")
    local path = chooseCsv("Choose Lureva Lightroom Structure Manifest", config.lureva_manifest_directory)
    if not path then return end
    local rows, err = readCsv(path)
    if not rows then LrDialogs.message("Nazariya", err, "critical"); return end
    if #rows == 0 then LrDialogs.message("Nazariya", "Structure CSV is empty.", "warning"); return end

    local catalog = LrApplication.activeCatalog()
    local collectionSetName = rows[1].collection_set or config.lureva_review_collection_set
    local root = ensureReviewRoot(catalog, config, collectionSetName)
    local batchesSet = ensureSet(catalog, root, "Batches")
    local groupsSet = ensureSet(catalog, root, "Groups")

    local batchIds, groupNames, keywordPaths = {}, {}, {}
    for _, row in ipairs(rows) do
        if row.batch_id and row.batch_id ~= "" then batchIds[row.batch_id] = true end
        if row.final_group and row.final_group ~= "" then
            groupNames[row.final_group .. " (" .. (row.candidate_group or "") .. ")"] = true
        end
        for _, key in ipairs({ "group_keyword", "batch_keyword", "primary_keyword", "alternate_keyword" }) do
            if row[key] and row[key] ~= "" then keywordPaths[row[key]] = true end
        end
    end

    for batchId in pairs(batchIds) do
        local batchSet = ensureSet(catalog, batchesSet, batchId)
        ensureCollection(catalog, batchSet, batchId .. " - All")
    end
    for groupName in pairs(groupNames) do
        ensureCollection(catalog, groupsSet, groupName)
    end
    for keywordPath in pairs(keywordPaths) do
        ensureKeywordPath(catalog, keywordPath)
    end

    LrDialogs.message("Nazariya", "Created review structure for " .. countMap(groupNames) .. " groups and " .. countMap(batchIds) .. " batches.", "info")
end

function M.applyBatchAssignments()
    local config = dofile(_PLUGIN.path .. "/Config.lua")
    local path = chooseCsv("Choose Lureva Batch Assignment Manifest", config.lureva_manifest_directory)
    if not path then return end
    local rows, err = readCsv(path)
    if not rows then LrDialogs.message("Nazariya", err, "critical"); return end
    if #rows == 0 then LrDialogs.message("Nazariya", "Assignment CSV is empty.", "warning"); return end

    local catalog = LrApplication.activeCatalog()
    local collectionSetName = rows[1].collection_set or config.lureva_review_collection_set
    local root = ensureReviewRoot(catalog, config, collectionSetName)
    local batchesSet = ensureSet(catalog, root, "Batches")
    local groupsSet = ensureSet(catalog, root, "Groups")

    local groupCollections, batchSets, batchCollections, keywordCache = {}, {}, {}, {}
    local actions, missing = {}, {}
    local relativeIndex = nil
    local progress = LrProgressScope({ title = "Prepare Lureva batch assignments" })

    for index, row in ipairs(rows) do
        local photo, matchedPath, matchMethod = findPhotoForRow(catalog, row, relativeIndex)
        if not photo and trim(row.archive_relative_path) ~= "" then
            progress:setCaption("Indexing catalog by archive-relative path")
            LrTasks.yield()
            relativeIndex = relativeIndex or buildArchiveRelativeIndex(catalog)
            photo, matchedPath, matchMethod = findPhotoForRow(catalog, row, relativeIndex)
        end
        if photo then
            local groupName = row.final_group .. " (" .. (row.candidate_group or "") .. ")"
            local batchId = row.batch_id or ""
            local batchName = batchId .. " - All"
            groupCollections[groupName] = groupCollections[groupName] or ensureCollection(catalog, groupsSet, groupName)
            if batchId ~= "" then
                batchSets[batchId] = batchSets[batchId] or ensureSet(catalog, batchesSet, batchId)
                batchCollections[batchName] = batchCollections[batchName] or ensureCollection(catalog, batchSets[batchId], batchName)
            end
            for _, key in ipairs({ "group_keyword", "batch_keyword", "role_keyword" }) do
                local keywordPath = row[key] or ""
                if keywordPath ~= "" and not keywordCache[keywordPath] then
                    keywordCache[keywordPath] = ensureKeywordPath(catalog, keywordPath)
                end
            end
            table.insert(actions, {
                photo = photo,
                groupCollection = groupCollections[groupName],
                batchCollection = batchCollections[batchName],
                groupKeyword = keywordCache[row.group_keyword or ""],
                batchKeyword = keywordCache[row.batch_keyword or ""],
                roleKeyword = keywordCache[row.role_keyword or ""],
                pickStatus = tonumber(row.initial_pick_status or "0") or 0,
                finalGroup = row.final_group or "",
                matchedPath = matchedPath or "",
                matchMethod = matchMethod or "",
            })
        else
            table.insert(missing, matchedPath or row.source_path or row.file_name or "<unknown>")
        end
        progress:setPortionComplete(index, #rows)
        progress:setCaption("Preparing " .. tostring(index) .. "/" .. tostring(#rows))
        LrTasks.yield()
    end
    progress:done()

    local applied = 0
    progress = LrProgressScope({ title = "Apply Lureva batch assignments" })
    catalog:withWriteAccessDo("Apply Lureva batch assignments", function()
        for index, action in ipairs(actions) do
            if action.groupCollection then action.groupCollection:addPhotos({ action.photo }) end
            if action.batchCollection then action.batchCollection:addPhotos({ action.photo }) end
            if action.groupKeyword then action.photo:addKeyword(action.groupKeyword) end
            if action.batchKeyword then action.photo:addKeyword(action.batchKeyword) end
            if action.roleKeyword then action.photo:addKeyword(action.roleKeyword) end
            action.photo:setRawMetadata("pickStatus", action.pickStatus)
            applied = applied + 1
            progress:setPortionComplete(index, #actions)
            progress:setCaption(action.finalGroup .. "  " .. tostring(index) .. "/" .. tostring(#actions))
            LrTasks.yield()
        end
    end)
    progress:done()

    local message = "Applied " .. tostring(applied) .. " batch assignments."
    if #missing > 0 then
        message = message .. "\nMissing photos: " .. tostring(#missing)
        message = message .. "\nFirst missing: " .. tostring(missing[1] or "")
    end
    LrDialogs.message("Nazariya", message, #missing > 0 and "warning" or "info")
end

return M
