local LrTasks = import "LrTasks"

local CandidateExport = require "CandidateExport"

LrTasks.startAsyncTask(function()
    CandidateExport.run()
end)
