local LrTasks = import "LrTasks"
local LurevaReview = require "LurevaReview"
LrTasks.startAsyncTask(function() LurevaReview.exportFinalSelection() end)
