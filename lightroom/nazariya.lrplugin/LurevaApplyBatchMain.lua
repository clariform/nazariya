local LrTasks = import "LrTasks"
LrTasks.startAsyncTask(function()
    dofile(_PLUGIN.path .. "/LurevaBatchReview.lua").applyBatchAssignments()
end)
