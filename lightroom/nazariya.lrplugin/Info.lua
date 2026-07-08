return {
    LrSdkVersion = 13.0,
    LrSdkMinimumVersion = 6.0,

    LrToolkitIdentifier = "com.clariform.nazariya",
    LrPluginName = "Nazariya",

    LrLibraryMenuItems = {
        {
            title = "Export Candidate CSV",
            file = "CandidateExportMain.lua",
        },
        {
            title = "Lureva: Build Review Structure",
            file = "LurevaBuildStructureMain.lua",
        },
        {
            title = "Lureva: Apply Batch Assignments",
            file = "LurevaApplyBatchMain.lua",
        },
        {
            title = "Lureva: Build 960 Review Collections",
            file = "LurevaBuildReviewMain.lua",
        },
        {
            title = "Lureva: Validate Review Selection",
            file = "LurevaValidateReviewMain.lua",
        },
        {
            title = "Lureva: Export Final 960 Manifest",
            file = "LurevaExportFinalMain.lua",
        },
    },

    VERSION = {
        major = 0,
        minor = 1,
        revision = 2,
        build = 0,
    },
}
