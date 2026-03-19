import { StateCreator } from 'zustand'
import { AppState, ModelSlice } from '../../types/store'

export const createModelSlice: StateCreator<AppState, [["zustand/immer", never]], [], ModelSlice> = (set) => ({
    models: [],
    recipes: [],
    setModels: (models) => set({ models }),
    setRecipes: (recipes) => set({ recipes }),

    startDownload: (modelId) => set((state) => {
        const model = state.models.find(m => m.id === modelId)
        if (model) {
            model.downloading = true
            model.downloadProgress = 0
            model.downloadError = undefined
            model.downloadPaused = false
            model.downloadState = "queued"
            model.downloadStage = "queued"
            model.downloadVerified = false
        }
    }),

    setDownloadProgress: (data) => set((state) => {
        const model = state.models.find(m => m.id === data.modelId)
        if (model) {
            model.downloadProgress = data.progress
            model.downloadSpeed = data.speed
            model.downloadEta = data.eta
            model.downloadError = undefined
            model.downloading = true
            model.downloadPaused = false
            model.downloadStage = data.stage
            model.downloadCurrentFile = data.currentFile
            model.downloadCurrentRelativePath = data.currentRelativePath
            model.downloadCurrentSource = data.currentSource
            model.downloadVerified = data.verified
            if (data.stage === "verifying") {
                model.downloadState = "verifying"
            } else if (data.stage === "preflighting") {
                model.downloadState = "preflighting"
            } else {
                model.downloadState = "downloading"
            }
        }
    }),

    completeDownload: (modelId) => set((state) => {
        const model = state.models.find(m => m.id === modelId)
        if (model) {
            model.downloading = false
            model.downloadProgress = 100
            model.downloadSpeed = undefined
            model.downloadEta = undefined
            model.downloadError = undefined
            model.downloadPaused = false
            model.downloadState = "verifying"
            model.downloadStage = "installed"
            model.downloadVerified = true
        }
    }),

    setDownloadError: (modelId, error) => set((state) => {
        const model = state.models.find(m => m.id === modelId)
        if (model) {
            model.downloading = false
            model.downloadProgress = 0
            model.downloadSpeed = undefined
            model.downloadEta = undefined
            model.downloadError = error
            model.downloadState = "failed"
            model.downloadStage = "failed"
        }
    }),

    pauseDownload: (modelId) => set((state) => {
        const model = state.models.find(m => m.id === modelId)
        if (model) {
            model.downloading = false
            model.downloadPaused = true
            model.downloadState = "paused"
            model.downloadStage = "paused"
        }
    }),

    resumeDownload: (modelId) => set((state) => {
        const model = state.models.find(m => m.id === modelId)
        if (model) {
            model.downloading = true
            model.downloadPaused = false
            model.downloadError = undefined
            model.downloadState = "queued"
            model.downloadStage = "queued"
        }
    }),

    setModelInstalled: (modelId, installed) => set((state) => {
        const model = state.models.find(m => m.id === modelId)
        if (model) {
            model.installed = installed
            model.installation = {
                ...(model.installation || {}),
                installed,
            }
            model.downloadState = installed ? "installed" : "idle"
            if (!installed) {
                model.downloadVerified = false
            }
        }
    }),

    upsertModel: (model) => set((state) => {
        const existingIndex = state.models.findIndex(entry => entry.id === model.id)
        if (existingIndex >= 0) {
            state.models[existingIndex] = { ...state.models[existingIndex], ...model }
            return
        }
        state.models.push(model)
    }),

    mergeModel: (modelId, patch) => set((state) => {
        const model = state.models.find(entry => entry.id === modelId)
        if (!model) return
        Object.assign(model, patch)
    }),
})
