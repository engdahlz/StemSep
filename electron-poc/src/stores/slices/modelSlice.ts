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
        }
    }),

    setDownloadProgress: (data) => set((state) => {
        const model = state.models.find(m => m.id === data.modelId)
        if (model) {
            model.downloadProgress = data.progress
            model.downloadSpeed = data.speed
            model.downloadEta = data.eta
            model.downloadError = undefined
        }
    }),

    completeDownload: (modelId) => set((state) => {
        const model = state.models.find(m => m.id === modelId)
        if (model) {
            model.installed = true
            model.downloading = false
            model.downloadProgress = 100
            model.downloadSpeed = undefined
            model.downloadEta = undefined
            model.downloadError = undefined
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
        }
    }),

    pauseDownload: (modelId) => set((state) => {
        const model = state.models.find(m => m.id === modelId)
        if (model) {
            model.downloading = false
            model.downloadPaused = true
        }
    }),

    resumeDownload: (modelId) => set((state) => {
        const model = state.models.find(m => m.id === modelId)
        if (model) {
            model.downloading = true
            model.downloadPaused = false
            model.downloadError = undefined
        }
    }),

    setModelInstalled: (modelId, installed) => set((state) => {
        const model = state.models.find(m => m.id === modelId)
        if (model) {
            model.installed = installed
        }
    }),
})
