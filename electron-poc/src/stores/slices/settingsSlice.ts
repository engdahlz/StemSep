import { StateCreator } from 'zustand'
import { AppState, SettingsSlice, PhaseParams } from '../../types/store'

export const createSettingsSlice: StateCreator<AppState, [["zustand/immer", never]], [], SettingsSlice> = (set) => ({
  history: [],
  settings: {
    theme: 'dark',
    phaseParams: {
      enabled: false,
      lowHz: 500,
      highHz: 5000,
      highFreqWeight: 0.8
    },
    advancedSettings: {
      shifts: 1,
      overlap: 0.25,
      segmentSize: 256,
      outputFormat: 'wav',
      bitrate: '320k'
    }
  },
  sessionToLoad: null,
  watchModeEnabled: false,
  watchPath: '',

  addToHistory: (item) => set((state) => {
    // Dedupe check: Don't add if same inputFile was added within last 10 seconds
    const tenSecondsAgo = Date.now() - 10000
    const isDuplicate = state.history.some(h =>
      h.inputFile === item.inputFile &&
      new Date(h.date).getTime() > tenSecondsAgo
    )

    if (isDuplicate) {
      console.log('[History] Skipping duplicate entry for:', item.inputFile)
      return // Don't add duplicate
    }

    state.history.unshift({
      ...item,
      id: crypto.randomUUID(),
      date: new Date().toISOString()
    })
    if (state.history.length > 50) {
      state.history = state.history.slice(0, 50)
    }
  }),

  removeFromHistory: (id) => set((state) => {
    state.history = state.history.filter(item => item.id !== id)
  }),

  toggleHistoryFavorite: (id) => set((state) => {
    const item = state.history.find(i => i.id === id)
    if (item) {
      item.isFavorite = !item.isFavorite
    }
  }),

  clearHistory: () => set((state) => {
    state.history = []
  }),

  loadSession: (item) => set((state) => {
    state.sessionToLoad = item
  }),

  clearLoadedSession: () => set((state) => {
    state.sessionToLoad = null
  }),

  setTheme: (theme) => set((state) => {
    state.settings.theme = theme
  }),

  setDefaultOutputDir: (path) => set((state) => {
    state.settings.defaultOutputDir = path
  }),

  setDefaultExportDir: (path) => set((state) => {
    state.settings.defaultExportDir = path
  }),

  setModelsDir: (path) => set((state) => {
    state.settings.modelsDir = path
  }),

  setDefaultModel: (modelId) => set((state) => {
    state.settings.defaultModelId = modelId
  }),

  setAdvancedSettings: (newSettings) => set((state) => {
    const currentSettings = state.settings.advancedSettings || {
      shifts: 1,
      overlap: 0.25,
      segmentSize: 256,
      outputFormat: 'wav',
      bitrate: '320k'
    }

    const entries = Object.entries(newSettings || {});
    const cleanUpdates = Object.fromEntries(
      entries.filter(([_, v]) => v !== undefined)
    )

    state.settings.advancedSettings = {
      ...currentSettings,
      ...cleanUpdates
    }
  }),

  setWatchMode: (enabled) => set((state) => {
    state.watchModeEnabled = enabled
  }),

  setWatchPath: (path) => set((state) => {
    state.watchPath = path
  }),

  setPhaseParams: (params: PhaseParams) => set((state) => {
    state.settings.phaseParams = params
  }),
})
