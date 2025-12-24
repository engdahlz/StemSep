import { StateCreator } from 'zustand'
import { AppState, SeparationSlice } from '../../types/store'

export const createSeparationSlice: StateCreator<AppState, [["zustand/immer", never]], [], SeparationSlice> = (set, get) => ({
  separation: {
    isProcessing: false,
    isPaused: false,
    progress: 0,
    message: '',
    logs: [],
    outputFiles: null,
    error: null,
    queue: [],
  },

  setQueue: (queue) => {
    set((state) => {
      state.separation.queue = queue
    })
    if (window.electronAPI?.saveQueue) {
      window.electronAPI.saveQueue(get().separation.queue)
    }
  },

  addToQueue: (items) => {
    set((state) => {
      const existingFiles = new Set(state.separation.queue.map(i => i.file))
      const uniqueNewItems = items.filter(i => !existingFiles.has(i.file))
      state.separation.queue.push(...uniqueNewItems)
    })
    if (window.electronAPI?.saveQueue) {
      window.electronAPI.saveQueue(get().separation.queue)
    }
  },

  removeFromQueue: (id) => {
    set((state) => {
      state.separation.queue = state.separation.queue.filter(item => item.id !== id)
    })
    if (window.electronAPI?.saveQueue) {
      window.electronAPI.saveQueue(get().separation.queue)
    }
  },

  updateQueueItem: (id, updates) => {
    set((state) => {
      const item = state.separation.queue.find(i => i.id === id)
      if (item) {
        Object.assign(item, updates)
      }
    })
    if (window.electronAPI?.saveQueue) {
      window.electronAPI.saveQueue(get().separation.queue)
    }
  },

  clearQueue: () => {
    set((state) => {
      state.separation.queue = []
      state.separation.outputFiles = null
      state.separation.progress = 0
      state.separation.logs = []
      state.separation.error = null
    })
    if (window.electronAPI?.saveQueue) {
      window.electronAPI.saveQueue([])
    }
  },

  startSeparation: () => set((state) => {
    state.separation.isProcessing = true
    state.separation.progress = 0
    state.separation.message = ''
    state.separation.outputFiles = null
    state.separation.error = null
    state.separation.startTime = Date.now()
    state.separation.logs = [`[${new Date().toLocaleTimeString()}] Starting batch processing...`]
  }),

  cancelSeparation: () => {
    set((state) => {
      state.separation.isProcessing = false
      const item = state.separation.queue.find(i => i.status === 'processing')
      if (item) {
        item.status = 'cancelled'
        item.error = 'Cancelled by user'
        if (window.electronAPI?.cancelSeparation) {
          window.electronAPI.cancelSeparation(item.id).catch(console.error)
        }
      }
      state.separation.logs.push(`[${new Date().toLocaleTimeString()}] Separation cancelled by user.`)
    })
    if (window.electronAPI?.saveQueue) {
      window.electronAPI.saveQueue(get().separation.queue)
    }
  },

  pauseQueue: () => {
    set((state) => {
      state.separation.isPaused = true
      if (window.electronAPI?.pauseQueue) window.electronAPI.pauseQueue()
    })
  },

  resumeQueue: () => {
    set((state) => {
      state.separation.isPaused = false
      if (window.electronAPI?.resumeQueue) window.electronAPI.resumeQueue()
    })
  },

  reorderQueue: (jobIds) => {
    set((state) => {
      const currentQueue = state.separation.queue
      const idMap = new Map(currentQueue.map(i => [i.id, i]))
      const newQ = []
      for (const id of jobIds) {
          const item = idMap.get(id)
          if (item) newQ.push(item)
      }
      // Add any missing (safety)
      for (const item of currentQueue) {
          if (!newQ.find(i => i.id === item.id)) newQ.push(item)
      }
      state.separation.queue = newQ
      
      if (window.electronAPI?.reorderQueue) window.electronAPI.reorderQueue(jobIds)
    })
  },

  setSeparationProgress: (progress, message) => set((state) => {
    state.separation.progress = progress
    state.separation.message = message
    
    // Also update the specific queue item
    const activeItem = state.separation.queue.find(i => i.status === 'processing')
    if (activeItem) {
        activeItem.progress = progress
    }
  }),

  addLog: (message) => set((state) => {
    state.separation.logs.push(`[${new Date().toLocaleTimeString()}] ${message}`)
  }),

  completeSeparation: (outputFiles) => set((state) => {
    state.separation.isProcessing = false
    state.separation.progress = 100
    state.separation.outputFiles = outputFiles
    state.separation.error = null
    state.separation.logs.push(`[${new Date().toLocaleTimeString()}] Batch processing complete!`)
  }),

  failSeparation: (error) => set((state) => {
    state.separation.isProcessing = false
    state.separation.error = error
    state.separation.logs.push(`[${new Date().toLocaleTimeString()}] Error: ${error}`)
  }),

  clearSeparation: () => set((state) => {
    state.separation.isProcessing = false
    state.separation.progress = 0
    state.separation.message = ''
    state.separation.outputFiles = null
    state.separation.error = null
    state.separation.startTime = undefined
    state.separation.logs = []
  }),

  loadQueue: async () => {
    if (!window.electronAPI?.loadQueue) return
    try {
      const savedQueue = await window.electronAPI.loadQueue()
      if (savedQueue && Array.isArray(savedQueue)) {
        // Sanitize queue: Reset 'processing' items to 'pending'
        // This handles cases where the app was closed/crashed during separation
        const sanitizedQueue = savedQueue.map(item => {
            if (item.status === 'processing') {
                return { 
                    ...item, 
                    status: 'pending', 
                    progress: 0, 
                    message: 'Interrupted - Ready to retry' 
                } 
            }
            return item
        })

        set((state) => {
          state.separation.queue = sanitizedQueue
        })
      }
    } catch (error) {
      console.error('Failed to load saved queue:', error)
    }
  },
})
