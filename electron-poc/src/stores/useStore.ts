import { create } from 'zustand'
import { immer } from 'zustand/middleware/immer'
import { persist } from 'zustand/middleware'
import { AppState } from '../types/store'
import { createModelSlice } from './slices/modelSlice'
import { createSeparationSlice } from './slices/separationSlice'
import { createSettingsSlice } from './slices/settingsSlice'

// Re-export types for components to use
export * from '../types/store'

export const useStore = create<AppState>()(
  persist(
    immer((...a) => ({
      ...createModelSlice(...a),
      ...createSeparationSlice(...a),
      ...createSettingsSlice(...a),
    })),
    {
      name: 'stemsep-storage',
      partialize: (state) => ({
        history: state.history,
        settings: state.settings
      }),
    }
  )
)