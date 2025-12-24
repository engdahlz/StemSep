import { createContext, useContext, useEffect, useState } from 'react'
import { useStore } from '../stores/useStore'

type Theme = 'light' | 'dark' | 'system'

type ThemeContextType = {
  theme: Theme
  setTheme: (theme: Theme) => void
}

const ThemeContext = createContext<ThemeContextType | undefined>(undefined)

export function ThemeProvider({ children }: { children: React.ReactNode }) {
  const storeTheme = useStore(state => state.settings.theme)
  const setStoreTheme = useStore(state => state.setTheme)

  const [theme, setThemeState] = useState<Theme>(storeTheme)

  useEffect(() => {
    const root = window.document.documentElement
    root.classList.remove('light', 'dark')

    if (theme === 'system') {
      const systemTheme = window.matchMedia('(prefers-color-scheme: dark)').matches
        ? 'dark'
        : 'light'
      root.classList.add(systemTheme)
    } else {
      root.classList.add(theme)
    }
  }, [theme])

  // Sync with store when store changes (e.g. from other tabs or init)
  useEffect(() => {
    setThemeState(storeTheme)
  }, [storeTheme])

  const setTheme = (newTheme: Theme) => {
    setThemeState(newTheme)
    setStoreTheme(newTheme)
  }

  return (
    <ThemeContext.Provider value={{ theme, setTheme }}>
      {children}
    </ThemeContext.Provider>
  )
}

export function useTheme() {
  const context = useContext(ThemeContext)
  if (!context) {
    throw new Error('useTheme must be used within ThemeProvider')
  }
  return context
}
