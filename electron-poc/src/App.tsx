import { useState, useEffect } from 'react'
import { ThemeProvider } from './contexts/ThemeContext'
import { ErrorBoundary } from './components/ErrorBoundary'
import { Sidebar } from './components/Sidebar'
import SeparatePage from './components/SeparatePage'
import { ModelsPage } from './components/ModelsPage'
import { PresetsPage } from './components/PresetsPage'
import { HistoryPage } from './components/HistoryPage'
import { SettingsPage } from './components/SettingsPage'
import { ConfigurePage, SeparationConfig } from './components/ConfigurePage'
import { logger } from './utils/logger'
import { useModelEvents } from './hooks/useModelEvents'
import { useModels } from './hooks/useModels'
import { useStore } from './stores/useStore'
import { Toaster, toast } from 'sonner'
import { ALL_PRESETS } from './presets'
import './App.css'

import { ResultsPage } from './components/ResultsPage'

type Page = 'home' | 'models' | 'settings' | 'history' | 'presets' | 'about' | 'results' | 'configure'

interface ConfigureFileInfo {
  name: string
  path: string
  presetId?: string
}

function App() {
  const [currentPage, setCurrentPage] = useState<Page>('home')
  const [preSelectedModel, setPreSelectedModel] = useState<string | undefined>()
  const [configureFile, setConfigureFile] = useState<ConfigureFileInfo | null>(null)

  // Initialize global model event listeners
  useModelEvents()
  // Load models globally
  useModels()

  const models = useStore(state => state.models)
  const recipes = useStore(state => state.recipes)

  // Load persistent queue on startup
  useEffect(() => {
    const loadQueue = async () => {
      await useStore.getState().loadQueue()
      const queue = useStore.getState().separation.queue
      const pendingCount = queue.filter(i => i.status === 'pending' || i.status === 'processing').length
      if (pendingCount > 0) {
        toast.info(`Resumed session with ${pendingCount} pending items in queue.`)
      }
    }
    loadQueue()
  }, [])

  useEffect(() => {
    logger.info('Application started', { version: '1.0.0' }, 'App')

    // Log page changes
    logger.debug(`Navigated to page: ${currentPage}`, undefined, 'Navigation')

    // Listen for backend errors
    if (window.electronAPI) {
      const removeBackendErrorListener = window.electronAPI.onBackendError((data) => {
        logger.error('Backend error received', data, 'App')
        toast.error('Backend Error', {
          description: data.error || 'An unexpected error occurred in the backend.',
          duration: 10000,
        })
      })

      return () => {
        removeBackendErrorListener()
      }
    }
  }, [currentPage])

  const handleNavigateToModels = (modelId?: string) => {
    setPreSelectedModel(modelId)
    setCurrentPage('models')
  }

  const handleNavigateToConfigure = (fileName: string, filePath: string, presetId?: string) => {
    setConfigureFile({ name: fileName, path: filePath, presetId })
    setCurrentPage('configure')
  }

  // Store pending separation config for SeparatePage to pick up
  const [pendingSeparationConfig, setPendingSeparationConfig] = useState<{ config: SeparationConfig, file: ConfigureFileInfo } | null>(null)

  const handleConfigureConfirm = (config: SeparationConfig) => {
    if (!configureFile) return

    logger.info('Separation configured', { config, file: configureFile }, 'App')

    // Store config for SeparatePage to use
    setPendingSeparationConfig({ config, file: configureFile })

    // Navigate back to home - SeparatePage will pick up the config
    setCurrentPage('home')
    setConfigureFile(null)
  }

  // Build availability map from recipes
  const availability: Record<string, any> = {}
  recipes.forEach(r => {
    availability[r.model_id] = {
      available: r.installed,
      model_id: r.model_id,
      model_name: r.name,
      model_exists: true,
      installed: r.installed,
      file_size: r.file_size || 0
    }
  })

  const renderPage = () => {
    switch (currentPage) {
      case 'home':
        return <SeparatePage
          onNavigateToModels={handleNavigateToModels}
          onNavigateToConfigure={handleNavigateToConfigure}
          onNavigateToResults={() => setCurrentPage('results')}
          pendingSeparationConfig={pendingSeparationConfig}
          onClearPendingConfig={() => setPendingSeparationConfig(null)}
        />
      case 'configure':
        if (!configureFile) {
          setCurrentPage('home')
          return null
        }
        return <ConfigurePage
          fileName={configureFile.name}
          filePath={configureFile.path}
          onBack={() => {
            setCurrentPage('home')
            setConfigureFile(null)
          }}
          onConfirm={handleConfigureConfirm}
          initialPresetId={configureFile.presetId}
          presets={ALL_PRESETS}
          availability={availability}
          models={models}
          onNavigateToModels={handleNavigateToModels}
        />
      case 'models':
        return <ModelsPage
          preSelectedModel={preSelectedModel}
          onModelDownloadComplete={() => {
            // Clear pre-selected model after download
            setPreSelectedModel(undefined)
          }}
          onBack={() => {
            setCurrentPage('home')
            setPreSelectedModel(undefined)
          }}
        />
      case 'results':
        return <ResultsPage onBack={() => setCurrentPage('home')} />
      case 'settings':
        return <SettingsPage />
      case 'history':
        return <HistoryPage onNavigate={setCurrentPage} />
      case 'presets':
        return <PresetsPage />
      case 'about':
        return <div className="p-8"><h1 className="text-3xl font-bold">About</h1><p className="text-muted-foreground mt-2">StemSep - Advanced Audio Stem Separation</p></div>
      default:
        return <SeparatePage />
    }
  }

  return (
    <ErrorBoundary>
      <ThemeProvider>
        <div className="flex h-screen w-screen overflow-hidden bg-background text-foreground">
          <Sidebar currentPage={currentPage} onPageChange={setCurrentPage} />
          <main className="flex-1 overflow-hidden">
            {renderPage()}
          </main>
        </div>
        <Toaster />
      </ThemeProvider>
    </ErrorBoundary>
  )
}

export default App

