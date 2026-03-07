import React from 'react'
import ReactDOM from 'react-dom/client'
import App from './App'
import './index.css'
import { installBrowserElectronApi } from './lib/browser/electronApiShim'

if (typeof window !== 'undefined' && !(window as any).electronAPI) {
  installBrowserElectronApi()
}

ReactDOM.createRoot(document.getElementById('root')!).render(
  <React.StrictMode>
    <App />
  </React.StrictMode>,
)
