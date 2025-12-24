import { useState, useEffect } from 'react'
import { X, Cpu, Zap, Sliders, Settings as SettingsIcon, AlertTriangle, FolderInput } from 'lucide-react'
import { useStore } from '../stores/useStore'
import { toast } from 'sonner'
import { Button } from './ui/button' // Need Button component

interface GpuDevice {
  id: string
  name: string
  type: 'cpu' | 'cuda'
  memory_gb: number | null
  index?: number
  recommended: boolean
}

interface SettingsDialogProps {
  isOpen: boolean
  onClose: () => void
}

export default function SettingsDialog({ isOpen, onClose }: SettingsDialogProps) {
  const [devices, setDevices] = useState<GpuDevice[]>([])
  const [selectedDevice, setSelectedDevice] = useState<string>('cpu')
  const [loading, setLoading] = useState(true)
  const [activeTab, setActiveTab] = useState<'general' | 'advanced' | 'automation'>('general')
  const [systemInfo, setSystemInfo] = useState<{
    has_cuda: boolean
    cuda_version: string | null
    gpu_count: number
    memory_total_gb?: number
    recommended_profile?: {
      profile_name: string
      settings: {
        batch_size?: number
        use_amp?: boolean
        chunk_size_modifier?: number
        description?: string
      }
      vram_gb: number
    }
  } | null>(null)

  // Automation state
  const watchEnabled = useStore(state => state.watchModeEnabled)
  const watchPath = useStore(state => state.watchPath)
  const setWatchMode = useStore(state => state.setWatchMode)
  const setWatchPath = useStore(state => state.setWatchPath)

  // Advanced settings state
  const advancedSettings = useStore(state => state.settings.advancedSettings)
  const setAdvancedSettings = useStore(state => state.setAdvancedSettings)

  const [localAdvanced, setLocalAdvanced] = useState({
    shifts: advancedSettings?.shifts || 1,
    overlap: advancedSettings?.overlap || 0.25,
    segmentSize: advancedSettings?.segmentSize || 256,
    outputFormat: advancedSettings?.outputFormat || 'wav',
    bitrate: advancedSettings?.bitrate || '320k'
  })

  // Update local state when store changes
  useEffect(() => {
    if (advancedSettings) {
      setLocalAdvanced({
        shifts: advancedSettings.shifts,
        overlap: advancedSettings.overlap,
        segmentSize: advancedSettings.segmentSize,
        outputFormat: advancedSettings.outputFormat,
        bitrate: advancedSettings.bitrate
      })
    }
  }, [advancedSettings])

  useEffect(() => {
    const loadInfo = async () => {
      if (!window.electronAPI) return

      try {
        const info = await window.electronAPI.getGpuDevices()
        // The API returns the info object directly ({ gpus: [...] })
        if (info && info.gpus) {
          const devicesList = info.gpus || []

          setDevices(devicesList)

          // Auto-select recommended
          const recommended = devicesList.find((d: any) => d.recommended)
          if (recommended) {
            setSelectedDevice(recommended.id)
          }

          setSystemInfo({
            has_cuda: info.has_cuda,
            cuda_version: info.cuda_version,
            gpu_count: devicesList.length,
            memory_total_gb: info.system_info?.memory_total_gb
          })
        } else {
          // toast.error('No GPU info returned') // Silent fail on init often better
        }
      } catch (error) {
        console.error('Failed to load GPU info:', error)
        // Fallback to CPU
        setDevices([{
          id: 'cpu',
          name: 'CPU (No GPU acceleration)',
          type: 'cpu',
          memory_gb: null,
          recommended: true
        }])
      } finally {
        setLoading(false)
      }
    }

    if (isOpen) {
      loadInfo()
    }
  }, [isOpen])

  const handleSave = async () => {
    localStorage.setItem('stemSepGpuDevice', selectedDevice)
    setAdvancedSettings(localAdvanced)

    // Handle Watch Mode
    if (watchEnabled && watchPath) {
      if (window.electronAPI?.startWatchMode) {
        await window.electronAPI.startWatchMode(watchPath)
        toast.success('Watch mode started')
      }
    } else {
      if (window.electronAPI?.stopWatchMode) {
        await window.electronAPI.stopWatchMode()
        if (watchEnabled && !watchPath) toast.error('Watch mode disabled: No folder selected')
      }
    }

    onClose()
  }

  const handleDeviceChange = (deviceId: string) => {
    setSelectedDevice(deviceId)
  }

  const handleBrowseWatch = async () => {
    if (window.electronAPI?.selectOutputDirectory) {
      const path = await window.electronAPI.selectOutputDirectory()
      if (path) setWatchPath(path)
    }
  }

  if (!isOpen) return null

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/60 backdrop-blur-sm">
      <div className="bg-neutral-900 border border-neutral-700 rounded-lg shadow-2xl w-full max-w-2xl max-h-[90vh] overflow-hidden flex flex-col">
        {/* Header */}
        <div className="flex items-center justify-between p-6 border-b border-neutral-700">
          <h2 className="text-2xl font-bold text-white">Settings</h2>
          <button
            onClick={onClose}
            className="p-2 rounded-lg hover:bg-neutral-800 transition-colors"
            aria-label="Close settings"
          >
            <X className="w-5 h-5 text-neutral-400" />
          </button>
        </div>

        {/* Tabs */}
        <div className="flex border-b border-neutral-700 px-6">
          <button
            onClick={() => setActiveTab('general')}
            className={`py-3 px-4 text-sm font-medium border-b-2 transition-colors flex items-center gap-2 ${activeTab === 'general'
              ? 'border-primary text-primary'
              : 'border-transparent text-neutral-400 hover:text-white'
              }`}
          >
            <SettingsIcon className="w-4 h-4" />
            General
          </button>
          <button
            onClick={() => setActiveTab('automation')}
            className={`py-3 px-4 text-sm font-medium border-b-2 transition-colors flex items-center gap-2 ${activeTab === 'automation'
              ? 'border-primary text-primary'
              : 'border-transparent text-neutral-400 hover:text-white'
              }`}
          >
            <FolderInput className="w-4 h-4" />
            Automation
          </button>
          <button
            onClick={() => setActiveTab('advanced')}
            className={`py-3 px-4 text-sm font-medium border-b-2 transition-colors flex items-center gap-2 ${activeTab === 'advanced'
              ? 'border-primary text-primary'
              : 'border-transparent text-neutral-400 hover:text-white'
              }`}
          >
            <Sliders className="w-4 h-4" />
            Advanced
          </button>
        </div>

        {/* Content */}
        <div className="flex-1 overflow-y-auto p-6">
          {activeTab === 'general' ? (
            /* GPU Selection Section */
            <div className="mb-8">
              <h3 className="text-lg font-semibold text-white mb-2 flex items-center gap-2">
                <Zap className="w-5 h-5 text-primary" />
                GPU Acceleration
              </h3>
              <p className="text-sm text-neutral-400 mb-4">
                Choose which device to use for audio separation. GPU acceleration significantly improves processing speed.
              </p>

              {loading ? (
                <div className="flex items-center justify-center py-8">
                  <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-primary"></div>
                </div>
              ) : (
                <div className="space-y-3">
                  {devices.map((device) => (
                    <button
                      key={device.id}
                      onClick={() => handleDeviceChange(device.id)}
                      className={`w-full p-4 rounded-lg border-2 transition-all text-left ${selectedDevice === device.id
                        ? 'border-primary bg-primary/10'
                        : 'border-neutral-700 bg-neutral-800 hover:border-neutral-600'
                        }`}
                    >
                      <div className="flex items-start justify-between">
                        <div className="flex items-start gap-3 flex-1">
                          {device.type === 'cpu' ? (
                            <Cpu className="w-5 h-5 mt-1 text-neutral-400" />
                          ) : (
                            <Zap className="w-5 h-5 mt-1 text-green-500" />
                          )}
                          <div className="flex-1">
                            <div className="font-medium text-white">{device.name}</div>
                            {device.memory_gb && (
                              <div className="text-sm text-neutral-400 mt-1">
                                {device.memory_gb}GB VRAM
                              </div>
                            )}
                            {device.type === 'cpu' && (
                              <div className="text-sm text-neutral-400 mt-1">
                                Slower processing but works on any system
                              </div>
                            )}
                          </div>
                        </div>
                        {device.recommended && (
                          <span className="px-2 py-1 text-xs font-semibold bg-green-500/20 text-green-400 rounded">
                            Recommended
                          </span>
                        )}
                      </div>
                    </button>
                  ))}
                </div>
              )}

              {/* System Info */}
              {systemInfo && (
                <div className="mt-4 p-3 bg-neutral-800 rounded-lg border border-neutral-700">
                  <div className="text-sm text-neutral-400">
                    <div className="flex items-center justify-between mb-1">
                      <span>System RAM:</span>
                      <span className="text-white">{systemInfo.memory_total_gb ? `${systemInfo.memory_total_gb} GB` : 'Unknown'}</span>
                    </div>
                    <div className="flex items-center justify-between mb-1">
                      <span>CUDA Available:</span>
                      <span className={systemInfo.has_cuda ? 'text-green-400' : 'text-neutral-500'}>
                        {systemInfo.has_cuda ? 'Yes' : 'No'}
                      </span>
                    </div>
                    {systemInfo.has_cuda && systemInfo.cuda_version && (
                      <div className="flex items-center justify-between mb-1">
                        <span>CUDA Version:</span>
                        <span className="text-white">{systemInfo.cuda_version}</span>
                      </div>
                    )}
                    <div className="flex items-center justify-between">
                      <span>GPU Count:</span>
                      <span className="text-white">{systemInfo.gpu_count}</span>
                    </div>
                    {systemInfo.recommended_profile && (
                      <div className="mt-3 pt-3 border-t border-neutral-700">
                        <div className="flex items-center gap-2 mb-2">
                          <Zap className="w-4 h-4 text-green-400" />
                          <span className="text-green-400 font-medium text-sm">Optimized Profile: {systemInfo.recommended_profile.profile_name}</span>
                        </div>
                        <div className="grid grid-cols-2 gap-x-4 gap-y-1 text-xs">
                          {systemInfo.recommended_profile.settings.batch_size && (
                            <>
                              <span className="text-neutral-500">Batch Size:</span>
                              <span className="text-neutral-300">{systemInfo.recommended_profile.settings.batch_size}</span>
                            </>
                          )}
                          <span className="text-neutral-500">AMP:</span>
                          <span className={systemInfo.recommended_profile.settings.use_amp ? 'text-green-400' : 'text-neutral-500'}>
                            {systemInfo.recommended_profile.settings.use_amp ? 'Enabled' : 'Disabled'}
                          </span>
                          <span className="text-neutral-500">VRAM:</span>
                          <span className="text-neutral-300">{systemInfo.recommended_profile.vram_gb.toFixed(1)} GB</span>
                        </div>
                        {systemInfo.recommended_profile.settings.description && (
                          <p className="text-neutral-500 text-xs mt-2 italic">
                            {systemInfo.recommended_profile.settings.description}
                          </p>
                        )}
                      </div>
                    )}
                  </div>
                </div>
              )}
            </div>
          ) : activeTab === 'automation' ? (
            /* Automation Section */
            <div className="space-y-6">
              <div className="bg-neutral-800/50 rounded-lg p-6 border border-neutral-700">
                <div className="flex items-center justify-between mb-6">
                  <div>
                    <h3 className="text-lg font-medium text-white">Watch Folder Mode</h3>
                    <p className="text-sm text-neutral-400">Automatically separate files added to a folder.</p>
                  </div>
                  <button
                    onClick={() => setWatchMode(!watchEnabled)}
                    className={`w-12 h-6 rounded-full transition-colors relative ${watchEnabled ? 'bg-primary' : 'bg-neutral-600'}`}
                  >
                    <div className={`w-4 h-4 bg-white rounded-full absolute top-1 transition-all ${watchEnabled ? 'left-7' : 'left-1'}`} />
                  </button>
                </div>

                <div className="space-y-4">
                  <div>
                    <label className="block text-sm font-medium text-white mb-2">Monitored Folder</label>
                    <div className="flex gap-2">
                      <input
                        type="text"
                        value={watchPath}
                        readOnly
                        className="flex-1 bg-neutral-900 border border-neutral-700 rounded-lg px-3 py-2 text-sm text-neutral-300 placeholder-neutral-600"
                        placeholder="Select a folder to watch..."
                      />
                      <Button variant="outline" size="sm" onClick={handleBrowseWatch}>Browse</Button>
                    </div>
                    <p className="text-xs text-neutral-500 mt-2">
                      Any audio file (mp3, wav, flac) added to this folder will be automatically processed using the currently selected default model.
                    </p>
                  </div>
                </div>
              </div>
            </div>
          ) : (
            /* Advanced Settings Section */
            <div className="space-y-6">
              <div className="bg-yellow-500/10 border border-yellow-500/20 rounded-lg p-4 flex items-start gap-3">
                <AlertTriangle className="w-5 h-5 text-yellow-500 shrink-0 mt-0.5" />
                <div className="text-sm text-yellow-200/80">
                  <p className="font-semibold text-yellow-500 mb-1">Performance Warning</p>
                  Increasing shifts or overlap significantly increases processing time. Use with caution, especially on CPU.
                </div>
              </div>

              {/* Shifts */}
              <div>
                <label className="block text-sm font-medium text-white mb-2">
                  Shifts ({localAdvanced.shifts})
                </label>
                <input
                  type="range"
                  min="1"
                  max="10"
                  step="1"
                  value={localAdvanced.shifts}
                  onChange={(e) => setLocalAdvanced(prev => ({ ...prev, shifts: parseInt(e.target.value) }))}
                  className="w-full h-2 bg-neutral-700 rounded-lg appearance-none cursor-pointer accent-primary"
                />
                <p className="text-xs text-neutral-400 mt-1">
                  Performs multiple predictions with random shifts to reduce artifacts. Higher = better quality, slower.
                </p>
              </div>

              {/* Overlap */}
              <div>
                <label className="block text-sm font-medium text-white mb-2">
                  Overlap ({(localAdvanced.overlap * 100).toFixed(0)}%)
                </label>
                <input
                  type="range"
                  min="0.1"
                  max="0.9"
                  step="0.05"
                  value={localAdvanced.overlap}
                  onChange={(e) => setLocalAdvanced(prev => ({ ...prev, overlap: parseFloat(e.target.value) }))}
                  className="w-full h-2 bg-neutral-700 rounded-lg appearance-none cursor-pointer accent-primary"
                />
                <p className="text-xs text-neutral-400 mt-1">
                  Amount of overlap between prediction segments. Higher values can smooth out transitions but increase time.
                </p>
              </div>

              {/* Segment Size */}
              <div>
                <label className="block text-sm font-medium text-white mb-2">
                  Segment Size ({localAdvanced.segmentSize})
                </label>
                <select
                  value={localAdvanced.segmentSize}
                  onChange={(e) => setLocalAdvanced(prev => ({ ...prev, segmentSize: parseInt(e.target.value) }))}
                  className="w-full bg-neutral-800 border border-neutral-700 rounded-lg p-2.5 text-white focus:ring-2 focus:ring-primary focus:border-transparent"
                >
                  <option value="256">256 (Default)</option>
                  <option value="320">320</option>
                  <option value="400">400</option>
                  <option value="512">512 (High VRAM)</option>
                </select>
                <p className="text-xs text-neutral-400 mt-1">
                  Length of audio segments processed at once. Larger segments require more VRAM.
                </p>
              </div>

              <div className="grid grid-cols-2 gap-4">
                {/* Output Format */}
                <div>
                  <label className="block text-sm font-medium text-white mb-2">
                    Output Format
                  </label>
                  <select
                    value={localAdvanced.outputFormat}
                    onChange={(e) => setLocalAdvanced(prev => ({ ...prev, outputFormat: e.target.value as any }))}
                    className="w-full bg-neutral-800 border border-neutral-700 rounded-lg p-2.5 text-white focus:ring-2 focus:ring-primary focus:border-transparent"
                  >
                    <option value="mp3">MP3</option>
                    <option value="wav">WAV</option>
                    <option value="flac">FLAC</option>
                  </select>
                </div>

                {/* Bitrate (MP3 only) */}
                <div>
                  <label className="block text-sm font-medium text-white mb-2">
                    Bitrate (MP3)
                  </label>
                  <select
                    value={localAdvanced.bitrate}
                    onChange={(e) => setLocalAdvanced(prev => ({ ...prev, bitrate: e.target.value }))}
                    disabled={localAdvanced.outputFormat !== 'mp3'}
                    className={`w-full bg-neutral-800 border border-neutral-700 rounded-lg p-2.5 text-white focus:ring-2 focus:ring-primary focus:border-transparent ${localAdvanced.outputFormat !== 'mp3' ? 'opacity-50 cursor-not-allowed' : ''
                      }`}
                  >
                    <option value="128k">128k</option>
                    <option value="192k">192k</option>
                    <option value="256k">256k</option>
                    <option value="320k">320k (High Quality)</option>
                  </select>
                </div>
              </div>
            </div>
          )}
        </div>

        {/* Footer */}
        <div className="flex items-center justify-end gap-3 p-6 border-t border-neutral-700">
          <button
            onClick={onClose}
            className="px-4 py-2 rounded-lg border border-neutral-600 text-white hover:bg-neutral-800 transition-colors"
          >
            Cancel
          </button>
          <button
            onClick={handleSave}
            className="px-4 py-2 rounded-lg bg-primary text-white font-semibold hover:bg-primary/90 transition-colors"
          >
            Save Changes
          </button>
        </div>
      </div>
    </div>
  )
}
