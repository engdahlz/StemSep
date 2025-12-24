import { useState } from 'react'
import { useStore } from '../stores/useStore'
import { useTheme } from '../contexts/ThemeContext'
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from './ui/card'
import { Button } from './ui/button'
import { Input } from './ui/input'
import { Monitor, Moon, Sun, FolderOpen, RefreshCw, Volume2, Settings2, HardDrive } from 'lucide-react'

type Tab = 'general' | 'audio' | 'advanced'

export function SettingsPage() {
    const [activeTab, setActiveTab] = useState<Tab>('general')
    const { theme, setTheme } = useTheme()
    const {
        settings,
        setTheme: setStoreTheme,
        setDefaultOutputDir,
        setModelsDir,
        setDefaultModel,
        models
    } = useStore()

    const handleThemeChange = (newTheme: 'light' | 'dark' | 'system') => {
        setTheme(newTheme)
        setStoreTheme(newTheme)
    }

    const handleBrowseOutputDir = async () => {
        try {
            const result = await window.electronAPI.selectOutputDirectory()
            if (result) {
                setDefaultOutputDir(result)
            }
        } catch (error) {
            console.error('Failed to open directory dialog:', error)
        }
    }

    const handleBrowseModelsDir = async () => {
        try {
            const result = await window.electronAPI.selectOutputDirectory()
            if (result) {
                const newPath = result

                // Show confirmation with explanation
                const confirmed = confirm(
                    `Change models directory to:\n${newPath}\n\n` +
                    `⚠️ Important:\n` +
                    `• Models will NOT be moved automatically\n` +
                    `• You will need to re-download your models\n` +
                    `• Restart the app for changes to take effect\n\n` +
                    `Continue?`
                )

                if (!confirmed) return

                setModelsDir(newPath)
                // Also save to app-config so main process can read at startup
                await window.electronAPI.saveAppConfig?.({ modelsDir: newPath })

                // Show restart reminder
                alert('Models directory updated!\n\nPlease restart the app and re-download your models.')
            }
        } catch (error) {
            console.error('Failed to open directory dialog:', error)
        }
    }

    return (
        <div className="h-full flex flex-col p-6 gap-6 overflow-hidden">
            <div>
                <h1 className="text-3xl font-bold flex items-center gap-3">
                    <Settings2 className="w-8 h-8" />
                    Settings
                </h1>
                <p className="text-muted-foreground mt-2">
                    Configure application preferences and defaults.
                </p>
            </div>

            {/* Custom Tabs */}
            <div className="flex gap-2 border-b border-border">
                <button
                    onClick={() => setActiveTab('general')}
                    className={`px-4 py-2 text-sm font-medium border-b-2 transition-colors ${activeTab === 'general'
                        ? 'border-primary text-primary'
                        : 'border-transparent text-muted-foreground hover:text-foreground'
                        }`}
                >
                    General
                </button>
                <button
                    onClick={() => setActiveTab('audio')}
                    className={`px-4 py-2 text-sm font-medium border-b-2 transition-colors ${activeTab === 'audio'
                        ? 'border-primary text-primary'
                        : 'border-transparent text-muted-foreground hover:text-foreground'
                        }`}
                >
                    Audio & Models
                </button>
                <button
                    onClick={() => setActiveTab('advanced')}
                    className={`px-4 py-2 text-sm font-medium border-b-2 transition-colors ${activeTab === 'advanced'
                        ? 'border-primary text-primary'
                        : 'border-transparent text-muted-foreground hover:text-foreground'
                        }`}
                >
                    Advanced
                </button>
            </div>

            <div className="flex-1 overflow-y-auto">
                <div className="max-w-3xl space-y-6">

                    {/* General Tab */}
                    {activeTab === 'general' && (
                        <>
                            <Card>
                                <CardHeader>
                                    <CardTitle className="flex items-center gap-2">
                                        <Monitor className="w-5 h-5" />
                                        Appearance
                                    </CardTitle>
                                    <CardDescription>
                                        Customize how the application looks.
                                    </CardDescription>
                                </CardHeader>
                                <CardContent className="space-y-4">
                                    <div className="grid grid-cols-3 gap-4">
                                        <button
                                            onClick={() => handleThemeChange('light')}
                                            className={`flex flex-col items-center gap-3 p-4 rounded-xl border-2 transition-all duration-200 btn-hover btn-active ${theme === 'light'
                                                ? 'border-primary bg-primary/10 ring-2 ring-primary/20 shadow-sm'
                                                : 'border-border hover:border-primary/50 hover:bg-accent/50'
                                                }`}
                                        >
                                            <Sun className={`w-8 h-8 ${theme === 'light' ? 'text-primary fill-primary/20' : 'text-muted-foreground'}`} />
                                            <span className={`font-medium ${theme === 'light' ? 'text-primary' : 'text-muted-foreground'}`}>Light</span>
                                        </button>
                                        <button
                                            onClick={() => handleThemeChange('dark')}
                                            className={`flex flex-col items-center gap-3 p-4 rounded-xl border-2 transition-all duration-200 btn-hover btn-active ${theme === 'dark'
                                                ? 'border-primary bg-primary/10 ring-2 ring-primary/20 shadow-sm'
                                                : 'border-border hover:border-primary/50 hover:bg-accent/50'
                                                }`}
                                        >
                                            <Moon className={`w-8 h-8 ${theme === 'dark' ? 'text-primary fill-primary/20' : 'text-muted-foreground'}`} />
                                            <span className={`font-medium ${theme === 'dark' ? 'text-primary' : 'text-muted-foreground'}`}>Dark</span>
                                        </button>
                                        <button
                                            onClick={() => handleThemeChange('system')}
                                            className={`flex flex-col items-center gap-3 p-4 rounded-xl border-2 transition-all duration-200 btn-hover btn-active ${theme === 'system'
                                                ? 'border-primary bg-primary/10 ring-2 ring-primary/20 shadow-sm'
                                                : 'border-border hover:border-primary/50 hover:bg-accent/50'
                                                }`}
                                        >
                                            <Monitor className={`w-8 h-8 ${theme === 'system' ? 'text-primary' : 'text-muted-foreground'}`} />
                                            <span className={`font-medium ${theme === 'system' ? 'text-primary' : 'text-muted-foreground'}`}>System</span>
                                        </button>
                                    </div>
                                </CardContent>
                            </Card>

                            <Card>
                                <CardHeader>
                                    <CardTitle className="flex items-center gap-2">
                                        <FolderOpen className="w-5 h-5" />
                                        Default Output
                                    </CardTitle>
                                    <CardDescription>
                                        Set the default directory where separated stems will be saved.
                                    </CardDescription>
                                </CardHeader>
                                <CardContent className="space-y-4">
                                    <div className="flex gap-2">
                                        <Input
                                            value={settings.defaultOutputDir || ''}
                                            readOnly
                                            placeholder="Select a directory..."
                                            className="font-mono text-sm"
                                        />
                                        <Button onClick={handleBrowseOutputDir} variant="outline">
                                            Browse
                                        </Button>
                                    </div>
                                </CardContent>
                            </Card>

                            <Card>
                                <CardHeader>
                                    <CardTitle className="flex items-center gap-2">
                                        <HardDrive className="w-5 h-5" />
                                        Models Directory
                                    </CardTitle>
                                    <CardDescription>
                                        Where AI models are downloaded and stored.
                                    </CardDescription>
                                </CardHeader>
                                <CardContent className="space-y-4">
                                    <div className="flex gap-2">
                                        <Input
                                            value={settings.modelsDir || '~/.stemsep/models (default)'}
                                            readOnly
                                            placeholder="Using default location..."
                                            className="font-mono text-sm"
                                        />
                                        <Button onClick={handleBrowseModelsDir} variant="outline">
                                            Browse
                                        </Button>
                                    </div>
                                    <p className="text-xs text-amber-500 flex items-center gap-1">
                                        ⚠️ Requires app restart to take effect. Existing models are not moved.
                                    </p>
                                </CardContent>
                            </Card>
                        </>
                    )}

                    {/* Audio Tab */}
                    {activeTab === 'audio' && (
                        <Card>
                            <CardHeader>
                                <CardTitle className="flex items-center gap-2">
                                    <Volume2 className="w-5 h-5" />
                                    Default Model
                                </CardTitle>
                                <CardDescription>
                                    Choose which model is selected by default when starting the app.
                                </CardDescription>
                            </CardHeader>
                            <CardContent>
                                <select
                                    value={settings.defaultModelId || ''}
                                    onChange={(e) => setDefaultModel(e.target.value)}
                                    className="w-full p-2 bg-background border border-input rounded-md text-foreground focus:ring-2 focus:ring-primary focus:border-transparent outline-none"
                                >
                                    <option value="">-- Select a default model --</option>
                                    {models.filter(m => m.installed).map(model => (
                                        <option key={model.id} value={model.id}>
                                            {model.name} ({model.category})
                                        </option>
                                    ))}
                                </select>
                                <p className="text-xs text-muted-foreground mt-2">
                                    Only installed models are shown.
                                </p>
                            </CardContent>
                        </Card>
                    )}

                    {/* Advanced Tab */}
                    {activeTab === 'advanced' && (
                        <Card>
                            <CardHeader>
                                <CardTitle className="flex items-center gap-2">
                                    <HardDrive className="w-5 h-5" />
                                    System
                                </CardTitle>
                                <CardDescription>
                                    Advanced system configurations.
                                </CardDescription>
                            </CardHeader>
                            <CardContent className="space-y-4">
                                <div className="flex items-center justify-between p-4 border rounded-lg">
                                    <div>
                                        <h4 className="font-medium">Reset Settings</h4>
                                        <p className="text-sm text-muted-foreground">
                                            Restore all settings to their default values.
                                        </p>
                                    </div>
                                    <Button
                                        variant="destructive"
                                        onClick={() => {
                                            if (confirm('Are you sure you want to reset all settings?')) {
                                                setTheme('system')
                                                setStoreTheme('system')
                                                setDefaultOutputDir('')
                                                setDefaultModel('')
                                            }
                                        }}
                                    >
                                        <RefreshCw className="w-4 h-4 mr-2" />
                                        Reset
                                    </Button>
                                </div>
                            </CardContent>
                        </Card>
                    )}
                </div>
            </div>
        </div>
    )
}
