import { useState } from 'react'
import { Home, Music, Settings, Clock, Bookmark, Info, ChevronLeft, ChevronRight, PlayCircle } from 'lucide-react'
import { Button } from './ui/button'
import { cn } from '../lib/utils'
import { SystemStatus } from './SystemStatus'
import { StemSepLogo } from './StemSepLogo'

type Page = 'home' | 'models' | 'settings' | 'history' | 'presets' | 'about' | 'results' | 'configure'

interface SidebarProps {
  currentPage: Page
  onPageChange: (page: Page) => void
}

export function Sidebar({ currentPage, onPageChange }: SidebarProps) {
  const [isCollapsed, setIsCollapsed] = useState(false)

  const menuItems = [
    { id: 'home' as Page, icon: Home, label: 'Separate' },
    { id: 'results' as Page, icon: PlayCircle, label: 'Results' },
    { id: 'models' as Page, icon: Music, label: 'Models' },
    { id: 'history' as Page, icon: Clock, label: 'History' },
    { id: 'presets' as Page, icon: Bookmark, label: 'Presets' },
    { id: 'settings' as Page, icon: Settings, label: 'Settings' },
    { id: 'about' as Page, icon: Info, label: 'About' },
  ]

  return (
    <div
      className={cn(
        "flex flex-col h-full bg-secondary border-r border-border transition-all duration-300",
        isCollapsed ? "w-16" : "w-64"
      )}
    >
      {/* Header */}
      <div className="flex items-center justify-between p-4 border-b border-border">
        {!isCollapsed && (
          <div className="flex items-center gap-2">
            <StemSepLogo className="h-6 w-6 text-foreground" />
            <h1 className="text-xl font-bold">StemSep</h1>
          </div>
        )}
        <Button
          variant="ghost"
          size="icon"
          onClick={() => setIsCollapsed(!isCollapsed)}
          className="ml-auto"
        >
          {isCollapsed ? <ChevronRight size={20} /> : <ChevronLeft size={20} />}
        </Button>
      </div>

      {/* Menu Items */}
      <nav className="flex-1 p-2 space-y-1">
        {menuItems.map((item) => {
          const Icon = item.icon
          const isActive = currentPage === item.id

          return (
            <Button
              key={item.id}
              variant={isActive ? "secondary" : "ghost"}
              className={cn(
                "w-full justify-start gap-3 btn-hover btn-active",
                isCollapsed && "justify-center",
                isActive && "bg-accent"
              )}
              onClick={() => onPageChange(item.id)}
            >
              <Icon size={20} />
              {!isCollapsed && <span>{item.label}</span>}
            </Button>
          )
        })}
      </nav>

      {/* System Status */}
      {!isCollapsed && <SystemStatus />}
    </div>
  )
}
