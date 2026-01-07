import { useState } from 'react'
import { Home, Music, Settings, Clock, Info, ChevronLeft, ChevronRight, PlayCircle } from 'lucide-react'
import { Button } from './ui/button'
import { cn } from '../lib/utils'
import { SystemStatus } from './SystemStatus'
import { StemSepLogo } from './StemSepLogo'
import type { Page } from '../types/navigation'

interface SidebarProps {
  currentPage: Page
  onPageChange: (page: Page) => void
}

export function Sidebar({ currentPage, onPageChange }: SidebarProps) {
  const [isCollapsed, setIsCollapsed] = useState(false)

  const navSections = [
    {
      title: 'Workspace',
      items: [
        { id: 'home' as Page, icon: Home, label: 'Home' },
        { id: 'results' as Page, icon: PlayCircle, label: 'Results' },
        { id: 'history' as Page, icon: Clock, label: 'History' },
      ],
    },
    {
      title: 'Storage',
      items: [{ id: 'models' as Page, icon: Music, label: 'Model Library' }],
    },
    {
      title: 'System',
      items: [
        { id: 'settings' as Page, icon: Settings, label: 'Settings' },
        { id: 'about' as Page, icon: Info, label: 'About' },
      ],
    },
  ]

  return (
    <div
      className={cn(
        "flex flex-col h-full bg-background border-r border-border transition-all duration-300",
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
      <nav className="flex-1 p-2 space-y-3">
        {navSections.map((section) => (
          <div key={section.title} className="space-y-1">
            {!isCollapsed && (
              <div className="px-3 pt-1 text-[11px] font-medium text-muted-foreground uppercase tracking-wider">
                {section.title}
              </div>
            )}
            <div className="space-y-1">
              {section.items.map((item) => {
                const Icon = item.icon
                const isActive = currentPage === item.id

                return (
                  <Button
                    key={item.id}
                    variant="ghost"
                    className={cn(
                      "w-full justify-start gap-3 text-muted-foreground hover:text-foreground",
                      isCollapsed && "justify-center",
                      isActive && "bg-muted text-foreground"
                    )}
                    onClick={() => onPageChange(item.id)}
                  >
                    <Icon size={18} />
                    {!isCollapsed && <span className="text-sm">{item.label}</span>}
                  </Button>
                )
              })}
            </div>
          </div>
        ))}
      </nav>

      {/* System Status */}
      {!isCollapsed && <SystemStatus />}
    </div>
  )
}
