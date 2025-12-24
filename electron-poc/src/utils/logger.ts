/**
 * Simple logging utility for frontend debugging
 * Logs to console and stores in memory for later inspection
 */

type LogLevel = 'debug' | 'info' | 'warn' | 'error'

interface LogEntry {
  timestamp: Date
  level: LogLevel
  message: string
  data?: any
  context?: string
}

class Logger {
  private logs: LogEntry[] = []
  private maxLogs = 1000
  private enabled = true

  constructor() {
    // Make logger accessible in dev tools
    if (typeof window !== 'undefined') {
      (window as any).__logger = this
    }
  }

  private log(level: LogLevel, message: string, data?: any, context?: string) {
    if (!this.enabled) return

    const entry: LogEntry = {
      timestamp: new Date(),
      level,
      message,
      data,
      context
    }

    // Add to memory store
    this.logs.push(entry)
    
    // Keep only last maxLogs entries
    if (this.logs.length > this.maxLogs) {
      this.logs.shift()
    }

    // Log to console with appropriate method
    const prefix = context ? `[${context}]` : '[App]'
    const timestamp = entry.timestamp.toISOString().split('T')[1].slice(0, -1)
    
    switch (level) {
      case 'debug':
        console.debug(`${timestamp} ${prefix} ${message}`, data || '')
        break
      case 'info':
        console.info(`${timestamp} ${prefix} ${message}`, data || '')
        break
      case 'warn':
        console.warn(`${timestamp} ${prefix} ${message}`, data || '')
        break
      case 'error':
        console.error(`${timestamp} ${prefix} ${message}`, data || '')
        break
    }
  }

  debug(message: string, data?: any, context?: string) {
    this.log('debug', message, data, context)
  }

  info(message: string, data?: any, context?: string) {
    this.log('info', message, data, context)
  }

  warn(message: string, data?: any, context?: string) {
    this.log('warn', message, data, context)
  }

  error(message: string, data?: any, context?: string) {
    this.log('error', message, data, context)
  }

  /**
   * Get all logs (for debugging)
   */
  getLogs(level?: LogLevel): LogEntry[] {
    if (level) {
      return this.logs.filter(log => log.level === level)
    }
    return [...this.logs]
  }

  /**
   * Clear all logs
   */
  clear() {
    this.logs = []
  }

  /**
   * Enable/disable logging
   */
  setEnabled(enabled: boolean) {
    this.enabled = enabled
  }

  /**
   * Export logs as JSON string
   */
  export(): string {
    return JSON.stringify(this.logs, null, 2)
  }

  /**
   * Download logs as file
   */
  download() {
    const blob = new Blob([this.export()], { type: 'application/json' })
    const url = URL.createObjectURL(blob)
    const a = document.createElement('a')
    a.href = url
    a.download = `stemsep-logs-${Date.now()}.json`
    document.body.appendChild(a)
    a.click()
    document.body.removeChild(a)
    URL.revokeObjectURL(url)
  }

  /**
   * Get summary statistics
   */
  getStats() {
    return {
      total: this.logs.length,
      debug: this.logs.filter(l => l.level === 'debug').length,
      info: this.logs.filter(l => l.level === 'info').length,
      warn: this.logs.filter(l => l.level === 'warn').length,
      error: this.logs.filter(l => l.level === 'error').length,
      oldest: this.logs[0]?.timestamp,
      newest: this.logs[this.logs.length - 1]?.timestamp,
    }
  }
}

// Export singleton instance
export const logger = new Logger()

// Global access for debugging in console:
// __logger.getLogs()
// __logger.getStats()
// __logger.download()
// __logger.clear()
