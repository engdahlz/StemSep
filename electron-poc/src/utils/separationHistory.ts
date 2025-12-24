export interface SeparationHistoryEntry {
    id: string;
    timestamp: number;
    inputFile: string;
    inputFileName: string;
    preset?: { id: string; name: string };
    outputDirectory: string;
    outputFiles: Record<string, string>;
    stems: string[];
    device: string;
    duration: number;
    success: boolean;
    error?: string;
}

export class SeparationHistory {
    private static STORAGE_KEY = 'separation_history';
    private static BACKUP_KEY = 'separation_history_backup';
    private static MAX_HISTORY_SIZE = 500;
    private static MAX_STORAGE_SIZE = 4 * 1024 * 1024; // 4MB limit

    /**
     * Validates that an entry has the required fields
     */
    private static isValidEntry(entry: unknown): entry is SeparationHistoryEntry {
        if (!entry || typeof entry !== 'object') return false;
        const e = entry as Record<string, unknown>;
        return (
            typeof e.id === 'string' &&
            typeof e.timestamp === 'number' &&
            typeof e.inputFile === 'string' &&
            typeof e.success === 'boolean'
        );
    }

    static getHistory(): SeparationHistoryEntry[] {
        try {
            const stored = localStorage.getItem(this.STORAGE_KEY);
            if (!stored) return [];

            const parsed = JSON.parse(stored);

            // Validate data structure
            if (!Array.isArray(parsed)) {
                console.error('History data corrupted - not an array. Backing up and resetting.');
                this.backupAndClear();
                return [];
            }

            // Filter out malformed entries
            const validEntries = parsed.filter(entry => {
                const isValid = this.isValidEntry(entry);
                if (!isValid) {
                    console.warn('Skipping malformed history entry:', entry);
                }
                return isValid;
            });

            // If we filtered out some entries, save the cleaned version
            if (validEntries.length !== parsed.length) {
                console.warn(`Cleaned ${parsed.length - validEntries.length} malformed entries from history`);
                this.saveHistory(validEntries);
            }

            return validEntries;
        } catch (e) {
            console.error('Failed to load history:', e);
            this.backupAndClear();
            return [];
        }
    }

    /**
     * Backup corrupted data before clearing
     */
    private static backupAndClear(): void {
        const corrupted = localStorage.getItem(this.STORAGE_KEY);
        if (corrupted) {
            try {
                localStorage.setItem(this.BACKUP_KEY, corrupted);
                console.info('Corrupted history backed up to', this.BACKUP_KEY);
            } catch (e) {
                console.error('Failed to backup corrupted history:', e);
            }
        }
        this.clearHistory();
    }

    static addEntry(entry: Omit<SeparationHistoryEntry, 'id' | 'timestamp'>): void {
        let history = this.getHistory();
        const newEntry: SeparationHistoryEntry = {
            ...entry,
            id: crypto.randomUUID(),
            timestamp: Date.now(),
        };

        history.unshift(newEntry);

        // Trim if exceeding max count
        if (history.length > this.MAX_HISTORY_SIZE) {
            console.info(`History exceeds ${this.MAX_HISTORY_SIZE} entries, trimming oldest`);
            history = history.slice(0, this.MAX_HISTORY_SIZE);
        }

        // Check storage size and trim further if needed
        let serialized = JSON.stringify(history);
        while (history.length > 10 && serialized.length > this.MAX_STORAGE_SIZE) {
            console.warn('History storage too large, removing oldest entries');
            history.pop();
            serialized = JSON.stringify(history);
        }

        this.saveHistory(history);
    }

    static clearHistory(): void {
        localStorage.removeItem(this.STORAGE_KEY);
    }

    static deleteEntry(id: string): void {
        const history = this.getHistory().filter(e => e.id !== id);
        this.saveHistory(history);
    }

    static getStats() {
        const history = this.getHistory();
        const totalSeparations = history.length;
        const successfulSeparations = history.filter(e => e.success).length;
        const failedSeparations = history.filter(e => !e.success).length;

        // Only count duration for successful separations
        const totalDuration = history
            .filter(e => e.success)
            .reduce((acc, e) => acc + (e.duration || 0), 0);

        const averageDuration = successfulSeparations > 0 ? totalDuration / successfulSeparations : 0;

        const presetCounts: Record<string, number> = {};
        const deviceCounts: Record<string, number> = {};

        history.forEach(e => {
            if (e.preset?.id) {
                presetCounts[e.preset.id] = (presetCounts[e.preset.id] || 0) + 1;
            }
            if (e.device) {
                deviceCounts[e.device] = (deviceCounts[e.device] || 0) + 1;
            }
        });

        const mostUsedPreset = Object.entries(presetCounts).sort((a, b) => b[1] - a[1])[0]?.[0] || null;
        const mostUsedDevice = Object.entries(deviceCounts).sort((a, b) => b[1] - a[1])[0]?.[0] || null;

        return {
            totalSeparations,
            successfulSeparations,
            failedSeparations,
            totalDuration,
            averageDuration,
            mostUsedPreset,
            mostUsedDevice,
        };
    }

    private static saveHistory(history: SeparationHistoryEntry[]): void {
        try {
            localStorage.setItem(this.STORAGE_KEY, JSON.stringify(history));
        } catch (e) {
            console.error('Failed to save history:', e);
            // If localStorage is full, try trimming and saving again
            if (e instanceof Error && e.name === 'QuotaExceededError' && history.length > 10) {
                console.warn('localStorage quota exceeded, trimming history');
                const trimmed = history.slice(0, Math.floor(history.length / 2));
                try {
                    localStorage.setItem(this.STORAGE_KEY, JSON.stringify(trimmed));
                } catch (e2) {
                    console.error('Failed to save even after trimming:', e2);
                }
            }
        }
    }
}
