import { describe, it, expect, beforeEach } from 'vitest';
import { SeparationHistory, type SeparationHistoryEntry } from '../utils/separationHistory';

// Mock entry data
const createMockEntry = (overrides: Partial<Omit<SeparationHistoryEntry, 'id' | 'timestamp'>> = {}): Omit<SeparationHistoryEntry, 'id' | 'timestamp'> => ({
  inputFile: 'test.mp3',
  inputFileName: 'test.mp3',
  preset: { id: 'preset1', name: 'Preset One' },
  outputDirectory: '/output',
  outputFiles: { 'vocals': '/output/vocals.wav' },
  stems: ['vocals'],
  device: 'cpu',
  duration: 60,
  success: true,
  ...overrides,
});

describe('SeparationHistory', () => {
  // Clear localStorage before each test
  beforeEach(() => {
    window.localStorage.clear();
  });

  it('should start with an empty history', () => {
    const history = SeparationHistory.getHistory();
    expect(history).toEqual([]);
  });

  it('should add a new entry to the history', () => {
    const mockEntry = createMockEntry();
    SeparationHistory.addEntry(mockEntry);

    const history = SeparationHistory.getHistory();
    expect(history.length).toBe(1);
    expect(history[0]).toMatchObject(mockEntry);
    expect(history[0].id).toBeDefined();
    expect(history[0].timestamp).toBeDefined();
  });

  it('should add new entries to the beginning of the array (most recent first)', () => {
    SeparationHistory.addEntry(createMockEntry({ inputFile: 'first.mp3' }));
    SeparationHistory.addEntry(createMockEntry({ inputFile: 'second.mp3' }));

    const history = SeparationHistory.getHistory();
    expect(history.length).toBe(2);
    expect(history[0].inputFile).toBe('second.mp3');
    expect(history[1].inputFile).toBe('first.mp3');
  });

  it('should clear the history', () => {
    SeparationHistory.addEntry(createMockEntry());
    SeparationHistory.clearHistory();
    const history = SeparationHistory.getHistory();
    expect(history.length).toBe(0);
  });

  it('should delete a specific entry by id', () => {
    SeparationHistory.addEntry(createMockEntry({ inputFile: 'to_keep.mp3' }));
    SeparationHistory.addEntry(createMockEntry({ inputFile: 'to_delete.mp3' }));
    
    const history_before = SeparationHistory.getHistory();
    const entryToDeleteId = history_before[0].id; // 'to_delete.mp3' is the first one

    SeparationHistory.deleteEntry(entryToDeleteId);

    const history_after = SeparationHistory.getHistory();
    expect(history_after.length).toBe(1);
    expect(history_after[0].inputFile).toBe('to_keep.mp3');
  });

  describe('getStats', () => {
    it('should return zero-values for empty history', () => {
      const stats = SeparationHistory.getStats();
      expect(stats).toEqual({
        totalSeparations: 0,
        successfulSeparations: 0,
        failedSeparations: 0,
        totalDuration: 0,
        averageDuration: 0,
        mostUsedPreset: null,
        mostUsedDevice: null,
      });
    });

    it('should correctly calculate statistics for a mixed history', () => {
      // Add some entries
      SeparationHistory.addEntry(createMockEntry({ preset: { id: 'p1', name: 'P1' }, device: 'cpu', duration: 100, success: true }));
      SeparationHistory.addEntry(createMockEntry({ preset: { id: 'p1', name: 'P1' }, device: 'cuda', duration: 50, success: true }));
      SeparationHistory.addEntry(createMockEntry({ preset: { id: 'p2', name: 'P2' }, device: 'cpu', duration: 120, success: true }));
      SeparationHistory.addEntry(createMockEntry({ preset: { id: 'p1', name: 'P1' }, device: 'cpu', success: false, error: 'failed' }));

      const stats = SeparationHistory.getStats();

      expect(stats.totalSeparations).toBe(4);
      expect(stats.successfulSeparations).toBe(3);
      expect(stats.failedSeparations).toBe(1);
      expect(stats.totalDuration).toBe(100 + 50 + 120); // 270
      expect(stats.averageDuration).toBe(270 / 3); // 90
      expect(stats.mostUsedPreset).toBe('p1');
      expect(stats.mostUsedDevice).toBe('cpu');
    });
  });
});
