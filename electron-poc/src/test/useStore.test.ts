import { describe, it, expect, beforeEach } from 'vitest';
import { useStore, type Model } from '../stores/useStore';
import { act } from '@testing-library/react';

const initialStoreState = useStore.getState();

beforeEach(() => {
  act(() => {
    useStore.setState(initialStoreState, true);
  });
});

const mockModels: Model[] = [
  { id: 'model1', name: 'Model One', installed: false, downloading: false, downloadProgress: 0 } as Model,
  { id: 'model2', name: 'Model Two', installed: true, downloading: false, downloadProgress: 0 } as Model,
];

describe('useStore - Zustand State Management', () => {
  
  describe('Model Actions', () => {
    it('should set models', () => {
      act(() => {
        useStore.getState().setModels(mockModels);
      });
      expect(useStore.getState().models).toEqual(mockModels);
    });

    it('should start a download', () => {
      act(() => {
        useStore.getState().setModels(mockModels);
        useStore.getState().startDownload('model1');
      });
      const model = useStore.getState().models.find(m => m.id === 'model1');
      expect(model?.downloading).toBe(true);
      expect(model?.downloadProgress).toBe(0);
      expect(model?.downloadError).toBeUndefined();
    });

    it('should set download progress', () => {
      act(() => {
        useStore.getState().setModels(mockModels);
        useStore.getState().startDownload('model1');
        useStore.getState().setDownloadProgress({ modelId: 'model1', progress: 50, speed: 1024, eta: 120 });
      });
      const model = useStore.getState().models.find(m => m.id === 'model1');
      expect(model?.downloadProgress).toBe(50);
      expect(model?.downloadSpeed).toBe(1024);
      expect(model?.downloadEta).toBe(120);
    });

    it('should complete a download', () => {
      act(() => {
        useStore.getState().setModels(mockModels);
        useStore.getState().startDownload('model1');
        useStore.getState().completeDownload('model1');
      });
      const model = useStore.getState().models.find(m => m.id === 'model1');
      expect(model?.installed).toBe(true);
      expect(model?.downloading).toBe(false);
      expect(model?.downloadProgress).toBe(100);
    });

    it('should set a download error', () => {
      act(() => {
        useStore.getState().setModels(mockModels);
        useStore.getState().startDownload('model1');
        useStore.getState().setDownloadError('model1', 'Network Error');
      });
      const model = useStore.getState().models.find(m => m.id === 'model1');
      expect(model?.downloading).toBe(false);
      expect(model?.downloadError).toBe('Network Error');
    });
    
    it('should set a model as installed', () => {
      act(() => {
        useStore.getState().setModels(mockModels);
        useStore.getState().setModelInstalled('model1', true);
      });
      const model = useStore.getState().models.find(m => m.id === 'model1');
      expect(model?.installed).toBe(true);
    });
  });

  describe('Separation Actions', () => {
    it('should start a separation', () => {
      act(() => {
        useStore.getState().startSeparation();
      });
      const { separation } = useStore.getState();
      expect(separation.isProcessing).toBe(true);
      expect(separation.progress).toBe(0);
      expect(separation.outputFiles).toBeNull();
      expect(separation.error).toBeNull();
    });

    it('should set separation progress', () => {
      act(() => {
        useStore.getState().startSeparation();
        useStore.getState().setSeparationProgress(50, 'Processing vocals...');
      });
      const { separation } = useStore.getState();
      expect(separation.progress).toBe(50);
      expect(separation.message).toBe('Processing vocals...');
    });

    it('should complete a separation', () => {
      const mockOutput = { 'vocals.wav': '/path/to/vocals.wav' };
      act(() => {
        useStore.getState().startSeparation();
        useStore.getState().completeSeparation(mockOutput);
      });
      const { separation } = useStore.getState();
      expect(separation.isProcessing).toBe(false);
      expect(separation.progress).toBe(100);
      expect(separation.outputFiles).toEqual(mockOutput);
    });

    it('should fail a separation', () => {
      act(() => {
        useStore.getState().startSeparation();
        useStore.getState().failSeparation('CUDA out of memory');
      });
      const { separation } = useStore.getState();
      expect(separation.isProcessing).toBe(false);
      expect(separation.error).toBe('CUDA out of memory');
    });

    it('should clear separation state', () => {
      const mockOutput = { 'vocals.wav': '/path/to/vocals.wav' };
      act(() => {
        useStore.getState().startSeparation();
        useStore.getState().completeSeparation(mockOutput);
        useStore.getState().clearSeparation();
      });
      const { separation } = useStore.getState();
      expect(separation.isProcessing).toBe(false);
      expect(separation.progress).toBe(0);
      expect(separation.outputFiles).toBeNull();
      expect(separation.error).toBeNull();
    });
  });
});
