export type LoopbackCaptureResult = {
  audioBytes: Uint8Array;
  captureSampleRate: number;
  captureChannels: number;
  captureStartAt: string;
  captureEndAt: string;
};

type LoopbackCaptureOptions = {
  expectedDurationSec?: number;
  maxDurationSec?: number;
  startTimeoutMs?: number;
  trailingSilenceMs?: number;
  minActiveRms?: number;
  onProgress?: (payload: {
    status: string;
    elapsedSec?: number;
    progress?: number;
  }) => void;
};

const DEFAULT_MIN_ACTIVE_RMS = 0.003;
const DEFAULT_START_TIMEOUT_MS = 15_000;
const DEFAULT_TRAILING_SILENCE_MS = 2_500;

function encodeWavFromFloat32(
  channels: Float32Array[],
  sampleRate: number,
): Uint8Array {
  const channelCount = channels.length;
  const frameCount = channels[0]?.length || 0;
  const bytesPerSample = 2;
  const dataSize = frameCount * channelCount * bytesPerSample;
  const buffer = new ArrayBuffer(44 + dataSize);
  const view = new DataView(buffer);

  const writeString = (offset: number, value: string) => {
    for (let index = 0; index < value.length; index += 1) {
      view.setUint8(offset + index, value.charCodeAt(index));
    }
  };

  writeString(0, "RIFF");
  view.setUint32(4, 36 + dataSize, true);
  writeString(8, "WAVE");
  writeString(12, "fmt ");
  view.setUint32(16, 16, true);
  view.setUint16(20, 1, true);
  view.setUint16(22, channelCount, true);
  view.setUint32(24, sampleRate, true);
  view.setUint32(28, sampleRate * channelCount * bytesPerSample, true);
  view.setUint16(32, channelCount * bytesPerSample, true);
  view.setUint16(34, bytesPerSample * 8, true);
  writeString(36, "data");
  view.setUint32(40, dataSize, true);

  let offset = 44;
  for (let frameIndex = 0; frameIndex < frameCount; frameIndex += 1) {
    for (let channelIndex = 0; channelIndex < channelCount; channelIndex += 1) {
      const sample = Math.max(
        -1,
        Math.min(1, channels[channelIndex]?.[frameIndex] || 0),
      );
      view.setInt16(
        offset,
        sample < 0 ? sample * 0x8000 : sample * 0x7fff,
        true,
      );
      offset += 2;
    }
  }

  return new Uint8Array(buffer);
}

function flattenChannelChunks(chunks: Float32Array[]) {
  const length = chunks.reduce((sum, chunk) => sum + chunk.length, 0);
  const merged = new Float32Array(length);
  let offset = 0;
  for (const chunk of chunks) {
    merged.set(chunk, offset);
    offset += chunk.length;
  }
  return merged;
}

export async function captureLoopbackAudioToWav(
  options: LoopbackCaptureOptions = {},
): Promise<LoopbackCaptureResult> {
  const expectedDurationSec = options.expectedDurationSec;
  const hardStopMs = Math.max(
    10_000,
    Math.round(
      ((options.maxDurationSec ||
        (expectedDurationSec ? expectedDurationSec + 8 : 180)) *
        1000),
    ),
  );
  const startTimeoutMs = options.startTimeoutMs || DEFAULT_START_TIMEOUT_MS;
  const trailingSilenceMs =
    options.trailingSilenceMs || DEFAULT_TRAILING_SILENCE_MS;
  const minActiveRms = options.minActiveRms || DEFAULT_MIN_ACTIVE_RMS;

  const captureStartAt = new Date().toISOString();
  const mediaStream = await navigator.mediaDevices.getDisplayMedia({
    audio: true,
    video: true,
  });
  const audioTracks = mediaStream.getAudioTracks();
  if (audioTracks.length === 0) {
    mediaStream.getTracks().forEach((track) => track.stop());
    throw new Error("No loopback audio stream was provided by Electron.");
  }

  const audioContext = new AudioContext();
  await audioContext.resume();
  const source = audioContext.createMediaStreamSource(mediaStream);
  const processor = audioContext.createScriptProcessor(4096, 2, 2);
  const sink = audioContext.createGain();
  sink.gain.value = 0;

  const channelChunks: Float32Array[][] = [[], []];
  let audioDetected = false;
  let captureEnded = false;
  let lastActiveAt = Date.now();
  let firstActiveAt = 0;
  let stopCapture!: () => void;

  const cleanup = async () => {
    if (captureEnded) return;
    captureEnded = true;
    processor.disconnect();
    source.disconnect();
    sink.disconnect();
    mediaStream.getTracks().forEach((track) => track.stop());
    await audioContext.close();
  };

  const resultPromise = new Promise<LoopbackCaptureResult>((resolve, reject) => {
    const startTimer = window.setTimeout(() => {
      void cleanup().finally(() => {
        reject(
          new Error(
            "No audio activity was detected. Start playback and try again.",
          ),
        );
      });
    }, startTimeoutMs);

    const hardStopTimer = window.setTimeout(() => {
      stopCapture();
    }, hardStopMs);

    stopCapture = () => {
      window.clearTimeout(startTimer);
      window.clearTimeout(hardStopTimer);
      void cleanup()
        .then(() => {
          const mergedChannels = channelChunks.map(flattenChannelChunks);
          const audioBytes = encodeWavFromFloat32(
            mergedChannels,
            audioContext.sampleRate,
          );
          resolve({
            audioBytes,
            captureSampleRate: audioContext.sampleRate,
            captureChannels: mergedChannels.length,
            captureStartAt,
            captureEndAt: new Date().toISOString(),
          });
        })
        .catch(reject);
    };

    processor.onaudioprocess = (event) => {
      if (captureEnded) return;
      const inputBuffer = event.inputBuffer;
      const elapsedSec = firstActiveAt
        ? (Date.now() - firstActiveAt) / 1000
        : 0;

      const frameCount = inputBuffer.length;
      const nextChannels = Array.from(
        { length: inputBuffer.numberOfChannels || 2 },
        (_, index) =>
          new Float32Array(
            inputBuffer.getChannelData(
              Math.min(index, inputBuffer.numberOfChannels - 1),
            ),
          ),
      );
      const rms =
        Math.sqrt(
          nextChannels[0].reduce((sum, sample) => sum + sample * sample, 0) /
            Math.max(1, frameCount),
        ) || 0;

      if (rms >= minActiveRms) {
        lastActiveAt = Date.now();
        if (!audioDetected) {
          audioDetected = true;
          firstActiveAt = Date.now();
          options.onProgress?.({ status: "capturing", elapsedSec: 0, progress: 0 });
        }
      }

      if (!audioDetected) {
        options.onProgress?.({ status: "awaiting_audio", elapsedSec: 0 });
        return;
      }

      if (nextChannels.length === 1) {
        nextChannels.push(new Float32Array(nextChannels[0]));
      }

      channelChunks[0].push(nextChannels[0]);
      channelChunks[1].push(nextChannels[1]);

      const progress =
        expectedDurationSec && expectedDurationSec > 0
          ? Math.max(0, Math.min(1, elapsedSec / expectedDurationSec))
          : undefined;

      options.onProgress?.({
        status: "capturing",
        elapsedSec,
        progress,
      });

      if (expectedDurationSec && elapsedSec >= expectedDurationSec + 2) {
        stopCapture();
        return;
      }

      if (Date.now() - lastActiveAt >= trailingSilenceMs && elapsedSec > 2) {
        stopCapture();
      }
    };
  });

  source.connect(processor);
  processor.connect(sink);
  sink.connect(audioContext.destination);

  try {
    return await resultPromise;
  } catch (error) {
    await cleanup();
    throw error;
  }
}
