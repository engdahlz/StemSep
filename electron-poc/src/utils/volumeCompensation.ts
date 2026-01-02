import type { VolumeCompensation } from '../types/separation'

export function bestVolumeCompensation(): VolumeCompensation {
    return {
        enabled: true,
        stage: 'both',
        dbPerExtraModel: 3,
    }
}
