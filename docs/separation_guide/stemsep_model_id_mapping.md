# StemSep model name → `model_id` mapping

This repo has a few “guide names” that don’t exactly match the **downloadable** model IDs in `StemSepApp/assets/models.json.bak`.

Use this page as the canonical translation when updating docs.

## Common mappings

| Guide / nickname (human) | Use in StemSep (model_id) | Notes |
|---|---|---|
| SCNet XL | `scnet-large` | “XL” is a guide nickname; the shipped ID is `scnet-large`. |
| HTDemucs 6s | `htdemucs-6s` | Shipped/built-in Demucs variant. |
| Becruily Karaoke | `mel-band-karaoke-becruily` | Karaoke-style vocal/instrumental separation. |
| UVR DeEcho + DeReverb | `anvuew-dereverb-room` | UVR “VR-Arch” model name; in StemSep use the shipped AnVuew dereverb model. |
| DeBleed MelBand Roformer | `gabox-denoise-debleed` | In StemSep, debleed is packaged with the GaBox denoise/debleed model. |
| Mel-Roformer De-Noise | `aufr33-denoise-std` | Use `aufr33-denoise-aggressive` if you need stronger suppression. |

## Not shipped / not downloadable (yet)

These may appear in the deton24 guide, but are **not currently downloadable** from the in-app registry (see `StemSepApp/assets/models.json.bak`).

- MVSEP-only / gated / manual-download models not present in the in-app registry (e.g. Celesta/Xylophone/Brass/Woodwind/Keys/Percussion/Plucked Strings/Lead-Rhythm Guitar entries, if you see those).
- “Becruily deux” (dual Mel-Roformer) models and “Gabox vocfv7beta3 / inst_fv7b / Inst_GaboxFv9” models are mentioned in the guide, but do not exist as shipped `model_id`s yet.
- Crowd-removal-specific models: we ship `mel-band-crowd` plus denoise alternatives (`aufr33-denoise-std`, `aufr33-denoise-aggressive`).
