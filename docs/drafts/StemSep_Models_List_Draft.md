# StemSep Models Database
*Models with measurable stats (SDR, Fullness, Bleedless)*

### Stats: fullness=21.82 bleedless=30.83 SDR=10.80 fullness=27.07 bleedless=47.49 SDR=16.71 

- Gabox released two new models (vocal and instrumental):
Mel vocfv7beta3 | yaml
Voc. fullness 21.82 | bleedless 30.83  | SDR 10.80
“beta 1 and 2... eh, pretty close to same instrumental bleed,
but beta 3 def a step up from the two songs I compared (...)
most songs so far, fv7beta3 is fuller than fv7beta1,
def less robotic sounding at times (when a voice gets quiet/hard to capture, and it just fails).
Just had another song where fv7beta1 was fuller than fv7beta3, but it was also a lot noisier
large majority of the songs I tested, fv7beta3 was fuller... I think fv7beta3 is usually a bit noisier than fv7beta1? But also sounds fuller in those cases, I'd say it's generally worth it
instrumental bleed, usually worse with fv7beta3 versus fv7beta1, but it depends
fv7beta2 is always less full/less noise, but only slightly less instrumental bleed than fv7beta1” - rainboomdash
Mel inst_fv7b | yaml
Inst. fullness 27.07 | bleedless 47.49 | SDR 16.71
“this may be the last beta before the final model” - Gabox
The highest bleedless metric out of all instrumental models so far. But fullness is worse than even most vocal Mel-Roformers (including BS-RoFormer SW and Mel Kim OG model).
“On the fuller side, somewhere around inst v1e+, maybe a tiny bit below. The main thing I notice is it captures more instruments than v1e+, but isn't muddy like [HyperACE] (which also captures more instruments) (...) It can add a lot of crackling noise, though, more than v1e+ (...) can be a little on the noisy side sometimes... but it at least isn't muddy and sounds natural (...) I'd still ensemble if you want the noise reduced - rainboomdash
(src)

---

### Stats: fullness=36.91 bleedless=38.77 SDR=17.27 bleedless=36.53 SDR=16.65 

- Unwa released BS-Roformer-HyperACE instrumental model | separate Colab
Inst. fullness 36.91 | bleedless 38.77 | SDR 17.27
(less fullness than v1e+: 37.89, but more bleedless: 36.53, SDR: 16.65)
Note: It uses its own inference script. “You can use this model by replacing the MSST repository's models/bs_roformer.py with the repository's bs_roformer.py.”
To not affect functionality of other BS-Roformer models by it, you can add it as new model_type by editing utils/settings.py and models/bs_roformer/init.py here (thx anvuew).
“Currently, this model holds the highest aura_mrstft score on the instrumental side of the Multisong dataset. (...)
Some components in the SegmModel module were implemented based on this paper: https://arxiv.org/abs/2506.17733
Simply put, it's a module that utilizes hypergraphs to capture global relationships and standard convolutions to capture local relationships, thereby generating the final “Correlation-Enhanced” feature map.
This weight is based on the following weights. Thank you, anvuew!
https://huggingface.co/anvuew/BS-RoFormer” - unwa

---

### Stats: SDR=9.21 SDR=9.02 

- (MVSEP) “We have released a new model 'MVSep Lead/Rhythm Guitar (lead-guitar, rhythm-guitar) '. It has two variants:
1) Two-stage model (SDR: 9.21) - Best guitar model applied, and then 2-stem model is used which can separate lead/rhythm guitar.
2) One-stage model (SDR: 9.02) - Single model is applied, which was trained on a 3 stem dataset.
They can give pretty different results, so worth trying both.
Demo: https://mvsep.com/result/20251120090832-f0bb276157-mixture.wav

---

### Stats: bleedless=31.55 fullness=20.44 SDR=10.87 

- Gabox released beta 2 of vocfv7 Mel-Roformer “fullness went down a little bit”
Voc. bleedless: 31.55, fullness: 20.44, SDR: 10.87
https://huggingface.co/GaboxR67/MelBandRoformers/blob/main/melbandroformers/experimental/vocfv7beta2.ckpt | yaml | Colab (TL;DR in the vocals section)
“still quite a bit fuller than big beta 6x, but has less noise than even fv4 (also a bit less fullness, of course)” at least when the instruments are loud, fv7beta2 is usually quite a bit less noisy than fv4, while still maintaining a decent amount of fullness... it is a bit less, but not too much (...) both are pretty noisy with fv4 (...) 
still gonna have an issue with backing vocals compared to fv7beta1 sometimes... (makes sense, it's a less full model). (...) Fv7beta2 has still been significantly better with BV than fv4, despite quite a bit less noise” but “significant issues on one song, while fv6/fv7beta1 didn't (...) Def an improvement over fv4. I'm really liking the balance of fullness and noise for most songs. fv4 and fv6/fv7beta1 are usually pretty noisy... this is less noisy, but still has a good amount of fullness.” “Where the noise was undesirable and I ensembled fv4/fv6/fv7beta1 with big beta 6x, now I can just use this instead”.
“Fv7beta2 has still been significantly better with BV than fv4, despite quite a bit less noise” but “significant issues on one song, while fv6/fv7beta1 didn't”

“If the noise isn't an issue, and you just want fullness, fv6/fv7beta1 are still the best models. I'd say fv6 and fv7beta1 are better models than fv4, fullness/noise aside. It depends with fv7beta1 versus fv7beta2, sometimes the noise can be pretty significant with fv7beta1, and fv7beta2 may have the fullness you desire.
fv6 is usually more noisy/full than fv7beta1, but it just depends... I've had instances where it's less noisy/full than fv7beta1. But if you really want high fullness, fv6 and fv7beta1 are the choices. Sometimes fv6 can be quite a bit more noisy and the gain in fullness isn't worth it”.

---

### Stats: SDR=10.22 

- (MVSEP) “The karaoke model by anvuew has been added under the algorithm "MVSep Karaoke (lead/back vocals)". It is available as the option "BS Roformer by anvuew (SDR: 10.22)" - ZFTurbo
For some reason it seems to give worse results than the ckpt anvuew shared.

---

### Stats: fullness=21.21 bleedless=30.81 SDR=10.96 fullness=21.33 bleedless=29.07 SDR=10.58 

- Gabox released voc_fv7 beta 1 Mel-Roformer model | yaml | Colab
Voc. fullness: 21.21, bleedless: 30.81, SDR: 10.96
"Just a better fv4 it seems, better bleedless" (fullness: 21.33, bleedless: 29.07, SDR 10.58)
vs voc_fv4 "It is noisier.. Kinda closer to beta 5e?” “It's slightly less noise and fullness than beta 5e but picking up the backing vocals REALLY well, significantly better than beta 5e”
But it's pulling the backing vocals out even better than 5e” “the backing vocals are so good!
“it does have significant synth bleed, too...   it at least wasn't coming through at full volume
when I say fullness, I specifically mean how muddy it sounds” - Raiboom Dash

---

### Stats: fullness=39.88 bleedless=32.56 SDR=14.35 

- neoculture released a Mel-Roformer instrumental model focused on preserving vocal chops
Inst. fullness 39.88, bleedless: 32.56, SDR: 14.35
https://huggingface.co/natanworkspace/melband_roformer/blob/main/Neo_InstVFX.ckpt (yaml) | Colab
“great model (at least for K-pop it achieved the clarity and quality that no other model managed to have) it should be noted that it has a bit of noise even in its latest update, its stability is impressive, how it captures vocal chops, in blank spaces it does not leave a vocal record, sometimes the voice on certain occasions tries to eliminate them confusing them with noise, but in general it was a model that impressed me. It captures the instruments very clearly” - billieoconnell.
“NOISY AF, this is probably the dumbest idea ever had for an instrumental model. Don’t use it as your main one, some vocals will leak because I added tracks with vocal chops to the dataset. Just use this model for songs that have vocal chops” - neoculture
It was trained on only RTX 4060 8GB.

---

### Stats: fullness=25.10 bleedless=37.13 SDR=14.32 fullness=13.24 bleedless=30.75 SDR=8.01 fullness=27.44 bleedless=46.56 SDR=17.32 bleedless=36.75 fullness=16.26 SDR=11.07 

- Aname Mel trained a Mel-Roformer model called Full Scratch
Inst. fullness: 25.10, bleedless: 37.13, SDR: 14.32
Voc. fullness: 13.24, bleedless: 30.75, SDR: 8.01
(“trained from scratch on a custom-built dataset targeting vocals. It can be used as a base model or for direct inference. Estimated Training cost: ~$100”)
For state_dict error, update MSST to the last repo version:
!rm -rf /content/Music-Source-Separation-Training
!git clone https://github.com/ZFTurbo/Music-Source-Separation-Training
“and you must reinstall main branch's requirement.txt. (before it, edit requirements.txt to remove wxpython)” - Essid
Kim Mel model for reference:
Inst. fullness 27.44, bleedless 46.56, SDR: 17.32
Voc. bleedless: 36.75, fullness: 16.26, SDR: 11.07

---

### Stats: fullness=28.83 bleedless=31.18 SDR=12.59 

- Essid reevaluated GAudio (a.k.a. GSEP) for the leaderboard.
https://mvsep.com/quality_checker/entry/9095
Inst fullness: 28.83, bleedless: 31.18, SDR: 12.59
The result would rather cover my observations that instrumentals rather have gotten worse over the years (at least since the last 2023 Bas' evaluation or even earlier, at least for certain songs). But it appears that the vocals might got better.
https://mvsep.com/quality_checker/multisong_leaderboard?algo_name_filter=Gsep&sort=instrum&ranking_metrics=
Despite the fact the metrics are worse than even the least bleedless free community models like even V1e, for specific songs where bleeding doesn't occur so badly, GSEP might be still interesting too try out to some limited extend, being a different architecture, sounding maybe less filtered. Also, mixdown of multi stem extraction instead, should rather have bigger bleedless metric, but since the appearance of instrumental Roformers, GSEP relevance for separation is rather faded.

---

### Stats: fullness=24.36 bleedless=46.52 SDR=17.15 fullness=28.03 bleedless=44.16 SDR=16.69 fullness=27.44 bleedless=46.56 SDR=17.39 fullness=29.96 bleedless=44.61 SDR=16.62 fullness=32.03 bleedless=42.87 SDR=17.60 

- Aname released Mel-Roformer duality model.
“it's odd why the model is named duality, but it has a single target (and the file size of the ckpt confirms it further)” - becruily 
It’s focused more on bleedless than fullness metric contrary to the unwa’s duality v2 model, but with bigger SDR.
Inst. fullness 24.36, bleedless 46.52, SDR: 17.15
“instrumental is really muddy” - Gabox
For comparison -
Mel Duality v2 by unwa
Inst. fullness 28.03, bleedless 44.16, SDR: 16.69
MelBand Roformer vocals by Kim
Inst. fullness 27.44, bleedless 46.56, SDR: 17.39
Instrumental public models with the biggest fullness metric -
Gabox Mel Roformer Inst_GaboxFv7z
Inst. fullness: 29.96, bleedless: 44.61, SDR: 16.62
Unwa BS-Roformer-Inst-FNO
Inst. fullness: 32.03, bleedless: 42.87, SDR: 17.60

---

### Stats: fullness=32.31 bleedless=38.15 SDR=17.20 

- (MVSEP) “I added new SCNet vocals model: SCNet XL IHF (high instrum fullness by becruily). It's high fullness version for instrumental prepared by becruily.”
Inst. fullness 32.31, inst. bleedless 38.15, SDR 17.20
“One of my favorite instrumental models, Roformer-like quality.
For busy songs it works great, for trap/acoustic etc. Roformer is better due to SCNet bleed” - becruily
“It's better than BS Roformer (mvsep 2025.07 and inst Resurrection) at low frequencies, but bad at highs due the bleeding. I think it has better phase understanding, because it keeps the harmonics that were masked behind vocals cleaner (but it might not necessary be true to the source, it might just interpret/make up the harmonics instead of actual unmasking)” - IntroC

---

### Stats: SDR=10.41 

- (MVSEP) “I added new Karaoke model: "BS Roformer by MVSep Team (SDR: 10.41)" it's available under option "MVSep MelBand Karaoke (lead/back vocals)". Metrics.
In contrast with other Karaoke models, it returns 3 stems: "lead", "back" and "instrumental".
Example: https://mvsep.com/result/20250915192251-53be20aa17-10seconds-song.wav” - ZFTurbo
“If I had to compare it to any of the models, it is similar to the frazer and becruily model. Sometimes it does not detect the lead vocals specially if there's some heavy hard panning, but when it does, there is almost no bleed, and it works very well with heavy harmonies in mono from what I tested.” - smilewasfound
“becruily & frazer is better a little when the main voice is stereo” - daylightgay
“On tracks I tested, harmony preservation was better in becruily & frazer (...) the new model isn't worse, I ended up finding examples like Chan Chan by Buena Vista Social Club or The Way I Are by Timbaland where it is better than the previous kar model. The thing is, with the Kar models, it's just track per track. Difficult to find a model for batch processing as it's really different from one track to another” - dca100fb8
“I also found the new model to not keep some BGVs, mainly mono/low octave ones, despite higher SDR” - becruily
“I think I've found a solution for people who don't like the new model.
If you put an audio file through the karaoke model and then put the lead vocal result through that, it usually picks up doubles. 
Which you can then put in your BGV stem if you'd like” - dynamic64
“it's definitely not as good as the one by frazer and becruily. SDR can be misleading sometimes” - ryanz48
becruily [“our model] uses 11.9 SDR vocal model as a base”
ZFTurbo “I started from SW weights”
“I've had fantastic results with it so far. Much MUCH better at holding the 'S' & 'T' sounds than the Rofo oke (for backing vox). Generally seems to provide fuller results .. but also the typical 'ghost' residue from the main vox can end up in the backing vox sometimes, but it's usually not enough to be an issue. I won't go so far as so say that it's replacing the other backing vox models for me entirely .. but it feels like the best of both worlds that Rofo and UVR2 provide.” - CC Karaoke

---

### Stats: SDR=9.53 

- (MVSEP) “New Karaoke model based on SCNet XL IHF was added on site in "MVSep MelBand Karaoke (lead/back vocals)". Name of model "SCNet XL IHF by becruily (SDR: 9.53, metrics)". It has slightly worse metrics than the top Roformer model, but since it's different architecture it can give better results in some cases where the Rofo failed.
Demo: https://mvsep.com/result/20250908072226-f0bb276157-mixture.wav” - ZFTurbo
Iirc it's BVE or IHF unpublic ZFTurbo model retrain, and ckpt won't be public till further notice, as becruily said.
“SCNet is more bleedy in general despite me trying to reduce the leakage
it's recommended for busy songs, often captures proper lead vocals better than Roformer. Another use case is to ensemble it with Roformer to improve fullness” - becruily
“Oh, might be related to the lead vocals panning, it seems this model doesn't like when it's not center (...) I'm indeed noticing this model works really great on some songs that the Mel Rofo Karaoke had trouble with (...) I noticed that, this model, instead of creating crossbleeding between LVs and BVs, make them both quieter. I prefer that compared to previous models Plus, it handle songs which have lead vocals in the sides and BVs also in the sides better”
To fix bleed in back-instrum stem, use “Extract vocals first, but, “I noticed a pattern that if you hear the lead vocals in the back-instrum track already (SCNet bleed), dont try to use Extract vocals first because there will be even more lead vocal bleed”  - dca
“Separates lead vocals better than Mel-Roformer karaoke becruily. It's not perfectly clean, sometimes a bit of the backing vocals slips through, but for now, scent karaoke model still the most reliable for lead vocals separation (imo)
https://pillows.su/f/df8c1791bceba5fe3ef6b16d310ec123
https://pillows.su/f/e1272a02c56e3d3eb7ba4007bbb0c4bd” - neoculture.
“the model seems to handle mono vocals better than melband but isn't as clean, lot of bleed” (extract vocals first was also used to test this) - Dry Paint Dealer Undr
Since the Mel Kar Becruily's model, the dataset is “larger” now, but still not “great”, and it might get eventually fixed, becruily said.

---

### Stats: SDR=9.45 

- “I added BS Roformer flute model. It's available in "MVSep Flute (flute, other)". It superior comparing to SCNet version. SDR: 9.45 vs 6.27. More than 3 SDR difference.
Example: https://mvsep.com/result/20250830211041-f0bb276157-mixture.wav” ZFTurbo

---

### Stats: fullness=37.66 bleedless=35.53 SDR=15.91 

- Gabox released experimental inst Mel-Roformer model (yaml) called just “fullness”.
“this isn't called fullness.ckpt for nothing.” - Musicalman
Inst. fullness: 37.66, bleedless: 35.53, SDR: 15.91 (thx Essid)

---

### Stats: bleedless=42.87 fullness=32.03 SDR=17.60 

- Unwa released BS-Roformer-Inst-FNO model (incompatible with UVR, use MSST and read special model installation instruction below).
inst. bleedless: 42.87, fullness: 32.03, SDR: 17.60
“very small amount of noise compared to other fullness inst models, while keeping enough fullness IMO. I don't even know if phase fix is needed. Maybe it's still needed a little bit.” dca
“seems less full than reserection, which I would expect given the MVSEP [metric] results. (...) I'd say it's roughly comparable to gabox inst v7”
“I replaced the MLP of the BS-Roformer mask estimator with FNO1d [Fourier Neural Operator], froze everything except the mask estimator, and trained it, which yielded good results. (...) While MLP is a universal function approximator, FNO learns mappings (operators) on function spaces.”
“(The base weight is Resurrection Inst)”
Installing the model - instructions:
1. “I had many errors with torch.load and load_state_dict, but I managed to solve them. 
PyTorch 2.6 and later have improved security when loading checkpoints, which causes the problem. torch._C_.nn.gelu must be set to exception”
> “Add the following line above torch.load (at utils/model_utils.py line 479; 531/532 in updated MSST):
with torch.serialization.safe_globals([torch._C._nn.gelu])

---

### Stats: SDR=7.29 

- (MVSEP) “I released new MVSep Violin (violin, other). It based on BS Roformer model with SDR: 7.29 for violin on my internal validation.
Link: https://mvsep.com/home?sep_type=65
Example: https://mvsep.com/result/20250809120109-f0bb276157-mixture.wav”- ZFTurbo
“I've only played around with it a little bit, but it can even separate violin quartets from cellos, so cool.” - smilewasfound
“Very neat model. (...) Sometimes the model does seem to pick up more than just violins imo, but yeah for separating high strings in particular it is really cool.” - Musicalman

---

### Stats: bleedless=26.61 fullness=24.93 SDR=10.64 bleedless=25.30 fullness=23.50 SDR=10.40 

- Gabox released experimental voc_fv6 model | yaml
“Sounds like b5e with vocal enhancer. Needs more training, some instruments are confused as vocals” - Gabox. “fv6 = fv4 but with better background vocal capture” - neoculture
bleedless: 26.61 | fullness: 24.93 | SDR: 10.64
For comparison:
SCNet XL very high fuillness on MVSEP has followin metrics:
Vocals bleedless: 25.30, fullness: 23.50, SDR: 10.40

---

### Stats: SDR=17.25 bleedless=40.14 fullness=34.93 

- Unwa released a new BS-Roformer Resurrection instrumental model | yaml | Colab
SDR: 17.25, bleedless: 40.14, fullness: 34.93
Compatible with UVR (model type v1). “Fast model to inference (204 MB only)”.
“One of my favorite fullness inst models ATM. Sounds like v1e to me, but cleaner. Especially with guitar/piano where v1e tended to add more phase distortion, I guess that's what you'd call it lol. This model preserves their purity better IMO” - Musicalman
“the way it sounds, is indeed the best fullness model, it's like between v1e and v1e+, so not so noisy and full enough, though it creates problems with instruments gone in the instrumental sadly, but apparently it seems Roformer inst models will always have problems with instruments it seems, seems like a rule. (...) Instrument preservation (...) is between v1e and v1e+ (...) Fixes crossbleeding of vocals in instrumental in a lot of songs, compared to previous models (...) No robotic voice bug at silent instrumental moments” - dca100fb8
“Some songs leaves vocal residue. It is heard little but felt” - Fabio
“Almost loses some sounds that v1e+ picks up just fine” - neoculture
Mushes some synths a bit in e.g. trap/drill tune compared to inst Mel-Roformers like INSTV7/Becruily/FVX/inst3, but the residues/vocal shells are a bit quieter, although the clarity is also decreased a bit. Kind of a trade.
So far, none models work for phase fixer/swapper besides 1296/1297 by viperx and unwa BS Large V1 to alleviate the remaining noise. ~ dca. SW model not tested.
Less crossbleeding than paid Dango 11.

---

### Stats: bleedless=27.58 fullness=15.24 bleedless=50.67 fullness=32.46 

- Gabox released a bunch of new models:
a) Gabox Inst_ExperimentalV1 model | yaml
b) Gabox Kar v2 Mel-Roformer | model | yaml
SDR is very similar with the v1 Gabox model: 9.7699 vs 9.7661.
Lead:
bleedless: 27.58 vs 28.18, fullness: 15.24 vs 14.79
Back-instrum:
bleedless: 50.67 vs 50.74, fullness: 32.46 vs 32.84
(but you’ll most likely get better results with Gabox denoise/debleed Mel-Roformer model instead ~Gabox, but it can’t remove vocal residues
model | yaml | Colab)
c) Gabox Lead Vocal De-Reverb Mel-Roformer | DL | config | Colab
“just use it on the mixture” - Gabox, “sounds great” - Rage313
Sometimes removes back vocals, especially if they're panned to the sides.
“(...) also a vocal/inst separator. Dry vocals go in vocal stem, everything else goes to reverb. Don't think anvuew's models do that.
I might still preprocess with vocal isolation before dereverb. But only really worth it if you're after high fullness vocals.” - Musicalman

---

### Stats: SDR=9.67 

- (MVSEP) MelBand Karaoke (lead/back vocals) Gabox model added (SDR: 9.67)

---

### Stats: SDR=9.85 

- Fused model of Gabox and Aufr33/viperx weights 0.5 + 0.5 added (SDR: 9.85)
It gives maybe only slightly worse results than normal ensembling, but with separation time of just one model “it doesn't have the same quality and definition as Gabox Karaoke, fused doesn't separate well.” - Billie O’Connell.
You can perform fusion of models using ZFTurbo script(src) or by Sucial script (they’re similar if not the same). “I think the models need to have at least the same dim and depth but I'm not sure about that” - mesk.
Despite the higher SDR, the fusion model seems to confuse lead/back vocals more.

---

### Stats: bleedless=33.13 fullness=18.98 SDR=10.98 

- Gabox released Mel-Roformer voc_gabox2 vocal model | yaml | Colab
Vocal bleedless: 33.13, fullness: 18.98, SDR: 10.98

---

### Stats: bleedless=39.99 fullness=15.14 SDR=11.34. 

- Unwa released a BS-Roformer vocal model called "Resurrection" | yaml which shares some similarities with the SW model (might be a retrain). The default chunk_size is pretty big, so if you run out of memory, decrease it to e.g. 523776.
Vocal bleedless: 39.99, fullness: 15.14, SDR: 11.34.
"Omg, this model is doing a really good job at capturing backing vocals (...)
Honestly, it sounds a bit muddy, and there's some instrumental bleeding into the vocal stems" neoculture
Not so good for speech denoising unlike some other models (Musicalman).

---

### Stats: bleedless=47.65 fullness=28.76 SDR=18.24 bleedless=36.30 fullness=17.73 SDR=11.93 

- MVSep Ensemble 11.93 (vocals, instrum) (2025.06.28) added.
Eventually surpassed sami-bytedance-v.1.1 on the multisong dataset SDR-wise.
Instrumental bleedless: 47.65, fullness: 28.76, SDR: 18.24
Vocal bleedless: 36.30, fullness: 17.73, SDR: 11.93

---

### Stats: bleedless=48.59 fullness=27.85 SDR=11.82 bleedless=37.83 fullness=17.30 SDR=18.12 

- (MVSEP) New BS Roformer model is now available on site - it’s called 2025.06 (don’t confuse it with SW).
Vocals bleedless: 48.59, fullness: 27.85, SDR: 11.82
Instrumental bleedless: 37.83, fullness: 17.30, SDR: 18.12
“It has +0.5 SDR to the previous best [24.08] model. We reached ByteDance's best model quality [only 0.1 SDR difference). It is also TOP1 on the Synth dataset. It's balanced between both [instrumental and vocals]. I used metal dataset during training as well"
Compared to previous models, picks up backing vocals and vocal chops greatly where 6X struggles, and fixes crossbleeding and reverbs where in some songs previous models struggled before. Sometimes you might still get better results with Beta 6X or voc_fv4 (depending on a song). “Very similar to SCNet very high fullness without the crazy noise” - dynamic64, “handles speech very well. Most models get confused by stuff like birds churping (they put it in the vocal stem), but this model keeps them out of the vocal stem way more than most. I love it!”
“not a fan of the inst result. I feel like unwa and gabox sound better despite being less accurate” - dynamic64. Might be better than Fv7n “I think gabox tends to sound better but the new BS-Roformer is more accurate” dynamic64, “instrumentals are muddy” - santilli_,
“I think the Gabox [fv7n] model sounded more crispier than BS” - REYYY. “[voc_]fv4 sounds better” - neoculture, “instrumentals sound very good” - GameAgainPL.
“it did things i never thought it could before” “this model is insane wtf (...) never seen a model accurately do the ayahuasca experience before” - mesk.
“the first model to not produce vocal bleed in instrumental for "Supersonic" by Jamiroquai (not even Dango does it). It is also the case with "Samsam (Chanson du générique)" and "Porcelain" by Moby.” and "In the Air Tonight" by Phil Collins, also “removes very most of Daft Punk vocoder vocals" - dca. “my new favorite for vocals. It sounds fantastic” - dynamics64. “for the first time ever it managed to remove the reverb from one specific song. it is not perfect, but still much better than previous attempts” - santilli_
“It even seems to handle speech very well. Most models get confused by stuff like birds churping (they put it in the vocal stem), but this model keeps them out of the vocal stem way more than most. I love it!”. “sometimes 6x is better sometimes bs is better” - isling “for me it's picked up a lot that 6x hadn't for backing vocals

---

### Stats: fullness=29.38 bleedless=44.95 Fullness=31.85 bleedless=41.73 

- Gabox released Inst_GaboxFv7z Mel Roformer | yaml
Inst. fullness: 29.38, bleedless: 44.95
“Focusing on the less amount of noise keeping fullness”
“The results were similar to INSTV7 but with less noise” - neoculture
Metrically better bleedless than Unwa v2 (although it’s even more muddy), for comparison:
Fullness: 31.85, bleedless: 41.73

---

### Stats: bleedless=28.31 fullness=17.98 

- (MVSEP) “I added a new SCNet vocal model. It's called SCNet XL IHF. It has a better SDR than previous versions. Very close to Roformers now".
Vocal bleedless is the best among all SCNet variants on MVSEP. Metrics. 
IHF stands for “Improved high frequencies”.
Vocal bleedless 28.31, fullness 17.98
“certainly sounds better than classic SCNet XL (...) less crossbleeding of vocals in instrumental so far, and handle complex vocals better (...) problems with instruments, compared to high fullness one. XL high fullness remain the one without too many instruments cut”, but some difficult songs used with previous models can yield better results - dca

---

### Stats: fullness=29.83 bleedless=39.36 SDR=16.51 

- Gabox released instv7plus bleedless model (experimental)
fullness: 29.83, bleedless: 39.36, SDR 16.51

---

### Stats: fullness=35.05 bleedless=36.90 SDR=16.59 

- And Inst_FV8b
fullness: 35.05, bleedless: 36.90, SDR 16.59
“Very clean” although muddier than V1E+.

---

### Stats: SDR=10.98 fullness=21.43 bleedless=30.51 

- (Unwa) “After a long time, I'm uploading a vocal model specialized in fullness.
Revive 3e is the opposite of version 2 — it pushes fullness to the extreme.
Also, the training dataset was provided by Aufr33. Many thanks for that.”
bs_roformer_revive3e | config | Colab (should be fixed now)
Voc. SDR: 10.98, fullness: 21.43, bleedless: 30.51

---

### Stats: bleedless=31.96 fullness=14.42 bleedless=58.68 fullness=49.85 bleedless=31.54 fullness=15.95 bleedless=49.36 fullness=31.57 

- Logic Pro updated their stem separation feature, which now incorporates guitar
Overall, it’s “surprisingly good” - dynamic64. And a piano separator was also added to it. More
“Guitar & Piano separation seems to be really on point. So far it separated super well, also didn’t confuse organs for guitars and certain piano sounds as well.” - Tobias51
“guitar model sounds better than demucs, mvsep, and moises” - Sausum
“it's not a fullness emphasis or anything, but it's shockingly good at understanding different types of instruments and keeping them consistent sounding” - becruily
You don’t need to process L and R for bleeding across channels like in other models, there isn’t any in this one - A5
Full evaluation on multisong dataset (besides instrumental):
SDR piano 7.79, bleedless 31.96, fullness 14.42
SDR other 19.90, bleedless 58.68, fullness 49.85
SDR guitar 9.00, bleedless 31.54, fullness 15.95
SDR other 15.94, bleedless 49.36, fullness 31.57
SDR drums 14.05 (although lower fullness than MVSep SCNet XL drums 14.26 vs 21.21), 
SDR bass 14.57 (-||-), other 8.66, vocals 11.27 (only that is not SOTA)
MVSep Piano Ensemble (SCNet + Mel) has only other fullness higher: 56.96 (click)

---

### Stats: bleedless=40.07 fullness=15.13 SDR=10.97 

- Unwa released Revive 2 variant of his BS-Roformer fine-tune of viperx 1297 model
Voc. bleedless: 40.07, fullness: 15.13, SDR: 10.97
“has a Bleedless score that surpasses the FT2 Bleedless” and fullness lower by 0.64.
“can keep the string well” better than viperx 1297 (...) in my country they have some song with Ethnic instruments. Only 1297 and Revive2 can keep them in Instrumental while other model notice them as Vocal” ~daylight
“it does capture more than viperx's” - mesk
It’s depth 12 and dim 512, so the inference is much slower than with some newer Mel-Roformers like voc_fv4 (even two times), with the exception of Mel 1143 which is as slow as BS 1297 (thx dca, neoculture).

---

### Stats: bleedless=38.80 fullness=15.48 SDR=11.03 

- BS-Roformer Revive unwa’s vocal model (viperx 1297 model fine-tuned) was released.
Voc. bleedless: 38.80, fullness: 15.48, SDR: 11.03
“Less instrument bleed in vocal track compared to BS 1296/1297” but it still has many issues, “has fewer problems with instruments bleeding it seems compared to Mel. (...) 1297 had very few instrument bleeding in vocal, and that Revive model is even better at this
(...). Works great as a phase fixer reference to remove Mel Roformer inst models noise” it doesn’t seem to remove instruments like FT3 Preview for phase fixing (thx dca100fb8)
Added to phase fixer Colab.

---

### Stats: bleedless=29.50 fullness=20.67 SDR=10.56 

- Gabox released voc_fv5 vocal model | yaml
voc bleedless: 29.50, fullness: 20.67, SDR: 10.56
“fv5 sounds a bit fuller than fv4, but the vocal chops end up in the vocal stem. In my opinion, fv4 is better for removing vocal chops from the vocal stem” - neoculture. Examples
“v5 is slightly fuller, v4 is less full but also slightly more careful about what it considers as vocals. I think b5e is the fullest overall, but it's a bit much sometimes. Pretty sure the gabox models are a little more accurate with vocal/instrument detection.” Musicalman
Passes the Gregory Brothers - Dudes a Beast test (before - trumpets in vocal stem at 0:51; unwa’s beta4 and inst v1e tested) - maxi74x1

---

### Stats: bleedless=38.06 fullness=35.57 SDR=16.51 

- Gabox released Inst_GaboxFv8 model (yaml) [weight has been replaced by v2]
Inst. bleedless: 38.06, fullness: 35.57, SDR: 16.51 [outdated]
Might have some “ugly vocal residues” at times (Phil Collins - In The Air Tonight) - 00:46, 02:56 - dca.
VS v1e ”it seems to pick up some instruments better” Gabox
“a bit cleaner-sounding and has less filtering/watery artifacts.
Both models are prone to very strange vocal leakage [“especially in the chorus.”].
And because Fv8 can be so clean at times, the leakage can be fairly obvious. For now, my vote is for Fv8, but I'll still probably be switching back and forth a lot” - Musicalman
“sometimes v1e+ have vocal residues which sound like you were speaking through a fan/low quality mp3” - dca

---

### Stats: bleedless=36.53 fullness=37.89 SDR=16.65 

- Unwa released a new V1e+ Mel-Roformer instrumental model | yaml | Colab
Inst bleedless: 36.53, fullness: 37.89, SDR: 16.65
Less noise than v1e (esp. in the lower frequencies), but it’s also less full - “somewhere between v1 and v1e.”. It has fewer problems with quiet vocals in instrumentals than the V1+, “issues with harmonica, saxophone, elec guitar and synth seem to have been fixed. Theremin and kazoo are still problematic [like] for models from MDX-Net or SCNet [archs]). Only dango seems to correctly detect kazoo as an instrument it seems” - dca, “The loss function was changed to be more fullness-oriented, and trained a further 50k steps from the v1+ test.” Unwa
“v1e keeps better instruments like trumps than v1e+
With v1e+ there is less noise, but some instruments are hidden” koseidon72
“v1e+ has a strange problem of almost vocoding the vocals and keeping them in quietly” even with phase fixer
“has some problems with cymbals bleed in vocals (not the case with other instrumental roformer models)” dca
“trained with additional phase loss which helps remove some of that metallic fullness noise, and also has higher sdr I believe” - becruily

---

### Stats: bleedless=38.26 fullness=35.31 SDR=16.72 

- Unwa released V1+ Mel-Roformer instrumental model | yaml | Colab
Inst. bleedless: 38.26, fullness: 35.31, SDR: 16.72
"It is based on v1e, but the Fullness is not as high as v1e, so it is positioned as an improved version of v1." Unwa
"very nice model, the multistft noise is gone"
It's probably due to:
"Unwrapped phase loss function added" Unwa
BTW. It was already proven before, that adding artificial noise to separations was increasing fullness metric.
"Seems to have significantly less sax and harmonica bleed in vocal, which is an awesome thing (...) It still struggles with other things like FX and Kazoo." dca
"It sounds clean. The only thing [is] that some instruments are deleted, and in some tracks leaves remnants of voice in the instrumental." Fabio
"Screams are not removed from the track" Halif
Training details
"I made a small improvement to the dataset and trained about 50k steps with a batch size of 2.
8192 was added to multi_stft_resolutions_window_sizes.
As it was, the memory usage increased too much, so it was rewritten to use hop_length = 147 when window_size is 4096 or less and 441 when window_size is greater than that." Unwa

---

### Stats: bleedless=48.81 fullness=42.85 SDR=13.7621 

- Mesk released a preview of his instrumental model retrained from Mel Kim on metal dataset consisting of a few thousands of songs.
https://huggingface.co/meskvlla33/metal_roformer_preview/tree/main | Colab
These are not multisong metrics, but made with private dataset!
Instr bleedless: 48.81, fullness: 42.85, SDR: 13.7621
"currently restarting from scratch because I think I know what all the problematic vocal tracks were, and I removed them, we'll see if it's gonna be better"
"vocals could follow if requested.
Should work fine for all genres of metal, but doesn't work on:

---

### Stats: bleedless=35.16 fullness=17.77 SDR=11.12 

- Unwa released Big Beta 6X vocal model (yaml)
Vocal bleedless: 35.16, fullness: 17.77, SDR: 11.12
“it is probably the highest SDR or log wmse score in my model to date.”
Some leaks into vocal might occur.
“dim 512, depth 12.
It is the largest Mel-Band Roformer model I have ever uploaded.”
“I've added dozens of samples and songs that use a lot of them to the dataset”

---

### Stats: bleedless=36.11 fullness=16.80 SDR=11.05 

- Unwa released FT3 preview vocal model | yaml
Vocal bleedless: 36.11, fullness: 16.80, SDR: 11.05
“primarily aimed at reducing leakage of wind instruments to vocals.
I will upload a further fine-tuned version as FT3 in the near future.”
For now, FT2 has less leakage for some songs (maybe till the next FT will be released).

---

### Stats: bleedless=41.69 fullness=32.13 

- Gabox released inst_gaboxBv3 instrumental model (B for bleedless)
Inst. bleedless: 41.69, fullness: 32.13
“can be muddy sometimes”

---

### Stats: bleedless=34.66 fullness=38.96 

- Gabox released instv7 beta 2
Inst. bleedless: 34.66, fullness: 38.96
and instv7 beta 3
“Both are noisy with small vocal residuals in places where music is low and deletions of some musical instruments.”

---

### Stats: bleedless=35.01 fullness=38.39 

- Gabox released instv7beta model yaml
Inst. bleedless: 35.01, fullness: 38.39
“sound is good, but sometimes some instruments are lowered or deleted”
“while the annoying buzzing/noise is still present, it seems to be more contained.”

---

### Stats: bleedless=32.98 fullness=18.83 

- New FullnessVocalModel (yaml) vocal model was released by Aname | Colab
Voc. bleedless: 32.98 (less than beta 4), fullness: 18.83 (less than big beta 5e/voc_fv4/becruily, more than beta 4)
“While it emphasizes fullness, the noise is well-balanced and does not interfere much. (...)
in sections without vocals, faint, rustling vocals can be heard.”
We have some report of very long separation of this model in UVR on Macs.
> Try to change chunk_size: 529200 to 112455 for that model/yaml (but it’s dim_t 256 equivalent, so something higher to test might be a better idea too)

---

### Stats: bleedless=29.07 fullness=21.33 

- Gabox released voc_fv4 | yaml | Colab
Voc. bleedless 29.07, fullness 21.33
“Very clean, non-muddy vocals. Loving this model so far” (mrmason347)
“lost some of the trumpet sound  while on Becruily model can keep it, but some also was lost”

---

### Stats: bleedless=32.63 fullness=41.68 

- New Gabox model released: INSTV6N (noisy) | yaml | Colab | SESA | metrics: 
inst bleedless: 32.63, fullness: 41.68 (more than v1e)
Interestingly, some people find it having less noise vs v1e, and more fullness.
Also, it has more fullness vs INSTV6, and more noise.
“v1e sounds like an "overall" noise on the song, while v6n kind of mixes into it.
v6n also sounds like two layers, one of noise that's just there. And the other one mixes into the song somehow.
Using the phase swap barely makes it any better than phase swapping with v1e though” - vernight
Also Kim model for phase swap seems to give less noise than unwa ft2 bleedless

---

### Stats: SDR=13.72 

- (added on MVSEP as SDR 13.72) ZFTurbo trained new SCNet XL model for drums.
“I have 2 versions: one is slightly higher SDR and avg Bleedless.
Second is better for fullness and L1Freq.
Previous best SDR model had 13.01 (it's SCNet Large).” Metrics
15.7180 (13.72) one has much better fullness metric.
“It's far superior to the other one, but I still hear some weird parts.
It still messes up on some percussion.
The drums stem sounds really weird.
The no drums is alright except for some bleeding but yeah the drums is quite muddy” - insling

---

### Stats: bleedless=39.30 fullness=15.77 SDR=11.05 

- Unwa released ft2 bleedless vocal model | Colab
https://huggingface.co/pcunwa/Kim-Mel-Band-Roformer-FT/tree/main
voc bleedless 39.30 | fullness 15.77 | SDR 11.05

---

### Stats: bleedless=33.49 

- instv5 model released by Gabox (39.40 inst fullness | inst. bleedless 33.49) link | yaml | Colab | x-minus
“it seems that most vocal leakage is gone, and the noise did significantly decrease, although there's still a bit more noise presence than v1e.
In terms of fullness though, for some reason it sounds as if it's actually less full than v1e, despite the higher instrumental fullness SDR.
Despite v4's significant amount of noise, it seems to be the only model that gave me a fuller sounding result compared to v1e that's actually perceivable by my ears.” Shintaro

---

### Stats: SDR=16.43 fullness=38.71 bleedless=35.62 

- Lots of new Gabox models added since then, including:
a) BS-Roformer instrumental variant, which doesn’t struggle so much with choirs like most Mel-Roformers, although may not help in all cases (link)
b) inst_gaboxFv3.ckpt - like v1e when it comes to fullness (added on x-minus)
Inst SDR 16.43 | inst. fullness 38.71 | inst. bleedless 35.62
It might pick up entire sax in vocal stem.

---

### Stats: bleedless=37.40 fullness=37.07 

- Gabox released Mel-Roformer instrumental model (Kim/Unwa/Becruily FT): https://huggingface.co/GaboxR67/MelBandRoformers/tree/main/melbandroformers
inst bleedless: 37.40 (better than v1e by 1.8), fullness 37.07 (better than unwa inst v1 and v2)
“It’s like the v1 model with phase fixer, but it gets more instruments,
like, it prevents some instruments getting into the vocals”, “sometimes both models don't get choirs”.

---

### Stats: bleedless=37.19 fullness=37.26 

- instrumental variant called fullness v1 (“noisier but fuller”)
inst bleedless: 37.19, fullness 37.26
(thanks for evaluation to Bas Curtiz and his GSheet with all models.)

---

### Stats: bleedless=34.66 fullness=18.10 

- voc_gabox.ckpt:
voc bleedless: 34.66 (better than 5e), fullness 18.10 (on pair with beta 4)

---

### Stats: bleedless=33.4013 fullness=19.3064 

- Vocal model F v2
voc bleedless: 33.4013, fullness: 19.3064

---

### Stats: SDR=13.03 

- Newer Mel-Roformer Male/Female model was added by ZFTurbo on MVSEP (SDR: 13.03 vs 11.83 - the previous SCNet one, and much better bleedless metric 41.9392 vs 26.0247 with only 0.2 fullness decrease)
“ I find it acts differently from Rofo or UVR2. Sometimes it's the one of the three that gets it right., and not strictly for male/female.” CC Karaoke

---

### Stats: SDR=17.2785 

- ZFTurbo released new models on MVSEP:
a) a new Male/Female separation model based on SCNet XL
SDR on the same dataset: 11.8346 vs 6.5259 (Sucial)
Model only works on vocals. If the track contains music, use the option to "extract vocals" first. Sometimes the old Sucial model might still do a better job at times, so feel free to experiment.
b) SCNet XL (vocals, instum)
Inst SDR: 17.2785
Vocals have similar SDR to viperx 1297 model,
and instrumental has a tiny bit worse score vs Mel-Kim model.

---

### Stats: SDR=17.81 

- “All Ensembles on MVSep were updated with latest release [SCNet XL] increasing vocals SDR to 11.50 -> 11.61 and instrum SDR: 17.81 -> 17.92”.

---

### Stats: SDR=16.4719 fullness=33.9763 bleedless=40.4849 SDR=10.5547 fullness=20.7284 bleedless=31.2549 

- Becruily’s released instrumental and vocal Mel-Roformer models | Colab | UVR beta |
Instrumental model files | Inst SDR 16.4719 | inst fullness 33.9763 | bleedless 40.4849
Vocal model file | Vocals SDR 10.5547 | voc fullness 20.7284 | bleedless 31.2549 | 
config with ensemble fix in UVR.
Instrumental model is as clean as unwa’s v1, but has less noise and, and it can be got rid well by Mel denoise and/or Roformer bleed suppressor. Inst variant “removed some of the faint vocals that even the bleed suppressor didn't manage to filter out” before”. Doesn’t require phase fix from Mel-Kim like unwa models below.
“it handles the busy instrumentals in a way that makes VR finally an arch of the past”
Correctly removes SFX voice. More instruments correctly recognized as instruments and not vocals, although not as much as Mel 2024.10 & BS 2024.08 on MVSEP, but still more than unwa’s inst v1e/v1/v2. (dca100fb8).
Trumpet or sax sound which on unwa model was lost, can be recovered
on becruily's model (hendry.setiadi)
The instrumental model pulled out more adlibs than the released vocal model variant - it pulled out nothing (isling).
“Vocal model pulling almost studio quality metal screams effortlessly. Wow, I've NEVER heard that scream so cleanly” (mesk)
The model was trained on dataset type 2 and single RTX 3090 for two days (although with months of experimentation beforehand). SDR metrics are lower than Mel-Kim model.
If you use lower dim_t like 256 at the bottom of config for slower GPU, these are the first models to have very bad results with that setting.

You can experiment with phase fixer with santilli_ suggestion “Using becruily's vocals as source and inst [model] as target, and changing high frequency weight from 0.8 to 2 makes for impressive results”.

---

### Stats: SDR=13.81. 

- New “MVSep Bass (bass, other)" SCNet model available on MVSEP
“It achieved SDR: 13.81. In Ensemble it gives 14.07 - which is a new record on the Leaderboard.” ZFTurbo
“It passes Food Mart - Tomodachi Life test. That's the first model to.”
“All bass models have problems with fretless bass”
There’s already an option to combine all SCNet+BS Roformer+HTDemucs bass models for 14.07 SDR.
Ensembles have been updated with this model too.

---

### Stats: SDR=7.16. 

- (MVSEP) “Guitar model was updated. I added BSRoformer model by viperx with SDR: 7.16. And

---

### Stats: SDR=12.76 SDR=12.04 SDR=13.05 SDR=11.41 SDR=11.99 

- (MVSEP) I finished my drums models. Results:
MelRoformer SDR: 12.76
Demucs4 (finetuned) SDR: 12.04
Ensemble Mel + Demucs4 SDR: 13.05
for comparison:
Old Best Demucs4 SDR: 11.41
Old Best Ensemble SDR: 11.99
New models will be added on site soon.” ZFTurbo
For comparison, the Mel-Roformer available on x-minus trained by viperx has 12.5375 SDR.

---

### Stats: SDR=12.5295 SDR=12.4964 SDR=11.86 SDR=11.20 

- (mvsep) Bass model is online. The metrics:
Single models:
HTDemucs4 bass SDR: 12.5295
BSRoformer bass SDR: 12.4964
MelRoformer bass SDR: 11.86
MDX23C bass SDR: 11.20
Models on site:
HTDemucs4 + BSRoformer Ensemble (It's available on site as MVSep Bass (bass, other)): 13.25
Ensemble 4 stems and All-In (from site): 13.34
For comparison:
Ripple lossless (bass): 13.49
Sami-ByteDance v1.0: 13.82

---

### Stats: SDR=15.86 

- MDX-Net HQ_4 model (SDR 15.86) released for UVR 5 GUI! Go to Models list>Download center>MDX-Net and pick HQ_4 for download. It is an improved and faster than HQ_3, trained for epoch 1149 (only in rare cases there’s more vocal bleeding, more often instrumental bleeding in vocals, but the model is made with instrumentals in mind.
Along with it, also UVR-MDX-NET Crowd HQ 1 has been added in download center.

---

### Stats: SDR=10.16 

- https://github.com/karnwatcharasupat/bandit
Better SDR for Cinematic Audio Source Separation (dialogue, effect, music) than Demucs 4 DNR model on MVSEP (mean SDR 10.16>11.47)

---

### Stats: SDR=10.17 SDR=16.48 

- New MDX23C model added exclusively on MVSEP:
vocals SDR 10.17 -> 10.36
instrum SDR 16.48 -> 16.66
Also ensemble 4 got updated by new model (10.32>10.44 for vocals)

---

### Stats: fullness=34.93 bleedless=40.14 SDR=17.25 

- Unwa BS-Roformer Resurrection inst (yaml) | a.k.a. “unwa high fullness inst" on MVSEP | uvronline.app/x-minus.pro | Colab | UVR (don’t confuse with Resurrection vocals variant)
Inst. fullness: 34.93, bleedless: 40.14, SDR: 17.25
MVSEP BS 2025.07 works as a reference for phase fix with 3000/5000 settings.
Only 200MB. Some people might prefer it over V1e+, although it’s more muddy.
“use if the others [below] are noisy”
Models working for phase fixer (to alleviate the noise) are only BS-Roformer 1296/1297 by viperx and BS Large V1 by unwa, but generally the model might require phase fixing less than other models here - dca
“One of my favorite fullness inst models ATM. Sounds like v1e to me, but cleaner. Especially with guitar/piano where v1e tended to add more phase distortion, I guess that's what you'd call it lol. This model preserves their purity better IMO” - Musicalman
“I like resurrection inst for segments of piano, a lot of other models are too noisy there (...) I also needed to turn overlap up for piano” (from 2 to 8). FNO was less noisy for it, but “the hit to fullness was extremely apparent” - rainboomdash
“The way it sounds, is indeed the best fullness model, it's like between v1e and v1e+, so not so noisy and full enough, though it creates problems with instruments gone in the instrumental sadly, but apparently it seems Roformer inst models will always have problems with instruments it seems, seems like a rule. (...) Instrument preservation (...) is between v1e and v1e+” - dca100fb8
“it seems to just nip some bits of random instruments like saxophone or guitar whereas v1e+ leaves them intact.” - dennis777
“Some songs leaves vocal residue. It is heard little but felt” - Fabio
“Almost loses some sounds that v1e+ picks up just fine” - neoculture
Mushes some synths a bit in e.g. trap/drill tune compared to inst Mel-Roformers like INSTV7/Becruily/FVX/inst3, but the residues/vocal shells are a bit quieter, although the clarity is also decreased a bit. Kind of a trade.
BS 2025.07/BS 2024.04/BS 2024.08/SW removes less noise than viperx models for phase fixer.

---

### Stats: fullness=36.91 bleedless=38.77 SDR=17.27 bleedless=36.53 SDR=16.65 

- Unwa BS-Roformer-HyperACE | separate Colab (doesn’t work in UVR)
Inst. fullness 36.91, bleedless 38.77, SDR 17.27
“sounding just like v1e+ after phase fix, but straight out of one single model
(...)  quite bleedy, but honestly it's a fair price to pay, I guess” - santilli_
Although for some people it can be even on pair with v1e+ bleed-wise, so check it out too (more fullness).
Note: It uses its own inference script. “You can use this model by replacing the MSST repository's models/bs_roformer.py with the repository's bs_roformer.py.”
To not affect functionality of other BS-Roformer models by it, you can add it as new model_type by editing utils/settings.py and models/bs_roformer/init.py here (thx anvuew).
Metrically less fullness than v1e+: 37.89, but more bleedless: 36.53, SDR: 16.65 (v1e+).
While using locally, consider changing overlap from default 4 to 2 in the yaml of the model. The difference won’t be really noticeable for most people, but it will be faster.
“Currently, this model holds the highest aura_mrstft score on the instrumental side of the Multisong dataset. (...)
This weight is based on the following weights. Thank you, anvuew!” - unwa
“Does seem like HyperACE is picking up more instruments than v1e+
does seem like slightly worse vocal bleed overall (still need to test this more, though)... haven't encountered the super tinny vocal bleed like v1e+, at least
still fails to pick up that brass instrument on one song... Not really any worse than v1e+, though (...) resurrection inst does sound more muddy, but also a lot less noise.. which makes sense... IDK, a little muddy for my tastes.
I did find one song/spot and resurrection inst was on par with hyperace in picking up the wind instrument, v1e+ lost it for a bit.
I have found in the past that resurrection inst generally picks up more instruments than v1e+ (...) fullness of HyperACE is much closer to v1e+ than resurrection inst (...) it gets pretty staticy compared to v1e+ [on some drums] (...) v1e+ does this to a lot less extent
it's not super common, though… (...) I'm very confident in saying bshyperace picks up more stuff than v1e+.
resurrection inst does pick it up much better than v1e+, but I think it's still too quiet
resurrection inst really does just pick up so much more instruments, despite having a lot less fullness” - rainboomdash
”fullness that is comparable to v1e+, but has significant more vocal crossbleeding in instrumental than BS Roformer Resurrection Inst, but still less than v1e+ and v1e” - dca100fb8

---

### Stats: fullness=37.89 bleedless=36.53 SDR=16.65 

- Unwa Mel-Roformer V1e+ model (yaml) | UVR (guide) | MVSEP | x-minus/uvronline | Colab | SESA Colab | Huggingface / 2
Inst. fullness: 37.89, bleedless: 36.53, SDR: 16.65
*) Phase fixer Colab (e.g. with FT3 as src)/UVR>Tools, or on x-minus with becruily vocal model used as reference model (premium) - for less noise.
*) introC script to get rid of vocal leakage in this model
*) “If you use Gabox Mel denoise/debleed model | yaml (Colab) on mixture then put the “denoised” inst stem of that into unwa inst v1e+ you get a very clean result with good fullness and very little noise” - 5b. But it can’t remove vocal residues, just vocal noise.
Might also sound interesting when using as target in Phase fixer, and with source set as Becruily inst model (overlap 50/chunk_size 112455 was used; very slow - gustownis).
Or bigbeta5e as a source to get rid of vocal residues - santilli_
(single model inference descriptions below)
“strange leakage [robot-like] in the vocal-only section with no instrumentation” - Unwa
”less noise than v1e (probably due to different loss function), but it’s also less full, “somewhere between v1e and v1”
“sometimes a detail piece of instrumental sound was lost, while on becruily inst [below] can pick that sound”. Might be too strong for MIDI sounds - kittykatkat_uwu.
Problems with broken lead-ins not happening in instv7 and v1e. Some issues with cymbals bleed in vocals - dca.
Better than v1+. ”has fewer problems with quiet vocals in instrumentals than the V1+, “issues with harmonica, saxophone, electric guitar and synth seem to have been fixed” - dca100fb8.
“has this faint pitched noise whenever vocals hit in dead silence, you may need to manually cut it out.” - dynamic64. Check out also BS_ResurrectioN later below, it’s like v1e++ (more fullness).

---

### Stats: fullness=39.40 bleedless=33.49 SDR=16.44 

- Gabox inst_fv4 (yaml) | Colab
Inst. fullness 39.40, bleedless 33.49, SDR 16.44
Don’t confuse it with inst_fv4noise - the regular variant was never released before (and with voc_fv4).
“Seems to be erasing a xylophone instrument. Does sound not too noisy and not muddy, I like it. (...) A little noisy with piano (I split the song up and process with resurrection inst there). (...) Does have some issues that resurrection inst doesn't have, but it doesn't sound muddy! It usually works great. (...) In my opinion, fv4 still has vocal traces, I don't know if in all of its songs and v1e plus doesn't have them, but the noise can bother you even though it's not much. Does have more vocal bleed at times. I think a lot of what I thought was vocal bleed was a synth, it did a pretty good job... There was one segment on a song where it caught vocal residues, though” - rainboomdash

---

### Stats: fullness=32.31 bleedless=38.15 SDR=17.20 

- MVSep SCNet vocals model: SCNet XL IHF (high instrum fullness by bercuily).
Inst. fullness 32.31, inst. bleedless 38.15, SDR 17.20
“One of my favorite instrumental models, Roformer-like quality.
For busy songs it works great, for trap/acoustic etc. Roformer is better due to SCNet bleed” - becruily
“bring[s] such near perfect instrumentals”
vs the previous XL models “It's high fullness version for instrumental prepared by becruily.”
It can also be an insane vocal model too.

---

### Stats: fullness=35.57 bleedless=38.06 SDR=16.51 

- Inst_GaboxFv8 v1 model (yaml) 
Inst. fullness: 35.57, bleedless: 38.06, SDR: 16.51
The OG link to the model changed to the v2 variant of the model, but the old link to the v1 was retrieved above.
It has “v1+ metallic noise” - Gabox
VS V1e+ “A bit cleaner-sounding and has less filtering/watery artifacts. Both models are prone to very strange vocal leakage [“especially in the chorus”].
And because Fv8 can be so clean at times, the leakage can be fairly obvious. For now, my vote is for Fv8, but I'll still probably be switching back and forth a lot. Still has ringing” - Musicalman. Although, you might still prefer it over V1e+.
Might have some “ugly vocal residues” at times (Phil Collins - In The Air Tonight) - 00:46, 02:56 - dca.
“Sometimes V1e+ has vocal residues which sound like you were speaking through a fan/low quality mp3” - dca
”Seems to pick up some instruments better” Gabox.

---

### Stats: fullness=35.05 bleedless=36.90 SDR=16.59 

- Gabox Inst_FV8b (yaml)
Inst. fullness: 35.05, bleedless: 36.90, SDR 16.59
If so, it’s called V8 there (at least it’s not INSTV8), maybe not fv8 v1.
Muddier than V1e+, but cleaner. Some people might prefer it over INSTV7.
“Preserves its volume stability to the original sound of the songs, it does not go down or lose strength, which is the most important thing, it manages to capture clear vocal chops, the voice is eliminated to 99 or 100% depending on its condition, it captures the entire instrumental and when making a mix it remains like the original that with other models the volume was lowered,” - Billie O’Connell

---

### Stats: fullness=33.22 bleedless=40.71 SDR=16.51 

- Gabox INSTV7 (yaml) | MVSEP | Colab | Huggingface / 2 | uvronline via special link for: free/premium | “F”, for fullness, “V” for version.
Inst. fullness: 33.22, bleedless: 40.71, SDR: 16.51
*) Phase fixer Colab/UVR’s Phase Swapper (for less noise; e.g. with FT3 by Unwa vocal model as source).
“I hear less noise compared to v1e, but it has a worse bleedless metric” and might be less full.
It might still have too much noise like v1e for some people, but less.
“Relatively full/noisy model. Fvx [below] is a sort of middle ground between v3 and v7.”
More fullness than V6, but vs v1e, sometimes “leaves noises throughout the song, sometimes vocal remnants in the verse of the song, and some instruments are erased.”
Less muddy than Mel 2024.10 on MVSEP, and V7 doesn’t preserve vocal chops/SFX.

---

### Stats: fullness=33.21 bleedless=40.73 SDR=16.57 fullness=35.57 bleedless=38.06 SDR=16.51 

- Inst_GaboxFv8 v2 model (yaml) | Colab
Inst. fullness: 33.21, bleedless: 40.73, SDR: 16.57
Usually refrerred as just Inst_GaboxFv8 without v2. Since its release, the checkpoint has been updated on 11.05.25 (same file name), metrics have changed (updated above).
“v8 from uvronline and Fv8 from huggingface are completely different models” - maybe it’s the v1 model. Also, don’t confuse with INSTV8.
“Good result for bleedless instead, fullness went down instead of up a little.”
Might be an interesting competitor to Unwa inst v2 which is muddier.
Inst. fullness: 35.57, bleedless: 38.06, SDR: 16.51 are the metrics of the old v1 model (unavailable). Unsure if uvronline uses the old fv8.

---

### Stats: fullness=33.98 bleedless=40.48 SDR=16.47 

- Becruily’s inst | Model files | Colab | Huggingface / 2 | on MVSEP a.k.a. Mel-Roformer “high fullness” | uvronline via special link for: free/premium (scroll down)
Inst. fullness 33.98, inst. bleedless 40.48, SDR 16.47
or on x-minus/uvronline (with optional phase correction feature in premium) | UVR
*) For less vocal residues use phase fixer Colab (also in UVR>Tools) and “becruily's vocals as source and inst as target”.
Alone, it’s as clean as unwa’s v1, but has less noise, and it can also be got rid well by:
*) Mel denoise and/or Roformer bleed suppressor by unwa/97chris. That model “removed some of the faint vocals that even the bleed suppressor didn't manage to filter out” before”. Doesn’t require phase fix. Try out denoising on a mixture first, then use the model.
On its own, the inst model correctly removes SFX voices. The instrumental model pulled out more adlibs than the released vocal model variant, when it can pull out nothing.
Currently, the only model capable of keeping vocal chops. 
“Struggles a lot with low passed vocals”
More instruments correctly recognized as instruments and not vocals, although not as much as Mel 2024.10 & BS 2024.08 on MVSEP, but still more than unwa’s inst v1e/v1/v2.

---

### Stats: fullness=32.03 bleedless=42.87 SDR=17.60 

- Unwa BS-Roformer-Inst-FNO
Inst. fullness: 32.03, bleedless: 42.87, SDR: 17.60
Incompatible with UVR, install MSST, then read model instructions here (requires modifying bs_roformer.py file in MSST, potentially also models_utils.py in some cases).
Actually similar results to BS-Resurrection inst model above, less fullness.
Some people even prefer Gabox BS_ResurrectioN instead.
“Very small amount of noise compared to other fullness inst models, while keeping enough fullness IMO. I don't even know if phase fix is needed. Maybe it's still needed a little bit.” dca
“seems less full than the Resurrection, which I would expect given the MVSEP [metric] results. (...) I'd say it's roughly comparable to Gabox inst v7”
“I replaced the MLP of the BS-Roformer mask estimator with FNO1d [Fourier Neural Operator], froze everything except the mask estimator, and trained it, which yielded good results. (...) While MLP is a universal function approximator, FNO learns mappings (operators) on function spaces.”
“(The base weight is Resurrection Inst)”

---

### Stats: fullness=29.96 bleedless=44.61 SDR=16.62 

- Gabox Inst_GaboxFv7z Mel Roformer (yaml) | Colab | uvronline.app/x-minus.pro
Inst. fullness: 29.96, bleedless: 44.61, SDR: 16.62
Becruily vocal used for phase fixer on x-minus.pro/uvronline (premium feature).
“Focusing on the less amount of noise, keeping fullness”
“the results were similar to INSTV7 but with less noise” but “the drums are totally fine with this model ”- neoculture
“it seems to capture some vocals better” - Gabox
In some songs, “it leaves a lot of reverb or noise from the vocals. unva v1e+  a little better” - GameAgainPL
“[one of the] best bleedless, good fullness, almost noiseless” - Aufr33

---

### Stats: fullness=27.07 bleedless=47.49 SDR=16.71 fullness=37.69 bleedless=35.93 SDR=16.50 fullness=36.83 bleedless=35.47 SDR=16.65 fullness=34.04 bleedless=35.15 SDR=16.60 fullness=31.95 bleedless=34.06 SDR=17.26 

- Gabox inst_fv7b Mel Roformer | yaml
Inst. fullness 27.07, bleedless 47.49, SDR 16.71
Fullness worse than even most vocal Mel-Roformers (incl. BS-RoFormer SW and Mel Kim OG model).
“on the fuller side, somewhere around inst v1e+, maybe a tiny bit below. The main thing I notice is it captures more instruments than v1e+, but isn't muddy like [HyperACE] (which also captures more instruments)
can be a little on the noisy side sometimes... but it at least isn't muddy and sounds natural (...) I'd still ensemble if you want the noise reduced - rainboomdash
(src)
Lower fullness models
(if you find the ones above too muddy, but here you get more noise)
0) Gabox inst_gabox3 (yaml) | Colab | Huggingface / 2 | Phase fixer Colab
Inst. fullness 37.69, bleedless 35.93, SDR 16.50
Actually worse fullness than v1e+ (37.89), and lower bleedless (36.53).
When used with Unwa’s beta 6 as reference for phase fixer (thx John UVR), slightly less muddy results than phase-fixed Becruily inst-voc results, but also slightly more vocal residues and a bit more inconsistent sound, fluctuations across the whole separation at times.
0) Gabox INSTV7N | Huggingface / 2
Inst. fullness 36.83, bleedless 35.47, SDR: 16.65
More noisy than INSTV7; “it's [even] closer to v7 than inst3”
__
0) SCNet XL model called “very high fullness” | MVSEP
Inst. fullness 34.04, bleedless 35.15, SDR 16.60
It might work better than Roformers for less noisy/loud/busy mixes or genres like alt-pop, orchestral tracks with choir, sometimes giving more full results than even v1e, but at the cost of more noise. Might struggle with some vocal reverbs or effects.
“Very hit or miss. When they're good they're really good but when they're bad there's nothing you can do other than use a different model”
Compared to the high fullness variant, more crossbleeding of vocals in instrumentals (along with SCNet XL basic model). Some songs which sound full enough even with basic SCNet XL (and HF variant) while others will sound muddy (dca)
“has a lot of noise/bleed, and I haven't found the best way to get rid of it, but it does tend to pick up harmonies and subtle BGV that other models don't.” dynamic64
0) MVSEP SCNet XL high fullness
Inst. fullness 31.95, bleedless 34.06, SDR 17.26
“I have a few examples where it's better than v1e+
Sometimes there is too much residue but most of the time it's fine” dca
“Really loving the way SCnet high fullness [variant] handles lower frequencies, below 2K [let’s] say. Roformers are better with the transients up high, but decay on guitars/keys on the SCnet is more natural”
“seems to also confuse less "difficult" instruments for vocals”
“I noticed classic SCNet XL preserves more instruments than the high fullness one, but has more vocal crossbleeding in instrumental compared to high fullness
So if you want instrument preservation use SCNet XL 1727 but if you want less crossbleeding of vocals in instrumental use SCNet XL high fullness
I ignore the very high fullness one because it has too much vocal residue” dca
(regular SCNet XL moved below)
_

---

### Stats: fullness=41.68 bleedless=32.63 fullness=40.40 bleedless=28.57 SDR=15.25 

- Gabox BS_ResurrectioN model | yaml
“It is a fine-tune of BS Roformer Resurrection Inst but with higher fullness (like v1e for example), it needs [MVSEP’s] BS 2025.07 (as a source/reference) phase fix
I requested it because I found some songs where Resur Inst was producing muddy instrum results (...) I requested it not just for me because I saw other people were looking for something like v1e++” - dca
Higher fullness (but with more noise)
(sorted by fullness)
0) Gabox INSTV6N (N for noise/fullness) | yaml | Colab | SESA | Huggingface / 2 | metrics: 
Inst. fullness: 41.68 (more than v1e), bleedless: 32.63 (N “noisier but fuller”)
To get rid of noise in INSTV6N, use Gabox denoise/debleed model (yaml) on mixture first, then use INSTV6N - “for some reason it gives cleaner results” (Gabox), but it can’t remove vocal residues.
Some people find it having less noise vs v1e and more fullness.
Also, it has more fullness vs INSTV6, and more noise, but some people might still prefer v1e.
“v1e sounds like an "overall" noise on the song, while v6n kind of mixes into it.
v6n also sounds like two layers, one of noise that's just there. And the other one mixes into the song somehow. Using the phase swap barely makes it any better than phase swapping with v1e though” - vernight
Also Kim model for phase swap seems to give less noise than unwa ft2 bleedless
“Comparing V6N with v1e and couldn't hear a fullness difference despite the metrics being approx 39 for v1e and 41 for V6N” - dca
“my all-time favorite” - ezequielcasas
0) Gabox inst_Fv4Noise | yaml | Colab | Huggingface / 2
Inst. fullness 40.40, bleedless 28.57, SDR 15.25
Can be better than INSTV6 for some people, but overkill for others. Bigger fullness metric than even v1e.
“Despite v4's significant amount of noise, it seems to be the only model [till 8 February] that gave me a fuller sounding result compared to v1e that's actually perceivable by my ears.” - Shintaro
“although the fullness metric increases when there is more noise, it doesn't always mean it's a better instrumental — an example of this is the fv4noise metrics” - Gabox

---

### Stats: fullness=39.88 bleedless=32.56 SDR=14.35 fullness=38.87 bleedless=35.59 SDR=16.37 fullness=38.71 bleedless=35.62 SDR=16.43 fullness=37.66 bleedless=35.53 SDR=15.91 

- Neo_InstVFX Mel-Roformer by neoculture | yaml | Colab | Huggingface / 2
Inst. fullness 39.88, bleedless: 32.56, SDR: 14.35
Focused on preserving vocal chops.
“great model (at least for K-pop it achieved the clarity and quality that no other model managed to have) it should be noted that it has a bit of noise even in its latest update, its stability is impressive, how it captures vocal chops, in blank spaces it does not leave a vocal record, sometimes the voice on certain occasions tries to eliminate them confusing them with noise, but in general it was a model that impressed me. It captures the instruments very clearly” - billieoconnell.
“NOISY AF, this is probably the dumbest idea ever had for an instrumental model. Don’t use it as your main one, some vocals will leak because I added tracks with vocal chops to the dataset. Just use this model for songs that have vocal chops” - neoculture
0) Unwa Inst V1e (don’t confuse with newer +/plus variant above) Model files (yaml from v1)
Inst. fullness 38.87, bleedless 35.59, SDR 16.37
Colab | MSST-GUI | UVR instructions | Huggingface / 2 | uvronline via special link for: free/premium (scroll down) | MVSEP
One of the first Mel Kim model fine-tunes trained with instrumental (other) target. High fullness metric, noisy at times and on some songs. To alleviate it, it can be used with automated phase fixer Colab or UVR>Tools (Kim Mel as reference removes more noise than 2024.10 vs muddier Unwa v1/2 on their own; optionally use VOCALS-MelBand-Roformer by Becruily or unwa's kim ft; you can also use FT2 as reference, but it “cuts instruments” vs FT3 which can be rather better alternative). Optionally, in Phase Fixer you can set 420 for low and 4200 for high or 500 for both and Mel-Kim model for source; and bleed suppressor (by unwa/97chris) to alleviate the noise further (e.g. phase fixer on its own works better with v1 model to alleviate the residues). Besides the default UVR default 500/5000 and Colab default 500/9000 values, you could potentially “even try like 200/1000 or even below for 2nd value.”  “I would say that the more noisy the input is, the lower you have to set the frequency for the phase fixer.”
V1e might catch more instruments and vocals than INSTV6N. Even fuller model with more noise is instfv4noise below by Gabox.
“The "e" stands for emphasis, indicating that this is a model that emphasizes fullness.”
“However, compared to v1, while the fullness score has increased, there is a possibility that noise has also increased.” “lighter compared to v2.” Like other unwa’s models, it can struggle with flute, sax and trumpet (unlike Mel 2024.10, and BS 2024.08 on MVSEP respectively - you can max ensemble all the three as a fix [dca100fb8]). Also, sometimes unwa's big beta5e can retrieve missing instruments vs v1e when those two above fails. Possible residues of dual layer vocals from suno songs.
0) inst_gaboxFv3 | yaml | Huggingface / 2 - F for fullness
inst. fullness 38.71, inst. bleedless 35.62 (“F” stands for fuillness) | Inst SDR 16.43
Like v1e when it comes to fullness, but less bleeding.
Vs v1e “it's slightly better with some instruments”, It might pick up an entire sax in the vocal stem.
It doesn't have that weird fullness noise that fullness models produce, but still gives pretty full results and the phase swapper (with big beta 6 as reference) gets rid of that weird buzzing sound” John UVR
0) Gabox experimental “fullness.ckpt” inst Mel-Roformer (yaml).
Inst. fullness: 37.66, bleedless: 35.53, SDR: 15.91
“this isn't called fullness.ckpt for nothing.” - Musicalman

---

### Stats: fullness=34.93 bleedless=40.14 SDR=17.25 fullness=35.69 bleedless=37.59 fullness=36.03 bleedless=38.02 fullness=33.98 bleedless=40.48 SDR=16.47 fullness=33.21 bleedless=40.73 SDR=16.57 fullness=29.83 bleedless=39.36 SDR=16.51 fullness=28.74 bleedless=39.42 SDR=17.27 fullness=28.87 bleedless=40.37 SDR=17.41 fullness=27.10 bleedless=41.47 SDR=17.05 fullness=32.13 bleedless=41.69 SDR=16.60 fullness=31.85 bleedless=41.73 fullness=32.03 bleedless=42.87 SDR=17.60 Fullness=29.96 bleedless=44.61 

Lower bleedless models/balanced
Still less noise even when using without phase fixer
0) Unwa BS-Roformer Resurrection inst (yaml) | a.k.a. “unwa high fullness inst" on MVSEP | uvronline.app/x-minus.pro | Colab | UVR (don’t confuse with Resurrection vocals variant)
Inst. fullness: 34.93, bleedless: 40.14, SDR: 17.25
(duplicate from the above, because it fits metrically and categorization-wise here, more info above)
0) Unwa Mel-Roformer inst v1 (yaml) | Colab | UVR installation | MVSEP | uvronline via special link for: free/premium (scroll down)
inst. fullness 35.69, bleedless 37.59
*) Denoising for v1/2/1e recommended with: 1) ensemble noise/phase fix option for x-minus premium 1b) Becruily phase fixer (also since UVR beta patch #7) 2) Mel-Roformer de-noise non-agg. (might be better solution) 3) UVR-Denoise medium aggression (default for free users) 4) minimum aggression for premium/link (damages some instruments less) 5) UVR-Denoise-Lite [agg. 4, no TTA] in UVR - more aggressive method 6) UVR-Denoise [agg. 30/25, hi-end proc., 320 w.s., p.pr.] - even more muddy but preserves trumpets better
v1 might have more instruments missing vs v1e and less noise
0) inst_gabox2 (yaml) | Huggingface / 2
inst. fullness 36.03, bleedless: 38.02
-
0) Becruily’s inst model (again, because it fits here metrically)
Inst. fullness 33.98, bleedless 40.48, SDR 16.47
Colab | on MVSEP a.k.a. “high fullness” (the same model) | x-minus (w/ optional phase correction feature in premium) | UVR
For less vocal residues use phase fixer Colab (also in UVR>Tools) and “becruily's vocals as source and inst as target”
0) Inst_GaboxFv8 v2 model (yaml) | Colab
Inst. fullness: 33.21, bleedless: 40.73, SDR: 16.57
(again, just for metrics)
0) Gabox instV7plus bleedless model (experimental)
inst. fullness: 29.83, bleedless: 39.36, SDR 16.51
_
0) MVSEP SCNet XL (don’t confuse with undertrained weights on ZFTurbo’s GitHub)
inst. fullness 28.74, bleedless 39.42, SDR 17.27
“I've come across a lot of songs where high fullness [SCNet variant above] gives that annoying static noise. I'm starting to like basic SCNet XL more to the high fullness [model]. And also, less vocal residues.” - dca. There is crossbleeding of vocals in some songs. You can find the dca’s list for that model in further parts of this section.
0) MVSEP SCNet XL IHF
inst. fullness 28.87, bleedless 40.37, SDR 17.41
Some songs struggling with previous models might yield better results.
0) MVSEP SCNet Large
inst. fullness 27.10, bleedless 41.47, SDR 17.05
Higher bleedless (not so full)
0) Gabox B/bleedless v3 (inst_gaboxBv3) | Huggingface / 2
Inst. fullness: 32.13, bleedless: 41.69, SDR 16.60
“can be muddy sometimes” but still fuller than the older one below
0) Unwa Mel-Roformer inst v2 (similar but fewer vocal residues (not always), muddier, bigger, heavier model)
Inst. fullness 31.85, bleedless 41.73 (less bleeding than Gabox instfv5/6)
Model files | Colab | Huggingface / 2 | uvronline via special link for: free/premium (scroll down) | MSST-GUI (or OG repo) | UVR Download Center)
Might miss flute. “Sounds very similar to v1 but has less noise, pretty good” “the aforementioned noise from the V1 is less noticeable to none at all, depending on the track”.  “V2 is more muddy than V1 (on some songs), but less muddy than the Kim model. (...) [As for V1,] sometimes it's better at high frequencies” Aufr33
Might miss some samples or adlibs while cleaning inverts. SDR got a bit bigger (16.845 vs 16.595).
“Significantly less noise than v1e, sounds full enough, despite the fullness inst score, and that it recognizes more instruments than v1 and v1e, added to the fact it has higher SDR so also slightly less vocal crossbleeding in instrumental.” - dca100fb8
0) Unwa BS-Roformer-Inst-FNO
Inst. fullness: 32.03, bleedless: 42.87, SDR: 17.60
(again, because it fits metrically, more info moved near the top to recent bleedless section)
0) Gabox Inst_GaboxFv7z Mel Roformer | yaml | Colab |  uvronline/x-minus.pro
Becruily vocal used for phase fixer on x-minus.pro/uvronline (premium).
Fullness: 29.96, bleedless: 44.61
(again, because it fits metrically, -||-)

---

### Stats: fullness=28.49 bleedless=42.38 SDR=16.67 

- (old) mesk’s “Rifforge” metal Mel-Roformer fine-tune instrumental model (focused more on bleedless).
Inst. fullness: 28.49, bleedless: 42.38, SDR 16.67
“training is still in progress, that's why it's a beta test of the model; It should work fine for a lot of things, but it HAS quirks on some tracks + to me there's some vocal stuff still audible on some tracks, I'm mostly trying to get feedback on how I could improve it” known issues.
https://drive.proton.me/urls/5XM3PR1M7G#F3UhCU8RDGhX
Be aware that MVSep’s BS-Roformer 2025.07 can be better for metal both for vocal and instrumentals than these mesk’s models, a lot of the times. It was also trained on mesk’s metal dataset.

---

### Stats: fullness=28.81 bleedless=42.16 SDR=16.66 

- Older Mesk metal Mel-Roformer preview instrumental model
Inst. fullness: 28.81, bleedless: 42.16, SDR 16.66
Retrained from Mel Kim on metal dataset consisting of a few thousands of songs.
https://huggingface.co/meskvlla33/metal_roformer_preview/tree/main | Colab
(previous metrics were made on private dataset)
“Should work fine for all genres of metal, but doesn't work on:

---

### Stats: fullness=37.62 bleedless=35.07 SDR=16.43 

- weird tracks (think Meshuggah's "The Ayahuasca Experience")”
P.S: Use the Colab “or training repo (MSST) if you want to [separate] with it. UVR will be abysmally slow (because of chunk_size [introduced since UVR Roformer beta #3])”
0) INSTV6 by Gabox | yaml | x-minus | Colab
Inst. fullness 37.62, bleedless 35.07, SDR 16.43
v1e still gives better fullness, but the noise in it is a problem
Opinions are divided whether v5 or v6 is better.
“Seems like a mix between brecuily and unwa's models”
“Slightly better than v5 (...) less muddy and also removes the vocals without adding that low EQ effect when the vocals would come in, so I feel it's better” zzz
Old viperx’ 12xx models have fewerproblems with sax.

---

### Stats: fullness=38.25 bleedless=35.35 SDR=16.49 

- Inst_GaboxFVX | yaml | Huggingface / 2
Inst. fullness 38.25, bleedless 35.35, SDR 16.49
“instv7+3” - fuller than instv3

---

### Stats: fullness=37.07 bleedless=37.40 

- Gabox instv10 (experimental) | yaml
Less noise and vocal residues than V7, but muddier
__
0) Gabox Mel-Roformer instrumental model “inst_gabox.ckpt” (Kim/Unwa/Becruily fine-tuned)
Gabox’ models repo | Colab | Huggingface / 2
inst fullness 37.07 (better than unwa inst v1 and v2), bleedless 37.40 (better than v1e by 1.8, slightly worse than unwa’s v1)
“It’s like the v1 model with phase fixer, but it gets more instruments,
like, it prevents some instruments from getting into the vocals”, “sometimes both models don't get choirs”.

---

### Stats: fullness=37.26 bleedless=37.19 fullness=37.46 bleedless=37.09 fullness=39.40 bleedless=33.49 fullness=35.03 bleedless=39.10 SDR=16.49 fullness=35.09 bleedless=38.38 SDR=16.49 

Older fullness models
0) Gabox F/fullness v1 | Huggingface / 2
inst fullness 37.26 | bleedless: 37.19
0)  Gabox F/fullness v2 | Huggingface / 2
inst fullness 37.46 | bleedless: 37.09
*) Gabox inst_Fv4 (F - fullness/v4) | (don’t confuse with vocal fv4) | yaml | Colab
inst fullness 39.40 | bleedless 33.49
Duplicate from the above
Others
0) intrumental_gabox | yaml | Huggingface / 2
__
0) Gabox B/bleedless v1 instrumental model | yaml | Huggingface / 2
Inst. fullness 35.03, bleedless 39.10, SDR 16.49
0) Gabox B/bleedless v2 instrumental model | yaml | Huggingface / 2
Inst. fullness 35.09, bleedless 38.38, SDR 16.49
(Gabox models repo)
0) Cut your song into fragments consisting from the best moments of e.g. v1e/v1/v2 into one (and optionally Mel-Roformer Bas Curtiz FT on MVSEP as it will give you even less vocal bleeding, but more muddiness if necessary)
0) Propositions of models for phase fixer to alleviate vocal residues (from the above)
a) Becruily voc with Becruily inst (muddy but very few residues if any)
b) FT3 with V1e+
c) Unwa Beta 6 with inst_gabox3 (although it might be less consistent than the top)
d) Unwa Revive model is also good with any instrumental model
e) Unwa Bigbeta5 used to be not bad either.
f) Or any of the vocal models above with e.g. V1e (it's pretty full, and it might be not enough for it nevertheless)
How to use the phase fixer in UVR?
Separate with vocal model, then with instrumental model. Go to Audio Tools>Phase swapper, and use vocal model result as reference, and instrumental as target
__
Ensembles
(for instrumentals; check out also DAW ensemble with the below)
If you find some phase fixer results (e.g. ensembled) unsatisfactory, use the Phase Fixer Colab - it's tweaked for better results than UVR and standalone scripts.
0) BS HyperACE + BS 2025.07 (Max FFT with BS 2025.07 as phase fix reference and 3000/5000 for the values)
—->My favorite ensemble right now. Though, I notice it produce vocal bleed sometimes and it can be noisy at some parts of the songs while the noise might be totally absent in other parts of the song.

---

### Stats: fullness=27.83 SDR=18.20 fullness=27.45 bleedless=47.41 SDR=17.67 fullness=27.84 bleedless=47.37 SDR=17.59 fullness=27.63 bleedless=45.90 SDR=16.89 fullness=28.36 bleedless=45.58 fullness=26.56 bleedless=47.48 SDR=17.62 fullness=27.44 bleedless=46.56 SDR=17.32 fullness=29.18 bleedless=45.36 SDR=17.32 

Muddier but cleaner single models (Roformer vocal models with fewer instrumental residues vs instrumental models without necessity of using phase fixer)
0c) MVSep BS-Roformer (2025.07.20)
Inst. fullness 27.83, vleedless 49.12, inst SDR 18.20
Probably a retrain of the SW model on a bigger dataset.
0c) BS-RoFormer SW 6 stem (MVSEP/Colab/undef13 splifft) / vocals only
Inst. fullness 27.45, bleedless 47.41, inst SDR 17.67
(use inversion from vocals and not mixed stems for better instrumental metrics)
Known for being good on some songs previously giving bad results.
0c) 10.2024 Mel-Roformer vocal model on MVSEP
Inst. fullness 27.84, bleedless 47.37, inst SDR 17.59
The cleanest, but muddy compared to models trained for instrumentals
Capable of detecting sax and trumpet, but still muddier than instrumental models above.
Bas Curtiz vocal model fine-tuned by ZFTurbo.
0c) Gabox voc_fv4 | yaml | Colab
Good for anime and RVC purposes (codename)
And also for instrumentals, if you need less vocal residues than typical instrumental Roformers (even less than Mel Kim, FT2 Bleedless, or Beta 6X - makidanyee).
0c) Unwa’s beta 5e model originally dedicated for vocals | Colab | MSST-GUI | UVR instr
Model files | yaml: big_beta5e.yaml or fixed for AttributeError in UVR
Inst. fullness: 27.63 (bigger than Mel-Kim) | bleedless 45.90 (bigger than Kim FT by unwa, worse than Mel-Kim) | Inst. SDR 16.89
Mainly for vocals, but can be still a decent all-rounder deprived of noise present in unwa’s inst v1e/v1/v2 models, also with fewer residues than in Kim FT by unwa, and also more consistent model than Kim Mel model in not muffling instrumental a bit in sudden moments.
The third highest bleedless instrumental metric after Mel-Kim model (after unwa ft2 bleedless in vocals).
It seems to fix some issues with trumpets in vocal stem (maxi74x1).
It handles reverb tails much better (jarredou/Rage123).
Noisier/grainier than beta 4 (a bit similarly to Apollo lew's vocal enhancer), but less muddy.
“The noise is terrible when that model is used for very intense songs” - unwa
Phase fixer for v1 inst model doesn’t help with the noise here (becruily).
“It's a miracle LMAO, slow instrumentation like violin, piano, not too many drums... 
it's perfect... but unfortunately it can't process Pop or Rock correctly” gilliaan
“the vocal stem of beta5e may have fullness and noise level like duality v1, but it may also suffer kind of robotic phase distortion, yet may also remove some kind of bleed present in other melrofo's.” Alisa/makidanyee
“particularly helpful when you invert an instrumental and then process the track with it.” gilliaan
0c) Unwa’s Kim Mel-Band Roformer FT2 | download | Colab
Inst. fullness: 28.36, bleedless: 45.58
Decent all-rounder too, sometimes less bleeding in instrumentals than 5e.
0c) Unwa Kim Mel-Band Roformer Bleedless FT2 | download | Colab
0c) Bas Curtiz' edition Mel-Roformer vocal model on MVSEP
(it was trained also on ZFTurbo dataset)
“Music sounds fuller than original Kim's one & the finetuned version from ZFTurbo [iirc below]. Even [though] the SDR is smaller than BS Roformer finetuned last version, but almost song has the best result in instrumental.” Henri
It can struggle with trumpets more than the other Mel-Roformer on MVSEP [whether 08.2024 or Mel-Kim, can’t remember].
0c) BS-Roformer 2024.08.07 vocal model on MVSEP
Inst. fullness 26.56 (less than Mel-Kim), Inst. bleedless 47.48 (the only single model with that better metric than Mel-Kim)
Inst SDR 17.62
vs 2024.04 model +0.1 SDR and “it seems to be much better at taking the vocals when there are a lot of vocal harmonies” also good for Dolby channels.
Capable of detecting flute correctly
0c) Mel-Roformer vocal model by KimberleyJSN - model | config | Colab
Inst. fullness 27.44 (worse than beta 5e and duality, but better than current BS-Roformers)
Inst. bleedless 46.56 (the best metric from public models)
Inst SDR 17.32
It became a base for many Mel-Roformer fine-tunes here.
(works in UVR beta Roformer/Colab/CML inference/x-minus/MDX23 2.5 (when weight is set only for Mel model)/simple model Colab (might have problems with mp3 files)
It’s less muddy than older viperx’ Roformer model, but can have more vocal residues e.g. in silent parts of instrumentals, plus, it can be more problematic with wind instruments putting them in vocals, and it might leave more instrumental residues in vocals. SDR is higher than viperx model (UVR/MVSEP) but lower than fine-tuned 2024.04 model on MVSEP.
0c) Unwa Revive 2 BS-Roformer (“my first impression is it may have less low end noise than fv4 but not the best in the overall quality and amount of residues in vocal” - makidanyee)
0c) BS-Roformer Large vocal model by unwa (viperx 1297 model fine-tune) download
Older BS model. It picks more instruments than 12xx models. More muddy than Kim’s Roformer, a bit less of vocal residues, a bit more artificial sound. Also tends to be more muddy than viperx 1297, sometimes muffling instrumental at times, but a bit less of vocal residues, a bit more artificial sound/a bit less musical. Sometimes it has more vocal residues than beta 5e.
Compared to BS, Mel-Roformers can be a good balance between muddiness and clarity for some instrumentals.
Compared to ZFTurbo (MVSEP) and viperx models, Kim’s trained on Aufr33’s and Anjok’s dataset.
UVR manual model installation (Model install option added in newer patches):
Place the model file to Ultimate Vocal Remover\models\MDX_Net_Models and the config to model_data\mdx_c_configs subfolder and “when it will ask you for the unrecognised model when you run it for the first time, you'll get some box that you'll need to tick "Roformer model" and choose its yaml” some models here are available in Download Center too.
Other unwa fine-tunes (originally vocal models)
0c) Mel-Roformer Kim | FT (by unwa) | Colab
https://huggingface.co/pcunwa/Kim-Mel-Band-Roformer-FT/tree/main
Inst. fullness 29.18 (lower than only unwa inst models)
Inst. bleedless 45.36 (lower than Beta 5e)
Inst. SDR 17.32
Has more vocal residues than Beta 5e

---

### Stats: fullness=24.36 bleedless=46.52 SDR=17.15 

- Aname Mel-Roformer duality model . 
It’s focused more on bleedless than fullness metric contrary to the unwa’s duality v2 model, but with bigger SDR.
Inst. fullness 24.36, bleedless 46.52, SDR: 17.15

---

### Stats: fullness=28.03 bleedless=44.16 SDR=16.69. SDR=16.67 

- Mel-Roformer unwa’s inst-voc model called “duality v1/2” (focused on both instrumental and vocal stem during training; two independent and not inversible stems inside one weight file).
https://huggingface.co/pcunwa/Mel-Band-Roformer-InstVoc-Duality | Colab | MVSEP
V1: Inst fullness 28.03, bleedless 44.16, SDR 16.69.
V2: Inst SDR 16.67
Outperformed in both metrics by the unwa’s Kim FT.
Vocals sound similar to beta 4 model, instrumentals are deprived of the noise present in inst v1/e models, but in result, they don't sound similarly muddy to previous Roformers.
Compared to beta 4 and BS-Roformer Large or other archs’ models, it has fewer problems with reverb residues, and vs v1e, with vocal residues in e.g. Suno AI songs.
"other" is output from model, "Instrumental" is inverted vocals against input audio.
The latter has lower SDR and more holes in the spectrum, using MSST-GUI, leave the checkbox “extract instrumental” disabled for duality models (now it’s also in the Colab with “extract_instrumental” option) and probably for inst vx models.
You can use it in the Bas Curtiz’ GUI for ZFTurbo script or with the OG ZF’s repo code.

---

### Stats: SDR=17.30 

- unwa’s Mel-Roformer fine-tuned beta 3 (based on Kim’s model)
https://huggingface.co/pcunwa/Mel-Band-Roformer-big/tree/main | Colab
Inst SDR: 17.30
Since beta 3 there’s no ringing issues in higher frequencies like in previous betas.
Sometimes better for instrumentals than beta 4 - but tends to be too muddy at times, but with fewer vocal residues than beta 5.

---

### Stats: SDR=17.2785 fullness=25.10 bleedless=37.13 

- unwa’s Mel-Roformer beta 4 (Kim’s model fine-tuned)
https://huggingface.co/pcunwa/Mel-Band-Roformer-big/tree/main | Colab
Outperformed in both metrics by beta 5e.
Be aware that the yaml config is different in this model.
“Metrics on my test dataset have improved over beta3, but are probably not accurate due to the small test dataset. (...) The high frequencies of vocals are now extracted more aggressively. However, leakage may have increased.” - unwa
“one of the best at isolating most vocals with very little vocal bleed and still doesn't sound muddy” Can be a better choice on its own than some ensembles.
0c) SCNet XL (vocals, instum)
Inst SDR: 17.2785
Vocals have similar SDR to viperx 1297 model,
and instrumental has a tiny bit worse score vs Mel-Kim model.
0c) Older SCNet Large vocal model on MVSEP
“just like the new BS-Roformer ft model, but with more bleed. [BS] catches vocals with more harmonies/bgv” - isling. “it's like improved HQ4” - dca100fb8
Issues with horizontal lines on spectrogram.
0d) Aname Mel model trained from scratch a.k.a. Full Scratch
Inst. fullness: 25.10, bleedless: 37.13
Models for older archs
0c) MDX23C 1666 model exclusively on mvsep.com
(vocal Roformers are much more muddy than MDX23C/MDX-Net in general, but can be cleaner)
0c) MDX23C 1648 model in UVR 5 GUI (a.k.a. MDX23C-InstVoc HQ / 8K FFT) and mvsep.com, also on x-minus.pro/uvronline.app
Both sometimes have more bleeding vs MDX-Net HQ_3, but also less muddiness.
Possible horizontal lines/resonances in the output - fix DC offset and or use overlap “starting from 7 and going multiples up - 14 and so on.” Artim Lusis
0c) MDX23C-InstVoc HQ 2 - VIP model for UVR 5. It's a slightly fine-tuned version of MDX23C-InstVoc HQ. “The SDR is a tiny bit lower, but I found that it leaves less vocal bleeding.” ~Anjok
It’s not always the case, sometimes it can be even the opposite, but as always, all may depend on a specific song.
0d) MDX-Net HQ_4/3/2 (UVR/MVSEP/x-minus/Colab/alt) - small amounts of vocal residues at times, while not muffling the sound too much like in old BS-Roformer v2 (2024.02) on MVSEP, although it still can be muddy at times (esp. vs MDX23C HQ models), HQ_4 tends to be the least muddy out of all HQ_X models (although not always), and is faster than HQ_3 and below, it tends to have less vocal residues vs MDX23C.
Final MDX-Net HQ_5 seems to be muddier for instrumentals, although slightly less noisy, but better for vocals than HQ_4.
0d) MDX HQ_5 final model in UVR (available in its Download center and Colab)
Versus HQ_4, less vocal residues, but also muddier at times and a bit lower, 21,5kHz cutoff. 
Sometimes even more muddy than narrowband inst 3 to the point it can spoil some hi hats occasionally.
Versus unwa’s v1e “HQ5 has less bleed but is prone to dips in certain situations. (...) Unwa has more stability, but the faint bleed is more audible. So I'd say it's situational. Use both. (...) Splice the two into one track depending on which part works better in whichever part of the song is what I'd do.” CC Karaoke

Model | config: "compensate": 1.010, "mdx_dim_f_set": 2560, "mdx_dim_t_set": 8, "mdx_n_fft_scale_set": 5120
0d) MDX HQ5 beta model on  uvronline via special link for: free/premium (scroll down)
Go to "music and vocals" and there you will see it
It's not a final model yet, the model was in training since April.
It seems to be muddier than HQ_4 (and more than Kim’s and MVSEP’s Mel-Roformer), it has less vocal bleeding than before, but more than Kim Mel-Roformer.
"Almost perfectly placed all the guitar in the vocal stem" it might get potentially fixed in the final version of the model.
0e) Other single MDX23C full band models on mvsep.com (queues for free unregistered users can be long)
(SDR is better when three or more of these models are ensembled on MVSEP; alternatively in UVR 5 GUI’s via “manual ensemble” of single models (worse SDR) or at best, weighted manually e.g. in DAW, but the MVSEP “ensemble” option is specific method - not all fullband MDX23C models on MVSEP, that’s including 04.24 BS-Roformer model are available in UVR)

---

### Stats: fullness=28.83 bleedless=31.18 SDR=12.59 

- GSEP (now paid) -
Inst fullness: 28.83, bleedless: 31.18, SDR: 12.59
Check out also 4-6 stem separation option and perform mixdown for instrumental manually, as it can contain less noise/residues vs 2 stem in light mix without bass and drums too (although more than first vocal fine-tunes of like MVSEP’s BS-Roformer v2 back then). Regular 2 stem option can be good for e.g. hip-hop, and 4/+ stems a bit too filtered for instrumentals with busy mix. GSEP tends to preserve flute or similar instruments better than some Roformers and HQ_X above (for this use cases, check out also kim inst and inst 3 models in UVR) and is not so aggressive in taking out vocal chops and loops from hip-hop beats. Sometimes might be good or even the best for instrumentals of more lo-fi hip-hop of the pre 2000s era, e.g. where vocals are not so bright but even still compressed/heavily processed/loud or when instrumental sound more specific to that era. For newer stuff from ~2014 onward, it produces vocal bleeding in instrumentals much sooner than the above models. "gsep loves to show off with loud synths and orchestra elements, every other mdx v2/demucs model fail with those types of things".
Older ensembles (among others from the leaderboard)
Q: How to ensemble BS-Roformer 1296 with Kim Mel-Roformer using UVR GUI?
I choose max/max vocal/instrumental, but on the list there is only 1296, and no Kim Mel-Roformer like in MDX-Net option [might have been fixed already]
A: “You have to set the stem pair to multi-stem ensemble, it can generate both vocal and instrumental from both models at the same time. Be sure to set the algorithm to max/max. Once that's done, find the ensemble folder and put the two instrumental files/two vocal files onto the input, provided that you have to go to audio tools first. Then set the algorithm to average and click on the start processing button” - imogen
0f. #4626:
MDX23C_D1581 + Voc FT
0g) #4595:
MDX23C_D1581 + HQ_3 (or HQ_4 now)
0h) Kim Vocal 2 + Kim Inst (a.k.a. Kim FT/other) + Inst Main + 406 + 427 + htdemucs_ft (avg/avg)
0i) Voc FT, inst HQ3, and Kim Inst
0j) Kim Inst + Kim Vocal 1 + Kim Vocal 2 + HQ 3 + voc_ft + htdemucs ft (avg/avg).
0k) MDX23C InstVoc HQ + MDX23C InstVoc HQ 2 + MDX23C InstVoc D1581 + UVR-MDX-NET-Inst HQ 3 (or HQ 4)
“A lot of that guitar/bass/drum/etc reverb ends up being preserved with Max Spec [in this ensemble]. The drawback is possible vocal bleed.” ~Anjok
0l) MDX23C InstVoc HQ + MDX23C InstVoc HQ 2 + UVR-MDX-Net Inst Main (496) + UVR-MDX-Net HQ 1
"This ensemble with Avg/Avg seems good to keep the instruments which are counted as vocals by other MDXv2/Demucs/VR models in the instrumental (like saxophone, harmonica) [but not flute in every case]" ~dca100fb8
0m) MDX23C InstVoc HQ + HQ4
0n) Ripple (no longer works) / Capcut.cn (uses SAMI-ByteDance a.k.a. BS-Roformer arch) - Ripple is for iOS 14.1 and US region set only - despite high SDR, it's better for vocals than instrumentals which are not so good due to noise in other stem (can be alleviated by decreasing volume by -3dB).
0n) Capcut (for Windows) allows separation only for the Chinese version above (and returns stems in worse quality). See more for a workaround. Sadly, it normalizes input already, so -3dB trick won’t work in Capcut. Also, it has worse quality than Ripple
The best single MDX-UVR non-Roformer models for instrumentals explained in more detail
(UVR 5 GUI/Colabs/MVSEP/x-minus):
0. full band MDX-Net HQ_4 - faster, and an improvement over HQ_3 (it was trained for epoch 1149). In rare cases there’s more vocal bleeding vs HQ_3 (sometimes “at points where only the vocal part starts without music then you can hear vocal residue, when the music starts then the voice disappears altogether”). Also, it can leave some vocal residues in fadeouts. More often instrumental bleeding in vocals, but the model is made mainly for instrumentals (like HQ_3 in general)
0b) full band MDX-Net HQ_5 - similarly fast, might be less noisy, but more muddy, although better for vocals, but “it seems it's the best workaround when there is vocal bleed caused by Roformers”
1. full band MDX-Net HQ_3 - like above, might be sometimes simply the best, pretty aggressive as for instrumental model, but still leaving small amounts of vocal residues at times - but not like BS-Roformer v2/viperx, so results are not so filtered like in these.
HQ_3 filters out flute into vocals. Can be still useful to this day for specific use cases “the only model that kept some gated FX vocals I wanted to keep”.
It all depends on a song, what’s the best - e.g. the one below might give better clarity:
2. full band MDX23C-InstVoc HQ (since UVR 5.60; 22kHz/fullband as well) - tends to have more vocal residues in instrumentals, but can give the best results for a lot of songs.
Added also in MDX23 2.2.2 Colab, possibly when weights include only that model, but UVR's implementation might be more correct for only that single model. Available also in KaraFan so it can be used there only as a solo model.
2b. MDX23C-InstVoc HQ 2 - worse SDR, sometimes less vocal residues
Older MDX models
2c. narrowband MDX23C_D1581 (model_2_stem_061321, 14.7kHz) - better SDR vs HQ_3 and voc_ft (single model file download [just for archiving purposes])
"really good, but (...) it filters some string and electric guitar sounds into the vocals output" also has more vocal residues vs HQ_3.
*. narrowband Kim inst (a.k.a. “ft other”, 17.7kHz) - for the least vocal residues than both above in some cases, and sometimes even vs HQ_3
*. narrowband inst 3 - similar results, a bit more muddy results, but also a bit more balanced in some cases

---

### Stats: bleedless=38.25 fullness=17.23 SDR=11.89 

- BS-Roformer 2025.07 only on MVSEP - free with longer queue
Vocals bleedless: 38.25, fullness: 17.23, SDR: 11.89
The biggest bleedless metric for a single model so far. Compared to previous models, picks up backing vocals and vocal chops greatly where 6X struggles, and fixes crossbleeding and reverbs where in some songs previous models struggled before. 
Sometimes you might still get better results with Beta 6X or voc_fv4 (depending on a song). “Very similar to SCNet very high fullness without the crazy noise” - dynamic64, “handles speech very well. Most models get confused by stuff like birds chirping (they put it in the vocal stem), but this model keeps them out of the vocal stem way more than most. I love it!”
Works the best for orchestral choirs out of the long list of other models (.elgiano).
It can be better for metal both for vocal and instrumentals than the mesk’s models, a lot of the times (and sometimes the best).
The first iteration of the model (2025.06: 37.83/17.30/11.82) received two small updates and was replaced by 2025.07.

---

### Stats: bleedless=37.80 fullness=17.07 SDR=11.28 bleedless=39.20 

- Mel-Roformer 2024.10 (Bas Curtiz model fine-tuned by ZFTurbo) on MVSEP 
Vocals bleedless: 37.80, fullness: 17.07, SDR 11.28
Small amounts of bleeding from instrumentals (inst. bleedless 39.20), might struggle with flute occasionally, good enough for creating RVC datasets.

---

### Stats: bleedless=39.20 fullness=16.24 SDR=11.18. 

- Mel-Roformer Bas Curtiz edition (/w Marekkon5) (trained on also ZFTurbo dataset) on MVSEP (older version of 2024.10 model)
Vocals bleedless: 39.20, fullness: 16.24, SDR 11.18.

---

### Stats: bleedless=39.30 fullness=15.77 SDR=11.05 

- Unwa Kim Mel-Band Roformer Bleedless FT2 | download | Colab | Huggingface / 2 | UVR instruction
Vocals bleedless 39.30 (better than Mel-Kim), fullness 15.77 | SDR 11.05
(voc. fullness is worse than Mel Kim - 16.26, 
inst. bleedless is still lower than base Mel-Kim model: 46.30 vs 46.56)
“I usually use big beta 6x, big beta 5e if that fails and FT2 bleedless if I want very low noise or instruments are quiet (it gets muddy quick)” - Rainboom Dash

---

### Stats: bleedless=39.99 fullness=15.14 SDR=11.34 

- Unwa’s BS-Roformer Resurrection | yaml | Colab *
Vocal bleedless: 39.99, fullness: 15.14, SDR: 11.34
Shares some similarities with the SW model, including small size (might be a retrain). The default chunk_size is pretty big, so if you run out of memory, decrease it to e.g. 523776.

---

### Stats: bleedless=40.07 fullness=15.13 SDR=10.97 

- Unwa’s Revive 2 BS-Roformer fine-tune of viperx 1297 model | config | Colab
Voc. bleedless: 40.07, fullness: 15.13, SDR: 10.97
“has a Bleedless score that surpasses the FT2 Bleedless”
“can keep the string well”
It’s depth 12 and dim 512, so the inference is much slower than some newer Mel-Roformers.

---

### Stats: bleedless=37.61 fullness=15.89 SDR=11.32 

- BS-Roformer 2024.08 (viperx model fine-tuned v2 by ZFTurbo) on MVSEP 
Vocals bleedless: 37.61, fullness: 15.89, SDR: 11.32
Good for inverts, Dolby, lots of harmonies, BGVs. Good or even the best vocal fullness for some genres ~Isling, decent all-rounder, but might be muddier than Mel models here, although it gives less vocal residues than all the Mel Kim fine-tune models here, can be also used for RVC). “I've found it very useful for extremely quiet vocals that Mel couldn't extract” - Dry Paint Dealer. It’s a second MVSEP’s fine-tune of viperx model.
Iirc, it’s used as a preprocessor model for "Extract from vocals part" feature on MVSEP.

---

### Stats: bleedless=36.30 fullness=17.73 SDR=11.93 

- MVSep Ensemble 11.93 (vocals, instrum) (2025.06.28) - only for paid premium users
Vocals bleedless: 36.30, fullness: 17.73, SDR: 11.93
Surpassed sami-bytedance-v.1.1 on the multisong dataset SDR-wise.

---

### Stats: bleedless=36.06 fullness=16.95 SDR=11.36 

- BS-Roformer SW 6 stem (MVSEP, Colab) / Vocals only *
Vocals bleedless: 36.06, fullness: 16.95, SDR 11.36
Good for some deep voices.
Bleedless #2 (less)

---

### Stats: bleedless=35.16 fullness=17.77 SDR=11.12 

- Unwa Mel-Roformer Big Beta 6X vocal model | yaml | Colab | AI Hub Colab | Huggingface | uvronline
voc bleedless: 35.16, fullness: 17.77, SDR: 11.12
“it is probably the highest SDR or log wmse score in my model to date.”
“There's some noise audible, it doesn't sound as clean when you compare to a more bleedless model (...) but it's certainly not fullness... (...) I think calling it bleedless wouldn't be crazy... makes more sense than "middle of the road" - rainboomdash
“Significantly better” than 5e for some people, although slower. Some leaks into vocal might occur, plus “The biggest problem with the model is the remaining background noise. If it were cleaner, it would already be an almost perfect result.” - musictrack
“6X has a lot less noise on vocals, but it's pretty muddy. I would prefer something in between [5e and 6X]. I tried to apply the phase [fixer/swapper] to the vocals and the noise was reduced, but only slightly.” - Aufr33
Some people might prefer fv5 instead [at least on some songs] ~5b
“6x is picking up BV just fine, where voc fv4 is failing” - Rainboom Dash
Training details:
“dim 512, depth 12. It is the largest Mel-Band Roformer model I have ever uploaded.” - “the same as Bas Curtiz Edition” model. It has a bigger SDR vs smaller depth 6 Big Beta 6 model. “I've added dozens of samples and songs that use a lot of them to the dataset”
Fullness models

---

### Stats: bleedless=30.51 fullness=21.43 SDR=10.98 

- * Unwa bs_roformer_revive3e | config | Colab
voc bleedless: 30.51, fullness: 21.43, SDR: 10.98‘
“A vocal model specialized in fullness.
Revive 3e is the opposite of version 2 — it pushes fullness to the extreme.
Also, the training dataset was provided by Aufr33. Many thanks for that.” - Unwa
“seems to sound better than beta5e, it sounds fuller, but this also means it sounds noisier” - gilliaan. For some people, it’s even the best.
More fullness less bleedless

---

### Stats: bleedless=26.61 fullness=24.93 SDR=10.64 

- * Gabox experimental Mel-Roformer voc_fv6 model | yaml | Colab
voc bleedless: 26.61, fullness: 24.93, SDR: 10.64 
“Definitely not bleedless” - rainboomdash, “Sounds like b5e with vocal enhancer. Needs more training, some instruments are confused as vocals” - Gabox. “fv6 = fv4 but with better background vocal capture” - neoculture
“very indecisive about whether to put vocal chops in the vocal stem or instrumental stem.
sometimes it plays in vocals and fades out into instrumental stem and sometimes it just splits it in half kinda and plays in both at the same time lol” - Isling
“I think is the fullest vocal model I've heard, aside from maybe the scnet high fullness ones lol/ Oh and revive 3e and b5e are full too but yeah.” - Musicalman

---

### Stats: bleedless=25.30 fullness=23.50 SDR=10.40 

- SCNet XL very high fullness on MVSEP
voc bleedless: 25.30, fullness: 23.50, SDR: 10.40

---

### Stats: bleedless=25.48 fullness=22.70 SDR=10.87 

- SCNet XL IHF (high instrum fullness by bercuily)
voc bleedless: 25.48, fullness: 22.70, SDR: 10.87
(it was made mainly for instrumentals, but “It can also be an insane vocal model too”)

---

### Stats: bleedless=28.31 fullness=17.98 SDR=11.11 

- MVSEP SCNet XL IHF
voc bleedless 28.31, fullness 17.98, SDR: 11.11
“It has a better SDR than previous versions. Very close to Roformers now.” also, vocal bleedless is the best among all SCNet variants on MVSEP. Metrics. IHF - “Improved high frequencies”.
“Certainly sounds better than classic SCNet XL (...) less crossbleeding of vocals in instrumental so far, and handle complex vocals better” - dca
Middle of the road #1 (lower fullness)

---

### Stats: bleedless=31.55 fullness=20.44 SDR=10.87 

- Gabox vocfv7 beta 2 Mel-Roformer model | yaml | Colab
voc bleedless: 31.55, fullness: 20.44, SDR: 10.87
“fullness went down a little bit” vs beta 1 (...) Definitely an improvement over fv4 (...) still quite a bit fuller than big beta 6x, but has less noise than even fv4 (...) at least when the instruments are loud, fv7beta2 is usually quite a bit less noisy than fv4, while still maintaining a decent amount of fullness... it is a bit less, but not too much (...) both are pretty noisy with fv4 (...) sometimes the noise can be pretty significant with fv7beta1, and fv7beta2 may have the fullness you desire. (...) “I'm really liking the balance of fullness and noise for most songs. fv4 and fv6/fv7beta1 are usually pretty noisy... this is less noisy, but still has a good amount of fullness.” still gonna have an issue with backing vocals compared to fv7beta1 sometimes… (...) “Fv7beta2 has still been significantly better with BV than fv4, despite quite a bit less noise” but “significant issues on one song, while fv6/fv7beta1 didn't” - rainboomdash

---

### Stats: bleedless=30.83 fullness=21.82 SDR=10.80 

- Gabox vocfv7beta3 Mel-Roformer | yaml | Colab
voc bleedless 30.83, fullness 21.82, SDR 10.80
“beta 1 and 2... eh, pretty close to same instrumental bleed,
but beta 3 def a step up from the two songs I compared (...)
most songs so far, fv7beta3 is fuller than fv7beta1,
def less robotic sounding at times (when a voice gets quiet/hard to capture, and it just fails).
Just had another song where fv7beta1 was fuller than fv7beta3, but it was also a lot noisier
large majority of the songs I tested, fv7beta3 was fuller... I think fv7beta3 is usually a bit noisier than fv7beta1? But also sounds fuller in those cases, I'd say it's generally worth it
instrumental bleed, usually worse with fv7beta3 versus fv7beta1, but it depends
fv7beta2 is always less full/less noise, but only slightly less instrumental bleed than fv7beta1” - rainboomdash

---

### Stats: bleedless=30.81 fullness=21.21 SDR=10.96 fullness=21.33 bleedless=29.07 SDR=10.58 

- Gabox Mel-Roformer voc_fv7 beta 1 (a.k.a. vocfv7beta1) model | yaml | Colab
voc bleedless: 30.81, fullness: 21.21, SDR: 10.96
“one step below the extreme fullness models (...) fv6 on average is more full” - rainboomdash. "Just a better fv4 it seems, better bleedless" (fullness: 21.33, bleedless: 29.07, SDR 10.58)
vs voc_fv4 "It is noisier... Kinda closer to beta 5e?” “It's slightly less noise and fullness than beta 5e but picking up the backing vocals REALLY well, significantly better than beta 5e”
But it's pulling the backing vocals out even better than 5e” “the backing vocals are so good!
“it does have significant synth bleed, too...   it at least wasn't coming through at full volume
when I say fullness, I specifically mean how muddy it sounds” - Raiboom Dash
_

---

### Stats: bleedless=29.07 fullness=21.33 SDR=10.58 

- Gabox Mel-Roformer voc_fv4 | yaml | Colab | Huggingface / 2 *
voc bleedless 29.07, fullness 21.33, SDR 10.58
“Very clean, non-muddy vocals. Loving this model so far” (mrmason347)
Good for anime and RVC purposes, currently the best public model for it (codename)
“The important thing for an RVC dataset is to get lead vocals so fv4 is good for that
The newer karaoke models are also helpful” - Ryan
Some might prefer voc_gabox2 instead, occasionally - chroniclaugh.
The opposite of Beta6x which has “lower noise but [is] less full/muddier (...) noise/muddiness seems between 6x and 5e, but even 6x is picking up BV just fine, where voc fv4 is failing”
Some people might want to test it with even overlap 32, and then:
“It's close to perfect, the only thing is it kinda struggled with picking up the adlibs and the delay, but the lead vocal is almost perfect I think. (...) on another song (...) 5e is just too noisy and 6x is muddy, fv4 is best of both worlds (...) has segments with constant significant vocal bleed (for the most part, it's not audible at all) (...) I was trying to get an acapella and every model failed except this one. It's not perfect, but I guess some songs are just too hard for the AI.” - Rainboom Dash
Good also for instrumentals, if you need less vocal residues than typical instrumental Roformers (even less than Mel Kim, FT2 Bleedless, or Beta 6X - makidanyee.
“even beta 6x is a lot better at pulling that background vocal out than voc fv4...
and that's a less full model. hmm, fv6 is noisier and also not picking up the backing vocals as full as the last mel band roformer” - Rainboom Dash

---

### Stats: bleedless=32.07 fullness=20.77 SDR=10.66 

- Unwa Mel big beta 5e vocal model | Colab | Huggingface / 2 | MVSEP | uvronline via special link free/premium | MSST-GUI | UVR
Model files | yaml: big_beta5e.yaml or fixed yaml for AttributeError in UVR
voc bleedless: 32.07, fullness: 20.77 (the biggest for now), vocals SDR: 10.66
“feel so full AF, but it has noticeable noise similar to lew's vocal enhancer”
You can alleviate some of this noise/residues by using phase fixer/swapper and using becruily vocals model as reference (imogen).
It seems to fix some issues with trumpets in vocal stem - maxi74x1.
“It's noisy and, IDK, grainy? When the accompaniment gets too loud. (...) Definitely not muddy though, which is a welcome change IMHO. I think I prefer beta 4 overall” - Musicalman “ending of the words also have a robotic noise” - John UVR
“Perhaps a phase problem is occurring” - unwa. Phase swapper doesn’t fix the issue (it works for inst unwa’s models).
If you try big beta 5e on a song that has lots of vocal chops, the vocal chops will be phasing in and out and sound muddy (Isling).
“Excellent for ASMR, for separating Whispers and noise, the quality is super good
That's good when your mic/pc makes a lot of noise. All the denoise models are a bit too harsh for ASMR (giliaan)”
Worse for RVC than Beta 4 model below (codename/NotEddy)

---

### Stats: bleedless=31.26 fullness=20.72 SDR=10.55 

- Mel-Roformer vocal by becruily model | config for ensemble in UVR | MVSEP | Colab
voc bleedless: 31.26, fullness: 20.72 (on pair with 5e), SDR: 10.55 | Huggingface / 2
Lower bleedless than 5e,“pulling almost studio quality metal screams effortlessly, wOw ive NEVER heard that scream so cleanly”
(on older UVR beta patches) If you use lower dim_t like 256 at the bottom of config for slower GPU these are the first models to have muddy results with it.
Consider setting 485100 chunk_size in the yaml for the highest SDR.
Currently used on x-minus/uvronline as a model for phase fixer.

---

### Stats: bleedless=29.50 fullness=20.67 SDR=10.56 

- Gabox Mel-Roformer voc_fv5 | yaml | Colab
voc bleedless: 29.50, fullness: 20.67, SDR: 10.56
“fv5 sounds a bit fuller than fv4, but the vocal chops end up in the vocal stem. In my opinion, fv4 is better for removing vocal chops from the vocal stem” - neoculture. Examples
Other/older models

---

### Stats: bleedless=33.13 fullness=18.98 SDR=10.98 

- Gabox Mel-Roformer voc_gabox2 model | yaml | Colab
Vocal bleedless: 33.13, fullness: 18.98, SDR: 10.98

---

### Stats: bleedless=32.15 fullness=19.97 

- Gabox Mel-Roformer Vocal F (fullness) v3 model | Colab | Huggingface / 2
voc bleedless: 32.15, fullness 19.97

---

### Stats: bleedless=33.40 fullness=19.31 

- Gabox Mel-Roformer Vocal F (fullness) v2 model | Colab | Huggingface / 2
voc bleedless: 33.40, fullness: 19.31

---

### Stats: bleedless=32.98 fullness=18.83 

- Aname Mel FullnessVocalModel (yaml) model | Colab | Huggingface / 2
Vocals bleedless: 32.98 (less than beta 4), fullness: 18.83 (less than big beta 5e/voc_fv4/becruily, more than beta 4)

---

### Stats: bleedless=34.66 fullness=18.10 

- Gabox Mel-Roformer voc_gabox (Kim/Unwa/Becruily FT) model | Colab Huggingface / 2
voc bleedless: 34.66 (better than 5e, beta 4 and becruily voc), fullness 18.10 (on pair with beta 4, worse than 5e and becruily)

---

### Stats: bleedless=33.76 fullness=18.09 

- Mel-Roformer unwa’s beta 4 (Kim’s model fine-tuned) download | Colab | Huggingface / 2
Vocals bleedless: 33.76, fullness: 18.09
“Clarity and fullness” - even compared to newer models above.
Beta 1/2 were more muddy than Kim’s Roformer, potentially a bit less of residues, a bit more artificial sound. Ringing issues in higher frequencies fixed in beta 3 and later. It’s good for RVC (and favourite codename’s public model for RVC before voc_fv4 was released). Fuller vocals than Bas Curtiz FT on MVSEP (but can bleed more synths) ~becruily
Unwa’s vocal models are capable of handling sidechain in songs - John UVR
Bleedless models #2

---

### Stats: bleedless=38.80 fullness=15.48 SDR=11.03 

- BS-Roformer Revive unwa’s vocal model experimental | yaml 
(viperx 1297 model fine-tuned)
Voc. bleedless: 38.80, fullness: 15.48, SDR: 11.03
“Less instrument bleed in vocal track compared to BS 1296/1297” but it still has many issues, “has fewer problems with instruments bleeding it seems compared to Mel. (...) 1297 had very few instrument bleeding in vocal, and that Revive model is even better at this.
Works great as a phase fixer reference to remove Mel Roformer inst models noise” (dca)

---

### Stats: bleedless=37.27 SDR=10.82 

- SYHFT V5 Beta - only on x-minus/uvronline (still available only with this link for premium users, and for free)
Vocal bleedless: 37.27, fullness, 16.18, SDR: 10.82
Other models #2

---

### Stats: bleedless=37.06 fullness=16.61 

- Unwa’s Kim Mel-Band Roformer FT2 | model | Colab
Vocals bleedless: 37.06, fullness: 16.61 (fullness worse vs the previous FT, but both metrics are better than Kim’s)
It tends to muddy instrumental outputs at times, similarly like the OG Kim’s model was doing, which didn’t happen in the previous FT below. Metrics

---

### Stats: bleedless=36.11 fullness=16.80 SDR=11.05 

- Unwa Kim Mel-Band Roformer FT3 Preview | model | yaml | Colab | uvronline via special link for: free/premium (scroll down)
Vocal bleedless: 36.11, fullness: 16.80, SDR: 11.05
“primarily aimed at reducing leakage of wind instruments to vocals.”
For now, FT2 has less leakage for some songs (maybe till the next FT will be released)

---

### Stats: bleedless=36.75 fullness=16.40 

- Unwa’s Kim Mel-Band Roformer FT vocal model | Colab
Enhanced both voc bleedless 36.75 (vs 36.95) and fullness 16.40 (vs 16.26) metric for vocals vs the original Mel Kim model. SDR-wise it’s a tad lower (10.97 vs 11.02).
Tips for separating vocals

---

### Stats: SDR=10.44 

- Models ensembled (inst, voc) available for premium users on mvsep.com
(SDR 10.44-11.93 and “High Vocal Fullness” variants)
RVC models choice by AI Hub (subject to change;
read their current docs too)
If you can separate with these models downloaded from above locally, see also here for the list of all cloud sites and Colabs.
“If you need to remove multiple noises, follow this pipeline for the best results:
Remove instrumental -> Remove reverb [probably on vocals] -> Extract main vocals -> Remove noise”
Or also Isling’s approach “gives insanely clean results”:
Vocals>De-reverb>Karaoke
Vocals

---

### Stats: SDR=9.85 

- Mel-Roformer-Karaoke-Aufr33-Viperx (surpassed by Becruily and Frazer Karaoke, but the first can be more consistent; anvuew's Karaoke model have fuller lead vocals; also older Model fuzed gabox & aufr33/viperx (SDR: 9.85) is mentioned in their MVSEP section)
De-noise

---

### Stats: bleedless=30.75 fullness=13.24 SDR=8.01 

- Aname Full Scratch Mel-Band Roformer model
bleedless 30.75 fullness 13.24, SDR: 8.01

---

### Stats: bleedless=36.75 fullness=16.26 SDR=11.07 

- Mel-Roformer model by Kim | model | config
Vocals bleedless: 36.75 | fullness: 16.26 | SDR: 11.07
(Colab/Huggingface/2/MVSEP/uvronline via special link for: free/premium (scroll down)/UVR beta Roformer (available in Download Center)/MSST-GUI/simple Colab/CML inference)
Usual base for lots of Mel fine-tunes on that list.
Sometimes might leave instrumental residues in vocals, but can be less muddy than other BS-Roformers - the same goes to any fine-tunes of this model vs BS 2024.08, so effectively all the Mel models above)
“godsend for voice modulated in synth/electronic songs” vs 1296 can be more problematic with wind instruments putting them in vocals.

---

### Stats: SDR=17.55 

- “ver. 2024.04” SDR 17.55 on MVSEP - fine-tuned viperx model v1 (can pick in adlibs better, occasionally picks some SFX’, sometimes one, sometimes the other is “slightly worse at pulling out difficult vocals”)

---

### Stats: SDR=17.17 

- BS-Roformer viperx 1297 model (UVR beta/MVSEP a.k.a. SDR 17.17 for “1296” variant iirc/called just “BS-Roformer” on uvronline via special link for: free/premium (scroll down)

---

### Stats: SDR=10.41 

- MVSEP’s BS Roformer by MVSep Team (SDR: 10.41)
under option "MVSep MelBand Karaoke (lead/back vocals)", metrics. Might be a fine-tune. Use the option extract vocals first.
(“In contrast with other Karaoke models, it returns 3 stems: "lead", "back" and "instrumental".)
“If I had to compare it to any of the models, it is similar to the frazer and becruily models. Sometimes it does not detect the lead vocals especially if there's some heavy hard panning, but when it does, there is almost no bleed, and it works very well with heavy harmonies in mono from what I tested.” - smilewasfound
“becruily & frazer is better a little when the main voice is stereo” - daylightgay
“On tracks I tested, harmony preservation was better in becruily & frazer (...) the new model isn't worse, I ended up finding examples like Chan Chan by Buena Vista Social Club or The Way I Are by Timbaland where it is better than the previous kar model. The thing is, with the Kar models, it's just track per track. Difficult to find a model for batch processing as it's really different from one track to another” - dca100fb8
As for MVsep Team: “It’s the only model that combines the lead vocal doubles with the lead vocals stem. It’s far more useful for dissecting harmonies on songs with vocal doubles like Backstreet Boys” - heuheu
“I also found the new model to not keep some BGVs, mainly mono/low octave ones, despite higher SDR” - becruily
“I think I've found a solution for people who don't like the new model.
If you put an audio file through the karaoke model and then put the lead vocal result through that, it usually picks up doubles.
Which you can then put in your BGV stem if you'd like” - dynamic64

---

### Stats: SDR=9.53 

- MVSep MelBand Karaoke (lead/back vocals) SCNet XL IHF by becruily (SDR: 9.53)
Worse SDR than the top performing Roformers, but works best in busy mix scenarios, and when Mel-Roformer models fail, generally bleedier arch. To fix the bleed in the back-instrum stem, use “Extract vocals first”, but “I noticed a pattern that if you hear the lead vocals in the back-instrum track already (SCNet bleed), don't try to use Extract vocals first because there will be even more lead vocal bleed” - dca (iirc it uses the biggest SDR BS-Roformer vocal model as preprocessor).
“Separates lead vocals better than Mel-Roformer karaoke becruily. It's not perfectly clean, sometimes a bit of the backing vocals slips through, but for now, scent karaoke model still the most reliable for lead vocals separation (imo)” - neoculture

---

### Stats: SDR=8.18 

- Aufr33 BS-Roformer Male/Female beta model | config | Colab | x-minus (uses Kim-Mel-Band-Roformer-FT2 as preprocessor) | MVSEP
(based on BS-RoFormer Chorus Male Female by Sucial) SDR 8.18

---

### Stats: SDR=13.72 

- Drums only/other SCNet XL model by ZFTurbo on MVSEP, SDR 13.72
“Very hit or miss. When they're good they're really good, but when they're bad there's nothing you can do other than use a different model” - dynamic64

---

### Stats: SDR=14.07 

- Ensemble of SCNet XL, BS and HTDemucs4 models (SDR 14.07); SCNet can be sometimes worse than Demucs which “considers not only spectrograms but also waveforms” - Unwa)
After separation, you might want to then apply Mel-RoFormer De-noise to remove the high noise, and finish with Apollo Universal by Lew (model | Colab) to get more clarity (Tobias51).

---

### Stats: SDR=9.48 

- Demucs_ft (4 stem) - the best single Demucs’ model (Colab / MVSEP / UVR5 GUI)
Multisong dataset SDR 9.48: bass: 12.24, drums: 11.41, other: 5.84, vocals: 8.43 
(shifts=1, overlap=0.95)
Better drums and vocals than in Demucs 6 stem model, decent acoustic guitar results in 6s. Good bass stem as Demucs “considers not only spectrograms but also waveforms”.
For 4 stems alternatively check MDX_extra, generally Demucs 6 stem model is worse than MDX-B (a.k.a. Leaderbord B) 4 stem model released with MDX-Net arch from MDX21 competition (kuielab_b_x.onnx in this Colab), and is also faster than Demucs 6s. 
For Demucs use overlap 0.1 if you have instrumental instead of mixture mixed with vocals as input (at least it works with ft model) and shifts 10 or higher. For normal use case (not instrumentals input) it will give more vocal residues, overlap 0.75 is max reasonable speed-wise, as a last resort 0.95, with shifts 10-20 max.

---

### Stats: SDR=9.29 

- SCNet-large_starrytong model (4 stems) (Colab or MSST-GUI)
Multisong dataset SDR 9.29: bass: 11.28, drums: 11.24, other: 5.58, vocals: 9.06 (overlap: 4)
It’s 3x faster than Demucs (Nvidia GPU or CPU-only) and sounds better for some people, except for bass. SDR-wise, vocals are better than in demucs_ft (which is low vs single vocal/inst models anyway). Better SDR than starrytong’s MUSDB18 and Mamba models.

---

### Stats: SDR=9.93 

- SCNet XL IHF model (4 stems) by ZFTurbo
Multisong dataset SDR 9.93: bass: 11.94, drums: 11.58, other: 6.49, vocals: 9.69

---

### Stats: SDR=9.72 

- SCNet XL model (4 stems) by ZFTurbo (Colab | MSST-GUI | MSST | UVR beta patch)
Multisong dataset SDR 9.72: bass: 11.87, drums: 11.49, other: 6.19, vocals: 9.32
Better metrics than the starrytong model but “downgrade to the Large model since it produces a f*** ton of buzzing” due to undertraining.
Only bass is better in Demucs_ft - 12.24, although drums might be still better in demucs_ft.
2 stem model on MVSEP is further trained iirc.

---

### Stats: SDR=9.38 

- BS-Roformer model (4 stems) by ZFTurbo | MSST-GUI
Multisong dataset SDR 9.38: bass: 11.08, drums: 11.29, other: 5.96, vocals: 9.19
Trained on MUSDB18HQ

---

### Stats: SDR=5.41 

- MVSEP Bowed Strings BS-Roformer (“doesn't disappoint”, SDR 5.41)

---

### Stats: SDR=3.84 

- MVSEP Strings MDX23C (it’s “weak”, SDR 3.84)

---

### Stats: SDR=2.87 

- x-minus.pro/Uvronline.app Mel-Roformer model by viperx (SDR 2.87) (sometimes with this link or this link)

---

### Stats: SDR=7.79 

- Logic Pro (paid; May 2025 update, SDR 7.79) / BS-Roformer 6 stems / MVSEP Piano SW
(“1000 times more efficient than the Lalal.ai piano model”)

---

### Stats: SDR=6.21 

- MVSep Piano Ensemble (Mel-Roformer  + SCNet Large piano models; SDR 6.21)
(Mel is viperx' iirc; “a [tiny] bit more bleed during the choruses and whatnot” vs x-minus, “works well maybe 7 times out of 10”; SCNet, “has a less watery sound, but more bleed” vs Mel)

---

### Stats: SDR=6.21 

- x-minus.pro (for paid users; cheap subscription; “more consistent than MVSep piano and demucs_6s” it knows well what piano is, but it sounds the best for other stem of piano separation, but e.g. on Carpenters - Yesterday Once More “while not terrible, the dropouts, underwater 'gurgles', and general lack of piano punch/presence remains noticeable” - Chris_tang1, while MVSEP Piano Ensemble: SCNet + Mel, SDR: 6.21, was much better in that case - might vary on a song)

---

### Stats: SDR=10.16 

- joowon bandit model: https://github.com/karnwatcharasupat/bandit
(better SDR for Cinematic Audio Source Separation (dialogue, effect, music) than DNR Demucs 4 below (SDR 10.16>11.47) - Colab / MVSEP)

---

### Stats: SDR=18.81 

- avuew v2 “less aggressive” variant - a bit lower SDR 18.81 | DL | config

---

### Stats: Fullness=27.07 Bleedless=47.49 Fullness=29.38 Bleedless=44.95 Fullness=32.03 Bleedless=42.87 Fullness=31.85 Bleedless=41.73 Fullness=32.13 Bleedless=41.69 Fullness=33.22 Bleedless=40.71 Fullness=33.98 Bleedless=40.49 Fullness=29.83 Bleedless=39.36 Fullness=36.91 Bleedless=38.77 Fullness=35.69 Bleedless=37.59 Fullness=38.71 Bleedless=35.62 Fullness=38.87 Bleedless=35.59 Fullness=39.40 Bleedless=33.49 Fullness=27.83 Bleedless=49.12 Fullness=27.17 Bleedless=47.94 Fullness=28.70 Bleedless=47.68 Fullness=27.73 Bleedless=47.48 Fullness=27.45 Bleedless=47.41 Fullness=28.02 Bleedless=47.24 Fullness=27.44 Bleedless=46.56 Fullness=27.78 Bleedless=46.31 Fullness=27.63 Bleedless=45.90 Fullness=28.36 Bleedless=45.58 Fullness=29.18 Bleedless=45.36 fullness=17.30 bleedless=37.83 fullness=17.73 bleedless=36.30 

Top metrics of publicly available Roformers for instrumentals available for download
(as for 18.06.25)
Instrumental models sorted by instrumental fullness metric:
INSTV6N (41.68)>inst_Fv4Noise (40.40)/INSTV7N (no metrics)/Inst V1e (38.87)>Inst Fv3 (38.71).
While V1e+ (37.89) might be already muddy in some cases
Instrumental models sorted by instrumental bleedless metric:
Gabox inst_fv7b
Fullness: 27.07 (worse than most vocal Mel-Roformers later below)
Bleedless: 47.49
Inst_GaboxFv7z
Fullness: 29.38
Bleedless: 44.95
Unwa BS-Roformer-Inst-FNO
Fullness: 32.03
Bleedless: 42.87
Unwa v2
Fullness: 31.85
Bleedless: 41.73
Inst_gaboxBv3
Fullness: 32.13
Bleedless: 41.69
Inst_GaboxFv8 (its replaced v2 variant)
Fullness: 33.22
Bleedless: 40.71
Becruily inst
Fullness: 33.98
Bleedless: 40.49
Gabox instv7plus
Fullness: 29.83
Bleedless: 39.36
Unwa HyperACE
Fullness: 36.91
Bleedless: 38.77
Unwa v1
Fullness: 35.69
Bleedless: 37.59
Gabox fv3
Fullness: 38.71
Bleedless: 35.62
Unwa v1e
Fullness: 38.87
Bleedless: 35.59
Gabox fv5
Fullness: 39.40
Bleedless: 33.49
Vocal models/ensembles sorted by instrumental bleedless metric:
(more muddy; Gabox and Unwa’s Revive models not evaluated yet):
Descriptions of the public models
MVSep BS-Roformer (2025.07.20) - the 2 previous versions got replaced on the site by it
Inst. Fullness 27.83
Inst. Bleedless 49.12
MVSep Ensemble 11.50 (2024.12.20)
Inst. Fullness 27.17
Inst. Bleedless 47.94
MVSep Ensemble (4 stem) 11.93 (2025.06.30)
Inst. Fullness 28.70
Inst. Bleedless 47.68
MVSep MelBand Roformer (2024.10)
Inst. Fullness 27.73
Inst. Bleedless 47.48
BS-RoFormer SW 6 stem (MVSEP/Colab/undef13 splifft)
Inst. Fullness 27.45
Inst. Bleedless 47.41
(use inversion from vocals and not mixed stems for better instrumental metrics)
MDX23 Colab fork v2.5 by jarredou
Inst. Fullness 28.02
Inst. Bleedless 47.24
(more noticeable bleeding/noise than MVSep Ensemble above)
voc_fv4
xx
xx
(Good if you need less vocal residues than typical instrumental Roformers (even less than Mel Kim, FT2 Bleedless, or Beta 6X - makidanyee).
MelBand Roformer Kim
Inst. Fullness 27.44
Inst. Bleedless 46.56
Kim | FT2 Bleedless (by Unwa)
Inst. Fullness 27.78
Inst. Bleedless 46.31
Beta 5e (by unwa)
Inst. Fullness 27.63 (bigger metric than Kim)
Inst. Bleedless 45.90
Kim | FT 2 (by unwa)
Inst. Fullness 28.36
Inst. Bleedless 45.58
Kim | FT (by unwa)
Inst. Fullness 29.18
Inst. Bleedless 45.36
MVSEP BS Roformer (2025.06)
Inst. fullness: 17.30
Inst. bleedless: 37.83
(can be still a good choice in case of some crossbleeding, vocal chops, or residues of reverbs or BGV)
MVSEP Ensemble 11.93 (also contains 2025.06)
Inst. fullness: 17.73
Inst. bleedless: 36.30

---

### Stats: Fullness=28.07 Bleedless=45.15 Fullness=29.08 Bleedless=43.26 Fullness=28.03 Bleedless=44.16 Fullness=28.25 Bleedless=40.95 Fullness=28.60 Bleedless=40.34 Fullness=28.48 Bleedless=44.81 Fullness=26.29 Bleedless=44.71 

Outperformed vocal models for instrumental bleedless
(still metrics for instrumental stem, so after inversion if not duality)
SYHFT V3 (by SYH99999)
Fullness 28.07
Bleedless 45.15
Duality v1 (by unwa)
Fullness 29.08
Bleedless 43.26
Duality v2 (by unwa)
Fullness 28.03
Bleedless 44.16
Mel Becruily vocal
Fullness 28.25
Bleedless 40.95
SYHFT V2.5 (by SYH99999)
Fullness 28.60
Bleedless 40.34
Big SYHFT V1 (by SYH99999)
Fullness 28.48
Bleedless 44.81
Unwa beta 4
Fullness 26.29
Bleedless 44.71
SYHFT V4 and V5 were never publicly released

---

### Stats: SDR=16.32 

- Bas’ unreleased fullband vocal model epoch 299 + voc_ft - SDR 16.32)

---

### Stats: SDR=17.26 SDR=18.13 SDR=18.75 

- Bytedance v.0.2 - inst. SDR 17.26, now it’s outperformed by v.0.3 and is 17.28, now called 1.0),
-"MSS" - is probably ByteDance 2.0, not multi source stable diffusion, as BD's test files which were published were starting with MSS name before, but the first doesn't necessarily contradict the latter, although they said to use novel arch - SDR 18.13, and probably another one by ByteDance - SDR 18.75, let's call it 2.1, but seeing inconsistent vocal result vs previous one here, we have some suspicions that the result was manipulated at least for vocals (or stems were given from different model).

---

### Stats: SDR=9 

1. Saxophone model for UVR.
2. "Karokee" model for MDX v2."
Also, completely rewritten UVR5 GUI version.
Among many new features - new denoiser for MDX models available and new Demucs 4 models (SDR 9).
Online sites and Colabs for separation - the best quality freebies you can currently get
Refrain from using lossy audio files for separation (e.g. downloaded from YouTube) for the best results.
See here for ripping lossless music from Tidal, Qobuz, Deezer, Apple Music or Amazon.
If you don't have a computer, or decent CPU/GPU and separation is too slow on your machine using UVR 5 GUI, or it doesn’t work correctly for you, you can use these online sites to separate for free:
mvsep.com (lots of the best UVR 5 GUI models incl. various Roformers, and some exclusive models not available in UVR, and ensemble of these for paid users)
The page by MDX23 code/Colab original author and models’ creator - ZFTurbo.
If you register an account on MVSep, you can output in .flac and .wav 32-bit float.
Since 28.07.25, now 32-bit float for WAV will be used only if gain level fall outside 1.0 range, otherwise 16 bit PCM will be used.
Also, now FLAC uses 16-bit instead of 24-bit.
If you have troubles with nulling due to the new changes in free version, consider decreasing volume of your mixtures by e.g. 5dB, and you won’t be affected, although it might slightly affect separation results.
If your credits are higher than 0, you have shorter queue (also users using the mobile app have “a bit higher priority”), 100 MB max file size/10 minutes (up to 10 concurrent separations in premium and 1GB/100MB in premium). You can disable using credits for non-ensembles in settings, for the cost of a longer queue again. Shorter queues seem to be currently around mornings of GMT +2/CEST (9 a.m.) or even early afternoon or late at night, depends - sometimes the queue goes crazy long randomly, but if you don’t care, you can just set your jobs and download it the next day.
Selecting “Extract from vocals part” uses the best BS-Roformer models as preprocessor for the chosen model (currently the ver. 2024.08 - subject to change).
For Mel Band Kar dim_t 801 and overlap 2 is used, and for Mel Becruily inst/voc, Mel 2024.10, Mel Rofo Decrowd: 1101 and 2.
If downloading from the site is too slow try out e.g. Free Download Manager (Win) or ADM (Android) and/or VPN, or if you have premium you can use your credits to pack to zip the separation after your separation is done.
Batch processing with API and GUI - click (Mac). You can use MVSEP download links as remote URL in order to further separate the result (e.g. MVSEP Drums>drumsep for more stems).
Q: Is there a way to turn off the normalization when using FLAC?
It's annoying when you have to combine the outputs later
A: “No, if you turn off normalization, FLAC will cut all above 1.0
And if it was normalized, it means you had these values.”
FLAC doesn’t support 32-bit float, it’s 32 int, so normalization is still needed.”
So if your stems don’t invert correctly, just use WAV.
Q: How multichannel is handled by MVSEP:
A: librosa script which performs stereo downmixing (for 5.1 or 7.1 inputs)
Q: I convert a song using v1e+ and use phase fix, then do another conversion using for example Gabox V7 and use phase fix, if I go back to upload the same song using v1e+ it gives the stems instantly but if I use phase fix it will process again, in the past it would remember
A: This may be a temporary issue. Sometimes that server may be unavailable, then processing will start on another server.
Q: What’s “include results from independent models”?
A: “When you use an ensemble, you will also get results from each model of the ensemble and not only the ensemble final result.”
Q: What means “Disown Expired Separations” option
A: “we do not delete expired separation data (they are needed for analytics), but just remove your ownership from expired separations
We could have written delete expired separations, but wanted to be more clear about your data”
Q: “So I understand, all the uploads are kept, regardless of 'disowning' or not. So what is the distinction between disowning and not disowning? Is there one?”
A: no uploads are kept, just settings. If you disown, you won't see your expired separations
Q: I will need a refresher in terms. Separations are created from (audio) uploads. Separations are also not kept? Only the settings used, i.e. kuielab_a_drums, aufr33-jarredou_DrumSep_model_mdx23c_ep_141_sdr_10.8059, and whatever segment, aggression, vocal only, etc are selected at the point of hitting 'do it'.. in a manner of speaking..
A: separation is when you choose settings and upload file, we just save the settings and delete file.
Q: How to use the same file over and over for different models in order to test them, but without reuploading the same file over and over
A: “You can use remote upload for this. Just use link on file from previous separation. So you will not need to upload anything. https://mvsep.com/remote”
x-minus.pro / uvronlione.app (-||-, 10 minute daily limit for free, very fast, mp3 192kbps output for free (lossless for premium), some exclusive models for paid users, Roformers will be back for free around 31 December 2024)
The site is made by one of the UVR creators and models creator - Aufr33 with dedicated
Overlap 2 used for Roformers. At subscription level standard and above, song limit for Roformers is 20 minutes. For Mel Karaoke model, dim_t 256 and overlap 2 is being used
“Mel-RoFormer by Kim & unwa ft3 and some other models are hidden. As before, you can find them here: https://uvronline.app/ai?hp&test” - Aufr33 (link for free users)
Model used for phase fixer/swapper/correction on the site is Mel-Roformer Becruily Vocal

---

### Stats: SDR=7.7 SDR=6.5 SDR=9 SDR=9 

Demucs 3
for 4 stems
(SDR 7.7 for 4 stems, it’s better than Spleeter (which is SDR 6.5-7), or better than MDX 4 stem. In most cases, it’s even better than Audioshake - at least on tracks without leading guitar)
Accompanied by MDX-UVR 9.7 vocal model, it gives very good 4 stem separation results
(For Demucs 4 a.k.a "htdemucs" check below)
https://colab.research.google.com/drive/1yyEe0m8t5b3i9FQkCl_iy6c9maF2brGx?usp=sharing (by txmutt), alternatively with float32 here
Or https://huggingface.co/spaces/akhaliq/demucs
Or https://mvsep.com/
Pick up from the list Demucs Model B there.
You can export result files in MP3 320kbps, WAV and FLAC. File limit is 100MB and has a 10 minute audio length limit.
To use Demucs 3 locally: https://discord.com/channels/708579735583588363/777727772008251433/909145349426384917
Currently, all the code uses now main branch which is Demucs 4 (previously HT) but these Colabs use old mdx_extra model.
Demucs 3 UVR models 2 stem only available on MVSEP.com or in UVR5 GUI (nice results in cases when you suffer vocal bleeding i regular UVR5, GSEP, MDX 9.7 - model 1 less aggressive, model 2 more destructive, model bag has more bleeding of all three).
In Colab, judging by quality of drums track, I prefer using overlap 0.1 (only for instrumentals), but default set by the author is 0.25 and is better for sound of instrumental as a whole.
But it still provides decent results with instrumentals.
Also, HV had overall better separation quality results using shifts=10, but it increases separation time (it's also reflected by MVSEP's SDR calculations). Later we found out it can be further increased to 20.
Also, I have a report that you may get better results in Demucs using previously separated instrumental from e.g. UVR.
Anjok’s tip for better instrumentals: “I recommend removing the drums with the Demucs, then removing the vocals and then mixing the drums back in”. Yields much better results than simple ensemble.
It works the best in cases when drums get muffled after isolation, e.g. in hip-hop. You need to ensure that tracks are aligned correctly. E.g. if you isolate drumless UVR track, isolate also regular track to align drumless UVR track easier with drums track from Demucs, otherwise there will be hard to find the same peaks. Then simply align drumless UVR the same as regular track is aligned and mute/delete UVR regular (instrumental) track.
Be aware! This is not a universal solution for the best isolation in every case. E.g. in tracks with busy mix like Eminem - Almost Famous, the guitar in the background can get impaired, and so even drums (UVR tends to impair guitars in general, but on drumless track it was even more prevalent - in that case normal UVR separation did better job).
Also, if you slow down the input file, it may allow you to separate more elements in the “other” stem.
It works either when you need an improvement in such instruments like snaps, human claps, etc.
Normally, the instrumental sounds choppy when you revert it to normal speed. The trick is - "do it in Audacity by changing sample rate of a track, and track only (track menu > rate), it won't resample, so there won't be any loss of quality, just remember to calculate your numbers
44100 > 33075 > 58800
48000 > 36000 >  64000
(both would result in x 0.75 speed)
etc.".
Also, there's dithering enabled in Audacity by default. Might be worth disabling it in some cases. Maybe not, but still, worth trying out. There should be less noise.
BTW. If you have some remains of drums in acapella using UVR or MDX, simply use Demucs, and invert drums track.
“The output will be a wave file encoded as int16. You can save as float32 wav files with --float32, or 24 bits integer wav with --int24” it doesn’t seem to work in Colab.
Demucs 4 (+ Colab) (4, 6 stem)
4 stem, SDR 9 for vocals on MUSDB HQ test, and SDR 9 for mixdowned instrumentals (5, 6 stem - experimental piano [bad] and guitar)
https://github.com/facebookresearch/demucs (all these models available in UVR 5 GUI or MVSEP [just x-minus doesn’t have ft model for at least free users, it was mmi model at some point, but then got replace by MDX-B which “ turned out to be not only higher quality, but also faster”])
Google Colab (all 4-6 stem models available, 16-32 bit output)
https://colab.research.google.com/drive/117SWWC0k9N2MBj7biagHjkRZpmd_ozu1
or Colab with upload script without Google Drive necessity:
https://colab.research.google.com/drive/1dC9nVxk3V_VPjUADsnFu8EiT-xnU1tGH?usp=sharing
or Colab by Bezio with batch processing, (only mp3 output and no overlap/shifts parameters beside model choice - choose demucs_ft for 4 stems):
https://colab.research.google.com/drive/15IscSKj8u6OrooR-B5GHxIvKE5YXyG_5?usp=sharing
or Colab with batch processing by jarredou (less friendly GUI, but should be usable too, lossless):
https://colab.research.google.com/drive/1KTkiBI21-07JTYcTdhlj_muSh_p7dP1d?usp=sharing
"I'd recommend using the “htdemucs_ft” model over normal “htdemucs” since IMHO it's a bit better", also SDR measurements confirm that. 6s might have more vocal residues than both, but will be a good choice in some cases (possibly songs with guitar).
All the best stock models:

---

### Stats: SDR=10.02 SDR=9 

- “Overlap is the percentage of the audio chunk that will be overlapped by the next audio chunk. So it's basically merging and averaging different audio chunk that have different start (& end) points.
For example, if audio chunk is `|---|` with overlap=0.5, each audio chunk will be half overlapped by next audio chunk:
```
|---|
|---|
|---| etc...
|---| (2nd audio chunk half overlapping previous one)
|---| (1st audio chunk)
```
-shifts is a random value between 0 and 0.5 seconds that will be used to pad the full audio track, changing its start(&end) point. When all "shifts" are processed, they are merged and average. (...)
It's to pad the full song with a silent of a random length between 0 and 0.5 sec. Each shift add a pass with a different random length of silence added before the song. When all shifts are done (and silences removed), the results are merged and averaged.
Shifts is performing lower than overlap because it is limited to that 0.5 seconds max value of shifting, when overlap is shifting progressively across the whole song. Both works because they are shifting the starting point of the separations. (Don't ask me why that works!)
But overlap with high values is kinda biased towards the end of the audio, it's caricatural here but first (chunk - overlap) will be 1 pass, 2nd (chunk - overlap) will be 2 passes, 3rd (chunk - overlap) will be 3 passes, etc…”
So Overlap has more impact on the results than shift.
“Side-note: Demucs overlap and MVSEP-MDX23 by ZFTurbo overlap features are not working in the same way. (...)
Demucs is kinda crossfading the chunks in their overlapping regions, while MVSep-MDX23 is doing avg/avg to mix them together”
Why is overlapping advantageous?
Because changing the starting point of the separation give slightly different results (I can't explain why!). The more you move the starting point, the more different the results are. That's why overlap performs better than shifts limited to 0-0.5sec range, like I said before.
Overlap in Demucs (and now UVR) is also crossfading overlapping chunks, that is probably also reducing the artifacts at audio chunks/segments boundaries.
[So technically, if you could load the entire track in at once, you wouldn't need overlap]
Shifts=10 vs 2 gives +0.2 SDR with overlap=0.25 (the setting they've used in their original paper), if you use higher value for overlap, the gain will be lower, as they both rely on the same "trick" to work.
Shifts=X can give little extra SDR as it's doing multiple passes, but will not degrade "baseline" quality (even with shifts=0)
Lower than recommanded values for segment will degrade "baseline" quality.
So in theory, you can equally set shifts to 0 and max out overlap.
Segments optimum (in UVR beta/new) is 256.
Gsep (2, 4, 5, 6 stem, karaoke)
https://studio.gaudiolab.io/
Paid (20 minutes free in mp3 - no credit card required)
7$/60 minutes
16$/240 minutes
50$/1200 minutes
Electric guitar (occasionally bad), good piano, output: mp3 320kbps (20kHz cutoff), wav only for paid users, accepted input: wav 16-32, flac 16, mp3, m4a, mp4, don’t upload files over 100MB (and also 11 minutes may fail on some devices with Chrome "aw snap" error), capable of isolating crowd in some cases, and sound effects. Ideally, upload 44kHz files with min. 320kbps bitrate to have always maximum mp3 320kbps output for free.
2025 metrics for 2 stems
https://mvsep.com/quality_checker/entry/9095
(outdated) About its SDR
10.02 SDR for vocal model (vs Byte Dance 8.079) on seemingly MDX21 chart, but non-SDR rated newer model(s) were available from 09.06.22, and later by the end of July, and now new model is released since 6 September (there were 4 or 5 different vocal/instrumental models in total so far, the last introduced somewhere in September and no models update was performed with later UI update). MVSEP SDR comparison chart on their dataset, shows it's currently around SDR 9 for both instrumental and vocals, but I think evaluation done on demixing challenge (first model) was more precise. Be aware that GSEP causes issue of cancelling different sounds which cannot be found in any stem.
Since May 2024 update there was an average of 0.13 SDR increase for mp3 output and first 19 songs from multisong dataset evaluation, but judging by no audible difference for most people, they could simply change some parameters of inference. Actually, it’s more muddy now, but in some songs there are a bit less of vocal residues, and in other songs, noticeably more. Inverting the mixture with vocals in WAV will muffle the sound in overall, e.g. snares, esp. in places of these residues, but the residues will disappear as well.
Uncheck vocals to download WAV file if WAV download doesn't work,
and uncheck instrumental to download vocals in WAV -
don't check all stems if you can't download WAV at all and download window simply disappears.
If you still can’t download your WAV files, go to Chrome DevTools>Network before starting downloading, and press CTR+R, now start download. Now both stems should be shown in DevTools>Network, starting with input file name, e.g. instrumental with ending name “result_accom.wav” (usually marked as “fail” in State column and xhr as type), click the entry with right mouse button and choose Open in new tab.
The download may fail frequently, forcing you to resume the download multiple times in browser manually, or wait a bit on the attempt to download the file at the start.
Free option of separating has been removed since the May 2024 update. There's only a 20-minute free trial with mp3 output.
Vocals and all other stems (including instrumentals/others) are paid, and length for each stem is taken from your account separately for each model.
No credit is not required for the trial.
For free, only mp3 output and 10 minutes input limit.
For paid users there's a 20 minutes limit, and mp3/wav output, plus paid users have faster queue, shareable links, and long term results storage.
Seems like there weren't any changes in the model
https://www.youtube.com/watch?v=OGWaoBOkiMg
The old files from previous separations on your account didn't get deleted so far if you have premium.
https://studio.gaudiolab.io/pricing
There was also added a new option for vocals called “Vocal remover” - good "for conservative vocals, it's fine it even has 15 best scoring on SDR." and 10.85 in vocals on multisong dataset.
Instruction
Log in, and re-enter into the link above if you feel lost on the landing page.
For instrumental with vocals, simply uncheck drums, choose vocal, and two stems will be available for download.
As for using 4/5 stem option for instrumental after mixing if you save the tracks mixed in 24 bit in DAW like Audacity, it currently produces less voice leftovers, but the instrumental have worse quality and spectrum probably due to noise cancellation (which is a possible cause of missing sounds in other stem). Use 5 stem, but cut silence in places when there is no guitar in the stem to get comparable quality to 4 stem in such places.
For 3-6 stem, you better don’t use dedicated stems mixing option - yes, it respects muting stems to get instrumental as well, but the output is always mp3 128kbps while you can perform mixdown from mp3s to even lossless 64 bit in free DAWs like Audacity or Cakewalk.
In some very specific cases you can get a bit better results for some songs by converting your input FLAC/WAV 16 to WAV 32 in e.g. Foobar2000.
Troubleshooting

---

### Stats: sdr=0.33 bleedless=9.29 fullness=16.58 

- Hit 'n' Mix RipX DAW Pro 7 released. For GPU acceleration, min. requirement is 8GB VRAM and 10XX card or newer (mentioned by the official document are: 1070, 1080, 2070, 2080, 3070, 3080, 3090, 40XX). Additionally, for GPU acceleration to work, exactly Nvidia CUDA Toolkit v.11.0 is necessary. Occasionally during transition from some older versions, separation quality of harmonies can increase. Separation time with GPU acceleration can decrease from even 40 minutes on CPU to 2 minutes on decent GPU.
They say it uses Demucs.
We have reports about crashes, at least on certain audio files. There are various RipX versions uploaded on archive.org, maybe one will work, but some keys work only on versions from 2 and up.
Spectralayers 10
Received an update of an AI, and they no longer use Spleeter, but Demucs 4 (6s), and they now also good kick, snare, cymbals separation too. Good opinions so far. Compared to drumsep sometimes it's better, sometimes it's not. Versus MDX23 Colab V2, instrumentals sometimes sound much worse, so rather don’t bother for instrumentals.
USS-Bytedance (any; esp. SFX)
https://github.com/bytedance/uss
(COMMAND: "conda install -c intel icc_rt" SOLVES the LLVM ERROR)
You provide e.g. a sample of any instrument or SFX, and the AI separates it solo from a song or movie fragment you choose to separate.
It works in mono. You need to process right and left channel separately.
Update 29.04.25 (Python No such file or directory fix; thx epiphery)
https://colab.research.google.com/drive/1rfl0YJt7cwxdT_pQlgobJNuX3fANyYmx?usp=sharing
(old) ByteDance USS with Colab by jazzpear94
https://colab.research.google.com/drive/1lRjlsqeBhO9B3dvW4jSWanjFLd6tuEO9?usp=share_link
(old) Probably mirror (fixed March 2024):
https://colab.research.google.com/drive/1f2qUITs5RR6Fr3MKfQeYaaj9ciTz93B2
errors out with:
“sed: can't read /usr/local/lib/python3.10/dist-packages/uss/config.py: No such file or directory”) (2025)
It works (much) better than zero-shot (not only “user-friendly wise”).
Better results, and It divides them into many categories.
Great for isolating SFX', worse for vocals than current vocal models. Even providing acapella didn't give better results than current instrumental models. It just serves well for other purposes.
"Queries [so exemplary samples] for ByteDance USS taken from the DNR dataset. Just download and put these on your drive to use them in the Colab as queries [as similarly sounding sounds from your songs to separate]."
https://www.dropbox.com/sh/fel3hunq4eb83rs/AAA1WoK3d85W4S4N5HObxhQGa?dl=0
Also, grab some crowd samples from here:
https://youtu.be/-FLgShtdxQ8
https://youtu.be/IKB3Qiglyro
https://youtu.be/Hheg88LKVDs
Q&A by Bas Curtis and jazzpear
Q: What is the difference between running with and without the usage of reference query audio?
A: Query audio lets you input audio for it to reference and extract similar songs based upon (like zeroshot but way better) whereas without a query auto splits many stems of all kinds without needing to feed it a query.
Q: Let's say there is this annoying flute you wanna get rid off...
and keep the vocals only....
You feed a snippet of the flute as reference, so it tries to ditch it from the input?
A: Quite the reverse. It extracts the flute only which ig you could use to invert and get rid of it
Zero Shot (any sample; esp. instruments)
https://github.com/RetroCirce/Zero_Shot_Audio_Source_Separation
(as USS Bytedance came out now, zero shot can be regarded as obsolete now, although zero-shot might is rather better for single instruments than for SFX)
You provide e.g. sample of any trumpet or any other instrument, and AI returns it from a song you choose to separate.
Guide and troubleshooting for local installation (get Discord invitation in footer first if necessary).
Google Colab troubleshooting and notebook (though it may not work at times when GDrive link resources are out of download limit, also it returns some torch issues after Colab updates in 2023).
Check out also this Colab alternative:
https://replicate.com/retrocirce/zero_shot_audio_source_separation
It's faster (mono input required).
Also available on https://mvsep.com/ in a form of 4 stems without custom queries, and it’s not better than Demucs in this form.
"Zero shot isn't meant to be used as a general model, that's why it accelerates on a specific class of sounds with some limitations in mind.... It mostly works the best when samples match the original input mixture, of course there are limitations"
"You don’t have to train any fancy models to get decent results [...] And it’s good at not destroying music". But it usually lefts some vocal bleeding, so process the result using MDX to get rid of these low volume vocals. Zero-shot is also capable of removing crowd from recordings pretty well.
As for drums separation, like for snares, it’s not so good as drumsep/FactorSynth/RipX, and it has cutoff.
"I did zero shot tests a week or two ago, and it was killing it, pulling harmonica down to -40dB, synth lines gone, guitars, anything. And the input sources were literally a few seconds of audio.
I've been pulling out whole synths and whistles and all sorts.
Knocks the wind model into the wind, zero shot with the right sample to form the model backbone works really well
The key is to give it about 10 seconds of a sample with a lot of variation, full scales, that kinda thing"
Dango.ai
Custom stem separation feature (paid, 10 seconds for free)
Expensive
Special method of separation by viperx (ACERVO DOS PLAYBACK) edited by CyberWaifu
Process music with Demucs to get drums and bass.
Process music with MDX to get vocals.
Separate left and right channels of vocals.
Process vocal channels through Zero-Shot with a noise sample from that channel.
Phase invert Zero-Shot's output to the channel to remove the noise.
Join the channels back together to get processed vocals.
Invert the processed vocals to music to get the instrumental.
Separate left and right channels of instrumental.
Process instrumental channels through Zero-Shot with a noise sample from that channel.
Phase invert Zero-Shot's output to the channel to remove the noise.
Join the channels back together to get processed instrumental.
Process instrumental with Demucs to get other.
Combine other with drums and bass to get better instrumental.
So it sounds like Zero-Shot is being used for noise removal.
As for how Zero-Shot and the noise sample works…
AudioSep
“I decided to try AudioSep: https://github.com/Audio-AGI/AudioSep on MultiSong Dataset.
I used prompt 'vocals'. I was sure it would be bad, but I didn't think it's so bad.
https://mvsep.com/quality_checker/entry/8408
I also tried it on the Guitar dataset - it's even worse - negative SDR. Maybe I'm doing something wrong. But I tried the example with cat from the demo page, and it worked the same as in there. So I think I have no errors.”
sdr: 0.33
si_sdr: -2.39
l1_freq: 17.62
log_wmse: 6.72
aura_stft: 3.66
aura_mrstft: 5.55
bleedless: 9.29
fullness: 16.58
Colab on GH probably gives unpickiling issue. You might be able to fix it be executing:
!pip install torch==2.5.0
After you execute all the installation-related cells.
Since then, probably something more about dependencies is also needed, like it ‘s coded now in the inference Colab.
Medley Vox (different vocalists)
Use already separated vocals as input (e.g. by Roformers, vox_ft or MDX23C fullband a.k.a. 1648 in UVR or 1666 on MVSEP).
Local installation video tutorial by Bas Curtiz:
https://youtu.be/VbM4qp0VP8
(NVIDIA GPU acceleration supported, or perhaps CPU - might be slow)
Cyrus version of MedleyVox Colab with chunking introduced, so you don't need to do chunking manually:
https://colab.research.google.com/drive/10x8mkZmpqiu-oKAd8oBv_GSnZNKfa8r2?usp=sharing (07.02.25 fork with fairseq fix and GDrive integration)
Currently, we have a duet/unison model 238 (default in Colab),
and main/rest 138 to uncomment in Colab.
Recommended model is located in vocals 238 folder (non ISR-net one).
While:
“The ISR_net is basically just a different type of model that attempts to make audio super resolution and then separate it. I only trained it because that's what the paper's author did, but it gives worse results than just the normal fine-tuned.”

MedleyVox is also available on MVSEP, but it has more bleeding and “doesn't work as well as the Colab iteration with duets”. (Isling/Ryanz)
The "duet/unison model 238" will be used by default.
``and main/rest 138 to uncomment in Colab`` if you need it.
Then go to the first cell again. To "uncomment" means to delete the "#" from the beginning of the line before the "!wget" so the line will be used to download the model files.
Do it for both pth and json lines
(you might be asked whether to replace existing pth and json files by the alternative model you just downloaded in the place of the previous one)
``Recommended model is located in vocals 238 folder (non ISR-net one).``
That's the model used in the Colab by default. You can ignore that information. It's for users using the MV on their own machine.
The output for 238 model is 24kHz sample rate (so 12kHz model in Spek).
You might want to upscale the results using e.g. AudioSR or maybe even Lew’s vocal enhancer location further below the linked section.
The output is mono.
You might want to create a "fake stereo" as input by copying the same channel over the two, then do the same with another channel, and then create the stereo result from both channels processed separately in dual mono with MV.
The AI will create a downmix from both input channels instead of processing channels separately.
Be aware that “dual mono processing with AI can often create incoherencies in stereo image (like the voice will be recognized in some part only in left channel and not the other, as they are processed independently)” jarredou
"The demos sound quite good (separating different voices, including harmonies or background [backing] vocals)"
It's for already separated or original acapellas.
The model is trained by Cyrus. The problem is, it was trained with 12kHz cutoff… “audiosr does almost perfect job [with upscaling it] already, but the hugging page doesn’t work with full songs, it runs out of memory pretty fast”.
It was possible at some point that later stages of the training, looking like over fitting were responsible for higher frequency output.
It’s sometimes already better than BVE models, and the model has already similar to demo results on their site.
Sadly, the training code is extremely messy and broken, but a fork by Cyrus with instructions is planned, with releasing datasets including the one behind geo-lock. Datasets are huge and heavy.
Original repo (Vinctekan fixed it - the video at the top contains it)
https://github.com/jeonchangbin49/medleyvox

---

### Stats: SDR=1 

- LANDR Stems (cheaper, also uses Audioshake; plugin, probably doesn’t work locally, free access won’t give you access to stems; “LANDR Stems is only included in Studio Standard + Studio Pro” it’s not included in trial; SDR: 1 | 2)

---

### Stats: SDR=9.4 

(outdated)
(for old MDX SDR 9.4 UVR model):
(input audio file on Colab can be max 44kHz and FLAC only).
Original MDX model B was updated and to get the best instrumental - you need to download invert instrumental from Colab.
Model A is 4 stem, so for instrumental, mix it, e.g. in Audacity without vocals stem (import all 3 tracks underneath and render). Isolation might take up to ~1 hour in Colab, but recently it takes below 20 minutes on 3.00 min+ track.
If you want to use it locally (no auto inversion):
https://discord.com/channels/708579735583588363/887455924845944873/887464098844016650
B 9.4 model:
https://github.com/Anjok07/ultimatevocalremovergui/releases/tag/MDX-Net
Or remotely (by CyberWaifu):
https://colab.research.google.com/drive/1R32s9M50tn_TRUGIkfnjNPYdbUvQOcfh?usp=sharing
Site version (currently includes 9.6 SDR model):
https://mvsep.com/
You can choose between MDX A or B, Spleeter 2/4/5 Stems), UnMix 2/4 stems, but output is mp3 only)
New MDX model released by UVR team on mvsep is currently also available. If you have any problems with separating in mobile browser (file type not supported) add for a file additional extension: trackname.flac.flac.
MDX is really worth checking. Even if you have some bleeding, and UVR model cuts some instruments in the background.
CyberWaifu Colab troubleshooting
If you have a problem with noise after a few seconds of the result files, try to use FLAC. After an unsuccessful attempt of isolation, you can try restoring default runtime to default state in options. The bug happened a few days after releasing the Colab suddenly one day and the is prevailing to this day (so WAV no longer works). If you run the first cell to upload, and afterwards after opening file view, one of the 001-00X wav files is distorted (000 after few second) it means it failed, and you need to start over till you get all the files played correctly. But after longer isolation, it may cause reaching GPU limit, and you will not be able to connect with GPU. To fix it, switch to another Google account. If you have a problem, that your stems are too long, and mixed with a different song, restore default runtime settings as well, or delete manually

---

### Stats: SDR=20 

- I always stop when [loss, avg_loss=] nan
Q: it went back to SDR 20 again
A: “that progress on valid updates per track, so some tracks are 20 SDR some will be like 2 SDR” (frazer)
Q: yea, some are still at 11 etc
A: “train until either avg loss nan or fixed sdr (example: 14 for me) with 5.0e-05
use best checkpoint from that run with 1.0e-05 to get some boost
idk why it works (and if it works im at step 1 rn xD)” - mesk
Q: is it normal that when i use a checkpoint with 12.53 sdr, then restart training, the results at epoch 0 and 1 drop back to 11.77?
A: yes
only if u restart with a higher LR
so i had a checkpoint that was 12SDR trained with 5e-5, then if i trained that with 1e-5, youd expect it to start at like 11.Xsdr
if i had a checkpoint 12SDR trained 1e-5, and i train with 5e-5, id expect it to either start at 12, then drop, then start to increase
keep it going 5e-5 for literally as long as u can
this is called overfitting - just make sure it doesnt do this
its the point where the model begins to not generalize but memorize the training set - happens on finetuning if u train too long
what u do is just keep the redline score if u still have it (pic) - frazer
Preparing dataset
Let’s get started.
First, check the -
Repository of stems - section of this document.
There you will find out that most stems are not equal in terms of loudness to contemporary standards, and clip when mixed together.
About sidechain stem limiting guide by Vinctekan
The sidechain limiting method might be not so beneficial for SDR as we thought initially, irc it’s explained in the interesting links section with the given paper.
Other useful links:
https://arxiv.org/pdf/2110.09958.pdf
https://github.com/darius522/dnr-utils/blob/main/config.py
“You can also just utilize this https://github.com/darius522/dnr-utils/blob/main/audio_utils.py
and make a script suited to your own, the one already on this repo is a bit difficult to repurpose.
I just concatenated a lot of sfx music and speech together into 1hr chunks and used audacity tho (set LUFS and mix)
oh and then further split into 60 second chunks after mixing them” - jowoon
“Aligned dataset is not a requirement to get performing models, so you can create a dataset with FL/Ableton with random beats for each stem. Or using loops (while they contain only 1 type of sound).
You create some tracks with only kick, some others with only snare, other with only...etc...
And you have your training dataset to use with random mixing dataloader (dataset type 2 in ZFTurbo script, one folder with all kick tracks, one folder with all snare tracks, one folder with… etc.
Then you have to create a validation dataset accordingly to the type of stems used in training, preferably with a kind of music close to the kind you want to separate, or "widespread", with a more general representation of current music, but this mean it has to be way larger.
The only requirements are:
44.1Hz stereo audio.
Lossless (wav/flac)
Only 1 type of sound by file (and no bleed like it would happen with real drums)
Audio length longer than 30s (current algos use mostly ~6/12 second chunks, but better to have some margin and longer tracks so they can be used in future when longer chunks can be handled by archs & hardware).” jarredou
“You can use flac too; saves space (though make them 44.1 / 16-bit / stereo, even if u use mp3's or whatever other format - convert upfront)
validation set however needs to remain in wav with mixture included.” Bas Curtiz
“A quite unknown Reaper script to randomize any automatable parameters on any VST/JS/ReaXXX plugin with MIDI notes. It's REALLY a must-have for dataset creation, adding sound diversity without hassle.
https://forum.cockos.com/showthread.php?t=234194” jarredou
(Guides for stem limiting moved to the end of the section for archival purposes - rather outdated approaches due to the statements in the paper above)
FAQ
You shouldn't compare training data against evaluation data, while those being the same.
You can use multisong dataset from MVSEP, and make sure you don't have any of those songs in your dataset.
Q: Does evaluation data matter for the final quality of the model?
A: Absolutely not. It's merely indication
SDR measurement is logarithmic, meaning that 1 SDR is 10x difference.
Q: Why I have negative SDR values (based on HTDemucs)
A: Make sure there are no empty stems in any training dataset and or validation dataset
Below is just a theory for now and probably wasn't strictly tested on any model yet, but seems promising
Q: Can you not calculate the average dB of the stems and fit one limiting value to them all?
A: the stems are divide-maxed prior
meaning they are made so, that when joined together, they won't clip
but are normalized
so they will be kinda standardized already
based on that, I should be able to just go with one static value for all
Example
https://www.youtube.com/watch?v=JYwslDs-t4k
Q: This is great, I actually used this method before with a few sets of stems, before I decided to try sidechain compression/ Voxengo elephant method, but I'm not too sure if I am on the right path. However, I'm pretty sure this only works best for evaluation, if the resulting mixture has consistent loudness like in today's music.
A: Yeah, it's a different approach than compression/voxengo indeed.
But the fact it scored high in SDR and UVR dataset is already compressed/elphanted
I think it's a good combo to use both in the set, a bit like new style tracks and oldies [so to use both approaches inside the dataset]
some tracks in real life are compressed like fuck - some aren't
so it mimics real life situation
Q: if it's true that's awesome, with that the model basically has the potential to work in multiple mixing styles, without having to create new data, or changing it, right?
While still adding new data
A: Yeah, since UVR dataset is already compressed - and then add these one of mines with the more delicate way of mastering (incl. divdemax prior)
Q: Does somebody know the best way to make dataset smaller? I have very huge dataset in flac format, so the one idea is to truncate part in the song where is only music without vocals? Also, I can convert it to opus format, does it worse it? Or maybe there is something better that I don't know?
A (jarredou): If you plan to use random mixing of stems during training (so non-aligned dataset), then you can remove all silent parts from stems pre-training, on instrumental it will not change a lot but for vocals it can save a lot of space (h/t Bas Curtiz for the idea)
Q: Currently dataset is aligned, but does this random mixing is standard approach? I am going to train official SCNet model, so maybe it will require modifications for this?
A: https://arxiv.org/abs/2402.18407 (Why does music source separation benefit from cacophony?)
https://www.merl.com/publications/docs/TR2024-030.pdf (same non-columns formatting)
“It thus appears that a small amount of random mixes composed of stems from a larger set of songs performs better than a large amount of random mixes composed of stems from a smaller set of songs.”
If needed, the training script that ZFTurbo has made does handle random and aligned dataset and has also SCNet implementation: https://github.com/ZFTurbo/Music-Source-Separation-Training
Q: As I know, SCNet supports only for inference here.
A: It does training too, ZFTurbo has recently trained a SCNet-large model on MUSDB18 dataset
Dataset types doc https://github.com/ZFTurbo/Music-Source-Separation-Training/blob/main/docs/dataset_types.md
(he only didn't update help string)
Creating dataset - guide by Bas Curtiz
(now also video available)
How to: Create a dataset for training

---

### Stats: SDR=9.1 SDR=10.2 

Why does training the bs_roformer model with 8s chunksize, 256 dim, 8 depth consume only 13GB of VRAM now, compared to 21GB last time [they could decrease VRAM since then]
Stuck troubleshooting of TPU training by Bas Curtiz (Q) and frazer, DJ NUO, jarredou and Cyclcrclicly (A):
Q: is it as simple as adding pytorch lightning though?
A: try using "xla" as the device instead of cuda and if you're lucky everything will Just Work™️
Q: The Pytorch's implementation of stft does not work on the XLA (TPU),
because it internally uses some unsupported functions.
There are not feasible workarounds for it available.
Only some 3x PhD discussion, which discusses the underlying function not working,
which would require forking pytorch to get it working, IF the solution was actually even feasible:
(hacky super slow workaround, or just "use different shit").
Only "realistic" solution I've found is porting the mel band roformer to tensorflow.
Which is bruh, but the thing is in their docs STFT says:
Implemented with TPU/GPU-compatible ops and supports gradients..
Also tensorflow is by google, the TPU as well, so yk, it might have better support.
The same error basically is described here:
https://github.com/pytorch/xla/issues/2241
A: As frazer said, you'll have better luck with jax than tensorflow
A: Can you try putting data to CPU & running it there, and then put the result back on TPU?
I encounter similar issues when running on Mac MPS (GPU), and this code helps to alleviate the issue:
stft_repr = torch.stft(raw_audio.cpu() if x_is_mps else raw_audio, **self.stft_kwargs, window=stft_window.cpu() if x_is_mps else stft_window, return_complex=True).to(device)
(of course, in your case the code might be a bit different, but it demonstrates the idea)
Q: obviously slow
it is called in a loop in the forward function (= very slow)
...if it was like only once / before each step, but not inside step.
we'll try anyways, thanks
Timed-out after 10 mins, 0 steps were finished.
Imagine doing 4032 steps.
JAX is like an optimizer/JIT.
STFT of it, is just Scipy's STFT but running under JAX.
Scipy's implementation is CPU-based.
So it expects CPU data. Not Tensor/GPU/TPU data.
A: Or this might help (custom implementation of STFT): https://github.com/MasayaKawamura/MB-iSTFT-VITS/blob/main/stft.py
A: There's also https://github.com/qiuqiangkong/torchlibrosa/ that has a stft implementation
Q: Hmm both use numpy which is cpu based
A: yeah its some weird operation in the torch spec, i use https://github.com/adobe-research/convmelspec anytime incompatibility occurs
Q: May be we need to replace mel spec with this in MelRoformer.
I got a boilerplate/minimal produduction ready, but 2  things...
no TPU for me right now to test - maybe someone else has better luck / paid Colab sub.
Last outcome, which might be fixed by now: RuntimeError: Attempted to call variable.set_data(tensor), but variable and tensor have incompatible tensor type.
A: you can use kaggle for tpuv3 with probably better availability
Q: https://github.com/qiuqiangkong/torchlibrosa/ result:
Calling "torch.nn.functional.fold" just gets stuck, when interrupting, the error stack has mentions of copying to CPU.
...smth to do with the fold function.
Numpy only in initialization (cpu), so that's fine.
https://github.com/MasayaKawamura/MB-iSTFT-VITS/blob/main/stft.py result:
Numpy and basically cpu all-the-way, so no/go.
https://github.com/adobe-research/convmelspec result:
Not a STFT library / whole spectrogram, don't wanna dissect it, the STFT part seems internal,
didn't notice (would have to double-check) the inverse, but wasted 2 days already. done with it.
A: Just a guess (have no experience with Tensorflow): what if STFT portion of the code can be executed by TensorFlow code -> convert result to numpy CPU -> convert to PyTorch tensor
Q: Problem is... It simply takes too much time, copying to cpu and back is expensive resource-wise
A: In some part of torchlibrosa they use a workaround for nn.functional.fold function, maybe that can be reproduced/adapted to the other failing part where fold used.
A: line 239 is the drop in, you have to make sure the settings are the same from what i remember https://github.com/adobe-research/convmelspec/blob/main/convmelspec/stft.py
Q: It got thru the whole ass forward step. But now it's stuck at backward step.
yk, recalculate the weights based on this step to improve the model.
replaced the backwards function of stft with empty one, and yet: stuck.
so since backwards step of stft/istft is disabled...
the problem is elsewhere.
No idea where, no idea how to debug, out of my expertise.
A: I might be 100% wrong here, but I think you should disable the backward pass through that class if it is type nn.Module
stft.requires_grad=False
or when you call stft use a decorator with indentation
with torch.no_grad():
x=stft(x)
Other archs
SCNet: Sparse Compression Network
Large models by ZFTurbo turned out to sound between Roformers and MDX.
“SCNet is maybe a bit more bleedy than MDX23c” and/or possibly noisy, judging by the MVSEP model(s). “seemingly impossible” to train guitars
[July 10th 2024] “Official SCNet repo has been updated by the author with training code. https://github.com/starrytong/SCNet”
“ZF's script already can train SCNet, but currently it doesn't give good results”
https://github.com/ZFTurbo/Music-Source-Separation-Training/releases/
The author’s checkpoint:
https://drive.google.com/file/d/1CdEIIqsoRfHn1SJ7rccPfyYioW3BlXcW/view
June 2025
ZFTurbo: “I added new version of SCNet with mask in main MSST repository. Available with key 'scnet_masked'. Thanks to becruily for help.”
“the main thing is removing the SCNet buzzing” - Dry Paint
How heavily undertrained weights looks on spectrograms with mask vs without: click
“One diff I see between author config and ZF's one, is that dev has used learning rate of 5e-04 while it's 4e-05 in ZF config. And main issue ZF was facing was slow progress (while author said it worked as expected using ZF training script https://github.com/starrytong/SCNet/issues/1#issuecomment-2063025663)”
The author:
“All our experiments are conducted on 8 Nvidia V100 GPUs.
When training solely on the MUSDB18-HQ dataset, the model is
trained for 130 epochs with the Adam [22] optimizer with an initial
learning rate of 5e-4 and batch size of 4 for each GPU. Nevertheless,
we adjust the learning rate to 3e-4 when introducing additional data
to mitigate potential gradient explosion.”
“Q: So that mean that you have to modulate the learning rate depending on the size of the dataset ?
I think it's first time I read something in that way
A: Yea, I suppose because the dataset is larger you need to ensure the model sees the whole distribution instead of just learning the first couple of batches”
Paper: https://arxiv.org/abs/2401.13276
https://cdn.discordapp.com/attachments/708579735583588366/1200415850277130250/image.png (dead)
On the same dataset (MUSDB18-HQ), it performs a lot better than Demucs 4 (Demucs HT).
“melband is still sota cause if you increase the feature dimensions and blocks it gets better
you can't scale up scnet cause it isn't a transformer
it's a good cheap alt version tho”
ZFTurbo “I trained small model because author post weights for small. Now I'm training large version of model, but it's slow and still not reach quality of small version.
I use the same dataset for both models
My SCNet large stuck at SDR 9.1 for vocals. I don't know why
My small SCNet has SDR 10.2
I added config of SCNet to train on MUSDB18:
https://github.com/ZFTurbo/Music-Source-Separation-Training/blob/main/configs/config_musdb18_scnet_large.yaml
Only changes comparing to small model are these parts:
Small:
dims:

---

### Stats: SDR=6 

- 0.403”
ZFTurbo eventually trained SCNet large model, but it turned out to sound similar to Roformers, but with more noise. You can test the model on MVSEP.com
SCNet Large turned out to be good for piano (vs MDX23C and MelRoformer) and also drums models according to ZFTurbo.
“He also said SCNet didn't work that well for strings, Aufr didn't have luck with BV model as well”
“MDX23c is already looking better on guitar after 5 epochs than scnet after 100 epochs”
“with SCNet I've had the fastest results with prodigy [optimizer]” becruily
Later, ZFTurbo released SCNet 4 stems (in his repo) and exclusive bass model on MVSEP.
There was also an older, an unofficial (not fully finished yet, it seems) implementation of SCNet: https://github.com/amanteur/SCNet-PyTorch
Experimental BS-Mamba
git clone https://github.com/mapperize/Music-Source-Separation-Training.git --branch workingmamba
TS-BSmamba2
https://arxiv.org/abs/2409.06245
https://github.com/baijinglin/TS-BSmamba2
Added to ZFTurbo training repo:
https://github.com/ZFTurbo/Music-Source-Separation-Training/
At this moment, training works only on Linux or WSL.
SDR seems to be higher than all the current archs, maybe besides Mel/BS Roformers (weren’t tested). “It's in between SCNet and Rofos but maybe more lightweight than them.
(...) From the scores from MelBand paper [it seems] the Rofos are still like +0.5 SDR average above the other archs when trained on musdb18 only.
But it's great to finally see some mamba-inspired MSS arch with great performance”.
As for 22.09.24, ZFTurbo had problems with low SDR during training.
https://discord.com/channels/708579735583588363/1220364005034561628/1286650425596186645
https://discord.com/channels/708579735583588363/1220364005034561628/1284221988294099102
Another three very promising archs for the moment:
Conformer
“performs just as well if not better than a standard Roformer”
https://arxiv.org/pdf/2005.08100
https://github.com/lucidrains/conformer
(people already train with it, and its implementation might be pushed to the MSST repo in not distant future)
https://github.com/ZFTurbo/Music-Source-Separation-Training/issues/169
Essid pretrain:
https://huggingface.co/Essid/MelBandConformer/tree/main
https://mvsep.com/quality_checker/entry/9087
“Due to cost issues, I'm discontinuing the Mel-Band-Conformer MUSDB18HQ-based train. I'm sharing the ckpt and config, so anyone who wants to continue can use them.”
It has shown steady improvement in training in the last 12 hours from epoch 0 to 83 (1 SDR increase on private validation dataset, probably on A100XL (80GB) on “thunder compute” $1.05/hour) and the shared weight is epoch 200+.
TF-Locoformer
https://arxiv.org/abs/2408.03440
https://github.com/merlresearch/tf-locoformer/blob/main/espnet2/enh/separator/tflocoformer_separator.py
“I see only now that the tf-locoformer repo was updated to include the variants published few months ago (TF-Locoformer-NoPE and BS-Locoformer)
https://github.com/merlresearch/tf-locoformer” - jarredou
dTTnet
https://github.com/junyuchen-cjy/DTTNet-Pytorch
“They report very good performance on vocals with low parameters” - Kim
Back in the end of 2023, one indie pop song from multisong dataset (of the two there) received the best SDR - Bas Curtiz
“better than scnet imo, remains to see if it can beat rofos”
“not fast to train. I'm back with vanilla mdx23c
Trying a config to train model with less than 4GB  VRAM, almost at 7 SDR for vocals in 8 hours of training (on moisesdb+musdb18, and using musdb18 eval, with my 1080Ti and batch_size=1, chunk_size is around 1.5sec) ”
Modification of the training code for MSST by becruily (DL; old).
Breaks compatibility with the authors' checkpoint.
“Also keep in mind authors trained with l1 loss only, default in msst is masked loss”
“from what I read, l1 loss when dataset is noisy, mse loss when dataset is clean”
“the loss is defined from msst, but in the original dttnet it was in the code itself
you can just --loss l1_loss”
@jarredou “I copied your tfc and tfc_tdf classes to my files (and used that latest stft/istft I sent) - and seems to be better, just like the og dttnet
the tfc/tdf fixed the nan issue for me” - becruily
Installation instruction:
“In the latest MSST [at least for 13.10.25]
add the ddtnet folder to "models" and replace your settings file in utils with this”
“The weird thing is, it sounds like a fullness model despite not being one, I barely can find dips in instrumentals
ddtnet vs kim melband, if anyone is curious https://drive.google.com/drive/folders/12an8wnKC-FKE48gVu9pHvUaLSxzpC6C8?usp=sharing
Keep in mind ddtnet was trained only with musdb and has 10-20x less params while being comparable in quality”
“the authors checkpoints had 16khz cutoff because dim_f was smaller than nfft/2
if you want to train model with cutoff it's fine, if you want fullband then dim_f must be half of nfft + 1”
“I’ve found the issue in my DTTNet version leading to the "noisy" outputs. It was just the * changed to a +  in forward here” - jarredou
Everything uploaded at the top.
Mesk’s config for training instrumental model (achieved from SDR 6 to 9.3 in a third epoch [counting the first as 0]:
python train.py --model_type dttnet --config_path config-ddtnet-other.yaml --start_check_point results/vocalsg32_ep4082.ckpt --results_path results/ --data_path [YOUR DATASET] --valid_path [YOUR VALIDATION DATASET] --dataset_type [TYPE 1/2/3/4/5] --num_workers 8 --device_ids 0 --metric_for_scheduler sdr --metrics fullness bleedless l1_freq
change these accordingly:
>data_path
>valid_path
>dataset_type
On the side.
ZLUDA is a translation layer for CUDA allowing to use any CUDA-written app to be used with AMD (and formerly Intel) GPUs, and without any modifications to such app.
Weaker GPUs than 7900 XT might show its weeknesses considerably, compared to better GPUs. The example came from ZLUDA in Blender, but rather from AMD period code, so before the takedown and rollback to pre-AMD codebase so now ZLUDA is more crippled. 
With never released code, at certain point it was even made to support Batman Arkham Knight, with general plans to support DLSS, but it will probably never see a day light.
Maybe this repo still has the old base forked - version 3 codebase is still being updated there. Utilizing it on 6700 XT in stable-diffusion-webui-amdgpu, it was performing slowly like DirectML, but on 7900 XT it sped up the process from 3-4 to ~1 minute. The first execution can be slow due to need of creating cache. Then it can surpass ROCm performance-wise if you manage to make it work. Plus, ZLUDA works on Windows and supports older AMD GPUs, like even RX 500 series (use lshqqytiger’s repo, check e.g. ROCm 5 version if your app doesn’t start, but it might crash anyway), while for ROCm on Linux and older GPUs, e.g. RX 5700 XT should work with some quirks (e.g. HIP 5.7 and ROCm around 5.2.* - src, although you can try out 6.21 or 6.2.x to ensure, as it could happen that some earlier 6.x wasn’t supporting RX 5700 XT correctly, while e.g. for RX 6000 ROCm 6.24 should be used at the moment). 
It could be interesting to see utilizing training repo using ZLUDA e.g. on Windows instead of ROCm Pytorch on Linux but Unwa notice in the ZLUDA repo fork, “PyTorch: torch.stft does not always return correct result.” and it might be problematic during training, so ZLUDA might be not a good solution for training currently, but who knows whether for inferencing on e.g. Windows using MSST or UVR, although the latter crashes for me during separation with nvcuda.dll. But I haven't tried messing with HIP SDK mentioned in the release page or other fork's ZLUDA versions than the newest. I don't even have anything in C:\Program Files\AMD\ROCm (if it wasn’t even futile without it), but I have amdhip64.dll v. 5.5 in system32 (if 5.7 isn’t shipped with newer drivers and required).
Also, didn't follow these instructions yet, and they might be useful and contain some older GPUs workaround:
https://github.com/vladmandic/sdnext/wiki/ZLUDA
All gfx names with corresponding GPU models:
https://llvm.org/docs/AMDGPUUsage.html#processors
More ZLUDA research and workarounds (may work for UVR, not have too):
https://github.com/comfyanonymous/ComfyUI#amd-gpus-experimental-windows-and-linux-rdna-3-35-and-4-only
(RDNA 3-4 instructions for Python manual installation using DirectML branch of UVR)
https://github.com/comfyanonymous/ComfyUI#for-amd-cards-not-officially-supported-by-rocm
(flags for RDNA 2-3)
https://github.com/patientx/ComfyUI-Zluda
(Instructions for GCN4-RDNA4;
RDNA 2 with HIP 6.2.4 and experimental 6.4.2)
Using hip 5.7.1 and corresponding ZLUDA should be possible on RDNA2 too
https://github.com/CS1o/Stable-Diffusion-Info/wiki/Webui-Installation-Guides#amd-fooocus-with-zluda
(Step 5 in "Setting up Zluda" a bit below - for GPUs below RX 6800 or 9070/60,
and instructions above the point are there for 6800 or higher too
https://github.com/CS1o/Stable-Diffusion-Info/wiki/Webui-Installation-Guides#rocm-hip-sdk-57-with-zluda-setup
(Instructions for GCN4 [RX 400/500]; it contains a step with
pip install torch==2.2.1 torchvision==0.17.1 torchaudio==2.2.1 --index-url https://download.pytorch.org/whl/cu118)
https://github.com/advanced-lvl-up/Rx470-Vega10-Rx580-gfx803-gfx900-fix-AMD-GPU#important-notes-on-installation
(Instructions for GCN4 [GFX803 & GFX900])
https://github.com/ROCm/ROCm/issues/4749#issuecomment-3117083336
(some old 7900 XTX (gfx11100) troubleshooting

---

