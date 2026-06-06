# Plan d'implémentation — Quantification int4/int8 (quanto) pour tenir sur une RTX 3090 (24 Go)

> Objectif : faire tourner les pipelines LTX-2 de Milimo sur **une seule RTX 3090 (24 Go)**
> en quantifiant le transformer via `optimum-quanto`, **sans** porter le GGUF de ComfyUI.
> Tout le code nécessaire existe déjà dans le repo (`ltx-trainer/quantization.py`), il faut
> juste le **câbler dans le chemin d'inférence**.

---

## 0. Pourquoi quanto et pas GGUF

| | GGUF (ComfyUI) | quanto (déjà présent) |
|---|---|---|
| Noms de tenseurs | Calibrés pour le graphe ComfyUI | Correspondent au transformer natif `ltx_core` |
| Travail d'intégration | Lecteur GGML + ops déquant + **remapping de clés** | Appeler une fonction existante |
| `EXCLUDE_PATTERNS` | À deviner | **Déjà écrits** pour cette archi |

Empreinte cible : LTX-2 19B en **int4 ≈ ~9-10 Go** (≈ Q4 GGUF), **int8 ≈ ~19 Go**.
Le `MemoryManager` gère déjà l'exclusion LTX↔Flux → un seul gros modèle en VRAM à la fois.

---

## 1. Chaîne de chargement actuelle (rappel du tracé)

```
backend/model_engine.py : ModelManager.load_pipeline(type)
  └─> TI2VidTwoStagesPipeline(checkpoint_path, fp8transformer=True, device, …)
        └─> ModelLedger(fp8transformer=…)              # ltx-pipelines/utils/model_ledger.py
              └─> ledger.transformer()                  # POINT D'INJECTION
                    ├─ if fp8transformer: fp8_builder.build() → X0Model
                    └─ else: builder.build(dtype) → X0Model
                          └─ build() renvoie un LTXModel (a .transformer_blocks)
                          └─ X0Model.velocity_model = ce LTXModel
```

Le quantizer existant :
```
ltx-trainer/src/ltx_trainer/quantization.py
  quantize_model(model, "int4-quanto", device=…)
    └─ si model a .transformer_blocks → _quantize_blockwise (bloc→GPU→quantize→freeze→CPU)
    └─ EXCLUDE_PATTERNS déjà ciblés (patchify_proj, *adaln*, proj_out, caption_projection, *norm*, audio_*)
```

---

## 2. Modifications (4 fichiers) — Phase 1 (quantif à la volée)

### 2.1 Relocaliser le quantizer dans `ltx-core` (éviter la dépendance au trainer)

`ltx-pipelines` ne doit pas dépendre de `ltx-trainer` (et le README n'installe pas le trainer).
Copier le fichier :

```
cp LTX-2/packages/ltx-trainer/src/ltx_trainer/quantization.py \
   LTX-2/packages/ltx-core/src/ltx_core/quantization.py
```

Dans la copie, remplacer l'import du logger :
```python
# avant : from ltx_trainer import logger
import logging
logger = logging.getLogger(__name__)
```

### 2.2 `ltx-core` → ajouter `optimum-quanto` en dépendance

Dans `LTX-2/packages/ltx-core/pyproject.toml`, section dependencies :
```toml
"optimum-quanto>=0.2.4",
```

### 2.3 `model_ledger.py` — ajouter `quant_mode` et la branche de quantif

**(a)** Constructeur — nouveau paramètre (après `fp8transformer`) :
```python
        fp8transformer: bool = False,
        quant_mode: str | None = None,   # NEW: "int4-quanto" | "int8-quanto" | None
    ):
        ...
        self.fp8transformer = fp8transformer
        self.quant_mode = quant_mode      # NEW
        self.build_model_builders()
```

**(b)** `with_loras()` — propager :
```python
            fp8transformer=self.fp8transformer,
            quant_mode=self.quant_mode,    # NEW
        )
```

**(c)** `transformer()` — la branche quanto **prend la priorité** sur fp8 :
```python
    def transformer(self) -> X0Model:
        if not hasattr(self, "transformer_builder"):
            raise ValueError("Transformer not initialized. ...")

        # NEW — quantification quanto (priorité sur fp8)
        if self.quant_mode:
            from ltx_core.quantization import quantize_model
            # build en pleine précision sur CPU (les LoRA sont fusionnées ici)
            inner = self.transformer_builder.build(device="cpu", dtype=self.dtype)
            # quantif bloc-par-bloc : chaque bloc va sur GPU puis revient en CPU
            quantize_model(inner, self.quant_mode, device=self.device)
            return X0Model(inner).to(self.device).eval()

        if self.fp8transformer:
            fp8_builder = replace(
                self.transformer_builder,
                module_ops=(UPCAST_DURING_INFERENCE,),
                model_sd_ops=LTXV_MODEL_COMFY_RENAMING_WITH_TRANSFORMER_LINEAR_DOWNCAST_MAP,
            )
            return X0Model(fp8_builder.build(device=self._target_device())).to(self.device).eval()
        else:
            return (
                X0Model(self.transformer_builder.build(device=self._target_device(), dtype=self.dtype))
                .to(self.device)
                .eval()
            )
```

> Note ordre LoRA→quant : `build()` fusionne déjà les LoRA dans les poids bf16, **puis**
> on quantifie. C'est le bon ordre (fuse-then-quantize). On laisse les LoRA en bf16 dans
> le chemin quanto (on n'utilise PAS le downcast map fp8).

### 2.4 Les 3 pipelines — accepter et propager `quant_mode`

Pour chacun : `ti2vid_two_stages.py`, `ic_lora.py`, `keyframe_interpolation.py`

**(a)** constructeur — nouveau param :
```python
        fp8transformer: bool = False,
        quant_mode: str | None = None,   # NEW
```

**(b)** à la création de chaque `ModelLedger(...)` dans le constructeur (il y en a 1 ou 2
selon le pipeline — ex. `stage_1_model_ledger`; `with_loras` propage déjà au stage 2) :
```python
        self.stage_1_model_ledger = ModelLedger(
            ...
            fp8transformer=fp8transformer,
            quant_mode=quant_mode,         # NEW
        )
```

### 2.5 `backend/model_engine.py` — exposer le mode et choisir le bon checkpoint

**(a)** Lire le mode (env/config). En haut de `load_pipeline`, après le calcul de `fp8` :
```python
            is_mps = (device == "mps")
            fp8 = False if is_mps else True

            # NEW — mode quanto via env (ex: MILIMO_QUANT=int4-quanto)
            quant_mode = os.environ.get("MILIMO_QUANT")  # None | "int4-quanto" | "int8-quanto"
            if quant_mode:
                fp8 = False  # quanto remplace fp8
```

**(b)** Passer `quant_mode=quant_mode` à chacun des 3 constructeurs de pipeline (à côté de
`fp8transformer=fp8`).

**(c)** Sélection du checkpoint : quanto part du **bf16 plein**, pas du fp8. Dans
`get_model_paths`, si `MILIMO_QUANT` est défini, forcer `ckpt_full` et avertir si absent.

---

## 3. Modèle requis & RAM

- quanto quantifie depuis le **checkpoint bf16 plein** : `ltx-2-19b-distilled.safetensors` (~38 Go disque).
- La quantif est **bloc-par-bloc** (un bloc à la fois sur GPU) → pic VRAM faible.
- Pic **RAM** : le modèle bf16 est chargé en CPU avant quantif. 19B bf16 ≈ 38 Go → **dépasse tes 32 Go de RAM**.
  → voir Phase 2 (sauvegarde du modèle déjà quantifié) qui supprime ce besoin après la 1ʳᵉ fois,
    ou quantifier une fois sur une machine à plus de RAM / via swap.

---

## 4. Phase 2 (recommandée) — sauvegarder/recharger le modèle quantifié

Évite de re-télécharger/recharger les 38 Go bf16 à chaque démarrage et résout le pic RAM.

**Sauvegarde** (une fois, juste après quantif dans `transformer()` ou script dédié) :
```python
from optimum.quanto import quantization_map
from safetensors.torch import save_file
import json
save_file(inner.state_dict(), "ltx-2-19b-distilled.int4.safetensors")
with open("ltx-2-19b-distilled.int4.qmap.json", "w") as f:
    json.dump(quantization_map(inner), f)
```

**Rechargement rapide** (squelette vide → requantize → load) :
```python
from optimum.quanto import requantize
inner = self.transformer_builder.build(device="cpu", dtype=self.dtype, weights_only_skeleton=...)  # ou build puis vider
with open("...qmap.json") as f: qmap = json.load(f)
state = load_file("...int4.safetensors")
requantize(inner, state, qmap, device=self.device)
return X0Model(inner).to(self.device).eval()
```
→ Le modèle int4 (~9-10 Go) se charge directement, sans jamais matérialiser 38 Go en RAM.

---

## 5. Côté Flux 2 (séparé, à traiter après)

`backend/models/flux_wrapper.py` (`FluxInpainter`) charge Flux 2 Klein 9B + Qwen3-8B
indépendamment. Même principe : appliquer `quantize_model` au transformer Flux et au text
encoder après chargement. Mais c'est un autre wrapper, hors périmètre de ce plan (à faire
en Phase 3). Le `MemoryManager` garantit que Flux et LTX ne sont jamais en VRAM en même temps.

---

## 6. Plan de test

1. Installer `optimum-quanto` dans le venv `milimov`.
2. `export MILIMO_QUANT=int4-quanto` puis lancer le backend.
3. Générer un clip court (ex. 512×320, ~49 frames) via `ti2vid`.
4. Vérifier `nvidia-smi` : pic VRAM transformer ≈ 9-10 Go (int4) vs ~19 Go (fp8).
5. Comparer qualité int4 vs int8 (`MILIMO_QUANT=int8-quanto`) — int8 = quasi sans perte si int4 dégrade trop.
6. Une fois validé : implémenter Phase 2 (save/load) pour des démarrages rapides.

---

## 7. Récap effort

| Tâche | Fichiers | Difficulté |
|---|---|---|
| Relocaliser quantizer | +1 fichier ltx-core | trivial |
| Dépendance quanto | pyproject + requirements | trivial |
| `quant_mode` dans ledger | model_ledger.py | faible |
| `quant_mode` dans 3 pipelines | 3 fichiers | faible (répétitif) |
| Câblage model_engine + env | model_engine.py | faible |
| **Phase 1 total** | **6 fichiers** | **~1 journée** |
| Phase 2 (save/load int4) | model_ledger + script | moyenne (~½ journée) |
| Phase 3 (Flux) | flux_wrapper.py | moyenne |
