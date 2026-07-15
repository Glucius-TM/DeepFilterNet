# DeepFilterNet Modern — Guía de compatibilidad e instalación

> Documento de referencia de la **compatibilidad** que aporta el trabajo de modernización
> (Fases 2–4 de la hoja de ruta) y de cómo instalar/usar el proyecto en el ecosistema actual.
>
> La matriz de abajo refleja combinaciones **verificadas de extremo a extremo** durante la
> modernización (build de wheels + `enhance` sobre audio real), no aspiraciones. Las
> capacidades las entregan los PRs de modernización; ver §5.

---

## 1. Matriz de compatibilidad

| Componente | Antes | Ahora (modernizado) |
|-----------|-------|---------------------|
| **Python** | 3.8–3.11 (wheels) | **3.8–3.13** (una sola wheel `abi3` por plataforma) |
| **NumPy** | `>=1.22,<2.0` | **1.x y 2.x** (`>=1.22`; en Python ≥3.10 el default es NumPy 2, ya que 3.8/3.9 no tienen wheels de NumPy 2) |
| **PyTorch** | pinneado a 2.1 (tareas `poe`) | **reciente** (probado con 2.5.x; ver nota torchaudio) |
| **pyo3 / rust-numpy** | 0.20 (API `GIL Ref`) | **0.22** (API `Bound`) |
| **HDF5 (entrenamiento)** | *fork* git no publicado | **`hdf5-metno`** (crates.io) |
| **Wheels** | 1 por versión de Python | **1 `abi3` por plataforma** |

Plataformas de wheel objetivo: Linux x86_64, Linux ARM64, Windows x86_64, macOS Intel,
macOS Apple Silicon.

### Combinaciones verificadas

| Python | NumPy | PyTorch / torchaudio | Resultado |
|--------|-------|----------------------|-----------|
| 3.12 | 2.5.1 | 2.5.1 / 2.5.1 | `enhance` OK (E/S vía torchaudio) |
| 3.12 | 2.5.1 | 2.13 / 2.11 | `enhance` OK (E/S vía *fallback* soundfile) |
| 3.12 | 1.26.4 | 2.5.1 / 2.5.1 | `libdf` OK (compatibilidad NumPy 1) |
| 3.12 | 2.5.1 | — | wheel `abi3` (`cp38-abi3`) instala e importa |

---

## 2. Instalación

Objetivo de la experiencia de instalación:

```bash
# 1) PyTorch (CPU o CUDA) desde pytorch.org, p. ej. CPU:
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cpu

# 2) DeepFilterNet (wheel abi3 precompilada; no requiere Rust ni compilar)
pip install deepfilternet
```

Uso rápido por línea de comandos:

```bash
deepFilter path/to/noisy_audio.wav -o out/
```

Desde Python (API de compatibilidad, sin cambios respecto al proyecto original):

```python
from df import enhance, init_df

model, df_state, _ = init_df()   # carga el modelo por defecto (DeepFilterNet3)
enhanced = enhance(model, df_state, noisy_audio)
```

---

## 3. Notas sobre PyTorch / torchaudio

Las versiones recientes de **torchaudio (≥2.9/2.11)** eliminaron el backend clásico de E/S
(`torchaudio.info/load/save`) y el tipo `AudioMetaData`. La capa de E/S se ha hecho
**agnóstica del backend**: usa torchaudio si aún ofrece esas funciones y, en caso contrario,
cae automáticamente a **`soundfile`** (ahora dependencia de base). Por eso `enhance`
funciona tanto con torchaudio "clásico" (2.1–2.8) como con los más recientes.

---

## 4. Entrenamiento / carga de datos (Linux)

El *dataloader* nativo (`libdfdata`) depende de HDF5. Ahora usa el *crate* publicado y
mantenido **`hdf5-metno`** en lugar de una revisión git no publicada, lo que mejora la
reproducibilidad y desbloquea la construcción en más plataformas.

```bash
# Requiere cabeceras de HDF5 del sistema (p. ej. Ubuntu: libhdf5-dev) o compilación estática
pip install deepfilternet[train]
```

> Nota: el *build* de la ruta de entrenamiento arrastra dependencias de compilación
> recientes; usar un Rust `stable` actual (como hace la CI).

---

## 5. Cómo se entrega esta compatibilidad

El objetivo se alcanza mediante cambios acotados y verificados por separado:

- **NumPy 2 + Python 3.13:** migración de los *bindings* nativos `pyDF`/`pyDF-data` de
  `pyo3`/`rust-numpy` 0.20 → 0.22 (API `Bound`) y levantado del tope `numpy<2.0`.
- **PyTorch reciente:** capa de E/S de audio con *fallback* a `soundfile`.
- **Distribución:** extensiones compiladas contra el ABI estable (`abi3-py38`) → una sola
  wheel por plataforma válida en 3.8–3.13+.
- **HDF5:** migración a `hdf5-metno`.

Para el detalle técnico y las prioridades, ver [`auditoria-fase1.md`](auditoria-fase1.md).
Para medir el coste del límite Rust↔Python, ver `scripts/bench_libdf.py` (el DSP + el cruce
Rust↔Python es ~2–3% del tiempo de `enhance`; domina la inferencia de la red).

---

## 6. Compatibilidad de la API

La API pública se mantiene intacta para facilitar la adopción:

- Funciones Python: `from df import enhance, init_df`, `df.version`, `df.config`.
- Módulo nativo: `libdf` con `DF`, `erb`, `erb_norm`, `unit_norm`, `unit_norm_init`.
- Los argumentos opcionales de las funciones nativas (`reset`, `db`, `state`, …) siguen
  siendo opcionales (declarados explícitamente con `#[pyo3(signature = ...)]`).

Doble licencia MIT/Apache-2.0, igual que el proyecto original.

---

## 7. Uso asíncrono

Para servir DeepFilterNet desde aplicaciones `asyncio` (FastAPI, aiohttp, …) sin bloquear el
bucle de eventos, existe `enhance_async`, que delega el cálculo (bloqueante) a un *executor*
de hilos. PyTorch libera el GIL durante el cómputo, así que el trabajo offloadeado puede
ejecutarse realmente en paralelo.

```python
import asyncio
from df import enhance_async, init_df
from df.enhance import load_audio

model, df_state, _, _ = init_df()
audio, _ = load_audio("noisy.wav", df_state.sr())
enhanced = asyncio.run(enhance_async(model, df_state, audio))
```

Los argumentos y el valor de retorno son idénticos a `enhance`. Se puede pasar un `executor`
propio (p. ej. un `ThreadPoolExecutor` acotado) para controlar el paralelismo.

> **Concurrencia:** `model` y `df_state` tienen estado interno (el modelo reinicia su estado
> oculto por llamada y `df_state` guarda los *buffers* STFT/ISTFT). No compartas el mismo par
> `model`/`df_state` entre tareas que se ejecuten concurrentemente: usa una instancia por
> tarea concurrente (o serializa el acceso). Encadenar `await` secuenciales con un par
> compartido es correcto.
