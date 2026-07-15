# DeepFilterNet Modern — Informe técnico de auditoría (Fase 1)

> Entregable de la **Fase 1 (Auditoría)** de la hoja de ruta de *DeepFilterNet Modern*.
> Objetivo: entender el estado real del proyecto antes de tocar nada, identificar deuda
> técnica y dependencias obsoletas, y establecer un plan de trabajo **priorizado** que
> minimice el riesgo de regresiones.
>
> Este documento describe el estado del repositorio; **no introduce cambios de código**.
> Cada recomendación indica el archivo afectado y el motivo, de modo que cualquier
> contribuidor pueda entender el porqué antes de implementarlo.

---

## 1. Resumen ejecutivo

DeepFilterNet sigue siendo técnicamente muy sólido, pero su cadena de instalación y
mantenimiento se ha quedado atrás respecto al ecosistema actual de Python. El problema
concreto y más urgente —incompatibilidad con **NumPy 2.x** y **Python 3.13**— no es un
detalle de configuración: está causado por versiones antiguas de los *bindings* Rust
(`pyo3` y `rust-numpy` en la 0.20), que además usan una API (`GIL Ref`) ya eliminada en
las versiones modernas.

Diagnóstico en una frase: **el núcleo de audio (Rust `libDF`) está sano; la deuda técnica
se concentra en la capa de *bindings* Python↔Rust, en el empaquetado (wheels) y en las
restricciones de versión declaradas.**

Bloqueos priorizados:

| Prioridad | Tema | Impacto |
|-----------|------|---------|
| **P0** | `pyo3`/`rust-numpy` 0.20 → 0.22 + migración a la API `Bound` | Desbloquea NumPy 2 y Python 3.13 |
| **P0** | Tope `numpy = ">=1.22,<2.0"` en el paquete Python | Impide instalar junto a NumPy 2 |
| **P1** | Wheels solo para CPython 3.8–3.11; sin 3.12/3.13; sin `abi3` | `pip install` falla o compila desde fuente |
| **P1** | Dependencia `hdf5` apuntando a un *fork* git no publicado | Riesgo de suministro y de build de entrenamiento |
| **P2** | Inconsistencias de versión entre *crates* (`env_logger`, `rubato`) | Deuda de mantenimiento, builds mayores |
| **P2** | CI: pruebas en una sola versión de Python; `poetry` dev-deps deprecado | Cobertura pobre, avisos |
| **P3** | ONNX / API asíncrona / documentación | Funcionalidades opcionales (Fase 6) |

---

## 2. Metodología y alcance

- **Alcance:** todo el *workspace* (código Rust y Python, sistema de build y CI).
- **Enfoque:** lectura estática del árbol de código y de los manifiestos de dependencias,
  contrastada con las notas de versión oficiales de `pyo3` y `rust-numpy`.
- **No incluido en esta fase:** cambios de código, ejecución de benchmarks o de la suite
  de pruebas (corresponde a fases posteriores). Aquí solo se *diagnostica*.

Referencias externas verificadas durante la auditoría:

- `rust-numpy` **0.22.0** añade soporte de **NumPy 2**, sube `pyo3` a 0.22 y soporta
  `ndarray` 0.16 (la 0.21 solo migró a la API `Bound`, sin NumPy 2).
- `pyo3` **0.22** añade soporte base de **Python 3.13**; `pyo3` **0.23** añade el build
  *free-threaded* (3.13t).

---

## 3. Mapa de arquitectura

El repositorio es un *workspace* de Cargo con cinco *crates* Rust y un paquete Python
gestionado con Poetry.

```
DeepFilterNet/  (workspace raíz, Cargo.toml)
├── libDF/          crate Rust "deep_filter": DSP, STFT/ISTFT, ERB, dataset, modelo tract
├── pyDF/           crate "DeepFilterLib" (módulo Python `libdf`)  ── binding pyo3
├── pyDF-data/      crate "DeepFilterDataLoader" (módulo `libdfdata`) ── binding pyo3 + HDF5
├── ladspa/         plugin LADSPA (supresión de ruido en tiempo real)
├── demo/           demo de escritorio (iced) + captura CLI
├── DeepFilterNet/  paquete Python `df` (Poetry): entrenamiento, inferencia, API pública
└── models/         modelos preentrenados (.tar.gz ONNX y .zip de checkpoints)
```

**Componentes y su papel:**

- **`libDF` (`deep_filter`)** — Núcleo. Procesado de señal (STFT/ISTFT vía `rustfft`/`realfft`),
  transformadas ERB, carga de dataset (HDF5) e inferencia con `tract` (ONNX). Es la base
  técnica y la parte más sana del proyecto. Expone además `crate-type = cdylib/rlib/staticlib`
  y una C-API (`capi`) y target `wasm`.
- **`pyDF` (módulo `libdf`)** — Envoltorio fino sobre el bucle STFT/ISTFT y las transformadas
  ERB de `libDF`. Es el que usa `df.enhance` en tiempo de inferencia. Archivo clave:
  `pyDF/src/lib.rs`.
- **`pyDF-data` (módulo `libdfdata`)** — Envoltorio del *dataloader* (HDF5) para entrenamiento
  con PyTorch. Solo Linux. Archivo clave: `pyDF-data/src/lib.rs`.
- **`DeepFilterNet` (paquete `df`)** — Código Python de alto nivel: modelos
  (`deepfilternet.py`, `deepfilternet2.py`, `deepfilternet3.py`), `enhance.py`, `train.py`,
  evaluación (PESQ/STOI/DNSMOS) y la **API pública de compatibilidad**:
  `from df import enhance, init_df`.
- **`ladspa` / `demo`** — Integraciones de tiempo real (PipeWire / escritorio).
- **`models`** — Modelos empaquetados. Los `.tar.gz` contienen el modelo ONNX + `config.ini`
  y los consume tanto la ruta Rust (`tract`) como la Python.

**API de compatibilidad a preservar** (contrato público con los usuarios actuales):

```py
from df import enhance, init_df
model, df_state, _ = init_df()
enhanced = enhance(model, df_state, noisy_audio)
```

Además, el módulo nativo se llama `libdf` (con clases/funciones `DF`, `erb`, `erb_norm`,
`unit_norm`). Cualquier renombrado del proyecto a *Modern* **no debe** romper estos
símbolos sin una capa de alias.

---

## 4. Sistema de compilación

### 4.1 Rust (Cargo workspace)

- `resolver = "2"`, perfiles `dev` / `release` / `release-lto` correctos.
- **MSRV declarado: 1.70** en `libDF`, `pyDF`, `pyDF-data`. `rust-numpy` 0.22 solo exige
  1.63, así que hay margen; conviene fijar y verificar la MSRV en CI.
- Los *crates* comparten `libDF` vía rutas relativas (bien).

### 4.2 Empaquetado Python

Dos mecanismos distintos conviven:

1. **`maturin`** (para `pyDF` y `pyDF-data`): `requires = ["maturin>=1.3,<1.5"]`.
   Construyen los módulos nativos `libdf` / `libdfdata`.
2. **`poetry`** (para el paquete `DeepFilterNet`): empaqueta el código Python puro `df` y
   declara `deepfilterlib` / `deepfilterdataloader` como dependencias por ruta.

El `pyproject.toml` raíz solo configura `black`/`isort` (no es un paquete).

**Observaciones de build:**

- Las wheels nativas se compilan **por versión de CPython** (cp38–cp311). No se usa el
  *stable ABI* (`abi3`) de `pyo3`, lo que multiplica la matriz de wheels y obliga a añadir
  manualmente cada nueva versión de Python.
- `[tool.poetry.dev-dependencies]` está **deprecado**; Poetry moderno usa
  `[tool.poetry.group.dev.dependencies]`.
- `target-version` de `black` es incoherente entre archivos: raíz `py37–py310`,
  `DeepFilterNet` `py38–py310`, `pyDF`/`pyDF-data` `py38–py311`.

---

## 5. Inventario de dependencias (actual → recomendado)

### 5.1 Bindings nativos (el bloqueo principal)

| Dependencia | Actual | Recomendado | Motivo |
|-------------|--------|-------------|--------|
| `pyo3` (`pyDF`, `pyDF-data`) | **0.20** | **0.22.x** | Python 3.13 + API `Bound` |
| `numpy` / `rust-numpy` | **0.20** | **0.22.x** | **Soporte NumPy 2** (llega en 0.22) |
| `ndarray` (todos los crates) | **0.15** | **0.16** | Requerido por `rust-numpy` 0.22 |

> Subir de 0.20 a 0.22 **cruza la migración a la API `Bound`** (introducida en 0.21). Por
> tanto no es un simple *bump* de versión: hay que reescribir `pyDF/src/lib.rs` y
> `pyDF-data/src/lib.rs` (ver §6).

### 5.2 Restricción de NumPy en el paquete Python

| Archivo | Actual | Recomendado |
|---------|--------|-------------|
| `DeepFilterNet/pyproject.toml` | `numpy = ">=1.22,<2.0"` | `numpy = ">=1.22"` (permitir 2.x) |
| `pyDF/pyproject.toml` | `numpy >= 1.22` | (sin tope, correcto) |
| `pyDF-data/pyproject.toml` | `numpy >= 1.22` | (sin tope, correcto) |

El tope `<2.0` del paquete raíz es lo que hace fallar `pip install deepfilternet` junto a
NumPy 2. Debe levantarse **solo después** de que los bindings soporten NumPy 2 (orden
importante para no publicar una versión rota).

### 5.3 Otras dependencias Rust

| Dependencia | Situación | Recomendación |
|-------------|-----------|---------------|
| `hdf5` | *fork* git `aldanor/hdf5-rust.git` fijado a `rev 26046fb` | Migrar al *fork* mantenido `hdf5-metno` (publicado en crates.io) |
| `env_logger` | `0.11` en `libDF`, `0.10` en `ladspa` y `demo` | Unificar a una versión |
| `rubato` | `0.14` en `libDF`, `0.15` en `demo` | Unificar |
| `tract-*` | `0.21.4` | Al día; verificar tras el resto |
| `itertools` | `0.12` | Al día |
| `clap` | `4.0` | Al día |

### 5.4 Dependencias Python

- `torch`/`torchaudio` no están fijadas en las dependencias del paquete, pero las tareas
  `poe` instalan **torch 2.1** (antiguo). Conviene ampliar a PyTorch reciente y probarlo.
- `packaging = ">=23,<25"`, `pesq`, `pystoi`, `onnxruntime ^1.15`, `soundfile <0.13`:
  revisables pero sin urgencia.

---

## 6. Deuda técnica detectada

### 6.1 API antigua de PyO3 (`GIL Ref`) — **crítico**

`pyDF/src/lib.rs` y `pyDF-data/src/lib.rs` usan el patrón previo a PyO3 0.21:

- Firmas `fn libdf(_py: Python, m: &PyModule)` y devolución de referencias con *lifetime*
  como `&'py PyArray3<...>`, `IntoPyArray::into_pyarray(py)` que devuelve `&PyArray`.
- Tipos como `PyReadonlyArray2<'py, f32>` con la semántica antigua.

En PyO3 0.22 esta API está **deprecada** y en 0.23 fue **eliminada**. La migración a
`Bound<'py, T>` (p. ej. `Bound<'py, PyModule>`, `Bound<'py, PyArray3<...>>`, `into_pyarray_bound`)
es mecánica pero extensa, y toca todas las funciones expuestas (`analysis`, `synthesis`,
`erb`, `erb_inv`, `erb_norm`, `unit_norm`, `unit_norm_init`, y el *dataloader*).

> Riesgo de regresión: los bloques `unsafe { input.as_array_mut() }` en `erb_norm`/`unit_norm`
> asumen exclusividad de acceso; al migrar hay que revisarlos con cuidado (y más aún si en
> el futuro se contempla el build *free-threaded* con PyO3 0.23+).

### 6.2 Dependencia `hdf5` sobre un *fork* git — **alto**

`libDF/Cargo.toml`:

```toml
hdf5 = { optional = true, git = "https://github.com/aldanor/hdf5-rust.git", rev = "26046fb" }
```

El *crate* `hdf5` original está prácticamente sin mantenimiento; el proyecto ya se ve
obligado a fijar un *fork* por *rev*. Esto:

- rompe la reproducibilidad si el repositorio remoto cambia o desaparece,
- complica el build de `pyDF-data` (entrenamiento) y su publicación,
- y es la razón por la que macOS está deshabilitado en algunos jobs de CI
  (`# Fails to install hdf5 from git`).

Recomendación: migrar a `hdf5-metno` (fork comunitario publicado en crates.io).

### 6.3 Cobertura de wheels y versiones de Python — **alto**

- `publish.yml` construye wheels nativas para **cp38, cp39, cp310, cp311** únicamente.
  No hay **3.12** ni **3.13**.
- No se usa `abi3`, por lo que cada versión nueva de Python obliga a ampliar la matriz.
- La instalación "simple" (`pip install`) solo funciona bien en las versiones con wheel
  precompilada; el resto intenta compilar desde fuente (y falla sin Rust/HDF5).

### 6.4 CI/CD — **medio**

- `test_df.yml` prueba en **una sola versión de Python (3.10)** y en Ubuntu + Windows
  (macOS está comentado por HDF5). No cubre la matriz 3.10–3.13 ni NumPy 1 vs 2.
- `python_lint.yml` usa Python 3.9 y ejecuta `flake8`/`black`/`isort` con instalación
  suelta (sin *pins*).
- `rust_lint.yml` usa `nightly` para `fmt`/`clippy` (aceptable, pero conviene fijarlo).
- Puntos buenos: las *actions* base están actualizadas (`checkout@v4`, `setup-python@v5`,
  `cache@v4`, `upload-artifact@v4`), hay caché de Cargo y de venv.

### 6.5 Inconsistencias menores

- Versiones divergentes de `env_logger` y `rubato` (ver §5.3).
- `target-version` de `black` incoherente entre `pyproject.toml` (ver §4.2).
- `[tool.poetry.dev-dependencies]` deprecado.

---

## 7. Modelos y carga

- `models/` contiene los pesos preentrenados: `DeepFilterNet.zip`, `DeepFilterNet2*`,
  `DeepFilterNet3*` (variantes `_onnx` y `_ll` de baja latencia).
- Los `.tar.gz` empaquetan modelo ONNX + `config.ini` y se cargan tanto por la ruta Rust
  (`tract`) como por la Python. El paquete `DeepFilterNet` incluye además checkpoints de
  DFN y DFN2 vía `include` en su `pyproject.toml`; DFN3 se descarga bajo demanda
  (`maybe_download_model`).
- **Nota positiva para la Fase 6:** la exportación ONNX **ya existe**
  (`DeepFilterNet/df/scripts/export.py`, con `onnx`, `onnxruntime` y verificación contra
  `tract`). La funcionalidad "exportación ONNX" está en gran parte cubierta; la tarea sería
  consolidarla y documentarla, no crearla de cero.
- No se detectaron alias NumPy obsoletos (`np.float`, `np.int`, `np.NaN`, etc.) en el código
  Python, por lo que la parte Python es en principio compatible con NumPy 2 una vez que los
  *bindings* nativos lo soporten.

---

## 8. Riesgos y estrategia anti-regresiones

1. **Orden de los cambios importa.** No levantar el tope `numpy < 2.0` del paquete Python
   antes de que `pyDF`/`pyDF-data` compilen y funcionen con `rust-numpy` 0.22. De lo
   contrario se publicaría una versión que se instala pero falla en tiempo de ejecución.
2. **Migración `Bound` acoplada.** `pyo3`, `rust-numpy` y `ndarray` deben subirse **a la vez**
   (0.22/0.22/0.16). Subir uno solo deja el *workspace* sin compilar.
3. **Validación numérica.** Tras migrar los *bindings*, comparar la salida de `enhance`
   contra una referencia (los tests de CI ya comparan métricas DNSMOS con umbrales fijos en
   `test_df.yml`): sirven como red de seguridad para detectar regresiones de audio.
4. **HDF5 aislado.** El cambio de `hdf5` afecta solo a `pyDF-data`/entrenamiento; puede
   abordarse de forma independiente del camino de inferencia.
5. **Compatibilidad de API.** Mantener los símbolos `df.enhance`, `df.init_df` y el módulo
   nativo `libdf`. Si se publica como `deepfilternet-modern`, ofrecer alias/*shims*.

---

## 9. Plan priorizado

### P0 — Desbloquear NumPy 2 y Python 3.13 (Fase 2)

1. Subir `pyo3` 0.20 → 0.22, `rust-numpy` 0.20 → 0.22, `ndarray` 0.15 → 0.16 en `pyDF` y
   `pyDF-data` (y `ndarray` en `libDF`).
2. Migrar `pyDF/src/lib.rs` y `pyDF-data/src/lib.rs` a la API `Bound`.
3. Revisar los bloques `unsafe` de `as_array_mut`.
4. Levantar el tope `numpy = ">=1.22,<2.0"` → `">=1.22"` en `DeepFilterNet/pyproject.toml`
   **una vez** validados los bindings.
5. Verificación: `maturin develop` + `df/scripts/test_df.py` con NumPy 1.x **y** 2.x.

**Criterio de aceptación:** el paquete instala y `enhance` produce salida equivalente
(dentro de tolerancia) con NumPy 1.x y 2.x, en Python 3.10–3.13.

### P1 — Distribución (Fases 2/4)

6. Añadir CPython **3.12** y **3.13** a la matriz de `publish.yml`.
7. Evaluar `abi3` (`pyo3` *feature* `abi3-py38`) para publicar una sola wheel por plataforma.
8. Migrar `hdf5` a `hdf5-metno` y reactivar macOS en CI si es viable.
9. Objetivo final: `pip install deepfilternet-modern` funcional en Linux x86_64/ARM64,
   Windows, macOS Intel y Apple Silicon.

### P2 — Calidad (Fase 3)

10. Ampliar `test_df.yml` a matriz Python 3.10–3.13 y NumPy 1/2.
11. Migrar `[tool.poetry.dev-dependencies]` → grupos; unificar `target-version` de `black`.
12. Unificar `env_logger`/`rubato`; fijar toolchain de `fmt`/`clippy`.
13. Añadir benchmarks de rendimiento (RTF) y de calidad (PESQ/STOI/DNSMOS) reproducibles.

### P3 — Optimización y opcionales (Fases 5/6)

14. Reducir copias de memoria Rust↔Python (revisar `to_owned()`/`into_owned()` en
    `pyDF/src/lib.rs`); perfilar el *streaming* de baja latencia.
15. Consolidar y documentar la exportación ONNX ya existente; explorar API asíncrona.
16. Renovar documentación y ejemplos.

---

## 10. Sobre el nombre y la compatibilidad

Propuesta de posicionamiento como **el fork de referencia**, no un fork personal:

- Nombre de distribución: `deepfilternet-modern` (PyPI), manteniendo el paquete importable
  `df` y el módulo nativo `libdf` para no romper el código existente.
- Documentar en el README el objetivo (compatibilidad con el ecosistema actual) y una tabla
  de compatibilidad Python × NumPy × PyTorch.
- Mantener la doble licencia MIT/Apache-2.0 y la atribución al proyecto original.

---

## 11. Conclusión

El proyecto tiene una base técnica excelente y el camino para modernizarlo está bien
delimitado: el 80% del valor inmediato se consigue con **una sola pieza de trabajo P0**
(subir `pyo3`/`rust-numpy`/`ndarray` y migrar los dos wrappers a la API `Bound`), tras la
cual se puede levantar el tope de NumPy y ampliar la matriz de wheels. El resto de la hoja
de ruta (calidad, distribución, optimización, opcionales) se apoya sobre esa base y puede
abordarse por etapas, con la red de seguridad de las comparaciones de métricas ya presentes
en CI.
