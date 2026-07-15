# Exportación a ONNX

> Guía de la funcionalidad de **exportación ONNX** de DeepFilterNet. Esta capacidad
> **ya existe** en el repositorio (`DeepFilterNet/df/scripts/export.py`); este documento la
> consolida y la documenta para que sea fácil de usar y de mantener. No sustituye al código:
> lo describe.

## 1. Qué hace

El script exporta un modelo entrenado (checkpoint PyTorch) a **tres grafos ONNX** que se
corresponden con las tres partes de la arquitectura DeepFilterNet:

| Archivo         | Submódulo   | Papel                                                  |
|-----------------|-------------|--------------------------------------------------------|
| `enc.onnx`      | `model.enc` | Codificador (features ERB + spec → embeddings)         |
| `erb_dec.onnx`  | `model.erb_dec` | Decodificador de la máscara ERB                    |
| `df_dec.onnx`   | `model.df_dec`  | Decodificador de los coeficientes de *deep filtering* |

Además genera:

- `config.ini` — copia de la configuración del modelo (se copia del directorio del modelo).
- `version.txt` — nombre del modelo y época (`<modelo>_epoch_<n>`).
- `<modelo>_onnx.tar.gz` — **paquete final** con los tres `.onnx`, `config.ini` y `version.txt`.
- `*_input.npz` / `*_output.npz` — tensores de referencia por etapa (útiles para depurar).

El `.tar.gz` resultante tiene **el mismo formato** que los modelos preentrenados de
`models/` (p. ej. `DeepFilterNet3_onnx.tar.gz`), por lo que puede consumirse directamente
por la ruta Rust (`tract`) y por la ruta Python.

## 2. Requisitos

La exportación es una tarea de desarrollo y necesita algunas dependencias adicionales que
**no** forman parte de la instalación de inferencia:

```bash
pip install onnx onnxruntime onnxsim MonkeyType
```

- `onnx`, `onnxruntime` — exportación, validación y comprobación numérica.
- `onnxsim` — *opcional*, simplificación del grafo (solo con `--simplify`).
- `MonkeyType` — requerido por el script (se usa junto a `torch.jit.script`).

`onnx` y `onnxruntime` ya están declarados como dependencias opcionales del paquete
(`onnxruntime` está en el extra `dnsmos-local`); `onnxsim` y `MonkeyType` se instalan aparte.

## 3. Uso

```bash
cd DeepFilterNet

# Exporta el modelo preentrenado DeepFilterNet3 al directorio ./export_out
python df/scripts/export.py -m DeepFilterNet3 ./export_out

# Con simplificación de grafo y un opset concreto
python df/scripts/export.py -m DeepFilterNet3 ./export_out --simplify --opset 12
```

Argumentos principales:

| Argumento              | Por defecto | Descripción                                                        |
|------------------------|-------------|--------------------------------------------------------------------|
| `-m`, `--model-base-dir` | DeepFilterNet2 | Nombre de un modelo preentrenado o ruta a un directorio de modelo. |
| `export_dir` (posicional) | —        | Directorio de salida de los `.onnx` y del `.tar.gz`.               |
| `-e`, `--epoch`        | `best`      | Época del checkpoint (`best`, `latest` o un entero).               |
| `--opset`              | `12`        | Versión de *opset* de ONNX.                                        |
| `--simplify`           | desactivado | Simplifica los grafos con `onnxsim`.                               |
| `--no-check`           | (comprueba) | Omite la verificación con `onnx.checker` y la comparación numérica. |

## 4. Verificación integrada (red de seguridad)

Por defecto (`check=True`), tras exportar cada grafo el script:

1. Valida el modelo con `onnx.checker.check_model(..., full_check=True)`.
2. Ejecuta el grafo con `onnxruntime` (CPU) y **compara la salida con la de PyTorch**
   mediante `np.testing.assert_allclose(..., rtol=1e-6, atol=1e-5)`.

Esto detecta regresiones de exportación antes de empaquetar el `.tar.gz`. Si algún grafo no
supera la tolerancia, el proceso falla de forma explícita.

## 5. Cómo se consume el resultado

- **Rust (`tract`)** — la ruta de inferencia del binario `deep-filter` carga el `.tar.gz`
  directamente:

```bash
cargo run -p deep_filter --profile=release \
  --features=tract,bin,wav-utils,transforms --bin deep-filter -- \
  ./assets/noisy_snr0.wav -m ./export_out/DeepFilterNet3_onnx.tar.gz -o out
```

- **Python** — el mismo `.tar.gz` se puede colocar donde `df.enhance`/`init_df` espera el
  modelo, manteniendo la API pública sin cambios.

## 6. Notas de mantenimiento

- La exportación fija `keep_initializers_as_inputs=False` y usa ejes dinámicos para la
  dimensión temporal (`S`), de modo que los grafos aceptan secuencias de longitud variable.
- El *opset* por defecto (12) es conservador; súbelo solo si una operación nueva lo requiere
  y vuelve a validar con la comprobación numérica del punto 4.
- La comprobación numérica es la garantía principal frente a regresiones: **no la desactives
  (`--no-check`) al publicar** un modelo nuevo.
