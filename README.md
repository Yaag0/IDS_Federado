# Quantum Security Federated IDS (QSF-IDS)
> Sistema de Detección de Intrusos (IDS) Federado, Robusto y Resistente a Ataques Cuánticos con Cifrado Híbrido PQC-AES.

## 📋 Descripción del Proyecto
Este repositorio contiene la implementación oficial del sistema de grado orientado a la seguridad de redes de nueva generación. El proyecto combina **Aprendizaje Federado (Federated Learning)** para el entrenamiento distribuido y privado de modelos de red en entornos descentralizados, complementado con un esquema de **Criptografía Post-Cuántica (PQC)** basado en el estándar FIPS 203 (**ML-KEM / Kyber-768**) y cifrado simétrico de alta velocidad (**AES-256-GCM**), mitigando tanto las amenazas clásicas como los futuros ataques de descifrado masivo mediante computación cuántica (algoritmo de Shor).

El sistema utiliza el conjunto de datos de referencia **UNSW-NB15** y se encuentra blindado contra ataques de envenenamiento de modelos mediante algoritmos de agregación robusta (**Trimmed Mean**).

---

## 🏗️ Arquitectura del Sistema

┌──────────────┐         ┌──────────────┐         ┌──────────────┐
│  Cliente 0   │         │  Cliente 1   │         │  Cliente 2   │
│ (Entrenamiento)│       │ (Entrenamiento)│       │ (Entrenamiento)│
└──────┬───────┘         └──────┬───────┘         └──────┬───────┘
│                        │                        │
│ Encapsulamiento PQC (Kyber-768) + Cifrado AES-256-GCM
▼                        ▼                        ▼
┌────────────────────────────────────────────────────────────────┐
│                        SERVIDOR CENTRAL                        │
│  1. Desencapsulación PQC & Descifrado de Gradientes            │
│  2. Defensa contra Envenenamiento (Trimmed Mean Aggregation)   │
│  3. Actualización y Distribución del Modelo Global             │
└───────────────────────────────┬────────────────────────────────┘
│
▼
┌───────────────────────┐
│  Dashboard en Vivo    │
│  (Monitoreo Dash/Plotly)│
└───────────────────────┘

---

## ⚙️ Decisiones de Ingeniería y Justificación Tecnológica

1. **Entorno de Ejecución Nativo (Unix / Arch Linux):** 
   El núcleo criptográfico (`liboqs`) requiere compilación de bajo nivel mediante `cmake` y `gcc` para garantizar el rendimiento óptimo de los algoritmos post-cuánticos. Se optó por un entorno nativo Linux en lugar de contenedores sobre sistemas operativos con restricciones corporativas estrictas (como Windows LTSC), optimizando la estabilidad de los punteros en memoria (`_ctypes`) y el rendimiento del compilador.
2. **Criptografía Híbrida Post-Cuántica:**
   * **Intercambio de Llaves:** ML-KEM (Kyber-768), estandarizado por el NIST para resistir ataques de computación cuántica.
   * **Cifrado de Carga Útil (Payload):** AES-256-GCM con nonces aleatorios de 12 bytes para garantizar confidencialidad e integridad autenticada de los gradientes de la red neuronal.
3. **Robustez ante Envenenamiento (*Model Poisoning*):**
   Incorporación de la estrategia de agregación **Trimmed Mean** en el servidor central para eliminar valores atípicos o maliciosos enviados por nodos comprometidos antes de promediar los pesos globales.

---

## 🛠️ Requisitos del Sistema y Dependencias

* **Sistema Operativo:** Linux (Recomendado: Arch Linux / Ubuntu con herramientas de compilación).
* **Python:** Versión 3.10 o superior.
* **Librerías principales:** 
  * `PyTorch` (Redes neuronales Profundas - IDSNet)
  * `liboqs-python` (Criptografía post-cuántica)
  * `cryptography` (AES-256-GCM)
  * `Scikit-Learn` (Preprocesamiento y normalización Min-Max)
  * `Dash` & `Plotly` (Monitorización en tiempo real)

---

## 🚀 Guía de Instalación y Ejecución

### 1. Clonar el Repositorio y Configurar el Entorno
```bash
# Clonar o acceder a la carpeta del proyecto
cd IDS_Federado

# Crear y activar el entorno virtual (Nota: usar activate.fish si usas Fish shell)
python -m venv venv
source venv/bin/activate.fish  # o source venv/bin/activate en Bash/Zsh

### 2. Instalar las Herramientas de Compilación Nativas
```bash
sudo pacman -S base-devel cmake python-virtualenv

### 3. Instalar Dependencias de Python y el Módulo PQC
```bash
pip install -r requirements.txt

### 4. Ejecución del Experimento Federado
```bash
python main.py

### 5. Lanzar el Panel de Control
```bash
python dashboard.py

