# Usamos una imagen ligera de Python como base
FROM python:3.10-slim

# Configuraciones para evitar archivos caché innecesarios y ver los logs en tiempo real
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Instalamos CMake, compiladores de C y Git (estrictamente necesarios para liboqs)
RUN apt-get update && apt-get install -y \
    cmake \
    build-essential \
    gcc \
    git \
    && rm -rf /var/lib/apt/lists/*

# Establecemos el directorio de trabajo dentro del contenedor
WORKDIR /app

# Copiamos primero el archivo de requerimientos
COPY requirements.txt .

# Instalamos las dependencias
RUN pip install --no-cache-dir --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt
# Instalamos la librería criptográfica que fallaba en Windows
RUN pip install --no-cache-dir liboqs-python

# Copiamos el resto de los archivos del proyecto al contenedor
COPY . .

# Comando por defecto al iniciar el contenedor
CMD ["python", "main.py"]