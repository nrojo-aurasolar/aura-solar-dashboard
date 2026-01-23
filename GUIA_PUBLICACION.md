# 🚀 GUÍA PASO A PASO: Publicar Dashboard en Streamlit Cloud

## ✅ PASO 1: Preparar archivos en tu computadora

1. Crea una carpeta llamada `aura-solar-dashboard`
2. Descarga y coloca estos archivos en la carpeta:
   - `app-local.py` (renómbralo a `app.py`)
   - `requirements.txt`
   - `.gitignore`
   - `README.md`
   - `credentials.json` (tu archivo de credenciales de Google)

3. Crea una subcarpeta `.streamlit` dentro de `aura-solar-dashboard`
4. Dentro de `.streamlit` crea un archivo `config.toml` con el contenido del archivo `.streamlit_config.toml`

**Estructura final:**
```
aura-solar-dashboard/
├── app.py
├── requirements.txt
├── .gitignore
├── README.md
├── credentials.json
└── .streamlit/
    └── config.toml
```

---

## ✅ PASO 2: Crear repositorio en GitHub

### Opción A: Con GitHub Desktop (MÁS FÁCIL)

1. Abre GitHub Desktop
2. Click en "File" → "New Repository"
3. Nombre: `aura-solar-dashboard`
4. Local Path: Selecciona la carpeta padre (NO la carpeta del proyecto)
5. Click "Create Repository"
6. Click "Publish Repository"
7. ✅ Desmarca "Keep this code private" SI quieres que sea público (gratis)
   O déjalo marcado si quieres privado ($20/mes en Streamlit)
8. Click "Publish Repository"

### Opción B: Con comandos (si prefieres terminal)

```bash
cd /Users/nataliarojo/Desktop/aura-solar-dashboard

# Inicializar Git
git init

# Agregar archivos
git add .

# Hacer commit
git commit -m "Initial commit - Aura Solar Dashboard"

# Crear repositorio en GitHub (ve a github.com y crea uno nuevo)
# Luego conecta:
git remote add origin https://github.com/TU_USUARIO/aura-solar-dashboard.git
git branch -M main
git push -u origin main
```

---

## ✅ PASO 3: Configurar Streamlit Cloud

1. Ve a: https://share.streamlit.io
2. Click "Sign up" y usa tu cuenta de GitHub
3. Click "New app"
4. Selecciona:
   - **Repository**: `aura-solar-dashboard`
   - **Branch**: `main`
   - **Main file path**: `app.py`
5. Click "Advanced settings"

---

## ✅ PASO 4: Configurar Secrets (MUY IMPORTANTE)

En "Advanced settings" → "Secrets", pega esto:

```toml
# Contraseña del dashboard
APP_PASSWORD = "4UR4-2026"

# ID de tu Google Sheet
SPREADSHEET_ID = "16_PyHTmy0IIwv17iM018vlF-gD_rt3KqQKTYNy4A63s"

# Credenciales de Google (copia EXACTAMENTE el contenido de tu credentials.json)
[gcp_service_account]
type = "service_account"
project_id = "TU_PROJECT_ID"
private_key_id = "TU_PRIVATE_KEY_ID"
private_key = "-----BEGIN PRIVATE KEY-----\nTU_PRIVATE_KEY_AQUI\n-----END PRIVATE KEY-----\n"
client_email = "TU_CLIENT_EMAIL@developer.gserviceaccount.com"
client_id = "TU_CLIENT_ID"
auth_uri = "https://accounts.google.com/o/oauth2/auth"
token_uri = "https://oauth2.googleapis.com/token"
auth_provider_x509_cert_url = "https://www.googleapis.com/oauth2/v1/certs"
client_x509_cert_url = "TU_CERT_URL"
```

**⚠️ IMPORTANTE**: Copia el contenido EXACTO de tu `credentials.json` reemplazando los valores arriba.

---

## ✅ PASO 5: Modificar app.py para usar Secrets

En tu `app.py`, cambia estas líneas:

**ANTES:**
```python
SPREADSHEET_ID = os.environ.get('SPREADSHEET_ID', '16_PyHTmy0IIwv17iM018vlF-gD_rt3KqQKTYNy4A63s')
APP_PASSWORD = os.environ.get('APP_PASSWORD', '4UR4-2026')

# Load credentials
credentials_path = os.environ.get('GOOGLE_APPLICATION_CREDENTIALS', 'credentials.json')
credentials = ServiceAccountCredentials.from_json_keyfile_name(
    credentials_path, 
    ['https://spreadsheets.google.com/feeds', 'https://www.googleapis.com/auth/drive']
)
```

**DESPUÉS:**
```python
import streamlit as st

# Get configuration from Streamlit secrets
SPREADSHEET_ID = st.secrets["SPREADSHEET_ID"]
APP_PASSWORD = st.secrets["APP_PASSWORD"]

# Load credentials from secrets
credentials = ServiceAccountCredentials.from_json_keyfile_dict(
    st.secrets["gcp_service_account"],
    ['https://spreadsheets.google.com/feeds', 'https://www.googleapis.com/auth/drive']
)
```

---

## ✅ PASO 6: Deploy!

1. Click "Deploy"
2. Espera 2-5 minutos
3. ¡Tu dashboard estará en línea!

**URL será algo como:**
`https://aura-solar-dashboard-XXXXX.streamlit.app`

---

## 🔄 PASO 7: Actualizar después de cambios

Cuando hagas cambios al código:

**Con GitHub Desktop:**
1. Abre GitHub Desktop
2. Verás los cambios en la lista
3. Escribe un mensaje en "Summary" (ej: "Actualicé colores")
4. Click "Commit to main"
5. Click "Push origin"
6. ¡Streamlit Cloud se actualizará automáticamente en 1-2 minutos!

**Con comandos:**
```bash
git add .
git commit -m "Descripción del cambio"
git push
```

---

## 🎯 RESUMEN RÁPIDO

1. ✅ Preparar archivos
2. ✅ Subir a GitHub
3. ✅ Crear app en Streamlit Cloud
4. ✅ Configurar secrets
5. ✅ Modificar app.py para usar secrets
6. ✅ Deploy
7. ✅ Compartir URL con tu equipo

---

## 🆘 PROBLEMAS COMUNES

**Error: "Credentials not found"**
→ Revisa que copiaste bien el contenido de credentials.json en los secrets

**Error: "Module not found"**
→ Revisa que requirements.txt esté correcto

**Error: "Authentication failed"**
→ Verifica que el service account tenga acceso al Google Sheet

**Cambios no se reflejan**
→ En Streamlit Cloud, click "Manage app" → "Reboot app"

---

## 📞 SOPORTE

Si tienes problemas:
1. Revisa los logs en Streamlit Cloud (click en "Manage app" → "Logs")
2. Verifica que todos los secrets estén correctos
3. Confirma que credentials.json tenga acceso al Sheet

---

¡Listo! Tu dashboard estará accesible 24/7 desde cualquier lugar. 🚀
