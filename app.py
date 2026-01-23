"""
Aura Solar Daily Briefing - Google Cloud Run Version
=====================================================
Reads from current month sheet (e.g., "Enero" for January)
Separate PML per facility, larger fonts and charts
"""

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
from typing import Optional
import gspread
from google.oauth2 import service_account
from google.auth import default
import os

# =============================================================================
# PAGE CONFIGURATION
# =============================================================================
st.set_page_config(
    page_title="Aura Solar | Daily Briefing",
    page_icon="☀️",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# =============================================================================
# CONFIGURATION
# =============================================================================
# Try to use Streamlit secrets first (for cloud deployment), fall back to env vars (for local)
try:
    SPREADSHEET_ID = st.secrets["SPREADSHEET_ID"]
    APP_PASSWORD = st.secrets["APP_PASSWORD"]
except:
    SPREADSHEET_ID = os.environ.get('SPREADSHEET_ID', '16_PyHTmy0IIwv17iM018vlF-gD_rt3KqQKTYNy4A63s')
    APP_PASSWORD = os.environ.get('APP_PASSWORD', '4UR4-2026')

MONTH_SHEETS = {
    1: 'Enero', 2: 'Febrero', 3: 'Marzo', 4: 'Abril',
    5: 'Mayo', 6: 'Junio', 7: 'Julio', 8: 'Agosto',
    9: 'Septiembre', 10: 'Octubre', 11: 'Noviembre', 12: 'Diciembre'
}

PLANTS = {
    'aura_i': {'name': 'Aura Solar I', 'inverters': '37 en 19 CTs', 'budget_daily': 197, 'capacity_mw': 30},
    'aura_iii': {'name': 'Aura Solar III', 'inverters': '560 en 10 CTs', 'budget_daily': 169, 'capacity_mw': 30}
}

# =============================================================================
# CSS - LARGER FONTS
# =============================================================================
CUSTOM_CSS = """
<style>
    .stApp { background-color: #0a0f1a; }
    .main-header { display: flex; justify-content: space-between; align-items: center; padding: 1.5rem 0; border-bottom: 1px solid #374151; margin-bottom: 2rem; }
    .header-left { display: flex; align-items: center; gap: 1.5rem; }
    .logo { width: 64px; height: 64px; background: linear-gradient(135deg, #f59e0b 0%, #ea580c 100%); border-radius: 14px; display: flex; align-items: center; justify-content: center; font-size: 28px; box-shadow: 0 4px 20px rgba(245, 158, 11, 0.3); }
    .header-subtitle { color: #94a3b8; font-size: 16px; display: flex; align-items: center; gap: 10px; }
    .live-dot { width: 10px; height: 10px; background: #10b981; border-radius: 50%; animation: pulse 2s infinite; }
    @keyframes pulse { 0%, 100% { opacity: 1; } 50% { opacity: 0.5; } }
    .date-big { font-size: 38px; font-weight: 700; color: #f8fafc; }
    .date-month { color: #f59e0b; }
    .timestamp { font-size: 14px; color: #f8fafc; }
    .kpi-container { display: grid; grid-template-columns: repeat(5, 1fr); gap: 1.25rem; margin-bottom: 2rem; }
    .kpi-card { background: linear-gradient(135deg, #1a2234 0%, #1e293b 100%); border: 1px solid #374151; border-radius: 14px; padding: 1.5rem; text-align: center; }
    .kpi-card.highlight { background: rgba(245, 158, 11, 0.1); border-color: rgba(245, 158, 11, 0.3); }
    .kpi-label { font-size: 13px; text-transform: uppercase; letter-spacing: 1px; color: #f8fafc; margin-bottom: 10px; }
    .kpi-value { font-size: 34px; font-weight: 700; color: #f8fafc; line-height: 1; }
    .kpi-value.positive { color: #10b981; }
    .kpi-value.warning { color: #f59e0b; }
    .kpi-value.negative { color: #ef4444; }
    .kpi-unit { font-size: 14px; color: #f8fafc; margin-top: 6px; }
    .kpi-delta { font-size: 13px; margin-top: 10px; padding: 3px 10px; border-radius: 6px; display: inline-block; }
    .kpi-delta.up { background: rgba(16, 185, 129, 0.15); color: #10b981; }
    .kpi-delta.down { background: rgba(239, 68, 68, 0.15); color: #ef4444; }
    .kpi-delta.neutral { background: rgba(245, 158, 11, 0.15); color: #f59e0b; }
    .facility-card { background: rgba(0,0,0,0.2); border-radius: 14px; padding: 1.5rem; border: 1px solid transparent; transition: all 0.3s ease; }
    .facility-card:hover { border-color: #f59e0b; background: rgba(245, 158, 11, 0.05); }
    .facility-header { display: flex; justify-content: space-between; align-items: center; margin-bottom: 1.25rem; }
    .facility-name { font-weight: 600; font-size: 18px; color: #f8fafc; }
    .facility-badge { font-size: 12px; padding: 5px 12px; border-radius: 20px; background: rgba(245, 158, 11, 0.15); color: #f59e0b; }
    .metric-row { display: flex; justify-content: space-between; align-items: center; padding: 10px 0; border-bottom: 1px solid rgba(255,255,255,0.05); }
    .metric-label { font-size: 15px; color: #f8fafc; }
    .metric-value { font-weight: 600; font-size: 16px; color: #f8fafc; }
    .price-panel { display: grid; grid-template-columns: repeat(4, 1fr); gap: 1rem; margin-bottom: 1.5rem; }
    .price-card { background: rgba(0,0,0,0.2); border-radius: 10px; padding: 1rem; text-align: center; }
    .price-label { font-size: 12px; color: #f8fafc; margin-bottom: 6px; }
    .price-value { font-size: 22px; font-weight: 700; font-family: monospace; }
    .price-subtext { font-size: 11px; color: #f8fafc; margin-top: 4px; }
    .gcp-badge { display: inline-flex; align-items: center; gap: 6px; background: rgba(66, 133, 244, 0.15); color: #4285f4; padding: 5px 14px; border-radius: 20px; font-size: 12px; }
    .section-title { font-size: 22px; font-weight: 600; color: #f8fafc; margin-bottom: 1rem; }
    #MainMenu {visibility: hidden;} footer {visibility: hidden;} header {visibility: hidden;}
    @media (max-width: 1200px) { .kpi-container { grid-template-columns: repeat(3, 1fr); } }
    @media (max-width: 768px) { .kpi-container { grid-template-columns: repeat(2, 1fr); } }
</style>
"""

# =============================================================================
# AUTHENTICATION
# =============================================================================
def check_password() -> bool:
    if "authenticated" not in st.session_state:
        st.session_state.authenticated = False
    if st.session_state.authenticated:
        return True
    
    st.markdown("""<style>.login-container{max-width:400px;margin:80px auto;padding:40px;background:linear-gradient(135deg,#1a2234 0%,#1e293b 100%);border-radius:16px;border:1px solid #374151;text-align:center}.login-logo{font-size:64px;margin-bottom:16px}.login-title{font-size:24px;font-weight:700;color:#f8fafc;margin-bottom:8px}.login-subtitle{font-size:14px;color:#94a3b8;margin-bottom:32px}</style><div class="login-container"><div class="login-logo">☀️</div><div class="login-title">Aura Solar</div><div class="login-subtitle">Daily Operations Briefing</div></div>""", unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        password = st.text_input("Contraseña", type="password", key="pwd")
        if st.button("Ingresar", use_container_width=True, type="primary"):
            if password == APP_PASSWORD:
                st.session_state.authenticated = True
                st.rerun()
            else:
                st.error("❌ Contraseña incorrecta")
    return False

def get_google_credentials():
    """Get Google credentials from Streamlit secrets or environment."""
    try:
        # Try Streamlit secrets first (for cloud deployment)
        if "gcp_service_account" in st.secrets:
            return service_account.Credentials.from_service_account_info(
                st.secrets["gcp_service_account"],
                scopes=['https://www.googleapis.com/auth/spreadsheets.readonly', 
                       'https://www.googleapis.com/auth/drive.readonly']
            )
        
        # Fall back to environment variable (for local development)
        creds_json = os.environ.get('GOOGLE_APPLICATION_CREDENTIALS')
        if creds_json and os.path.isfile(creds_json):
            return service_account.Credentials.from_service_account_file(
                creds_json, 
                scopes=['https://www.googleapis.com/auth/spreadsheets.readonly', 
                       'https://www.googleapis.com/auth/drive.readonly']
            )
        
        # Try default credentials (for Cloud Run)
        credentials, _ = default(scopes=['https://www.googleapis.com/auth/spreadsheets.readonly', 
                                        'https://www.googleapis.com/auth/drive.readonly'])
        return credentials
    except Exception as e:
        st.error(f"Error de autenticación: {e}")
        return None

@st.cache_resource(ttl=3600)
def get_sheets_client():
    credentials = get_google_credentials()
    if credentials is None:
        return None
    try:
        return gspread.authorize(credentials)
    except Exception as e:
        st.error(f"Error conectando a Google Sheets: {e}")
        return None

# =============================================================================
# DATA LOADING
# =============================================================================
def get_current_month_sheet() -> str:
    return MONTH_SHEETS.get(datetime.now().month, 'Enero')

def find_column(df, possible_names, default=None):
    for name in possible_names:
        if name in df.columns:
            return name
    return default



def load_hourly_data_from_historical(_client, spreadsheet_id: str, target_date: datetime):
    """Load hourly data from 'Base de Datos Horarios' sheet for a specific date."""
    try:
        spreadsheet = _client.open_by_key(spreadsheet_id)
        worksheet = spreadsheet.worksheet("Base de Datos Horarios")
        
        # Get all values
        all_values = worksheet.get_all_values()
        
        # Find header row
        header_row_idx = None
        for i, row in enumerate(all_values):
            row_lower = [str(cell).lower().strip() for cell in row]
            if 'fecha' in row_lower:
                header_row_idx = i
                break
        
        if header_row_idx is None:
            print("No header found in Base de Datos Horarios")
            return None
        
        # Column structure:
        # E (idx 4) = Fecha (in format m/d/yyyy - American format)
        # F (idx 5) = Hora
        # G (idx 6) = PML MTR
        # H (idx 7) = PML MDA  
        # J (idx 9) = Pronóstico Aura I
        # K (idx 10) = Pronóstico Aura III
        # L (idx 11) = Producción Aura I
        # M (idx 12) = Producción Aura III
        
        # Format target date as m/d/yyyy (American format without leading zeros)
        target_date_str = f"{target_date.month}/{target_date.day}/{target_date.year}"
        
        print(f"Debug: Looking for date '{target_date_str}' in Base de Datos Horarios")
        
        hourly_data = []
        
        for row in all_values[header_row_idx + 1:]:
            if len(row) > 12:
                try:
                    # Column E (index 4) - Fecha
                    fecha_str = str(row[4]).strip() if len(row) > 4 else ''
                    if not fecha_str or fecha_str != target_date_str:
                        continue
                    
                    # Column F (index 5) - Hora
                    hora = str(row[5]).strip() if len(row) > 5 else '0'
                    try:
                        hora_num = int(float(hora))
                    except:
                        hora_num = 0
                    
                    def parse_val(val):
                        try:
                            # Remove spaces, commas, dollar signs
                            cleaned = str(val).strip().replace(',', '').replace('$', '').replace(' ', '')
                            return float(cleaned) if cleaned and cleaned != '' and cleaned != '-' else 0
                        except:
                            return 0
                    
                    hourly_data.append({
                        'hour': hora_num,
                        'pml_mtr': parse_val(row[6] if len(row) > 6 else 0),
                        'pml_mda': parse_val(row[7] if len(row) > 7 else 0),
                        'forecast_ai': parse_val(row[9] if len(row) > 9 else 0),
                        'forecast_aiii': parse_val(row[10] if len(row) > 10 else 0),
                        'actual_ai': parse_val(row[11] if len(row) > 11 else 0),
                        'actual_aiii': parse_val(row[12] if len(row) > 12 else 0)
                    })
                    
                except Exception as e:
                    continue
        
        print(f"Debug: Loaded {len(hourly_data)} hours from Base de Datos Horarios for {target_date_str}")
        return hourly_data if hourly_data else None
        
    except Exception as e:
        print(f"Error loading from Base de Datos Horarios: {e}")
        import traceback
        traceback.print_exc()
        return None


@st.cache_data(ttl=300)
def load_hourly_data(_client, spreadsheet_id: str, sheet_name: str = None):
    if sheet_name is None:
        sheet_name = get_current_month_sheet()
    try:
        spreadsheet = _client.open_by_key(spreadsheet_id)
        worksheet = spreadsheet.worksheet(sheet_name)
        
        # Get all values
        all_values = worksheet.get_all_values()
        
        if len(all_values) < 2:
            st.error("La hoja no tiene suficientes datos")
            return None
        
        # Find the header row by looking for 'Fecha' in any row
        header_row_idx = None
        for i, row in enumerate(all_values):
            # Check if this row contains 'Fecha' (case insensitive)
            row_lower = [str(cell).lower().strip() for cell in row]
            if 'fecha' in row_lower:
                header_row_idx = i
                break
        
        if header_row_idx is None:
            st.error(f"❌ No se encontró fila de encabezados con 'Fecha'")
            return None
        
        # Headers found - continue silently
        
        # Get headers from the found row
        headers = all_values[header_row_idx]
        
        # Find where the main table ends (before any secondary table)
        # Look for the first empty column after 'Fecha' to determine table width
        fecha_idx = next(i for i, h in enumerate(headers) if 'fecha' in str(h).lower().strip())
        
        # Find table end - look for large gap of empty columns
        table_end = len(headers)
        empty_count = 0
        for i in range(fecha_idx + 1, len(headers)):
            if str(headers[i]).strip() == '':
                empty_count += 1
                if empty_count >= 2:  # Two consecutive empty columns = end of table
                    table_end = i - 1
                    break
            else:
                empty_count = 0
        
        # Trim headers to main table only
        headers = headers[:table_end]
        
        # Clean headers - make unique
        seen = {}
        clean_headers = []
        for i, h in enumerate(headers):
            h = str(h).strip()
            if h == '':
                h = f'_empty_{i}'
            elif h in seen:
                seen[h] = seen.get(h, 0) + 1
                h = f'{h}_{seen[h]}'
            seen[h] = seen.get(h, 0)
            clean_headers.append(h)
        
        # Get data rows (everything after header row), trimmed to table width
        data_rows = [row[:table_end] for row in all_values[header_row_idx + 1:]]
        
        # Create DataFrame
        df = pd.DataFrame(data_rows, columns=clean_headers)
        
        # Remove empty columns
        df = df.loc[:, ~df.columns.str.startswith('_empty_')]
        
        # Find and standardize date column name
        date_col = None
        for col in df.columns:
            if 'fecha' in col.lower():
                date_col = col
                break
        
        if date_col and date_col != 'Fecha':
            df = df.rename(columns={date_col: 'Fecha'})
        
        # Convert date column
        df['Fecha'] = pd.to_datetime(df['Fecha'], errors='coerce')
        
        # Remove rows with no date
        df = df[df['Fecha'].notna()]
        
        # Convert ALL numeric columns - be comprehensive
        # First, convert Hora
        if 'Hora' in df.columns:
            df['Hora'] = pd.to_numeric(df['Hora'].astype(str).str.replace(',', ''), errors='coerce').fillna(0)
        
        # Convert all columns that look like they should be numeric
        for col in df.columns:
            if col in ['Fecha', 'Hora']:
                continue
            # Try to convert - if it fails, leave as string
            try:
                # Check if column has numeric-like values
                sample = df[col].astype(str).str.replace(',', '').str.replace('$', '').str.replace('-', '').str.strip()
                # If most non-empty values are numeric, convert the column
                non_empty = sample[sample != '']
                if len(non_empty) > 0:
                    numeric_count = pd.to_numeric(non_empty, errors='coerce').notna().sum()
                    if numeric_count / len(non_empty) > 0.5:  # More than 50% are numeric
                        df[col] = pd.to_numeric(
                            df[col].astype(str).str.replace(',', '').str.replace('$', '').str.replace('-', '').str.strip().replace('', '0'),
                            errors='coerce'
                        ).fillna(0)
            except:
                pass
        
        return df
        
    except gspread.exceptions.WorksheetNotFound:
        st.error(f"❌ Hoja '{sheet_name}' no encontrada.")
        return None
    except Exception as e:
        st.error(f"Error cargando datos: {e}")
        import traceback
        st.code(traceback.format_exc())
        return None

def get_latest_date(df):
    """Get the most recent date that has actual generation data."""
    try:
        # Try to find a column with generation data
        gen_col = None
        for col in df.columns:
            if 'generada' in col.lower() or 'entregada' in col.lower():
                gen_col = col
                break
        
        if gen_col is None:
            # Fallback: just get the max date
            return pd.to_datetime(df['Fecha'].max())
        
        df_valid = df[(df[gen_col].notna()) & (df[gen_col] > 0) & (df['Fecha'].notna())]
        return pd.to_datetime(df_valid['Fecha'].max()) if len(df_valid) > 0 else pd.to_datetime(df['Fecha'].max())
    except:
        return datetime.now()

def get_available_dates(df):
    """Get list of dates that have data."""
    try:
        dates = df[df['Fecha'].notna()]['Fecha'].dt.date.unique()
        return sorted(dates, reverse=True)
    except:
        return []



def process_historical_data(hourly_data_list, target_date):
    """Process hourly data from Base de Datos Horarios into dashboard format."""
    if not hourly_data_list:
        return None
    
    # Calculate totals
    total_ai = sum(h['actual_ai'] for h in hourly_data_list)
    total_aiii = sum(h['actual_aiii'] for h in hourly_data_list)
    forecast_ai_total = sum(h['forecast_ai'] for h in hourly_data_list)
    forecast_aiii_total = sum(h['forecast_aiii'] for h in hourly_data_list)
    
    # Convert kWh to MWh
    gen_i = total_ai / 1000
    gen_iii = total_aiii / 1000
    forecast_i = forecast_ai_total / 1000 if forecast_ai_total > 0 else PLANTS['aura_i']['budget_daily']
    forecast_iii = forecast_aiii_total / 1000 if forecast_aiii_total > 0 else PLANTS['aura_iii']['budget_daily']
    
    # Calculate PML averages (excluding zeros)
    pml_mtr_vals = [h['pml_mtr'] for h in hourly_data_list if h['pml_mtr'] > 0]
    pml_mda_vals = [h['pml_mda'] for h in hourly_data_list if h['pml_mda'] > 0]
    
    pml_mtr_avg = sum(pml_mtr_vals) / len(pml_mtr_vals) if pml_mtr_vals else 0
    pml_mda_avg = sum(pml_mda_vals) / len(pml_mda_vals) if pml_mda_vals else 0
    
    # Accuracy
    acc_i = min(gen_i/forecast_i, forecast_i/gen_i) if forecast_i > 0 and gen_i > 0 else 0.95
    acc_iii = min(gen_iii/forecast_iii, forecast_iii/gen_iii) if forecast_iii > 0 and gen_iii > 0 else 0.95
    
    # Build hourly data for charts
    hourly = []
    for h in hourly_data_list:
        hourly.append({
            'hour': h['hour'],
            'actual_i': h['actual_ai'] / 1000,  # kWh to MWh
            'forecast_i': h['forecast_ai'] / 1000,
            'actual_iii': h['actual_aiii'] / 1000,
            'forecast_iii': h['forecast_aiii'] / 1000,
            'pml_mtr_ai': h['pml_mtr'],
            'pml_mda_ai': h['pml_mda'],
            'pml_mtr_aiii': h['pml_mtr'],
            'pml_mda_aiii': h['pml_mda'],
        })
    
    return {
        'date': target_date,
        'aura_i': {
            **PLANTS['aura_i'], 
            'generated': gen_i, 
            'forecast': forecast_i, 
            'accuracy': acc_i, 
            'pml_mtr': pml_mtr_avg, 
            'pml_mda': pml_mda_avg,
            'curtailment': 5, 
            'notified_pct': 0.735
        },
        'aura_iii': {
            **PLANTS['aura_iii'], 
            'generated': gen_iii, 
            'forecast': forecast_iii, 
            'accuracy': acc_iii, 
            'pml_mtr': pml_mtr_avg, 
            'pml_mda': pml_mda_avg,
            'curtailment': 0, 
            'notified_pct': 0.810
        },
        'hourly': hourly,
        'last_updated': datetime.now(),
    }


def process_daily_data(df, target_date):
    df_day = df[df['Fecha'].dt.date == target_date.date()].copy()
    if len(df_day) == 0:
        return None
    df_day = df[df['Fecha'].dt.date == target_date.date()].copy()
    if len(df_day) == 0:
        return None
    
    # Based on actual column names from the sheet:
    # Aura I columns (in order): Energía Pronosticada, Energía Generada, Energía Entregada, 
    #                            Energía no notificada (@95% PML), Energía no notificada (@98% PML),
    #                            Ingreso a 0.95PML (MX$), Ingreso a 0.98PML (MX$), Total (MX$)
    # Then Aura III has similar columns
    
    # The columns we need - map from what we expect to what might exist
    # For generation, look for "Energía Generada" or "Energía Entregada"
    
    # Find Aura I generation column
    prod_ai = find_column(df_day, [
        'Energía Generada',  # Your actual column name
        'Energía Entregada',
        'Producción AI (kwh)', 
        'Energía Generada (kWh)'
    ])
    
    # For Aura III - columns might have "_1" suffix if duplicated
    prod_aiii = find_column(df_day, [
        'Energía Generada_1',  # Duplicate column renamed
        'Energía Entregada_1',
        'Producción AIII (kwh)',
        'Energía Generada (kWh)_1'
    ])
    
    # Forecast columns
    fc_ai = find_column(df_day, [
        'Energía Pronosticada',
        'Pronóstico AI', 
        'Energía Pronosticada (kWh)'
    ])
    
    fc_aiii = find_column(df_day, [
        'Energía Pronosticada_1',
        'Pronóstico AIII',
        'Energía Pronosticada (kWh)_1'
    ])
    
    # PML columns - these appear before the Aura sections
    pml_mtr = find_column(df_day, ['PML MTR', 'PML MTR (MXN/MWh)'])
    pml_mda = find_column(df_day, ['PML MDA', 'PML MDA (MXN/MWh)'])
    
    # Calculate generation (data is in kWh, convert to MWh)
    gen_i = df_day[prod_ai].sum() / 1000 if prod_ai and prod_ai in df_day.columns else 0
    gen_iii = df_day[prod_aiii].sum() / 1000 if prod_aiii and prod_aiii in df_day.columns else 0
    
    # If Aura III not found separately, try to estimate or use 0
    if gen_iii == 0 and prod_ai:
        # Check if there's a second generation column we missed
        gen_cols = [c for c in df_day.columns if 'generada' in c.lower() or 'entregada' in c.lower()]
        if len(gen_cols) >= 2:
            prod_aiii = gen_cols[1]
            gen_iii = df_day[prod_aiii].sum() / 1000
    
    forecast_i = df_day[fc_ai].sum() / 1000 if fc_ai and fc_ai in df_day.columns else PLANTS['aura_i']['budget_daily']
    forecast_iii = df_day[fc_aiii].sum() / 1000 if fc_aiii and fc_aiii in df_day.columns else PLANTS['aura_iii']['budget_daily']
    
    # If forecast columns not found, try alternative
    if forecast_i == 0:
        fc_cols = [c for c in df_day.columns if 'pronostic' in c.lower()]
        if len(fc_cols) >= 1:
            fc_ai = fc_cols[0]
            forecast_i = df_day[fc_ai].sum() / 1000
        if len(fc_cols) >= 2:
            fc_aiii = fc_cols[1]
            forecast_iii = df_day[fc_aiii].sum() / 1000
    
    def calc_avg(col):
        if col and col in df_day.columns:
            vals = df_day[col].replace(0, pd.NA).dropna()
            return vals.mean() if len(vals) > 0 else 0
        return 0
    
    pml_mtr_val = calc_avg(pml_mtr)
    pml_mda_val = calc_avg(pml_mda)
    
    # Accuracy calculation
    acc_i = min(gen_i/forecast_i, forecast_i/gen_i) if forecast_i > 0 and gen_i > 0 else 0.95
    acc_iii = min(gen_iii/forecast_iii, forecast_iii/gen_iii) if forecast_iii > 0 and gen_iii > 0 else 0.95
    
    # Build hourly data
    hourly = []
    for _, row in df_day.iterrows():
        hourly.append({
            'hour': int(row.get('Hora', 0)) if pd.notna(row.get('Hora')) else 0,
            'actual_i': (row.get(prod_ai, 0) or 0) / 1000 if prod_ai else 0,
            'forecast_i': (row.get(fc_ai, 0) or 0) / 1000 if fc_ai else 0,
            'actual_iii': (row.get(prod_aiii, 0) or 0) / 1000 if prod_aiii else 0,
            'forecast_iii': (row.get(fc_aiii, 0) or 0) / 1000 if fc_aiii else 0,
            'pml_mtr_ai': row.get(pml_mtr, 0) or 0 if pml_mtr else 0,
            'pml_mda_ai': row.get(pml_mda, 0) or 0 if pml_mda else 0,
            'pml_mtr_aiii': row.get(pml_mtr, 0) or 0 if pml_mtr else 0,  # Same PML for both
            'pml_mda_aiii': row.get(pml_mda, 0) or 0 if pml_mda else 0,
        })
    
    return {
        'date': target_date,
        'aura_i': {**PLANTS['aura_i'], 'generated': gen_i, 'forecast': forecast_i or PLANTS['aura_i']['budget_daily'], 'accuracy': acc_i, 'pml_mtr': pml_mtr_val, 'pml_mda': pml_mda_val, 'curtailment': 5, 'notified_pct': 0.735},
        'aura_iii': {**PLANTS['aura_iii'], 'generated': gen_iii, 'forecast': forecast_iii or PLANTS['aura_iii']['budget_daily'], 'accuracy': acc_iii, 'pml_mtr': pml_mtr_val, 'pml_mda': pml_mda_val, 'curtailment': 0, 'notified_pct': 0.810},
        'hourly': hourly,
        'last_updated': datetime.now(),
    }

def get_weekly_data(_client, spreadsheet_id: str, sheet_name: str, end_date):
    """Get last 7 days of daily generation totals from the summary table (columns W, Z, AA)."""
    try:
        spreadsheet = _client.open_by_key(spreadsheet_id)
        worksheet = spreadsheet.worksheet(sheet_name)
        
        # Get all values
        all_values = worksheet.get_all_values()
        
        # The summary table starts at row 24 (index 23)
        # Column W (index 22) = Día (day number)
        # Column Z (index 25) = Producción Aura I (MWh)
        # Column AA (index 26) = Producción Aura III (MWh)
        
        # Get the current month and year from end_date
        current_month = end_date.month
        current_year = end_date.year
        current_day = end_date.day
        
        # Calculate which days to include (last 7 days)
        target_days = []
        for i in range(7):
            day_offset = current_day - i
            if day_offset > 0:
                target_days.append(day_offset)
        
        result = []
        
        # Start reading from row 24 (index 23)
        for row in all_values[23:]:  # Row 24 onwards (where summary table starts)
            if len(row) > 26:
                try:
                    # Column W (index 22) - Day number
                    day_str = str(row[22]).strip()
                    if not day_str or day_str == '' or day_str == 'Día':
                        continue
                    
                    try:
                        day_num = int(day_str)
                    except:
                        continue
                    
                    # Only include the last 7 days
                    if day_num not in target_days:
                        continue
                    
                    # Column Z (index 25) - Aura I Production (MWh)
                    aura_i_val = str(row[25]).replace(',', '').replace('-', '0').strip()
                    try:
                        aura_i = float(aura_i_val) if aura_i_val and aura_i_val != '' else 0
                    except:
                        aura_i = 0
                    
                    # Column AA (index 26) - Aura III Production (MWh)
                    aura_iii_val = str(row[26]).replace(',', '').replace('-', '0').strip()
                    try:
                        aura_iii = float(aura_iii_val) if aura_iii_val and aura_iii_val != '' else 0
                    except:
                        aura_iii = 0
                    
                    # Create date object for this day
                    from datetime import datetime, date
                    date_obj = date(current_year, current_month, day_num)
                    
                    result.append({
                        'date': date_obj,
                        'aura_i': aura_i,
                        'aura_iii': aura_iii
                    })
                    
                except Exception as e:
                    continue
        
        # Sort by date (oldest to newest)
        result = sorted(result, key=lambda x: x['date'])
        
        return result
        
    except Exception as e:
        print(f"Error loading weekly data: {e}")
        import traceback
        traceback.print_exc()
        return []


def get_daily_summary_data(_client, spreadsheet_id: str, sheet_name: str, day_number: int):
    """Get summary data for a specific day from the daily summary table (columns W-AE)."""
    try:
        spreadsheet = _client.open_by_key(spreadsheet_id)
        worksheet = spreadsheet.worksheet(sheet_name)
        
        # Get all values
        all_values = worksheet.get_all_values()
        
        # The summary table starts at row 24 (index 23)
        # Column W (index 22) = Día
        # Column X (index 23) = Precio Promedio Aura I
        # Column Y (index 24) = Precio Promedio Aura III
        # Column Z (index 25) = Producción Aura I (MWh)
        # Column AA (index 26) = Producción Aura III (MWh)
        # Column AD (index 29) = Ingreso Diario Aura I (MX$)
        # Column AE (index 30) = Ingreso Diario Aura III (MX$)
        
        # Start reading from row 24 (index 23)
        for row in all_values[23:]:
            if len(row) > 30:
                try:
                    # Column W (index 22) - Day number
                    day_str = str(row[22]).strip()
                    if not day_str or day_str == '' or day_str == 'Día':
                        continue
                    
                    try:
                        day_num = int(day_str)
                    except:
                        continue
                    
                    if day_num != day_number:
                        continue
                    
                    # Column X (index 23) - Precio Promedio Aura I
                    precio_ai = str(row[23]).replace(',', '').replace('$', '').strip()
                    try:
                        precio_ai_val = float(precio_ai) if precio_ai and precio_ai != '' and precio_ai != '-' else 0
                    except:
                        precio_ai_val = 0
                    
                    # Column Y (index 24) - Precio Promedio Aura III
                    precio_aiii = str(row[24]).replace(',', '').replace('$', '').strip()
                    try:
                        precio_aiii_val = float(precio_aiii) if precio_aiii and precio_aiii != '' and precio_aiii != '-' else 0
                    except:
                        precio_aiii_val = 0
                    
                    # Column AD (index 29) - Ingreso Aura I
                    ingreso_ai = str(row[29]).replace(',', '').replace('$', '').strip()
                    try:
                        ingreso_ai_val = float(ingreso_ai) if ingreso_ai and ingreso_ai != '' and ingreso_ai != '-' else 0
                    except:
                        ingreso_ai_val = 0
                    
                    # Column AE (index 30) - Ingreso Aura III
                    ingreso_aiii = str(row[30]).replace(',', '').replace('$', '').strip()
                    try:
                        ingreso_aiii_val = float(ingreso_aiii) if ingreso_aiii and ingreso_aiii != '' and ingreso_aiii != '-' else 0
                    except:
                        ingreso_aiii_val = 0
                    
                    return {
                        'precio_ai': precio_ai_val,
                        'precio_aiii': precio_aiii_val,
                        'ingreso_ai': ingreso_ai_val,
                        'ingreso_aiii': ingreso_aiii_val
                    }
                    
                except Exception as e:
                    continue
        
        # If day not found, return zeros
        return {
            'precio_ai': 0,
            'precio_aiii': 0,
            'ingreso_ai': 0,
            'ingreso_aiii': 0
        }
        
    except Exception as e:
        print(f"Error loading daily summary: {e}")
        return {
            'precio_ai': 0,
            'precio_aiii': 0,
            'ingreso_ai': 0,
            'ingreso_aiii': 0
        }


def get_ytd_performance_ratio(_client, spreadsheet_id: str):
    """Get Performance Ratio data from YTD sheet for both plants and all years."""
    try:
        spreadsheet = _client.open_by_key(spreadsheet_id)
        worksheet = spreadsheet.worksheet("YTD")
        
        all_values = worksheet.get_all_values()
        
        # Performance Ratio Aura I: X11:Z25 (rows 10-24, cols 23-25)
        # Performance Ratio Aura III: X50:Z64 (rows 49-63, cols 23-25)
        
        # Column X (23) = Mes
        # Column Y (24) = 2026 (futuro - vacío)
        # Column Z (25) = 2025 (actual - con datos)
        
        pr_data = {
            'aura_i': {'months': [], '2025': [], '2026': []},
            'aura_iii': {'months': [], '2025': [], '2026': []}
        }
        
        # Read Aura I PR (rows 11-25, indices 10-24) - solo primeros 12
        for idx, row in enumerate(all_values[10:22]):  # Cambié 25 por 22 para solo 12 meses
            if len(row) > 25:
                month = str(row[23]).strip()
                if month and month != 'Mes' and month.lower() not in ['total', 'promedio', 'suma']:
                    pr_data['aura_i']['months'].append(month)
                    
                    # 2025 - Columna Z (índice 25)
                    val_2025 = str(row[25]).replace('%', '').replace(',', '.').strip()
                    try:
                        pr_data['aura_i']['2025'].append(float(val_2025) if val_2025 and val_2025 != '' else None)
                    except:
                        pr_data['aura_i']['2025'].append(None)
                    
                    # 2026 - Columna Y (índice 24)
                    val_2026 = str(row[24]).replace('%', '').replace(',', '.').strip()
                    try:
                        pr_data['aura_i']['2026'].append(float(val_2026) if val_2026 and val_2026 != '' else None)
                    except:
                        pr_data['aura_i']['2026'].append(None)
        
        # Read Aura III PR (rows 50-64, indices 49-63) - solo primeros 12
        for idx, row in enumerate(all_values[49:61]):  # Cambié 64 por 61 para solo 12 meses
            if len(row) > 25:
                month = str(row[23]).strip()
                if month and month != 'Mes' and month.lower() not in ['total', 'promedio', 'suma']:
                    pr_data['aura_iii']['months'].append(month)
                    
                    # 2025 - Columna Z (índice 25)
                    val_2025 = str(row[25]).replace('%', '').replace(',', '.').strip()
                    try:
                        pr_data['aura_iii']['2025'].append(float(val_2025) if val_2025 and val_2025 != '' else None)
                    except:
                        pr_data['aura_iii']['2025'].append(None)
                    
                    # 2026 - Columna Y (índice 24)
                    val_2026 = str(row[24]).replace('%', '').replace(',', '.').strip()
                    try:
                        pr_data['aura_iii']['2026'].append(float(val_2026) if val_2026 and val_2026 != '' else None)
                    except:
                        pr_data['aura_iii']['2026'].append(None)
        
        return pr_data
        
    except Exception as e:
        print(f"Error loading PR data: {e}")
        return None


def get_ytd_budget_vs_real(_client, spreadsheet_id: str):
    """Get Budget vs Real data from YTD sheet for both plants - Year 2026."""
    try:
        spreadsheet = _client.open_by_key(spreadsheet_id)
        worksheet = spreadsheet.worksheet("YTD")
        
        all_values = worksheet.get_all_values()
        
        # Budget Aura I: N11:V25 (rows 10-24, cols 13-21)
        # Real Aura I: D11:L25 (rows 10-24, cols 3-11)
        # Budget Aura III: N50:V64 (rows 49-63, cols 13-21)
        # Real Aura III: D50:L64 (rows 49-63, cols 3-11)
        
        # Columns for both Budget and Real:
        # D/N (3/13) = Mes
        # E/O (4/14) = Producción (MWh/m) - TOTAL MENSUAL
        # F/P (5/15) = Producción (MWh/d) - diaria
        # G/Q (6/16) = Producción (MWh/MWp)
        # H/R (7/17) = Energía @98% PML (%)
        # I/S (8/18) = Limitaciones (MWh)
        # J/T (9/19) = Limitaciones (%)
        # K/U (10/20) = Precio (MX$/MWh)
        # L/V (11/21) = Ingreso (MM MX$)
        
        data = {
            'aura_i': {
                'months': [],
                'budget': {'produccion_mwh': [], 'precio': [], 'ingreso': []},
                'real': {'produccion_mwh': [], 'precio': [], 'ingreso': []}
            },
            'aura_iii': {
                'months': [],
                'budget': {'produccion_mwh': [], 'precio': [], 'ingreso': []},
                'real': {'produccion_mwh': [], 'precio': [], 'ingreso': []}
            }
        }
        
        # Read Aura I (rows 11-22, indices 10-21) - solo 12 meses
        for row in all_values[10:22]:
            if len(row) > 21:
                month = str(row[3]).strip()
                if month and month != 'Mes' and month.lower() not in ['total', 'promedio', 'suma', '2026']:
                    data['aura_i']['months'].append(month)
                    
                    # Real - Producción TOTAL (MWh/m) col E (4)
                    real_prod = str(row[4]).replace(',', '').strip()
                    try:
                        data['aura_i']['real']['produccion_mwh'].append(float(real_prod) if real_prod and real_prod != '' else 0)
                    except:
                        data['aura_i']['real']['produccion_mwh'].append(0)
                    
                    # Real - Precio col K (10)
                    real_precio = str(row[10]).replace(',', '').replace('$', '').strip()
                    try:
                        data['aura_i']['real']['precio'].append(float(real_precio) if real_precio and real_precio != '' else 0)
                    except:
                        data['aura_i']['real']['precio'].append(0)
                    
                    # Real - Ingreso col L (11)
                    real_ingreso = str(row[11]).replace(',', '').replace('$', '').strip()
                    try:
                        data['aura_i']['real']['ingreso'].append(float(real_ingreso) if real_ingreso and real_ingreso != '' else 0)
                    except:
                        data['aura_i']['real']['ingreso'].append(0)
                    
                    # Budget - Producción TOTAL (MWh/m) col O (14)
                    budget_prod = str(row[14]).replace(',', '').strip()
                    try:
                        data['aura_i']['budget']['produccion_mwh'].append(float(budget_prod) if budget_prod and budget_prod != '' else 0)
                    except:
                        data['aura_i']['budget']['produccion_mwh'].append(0)
                    
                    # Budget - Precio col U (20)
                    budget_precio = str(row[20]).replace(',', '').replace('$', '').strip()
                    try:
                        data['aura_i']['budget']['precio'].append(float(budget_precio) if budget_precio and budget_precio != '' else 0)
                    except:
                        data['aura_i']['budget']['precio'].append(0)
                    
                    # Budget - Ingreso col V (21)
                    budget_ingreso = str(row[21]).replace(',', '').replace('$', '').strip()
                    try:
                        data['aura_i']['budget']['ingreso'].append(float(budget_ingreso) if budget_ingreso and budget_ingreso != '' else 0)
                    except:
                        data['aura_i']['budget']['ingreso'].append(0)
        
        # Read Aura III (rows 50-61, indices 49-60) - solo 12 meses
        for row in all_values[49:61]:
            if len(row) > 21:
                month = str(row[3]).strip()
                if month and month != 'Mes' and month.lower() not in ['total', 'promedio', 'suma', '2026']:
                    data['aura_iii']['months'].append(month)
                    
                    # Real - Producción TOTAL col E (4)
                    real_prod = str(row[4]).replace(',', '').strip()
                    try:
                        data['aura_iii']['real']['produccion_mwh'].append(float(real_prod) if real_prod and real_prod != '' else 0)
                    except:
                        data['aura_iii']['real']['produccion_mwh'].append(0)
                    
                    real_precio = str(row[10]).replace(',', '').replace('$', '').strip()
                    try:
                        data['aura_iii']['real']['precio'].append(float(real_precio) if real_precio and real_precio != '' else 0)
                    except:
                        data['aura_iii']['real']['precio'].append(0)
                    
                    real_ingreso = str(row[11]).replace(',', '').replace('$', '').strip()
                    try:
                        data['aura_iii']['real']['ingreso'].append(float(real_ingreso) if real_ingreso and real_ingreso != '' else 0)
                    except:
                        data['aura_iii']['real']['ingreso'].append(0)
                    
                    # Budget - Producción TOTAL col O (14)
                    budget_prod = str(row[14]).replace(',', '').strip()
                    try:
                        data['aura_iii']['budget']['produccion_mwh'].append(float(budget_prod) if budget_prod and budget_prod != '' else 0)
                    except:
                        data['aura_iii']['budget']['produccion_mwh'].append(0)
                    
                    budget_precio = str(row[20]).replace(',', '').replace('$', '').strip()
                    try:
                        data['aura_iii']['budget']['precio'].append(float(budget_precio) if budget_precio and budget_precio != '' else 0)
                    except:
                        data['aura_iii']['budget']['precio'].append(0)
                    
                    budget_ingreso = str(row[21]).replace(',', '').replace('$', '').strip()
                    try:
                        data['aura_iii']['budget']['ingreso'].append(float(budget_ingreso) if budget_ingreso and budget_ingreso != '' else 0)
                    except:
                        data['aura_iii']['budget']['ingreso'].append(0)
        
        return data
        
    except Exception as e:
        print(f"Error loading budget vs real data: {e}")
        return None


def get_pmls_bcs_data(_client, spreadsheet_id: str):
    """Get last 15 days of PMLs BCS and New Fortress data."""
    try:
        spreadsheet = _client.open_by_key(spreadsheet_id)
        
        # Try to find the sheet - try different possible names
        possible_names = [
            "PMLs BCS y New Fortress",
            "PMLs BCS",
            "PMLS BCS y New Fortress",
            "PMls BCS y New Fortress"
        ]
        
        worksheet = None
        for name in possible_names:
            try:
                worksheet = spreadsheet.worksheet(name)
                print(f"Debug: Found sheet '{name}'")
                break
            except:
                continue
        
        if worksheet is None:
            print("Error: Could not find PMLs BCS sheet with any expected name")
            print(f"Available sheets: {[ws.title for ws in spreadsheet.worksheets()]}")
            return []
        
        # Get all values
        all_values = worksheet.get_all_values()
        print(f"Debug: Sheet has {len(all_values)} rows")
        
        # Header is at row 2 (index 2), data starts at row 3 (index 3)
        # Skip the header detection and start from row 3
        header_row_idx = 2
        
        # Column C (index 2) = Fecha
        # Column E (index 4) = PML La Paz (07OLA-115)
        # Column P (index 15) = Componente de congestión Los Cabos (07TCB-115)
        # Column Q (index 16) = Generación New Fortress (MW)
        
        daily_data = {}
        rows_processed = 0
        
        for row in all_values[3:]:  # Start from row 3 (skip headers)
            if len(row) > 16:
                try:
                    # Get date from column C (index 2)
                    date_str = str(row[2]).strip() if len(row) > 2 else ''
                    if not date_str or date_str == '':
                        continue
                    
                    # Parse date
                    import re
                    date_str = re.sub(r'\s.*', '', date_str)
                    
                    # Column E (index 4) - PML La Paz
                    pml_paz = str(row[4]).replace(',', '').replace('$', '').replace('-', '0').strip() if len(row) > 4 else '0'
                    try:
                        pml_paz_num = float(pml_paz) if pml_paz and pml_paz != '' else 0
                    except:
                        pml_paz_num = 0
                    
                    # Column P (index 15) - Congestión Cabos (puede ser negativa)
                    cong_cabos = str(row[15]).replace(',', '').replace('$', '').strip() if len(row) > 15 else '0'
                    try:
                        cong_cabos_num = float(cong_cabos) if cong_cabos and cong_cabos != '' and cong_cabos != '-' else 0
                    except:
                        cong_cabos_num = 0
                    
                    # Column Q (index 16) - Gen New Fortress
                    gen_nf = str(row[16]).replace(',', '').replace('$', '').strip() if len(row) > 16 else '0'
                    try:
                        gen_nf_num = float(gen_nf) if gen_nf and gen_nf != '' and gen_nf != '-' else 0
                    except:
                        gen_nf_num = 0
                    
                    # Only add if PML La Paz has data or congestión is non-zero
                    if pml_paz_num > 0 or abs(cong_cabos_num) > 0:
                        # Accumulate daily data (in case there are multiple rows per day, we'll average)
                        if date_str not in daily_data:
                            daily_data[date_str] = {
                                'count': 0,
                                'pml_paz': 0,
                                'cong_cabos': 0,
                                'gen_nf': 0
                            }
                        
                        daily_data[date_str]['count'] += 1
                        daily_data[date_str]['pml_paz'] += pml_paz_num
                        daily_data[date_str]['cong_cabos'] += cong_cabos_num
                        daily_data[date_str]['gen_nf'] += gen_nf_num
                        rows_processed += 1
                    
                except Exception as e:
                    continue
        
        print(f"Debug: Processed {rows_processed} rows with data")
        
        # Calculate averages and prepare result
        result = []
        for date_str, data in daily_data.items():
            if data['count'] > 0:
                result.append({
                    'date': date_str,
                    'pml_paz': data['pml_paz'] / data['count'],
                    'cong_cabos': data['cong_cabos'] / data['count'],
                    'gen_nf': data['gen_nf'] / data['count']
                })
        
        print(f"Debug: Found {len(result)} days with PMLs BCS data")
        
        # Convert dates to datetime for proper sorting
        from datetime import datetime
        for item in result:
            try:
                # Parse date in format d/m/yyyy or m/d/yyyy
                date_parts = item['date'].split('/')
                if len(date_parts) == 3:
                    # Try both formats
                    try:
                        # Try d/m/yyyy first
                        item['date_obj'] = datetime(int(date_parts[2]), int(date_parts[1]), int(date_parts[0]))
                    except:
                        # Try m/d/yyyy
                        item['date_obj'] = datetime(int(date_parts[2]), int(date_parts[0]), int(date_parts[1]))
                else:
                    item['date_obj'] = datetime.now()
            except:
                item['date_obj'] = datetime.now()
        
        # Sort by actual date and get last 15 days with data
        result = sorted(result, key=lambda x: x['date_obj'], reverse=True)[:15]
        result.reverse()  # Oldest to newest for chart
        
        # Remove the date_obj helper field
        for item in result:
            item.pop('date_obj', None)
        
        return result
        
    except Exception as e:
        print(f"Error loading PMLs BCS data: {e}")
        import traceback
        traceback.print_exc()
        return []


def get_monthly_hourly_max(_client, spreadsheet_id: str, sheet_name: str):
    """Get hourly production from the day with maximum total generation for each plant."""
    try:
        spreadsheet = _client.open_by_key(spreadsheet_id)
        worksheet = spreadsheet.worksheet(sheet_name)
        
        # Get all values
        all_values = worksheet.get_all_values()
        
        # Find header row
        header_row_idx = None
        for i, row in enumerate(all_values):
            row_lower = [str(cell).lower().strip() for cell in row]
            if 'fecha' in row_lower or 'hora' in row_lower:
                header_row_idx = i
                break
        
        if header_row_idx is None:
            print("Error: No se encontró fila de encabezados")
            return {'aura_i': {}, 'aura_iii': {}}
        
        # Based on the sheet structure:
        # Column B (index 1) = Fecha
        # Column C (index 2) = Hora  
        # Column H (index 7) = Energía Generada Aura I (kWh)
        # Column P (index 15) = Energía Entregada Aura III (kWh)
        
        # First, calculate daily totals to find the maximum day for each plant
        daily_totals_ai = {}
        daily_totals_aiii = {}
        
        for row in all_values[header_row_idx + 1:]:
            if len(row) > 15:
                try:
                    # Get date from column B (index 1)
                    date_str = str(row[1]).strip() if len(row) > 1 else ''
                    if not date_str or date_str == '':
                        continue
                    
                    # Normalize date format - convert any format to just the date part
                    import re
                    date_str = re.sub(r'\s.*', '', date_str)  # Remove time part if exists
                    
                    # Column H (index 7) - Aura I - Energía Generada
                    val_ai = str(row[7]).replace(',', '').replace('$', '').replace('-', '0').strip() if len(row) > 7 else '0'
                    try:
                        val_ai_num = float(val_ai) if val_ai and val_ai != '' else 0
                    except:
                        val_ai_num = 0
                    
                    # Column P (index 15) - Aura III - Energía Entregada
                    val_aiii = str(row[15]).replace(',', '').replace('$', '').replace('-', '0').strip() if len(row) > 15 else '0'
                    try:
                        val_aiii_num = float(val_aiii) if val_aiii and val_aiii != '' else 0
                    except:
                        val_aiii_num = 0
                    
                    # Accumulate daily totals
                    if date_str not in daily_totals_ai:
                        daily_totals_ai[date_str] = 0
                        daily_totals_aiii[date_str] = 0
                    
                    daily_totals_ai[date_str] += val_ai_num
                    daily_totals_aiii[date_str] += val_aiii_num
                except Exception as e:
                    continue
        
        print(f"Debug: Found {len(daily_totals_ai)} days with data for Aura I")
        print(f"Debug: Found {len(daily_totals_aiii)} days with data for Aura III")
        
        if not daily_totals_ai or not daily_totals_aiii:
            print("Error: No se encontraron datos diarios")
            return {'aura_i': {}, 'aura_iii': {}}
        
        # Find the day with maximum generation for each plant
        max_day_ai = max(daily_totals_ai.items(), key=lambda x: x[1])[0] if daily_totals_ai else None
        max_day_aiii = max(daily_totals_aiii.items(), key=lambda x: x[1])[0] if daily_totals_aiii else None
        
        print(f"Debug: Max day for Aura I: {max_day_ai} ({daily_totals_ai.get(max_day_ai, 0):.0f} kWh)")
        print(f"Debug: Max day for Aura III: {max_day_aiii} ({daily_totals_aiii.get(max_day_aiii, 0):.0f} kWh)")
        
        # Now get hourly values for those maximum days
        result_ai = {}
        result_aiii = {}
        
        for row in all_values[header_row_idx + 1:]:
            if len(row) > 15:
                try:
                    # Date from column B (index 1)
                    date_str = str(row[1]).strip() if len(row) > 1 else ''
                    import re
                    date_str = re.sub(r'\s.*', '', date_str)
                    
                    # Hour from column C (index 2)
                    hour_str = str(row[2]).replace(',', '').strip() if len(row) > 2 else ''
                    if not hour_str:
                        continue
                    
                    try:
                        hour = int(float(hour_str))
                    except:
                        continue
                    
                    if hour < 0 or hour > 23:
                        continue
                    
                    # Get Aura I values for its max day
                    if date_str == max_day_ai:
                        val_ai = str(row[7]).replace(',', '').replace('$', '').replace('-', '0').strip() if len(row) > 7 else '0'
                        try:
                            val_ai_num = float(val_ai) if val_ai and val_ai != '' else 0
                            result_ai[hour] = val_ai_num / 1000  # Convert kWh to MWh
                        except:
                            pass
                    
                    # Get Aura III values for its max day
                    if date_str == max_day_aiii:
                        val_aiii = str(row[15]).replace(',', '').replace('$', '').replace('-', '0').strip() if len(row) > 15 else '0'
                        try:
                            val_aiii_num = float(val_aiii) if val_aiii and val_aiii != '' else 0
                            result_aiii[hour] = val_aiii_num / 1000  # Convert kWh to MWh
                        except:
                            pass
                except:
                    continue
        
        print(f"Debug: Loaded {len(result_ai)} hours for Aura I max day")
        print(f"Debug: Loaded {len(result_aiii)} hours for Aura III max day")
        
        return {'aura_i': result_ai, 'aura_iii': result_aiii}
    except Exception as e:
        print(f"Error getting monthly max: {e}")
        import traceback
        traceback.print_exc()
        return {'aura_i': {}, 'aura_iii': {}}

# =============================================================================
# CHARTS - LARGER
# =============================================================================
def create_generation_chart(hourly, monthly_max=None, plant_view="Ambas plantas"):
    df = pd.DataFrame(hourly)
    df = df[df['hour'] >= 6]
    
    fig = go.Figure()
    
    # Determine which plants to show
    show_aura_i = plant_view in ["Ambas plantas", "Solo Aura I"]
    show_aura_iii = plant_view in ["Ambas plantas", "Solo Aura III"]
    
    # Add traces for selected plants first (so max lines appear on top)
    if show_aura_i:
        fig.add_trace(go.Scatter(x=df['hour'], y=df['actual_i'], name='Aura I Real', fill='tozeroy', fillcolor='rgba(245, 158, 11, 0.2)', line=dict(color='#f59e0b', width=3), mode='lines+markers', marker=dict(size=8)))
        fig.add_trace(go.Scatter(x=df['hour'], y=df['forecast_i'], name='Aura I Pronóstico', line=dict(color='#f59e0b', width=2, dash='dash'), mode='lines'))
    
    if show_aura_iii:
        fig.add_trace(go.Scatter(x=df['hour'], y=df['actual_iii'], name='Aura III Real', fill='tozeroy', fillcolor='rgba(16, 185, 129, 0.2)', line=dict(color='#10b981', width=3), mode='lines+markers', marker=dict(size=8)))
        fig.add_trace(go.Scatter(x=df['hour'], y=df['forecast_iii'], name='Aura III Pronóstico', line=dict(color='#10b981', width=2, dash='dash'), mode='lines'))
    
    # Add monthly max reference lines on top with higher visibility
    if monthly_max and isinstance(monthly_max, dict):
        # Aura I max line
        if 'aura_i' in monthly_max and show_aura_i and len(monthly_max['aura_i']) > 0:
            max_hours_i = sorted([h for h in monthly_max['aura_i'].keys() if h >= 6 and h <= 18])
            max_values_i = [monthly_max['aura_i'].get(h, 0) for h in max_hours_i]
            if max_values_i and max(max_values_i) > 0:
                fig.add_trace(go.Scatter(
                    x=max_hours_i, y=max_values_i, name='Máx. Mes Aura I',
                    line=dict(color='rgba(255, 255, 255, 0.6)', width=2, dash='dot'),
                    mode='lines', showlegend=True
                ))
        
        # Aura III max line
        if 'aura_iii' in monthly_max and show_aura_iii and len(monthly_max['aura_iii']) > 0:
            max_hours_iii = sorted([h for h in monthly_max['aura_iii'].keys() if h >= 6 and h <= 18])
            max_values_iii = [monthly_max['aura_iii'].get(h, 0) for h in max_hours_iii]
            if max_values_iii and max(max_values_iii) > 0:
                fig.add_trace(go.Scatter(
                    x=max_hours_iii, y=max_values_iii, name='Máx. Mes Aura III',
                    line=dict(color='rgba(255, 255, 255, 0.6)', width=2, dash='dot'),
                    mode='lines', showlegend=True
                ))
    
    fig.update_layout(template='plotly_dark', paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', height=450,
        margin=dict(l=60, r=40, t=40, b=60), legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='center', x=0.5, font=dict(size=14, color='#f8fafc')),
        xaxis=dict(title=dict(text='Hora', font=dict(size=16)), gridcolor='rgba(55, 65, 81, 0.5)', tickvals=list(range(6, 19)), ticktext=[f'{h}:00' for h in range(6, 19)], tickfont=dict(size=14)),
        yaxis=dict(title=dict(text='MW', font=dict(size=16)), gridcolor='rgba(55, 65, 81, 0.5)', tickfont=dict(size=14)), hovermode='x unified')
    return fig

def create_price_chart(hourly, facility='aura_i'):
    df = pd.DataFrame(hourly)
    df = df[df['hour'] >= 6]
    
    mtr_col = f'pml_mtr_{facility.replace("aura_", "a")}'
    mda_col = f'pml_mda_{facility.replace("aura_", "a")}'
    if mtr_col not in df.columns:
        mtr_col, mda_col = 'pml_mtr_ai', 'pml_mda_ai'
    
    df = df[(df[mtr_col] > 0) | (df[mda_col] > 0)]
    if len(df) == 0:
        return go.Figure()
    
    df['spread'] = ((df[mtr_col] - df[mda_col]) / df[mda_col] * 100).fillna(0)
    
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    fig.add_trace(go.Scatter(x=df['hour'], y=df[mtr_col], name='MTR', fill='tozeroy', fillcolor='rgba(245, 158, 11, 0.15)', line=dict(color='#f59e0b', width=2)), secondary_y=False)
    fig.add_trace(go.Scatter(x=df['hour'], y=df[mda_col], name='MDA', fill='tozeroy', fillcolor='rgba(139, 92, 246, 0.15)', line=dict(color='#8b5cf6', width=2)), secondary_y=False)
    fig.add_trace(go.Scatter(x=df['hour'], y=df['spread'], name='Spread %', line=dict(color='#ef4444', width=2, dash='dot'), mode='lines+markers', marker=dict(size=6)), secondary_y=True)
    fig.add_hline(y=0, line_dash='dash', line_color='rgba(255,255,255,0.3)', secondary_y=True)
    
    fig.update_layout(template='plotly_dark', paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', height=350,
        margin=dict(l=60, r=60, t=40, b=60), legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='center', x=0.5, font=dict(size=13, color='#f8fafc')),
        xaxis=dict(gridcolor='rgba(55, 65, 81, 0.5)', tickfont=dict(size=13), title=dict(text='Hora', font=dict(size=14))), hovermode='x unified')
    fig.update_yaxes(title_text="PML (MX$/MWh)", tickformat='$,.0f', gridcolor='rgba(55, 65, 81, 0.5)', tickfont=dict(size=13), title_font=dict(size=14), secondary_y=False)
    fig.update_yaxes(title_text="Spread %", tickformat='.1f', tickfont=dict(size=13), title_font=dict(size=14), secondary_y=True)
    return fig

def create_weekly_chart(weekly):
    df = pd.DataFrame(weekly)
    fig = go.Figure()
    fig.add_trace(go.Bar(x=df['date'], y=df.get('aura_i', []), name='Aura I', marker_color='rgba(245, 158, 11, 0.8)'))
    if 'aura_iii' in df.columns:
        fig.add_trace(go.Bar(x=df['date'], y=df['aura_iii'], name='Aura III', marker_color='rgba(16, 185, 129, 0.8)'))
    
    fig.update_layout(template='plotly_dark', paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', height=400,
        margin=dict(l=60, r=40, t=40, b=60), legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='center', x=0.5, font=dict(size=14, color='#f8fafc')),
        xaxis=dict(gridcolor='rgba(55, 65, 81, 0.5)', tickfont=dict(size=13, color='#f8fafc')),
        yaxis=dict(title=dict(text='MWh', font=dict(size=16)), gridcolor='rgba(55, 65, 81, 0.5)', tickfont=dict(size=14)), barmode='group')
    return fig


def create_pmls_bcs_chart(pmls_data):
    """Create chart for PMLs BCS showing congestion, PML La Paz, and New Fortress generation."""
    if not pmls_data:
        return go.Figure()
    
    df = pd.DataFrame(pmls_data)
    
    # Format dates as día/mes/año
    df['date_formatted'] = df['date'].apply(lambda x: x if '/' not in x else 
        f"{x.split('/')[0]}/{x.split('/')[1]}/{x.split('/')[2]}")
    
    # Create figure with secondary y-axis
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    
    # Add PML La Paz (07OLA-115)
    fig.add_trace(
        go.Scatter(
            x=df['date_formatted'], 
            y=df['pml_paz'], 
            name='PML La Paz',
            line=dict(color='#10b981', width=2),
            mode='lines+markers',
            marker=dict(size=6)
        ),
        secondary_y=False
    )
    
    # Add Congestión Los Cabos (07TCB-115)
    fig.add_trace(
        go.Scatter(
            x=df['date_formatted'], 
            y=df['cong_cabos'], 
            name='Congestión Los Cabos',
            line=dict(color='#f59e0b', width=2),
            mode='lines+markers',
            marker=dict(size=6)
        ),
        secondary_y=False
    )
    
    # Add New Fortress generation on secondary axis
    fig.add_trace(
        go.Scatter(
            x=df['date_formatted'], 
            y=df['gen_nf'], 
            name='Gen. New Fortress',
            line=dict(color='#8b5cf6', width=2, dash='dash'),
            mode='lines+markers',
            marker=dict(size=6)
        ),
        secondary_y=True
    )
    
    # Update layout
    fig.update_layout(
        template='plotly_dark',
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        height=350,
        margin=dict(l=60, r=60, t=40, b=60),
        legend=dict(
            orientation='h',
            yanchor='bottom',
            y=1.02,
            xanchor='center',
            x=0.5,
            font=dict(size=13, color='#f8fafc')
        ),
        xaxis=dict(
            gridcolor='rgba(55, 65, 81, 0.5)',
            tickfont=dict(size=12, color='#f8fafc'),
            tickangle=-45
        ),
        hovermode='x unified'
    )
    
    # Update y-axes
    fig.update_yaxes(
        title_text="PML / Congestión (MX$/MWh)",
        tickformat='$,.0f',
        gridcolor='rgba(55, 65, 81, 0.5)',
        tickfont=dict(size=13),
        title_font=dict(size=14),
        secondary_y=False
    )
    
    fig.update_yaxes(
        title_text="Gen. New Fortress (MWh)",
        tickformat=',.0f',
        tickfont=dict(size=13),
        title_font=dict(size=14),
        secondary_y=True
    )
    
    return fig


def create_pr_chart(pr_data, plant_name):
    """Create Performance Ratio comparison chart for a plant"""
    if not pr_data:
        return None
    
    plant_key = 'aura_i' if 'I' in plant_name else 'aura_iii'
    data = pr_data[plant_key]
    
    fig = go.Figure()
    
    # Add 2025 trace
    fig.add_trace(go.Scatter(
        x=data['months'],
        y=data['2025'],
        name='2025',
        line=dict(color='#10b981', width=3),
        mode='lines+markers',
        marker=dict(size=8)
    ))
    
    # Add 2026 trace
    fig.add_trace(go.Scatter(
        x=data['months'],
        y=data['2026'],
        name='2026',
        line=dict(color='#f59e0b', width=3),
        mode='lines+markers',
        marker=dict(size=8)
    ))
    
    fig.update_layout(
        template='plotly_dark',
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        height=350,
        margin=dict(l=60, r=40, t=40, b=60),
        hovermode='x unified',
        showlegend=True,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="center",
            x=0.5,
            font=dict(color='#f8fafc', size=13)
        )
    )
    
    fig.update_xaxes(
        title=None,
        showgrid=True,
        gridwidth=1,
        gridcolor='rgba(255,255,255,0.1)',
        tickfont=dict(color='#f8fafc', size=12)
    )
    
    fig.update_yaxes(
        title_text="Performance Ratio (%)",
        title_font=dict(color='#f8fafc', size=13),
        showgrid=True,
        gridwidth=1,
        gridcolor='rgba(255,255,255,0.1)',
        tickfont=dict(color='#f8fafc', size=12),
        range=[40, 100]
    )
    
    return fig


def create_budget_vs_real_chart(ytd_data, plant_name, metric):
    """Create Budget vs Real comparison chart with LINES"""
    if not ytd_data:
        return None
    
    plant_key = 'aura_i' if 'I' in plant_name else 'aura_iii'
    data = ytd_data[plant_key]
    
    fig = go.Figure()
    
    # Determine which data to show based on metric
    if metric == 'produccion':
        real_data = data['real']['produccion_mwh']
        budget_data = data['budget']['produccion_mwh']
        y_title = "Producción (MWh/mes)"
        real_color = '#10b981'
        budget_color = '#8b5cf6'
    elif metric == 'precio':
        real_data = data['real']['precio']
        budget_data = data['budget']['precio']
        y_title = "Precio Promedio (MX$/MWh)"
        real_color = '#f59e0b'
        budget_color = '#8b5cf6'
    else:  # ingreso
        real_data = data['real']['ingreso']
        budget_data = data['budget']['ingreso']
        y_title = "Ingreso (MM MX$)"
        real_color = '#10b981'
        budget_color = '#8b5cf6'
    
    # Add Budget line
    fig.add_trace(go.Scatter(
        x=data['months'],
        y=budget_data,
        name='Presupuesto 2026',
        line=dict(color=budget_color, width=3, dash='dash'),
        mode='lines+markers',
        marker=dict(size=8)
    ))
    
    # Add Real line
    fig.add_trace(go.Scatter(
        x=data['months'],
        y=real_data,
        name='Real 2026',
        line=dict(color=real_color, width=3),
        mode='lines+markers',
        marker=dict(size=8)
    ))
    
    fig.update_layout(
        template='plotly_dark',
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        height=350,
        margin=dict(l=60, r=40, t=40, b=60),
        hovermode='x unified',
        showlegend=True,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="center",
            x=0.5,
            font=dict(color='#f8fafc', size=13)
        )
    )
    
    fig.update_xaxes(
        title=None,
        showgrid=True,
        gridwidth=1,
        gridcolor='rgba(255,255,255,0.1)',
        tickfont=dict(color='#f8fafc', size=12)
    )
    
    fig.update_yaxes(
        title_text=y_title,
        title_font=dict(color='#f8fafc', size=13),
        showgrid=True,
        gridwidth=1,
        gridcolor='rgba(255,255,255,0.1)',
        tickfont=dict(color='#f8fafc', size=12)
    )
    
    return fig



    
    return fig

# =============================================================================
# MAIN APPLICATION
# =============================================================================
def main():
    st.markdown(CUSTOM_CSS, unsafe_allow_html=True)
    
    if not check_password():
        return
    
    client = get_sheets_client()
    if client is None:
        st.error("❌ No se pudo conectar a Google Sheets")
        st.stop()
    
    # Date selector - show calendar first
    col_date1, col_date2, col_date3 = st.columns([2, 1, 2])
    with col_date2:
        # Default to yesterday (since today might not have data yet)
        default_date = (datetime.now() - timedelta(days=1)).date()
        
        selected_date = st.date_input(
            "📅 Seleccionar fecha",
            value=default_date,
            min_value=datetime(2020, 1, 1).date(),
            max_value=datetime.now().date(),
            key="date_selector"
        )
        target_date = datetime.combine(selected_date, datetime.min.time())
    
    # Load data from Base de Datos Horarios
    with st.spinner(f"Cargando datos para {target_date.strftime('%d/%m/%Y')}..."):
        hourly_data_list = load_hourly_data_from_historical(client, SPREADSHEET_ID, target_date)
    
    if not hourly_data_list:
        st.error(f"❌ No se encontraron datos para {target_date.strftime('%d/%m/%Y')}")
        st.stop()
    
    # Process the data
    data = process_historical_data(hourly_data_list, target_date)
    
    if data is None:
        st.warning(f"No hay datos para {target_date.strftime('%Y-%m-%d')}. Selecciona otra fecha.")
        st.stop()
    
    # Determine current sheet name for other functions that need it
    current_sheet = MONTH_SHEETS.get(target_date.month, 'Enero')
    
    weekly = get_weekly_data(client, SPREADSHEET_ID, current_sheet, target_date)
    monthly_max = get_monthly_hourly_max(client, SPREADSHEET_ID, current_sheet)  # For reference line in chart
    pmls_bcs_data = get_pmls_bcs_data(client, SPREADSHEET_ID)  # Load PMLs BCS data
    
    # Load daily summary data for prices and revenue
    daily_summary = get_daily_summary_data(client, SPREADSHEET_ID, current_sheet, target_date.day)
    
    # Load YTD data for annual comparisons
    pr_data = get_ytd_performance_ratio(client, SPREADSHEET_ID)
    budget_vs_real = get_ytd_budget_vs_real(client, SPREADSHEET_ID)
    
    # Debug: Show what monthly_max contains
    if monthly_max:
        ai_hours = len(monthly_max.get('aura_i', {}))
        aiii_hours = len(monthly_max.get('aura_iii', {}))
        if ai_hours > 0 or aiii_hours > 0:
            st.info(f"✓ Máximos mensuales cargados: Aura I ({ai_hours} horas), Aura III ({aiii_hours} horas)")
        else:
            st.warning("⚠️ No se pudieron cargar los máximos mensuales")
    
    ai, aiii = data['aura_i'], data['aura_iii']
    
    total_gen = ai['generated'] + aiii['generated']
    total_budget = ai['budget_daily'] + aiii['budget_daily']
    avg_accuracy = (ai['accuracy'] + aiii['accuracy']) / 2
    budget_delta = ((total_gen / total_budget) - 1) * 100 if total_budget > 0 else 0
    
    # Use prices from daily summary table instead of hourly averages
    precio_ai = daily_summary['precio_ai']
    precio_aiii = daily_summary['precio_aiii']
    avg_pml = (precio_ai + precio_aiii) / 2 if precio_ai > 0 and precio_aiii > 0 else max(precio_ai, precio_aiii)
    
    # Use revenue from daily summary table
    revenue_ai = daily_summary['ingreso_ai']
    revenue_aiii = daily_summary['ingreso_aiii']
    revenue_total = revenue_ai + revenue_aiii
    
    # Keep old calculations for spread (using hourly data)
    avg_pml_mtr = (ai['pml_mtr'] + aiii['pml_mtr']) / 2 if ai['pml_mtr'] > 0 and aiii['pml_mtr'] > 0 else max(ai['pml_mtr'], aiii['pml_mtr'])
    avg_pml_mda = (ai['pml_mda'] + aiii['pml_mda']) / 2 if ai['pml_mda'] > 0 and aiii['pml_mda'] > 0 else max(ai['pml_mda'], aiii['pml_mda'])
    spread_pct = ((avg_pml_mtr - avg_pml_mda) / avg_pml_mda * 100) if avg_pml_mda > 0 else 0
    
    months_es = ['enero', 'febrero', 'marzo', 'abril', 'mayo', 'junio', 'julio', 'agosto', 'septiembre', 'octubre', 'noviembre', 'diciembre']
    
    # HEADER
    st.markdown(f"""
    <div class="main-header">
        <div class="header-left">
            <div class="logo">☀️</div>
            <div>
                <h1 style="font-size: 32px; font-weight: 700; color: #f8fafc; margin: 0;">Aura Solar Daily Briefing</h1>
                <div class="header-subtitle">
                    <span class="live-dot"></span> La Paz, BCS
                    <span class="gcp-badge">☁️ Google Cloud</span>
                    <span style="color: #f8fafc; font-size: 13px;">Datos: {current_sheet}</span>
                </div>
            </div>
        </div>
        <div style="text-align: right;">
            <div class="date-big">{data['date'].day} <span class="date-month">{months_es[data['date'].month-1]}</span> {data['date'].year}</div>
            <div class="timestamp">Actualizado: {data['last_updated'].strftime('%H:%M')} hrs</div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # KPIs
    pml_label = "PML Promedio" if avg_pml > 0 else "PML (N/D)"
    
    # Pre-calculate display values to avoid complex f-string expressions
    if avg_pml > 0:
        pml_value_display = f"${avg_pml:,.0f}"
        pml_unit_display = "MX$/MWh"
    else:
        pml_value_display = "N/D"
        pml_unit_display = "Datos pendientes"
    
    # Show spread only if we have MTR data
    show_spread = avg_pml_mtr > 0
    spread_style = "display: none;" if not show_spread else ""
    
    if show_spread:
        spread_value_display = f"{spread_pct:+.1f}%"
        spread_unit_display = "Favorable" if spread_pct >= 0 else "Desfavorable"
    else:
        spread_value_display = "N/D"
        spread_unit_display = "Datos pendientes"
    
    st.markdown(f"""
    <div class="kpi-container">
        <div class="kpi-card highlight">
            <div class="kpi-label">Generación Total</div>
            <div class="kpi-value">{total_gen:,.0f}</div>
            <div class="kpi-unit">MWh</div>
            <div class="kpi-delta {'up' if budget_delta >= 0 else 'down'}">{'↑' if budget_delta >= 0 else '↓'} {budget_delta:+.1f}% vs budget</div>
        </div>
        <div class="kpi-card">
            <div class="kpi-label">Asertividad</div>
            <div class="kpi-value positive">{avg_accuracy*100:.1f}%</div>
            <div class="kpi-unit">forecast accuracy</div>
            <div class="kpi-delta up">{'✓ OK' if avg_accuracy >= 0.95 else '⚠️'}</div>
        </div>
        <div class="kpi-card">
            <div class="kpi-label">{pml_label}</div>
            <div class="kpi-value">{pml_value_display}</div>
            <div class="kpi-unit">{pml_unit_display}</div>
            <div class="kpi-delta neutral">AI: ${precio_ai:,.0f} | III: ${precio_aiii:,.0f}</div>
        </div>
        <div class="kpi-card" style="{spread_style}">
            <div class="kpi-label">Spread MTR-MDA</div>
            <div class="kpi-value {'positive' if spread_pct >= 0 else 'negative'}">{spread_value_display}</div>
            <div class="kpi-unit">{spread_unit_display}</div>
            <div class="kpi-delta {'up' if spread_pct >= 0 else 'down'}">{'↑' if spread_pct >= 0 else '↓'} vs MDA</div>
        </div>
        <div class="kpi-card">
            <div class="kpi-label">Ingreso Diario</div>
            <div class="kpi-value positive">${revenue_total/1000:,.0f}K</div>
            <div class="kpi-unit">MXN</div>
            <div class="kpi-delta neutral">AI: ${revenue_ai/1000:,.0f}K | III: ${revenue_aiii/1000:,.0f}K</div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # ROW 1
    col1, col2 = st.columns([1, 1.5])
    
    with col1:
        st.markdown('<div class="section-title">⚡ Performance por Planta</div>', unsafe_allow_html=True)
        fc1, fc2 = st.columns(2)
        
        with fc1:
            st.markdown(f"""
            <div class="facility-card">
                <div class="facility-header"><span class="facility-name">{ai['name']}</span><span class="facility-badge">{ai['inverters']}</span></div>
                <div class="metric-row"><span class="metric-label">Budget</span><span class="metric-value">{ai['budget_daily']} MWh</span></div>
                <div class="metric-row"><span class="metric-label">Generada</span><span class="metric-value" style="color: #10b981">{ai['generated']:.0f} MWh</span></div>
                <div class="metric-row"><span class="metric-label">Asertividad</span><span class="metric-value">{ai['accuracy']*100:.0f}%</span></div>
                <div class="metric-row"><span class="metric-label">PML MTR</span><span class="metric-value" style="color: #f59e0b">${ai['pml_mtr']:,.0f}</span></div>
                <div class="metric-row"><span class="metric-label">PML MDA</span><span class="metric-value" style="color: #8b5cf6">${ai['pml_mda']:,.0f}</span></div>
            </div>
            """, unsafe_allow_html=True)
            st.progress(ai['notified_pct'], text=f"Notificada: {ai['notified_pct']*100:.1f}%")
        
        with fc2:
            st.markdown(f"""
            <div class="facility-card">
                <div class="facility-header"><span class="facility-name">{aiii['name']}</span><span class="facility-badge">{aiii['inverters']}</span></div>
                <div class="metric-row"><span class="metric-label">Budget</span><span class="metric-value">{aiii['budget_daily']} MWh</span></div>
                <div class="metric-row"><span class="metric-label">Generada</span><span class="metric-value" style="color: #10b981">{aiii['generated']:.0f} MWh</span></div>
                <div class="metric-row"><span class="metric-label">Asertividad</span><span class="metric-value">{aiii['accuracy']*100:.0f}%</span></div>
                <div class="metric-row"><span class="metric-label">PML MTR</span><span class="metric-value" style="color: #f59e0b">${aiii['pml_mtr']:,.0f}</span></div>
                <div class="metric-row"><span class="metric-label">PML MDA</span><span class="metric-value" style="color: #8b5cf6">${aiii['pml_mda']:,.0f}</span></div>
            </div>
            """, unsafe_allow_html=True)
            st.progress(aiii['notified_pct'], text=f"Notificada: {aiii['notified_pct']*100:.1f}%")
    
    with col2:
        st.markdown('<div class="section-title">📈 Generación Horaria</div>', unsafe_allow_html=True)
        
        # Add plant selector dropdown
        plant_view = st.selectbox(
            "Vista:",
            options=["Ambas plantas", "Solo Aura I", "Solo Aura III"],
            key="plant_selector",
            label_visibility="collapsed"
        )
        
        st.plotly_chart(create_generation_chart(data['hourly'], monthly_max, plant_view), use_container_width=True, config={'displayModeBar': False})
    
    # ROW 2
    col3, col4 = st.columns([1.2, 1])
    
    with col3:
        st.markdown('<div class="section-title">💰 Precios Marginales Locales</div>', unsafe_allow_html=True)
        
        # Use MDA for peak instead of MTR
        peak_data = max(data['hourly'], key=lambda x: x.get('pml_mda_ai', 0) or 0)
        peak_price, peak_hour = peak_data.get('pml_mda_ai', 0), peak_data.get('hour', 0)
        
        # Check if MTR data is available
        mtr_available = avg_pml_mtr > 0
        mtr_display = f"${avg_pml_mtr:,.0f}" if mtr_available else "N/D"
        mda_display = f"${avg_pml_mda:,.0f}" if avg_pml_mda > 0 else "N/D"
        spread_display = f"{spread_pct:+.1f}%" if mtr_available and avg_pml_mda > 0 else "N/D"
        peak_display = f"${peak_price:,.0f}" if peak_price > 0 else "N/D"
        subtext_mtr = "MX$/MWh • Tiempo Real" if mtr_available else "Datos no disponibles"
        subtext_spread = "Favorable" if spread_pct >= 0 else "Desfavorable" if mtr_available else "Pendiente"
        
        st.markdown(f"""
        <div class="price-panel">
            <div class="price-card"><div class="price-label">MTR Promedio</div><div class="price-value" style="color: {'#f59e0b' if mtr_available else '#f8fafc'}">{mtr_display}</div><div class="price-subtext">{subtext_mtr}</div></div>
            <div class="price-card"><div class="price-label">MDA Promedio</div><div class="price-value" style="color: #8b5cf6">{mda_display}</div><div class="price-subtext">MX$/MWh • Día Adelante</div></div>
            <div class="price-card"><div class="price-label">Spread MTR-MDA</div><div class="price-value" style="color: {'#10b981' if spread_pct >= 0 and mtr_available else '#f8fafc'}">{spread_display}</div><div class="price-subtext">{subtext_spread}</div></div>
            <div class="price-card"><div class="price-label">Pico del Día</div><div class="price-value" style="color: {'#10b981' if peak_price > 0 else '#f8fafc'}">{peak_display}</div><div class="price-subtext">{peak_hour:02d}:00 hrs</div></div>
        </div>
        """, unsafe_allow_html=True)
        
        if mtr_available:
            st.plotly_chart(create_price_chart(data['hourly'], 'aura_i'), use_container_width=True, config={'displayModeBar': False})
        else:
            st.info("📊 Datos de MTR no disponibles aún. Se actualizarán cuando estén disponibles.")
    
    with col4:
        st.markdown('<div class="section-title">📊 Tendencia Semanal</div>', unsafe_allow_html=True)
        if weekly:
            st.plotly_chart(create_weekly_chart(weekly), use_container_width=True, config={'displayModeBar': False})
        else:
            st.info("Datos semanales no disponibles")
    
    # ROW 3 - PMLs BCS y New Fortress
    st.markdown('<div class="section-title">🌊 PMLs BCS y New Fortress (Últimos 15 días)</div>', unsafe_allow_html=True)
    if pmls_bcs_data:
        st.plotly_chart(create_pmls_bcs_chart(pmls_bcs_data), use_container_width=True, config={'displayModeBar': False})
    else:
        st.info("Datos de PMLs BCS no disponibles")
    
    # ROW 4 - ANÁLISIS ANUAL
    st.markdown("---")
    st.markdown('<div class="section-title" style="font-size: 26px; margin-top: 2rem;">📈 Análisis Año a Año</div>', unsafe_allow_html=True)
    
    # Performance Ratio Comparison
    st.markdown('<div class="section-title">⚡ Performance Ratio - Comparativa Anual</div>', unsafe_allow_html=True)
    if pr_data:
        col_pr1, col_pr2 = st.columns(2)
        with col_pr1:
            st.markdown("**Aura Solar I**")
            st.plotly_chart(create_pr_chart(pr_data, 'Aura I'), use_container_width=True, config={'displayModeBar': False}, key='pr_aura_i')
        with col_pr2:
            st.markdown("**Aura Solar III**")
            st.plotly_chart(create_pr_chart(pr_data, 'Aura III'), use_container_width=True, config={'displayModeBar': False}, key='pr_aura_iii')
    else:
        st.info("Datos de Performance Ratio no disponibles")
    
    # Budget vs Real Comparison - Aura I
    st.markdown('<div class="section-title" style="margin-top: 2rem;">💰 Presupuesto vs Real - Aura Solar I</div>', unsafe_allow_html=True)
    if budget_vs_real:
        col_b1, col_b2, col_b3 = st.columns(3)
        with col_b1:
            st.markdown("**Producción (MWh/mes)**")
            st.plotly_chart(create_budget_vs_real_chart(budget_vs_real, 'Aura I', 'produccion'), use_container_width=True, config={'displayModeBar': False}, key='budget_ai_prod')
        with col_b2:
            st.markdown("**Precio Promedio (MX$/MWh)**")
            st.plotly_chart(create_budget_vs_real_chart(budget_vs_real, 'Aura I', 'precio'), use_container_width=True, config={'displayModeBar': False}, key='budget_ai_precio')
        with col_b3:
            st.markdown("**Ingreso (MM MX$)**")
            st.plotly_chart(create_budget_vs_real_chart(budget_vs_real, 'Aura I', 'ingreso'), use_container_width=True, config={'displayModeBar': False}, key='budget_ai_ingreso')
    else:
        st.info("Datos de presupuesto vs real no disponibles")
    
    # Budget vs Real Comparison - Aura III
    st.markdown('<div class="section-title" style="margin-top: 2rem;">💰 Presupuesto vs Real - Aura Solar III</div>', unsafe_allow_html=True)
    if budget_vs_real:
        col_b4, col_b5, col_b6 = st.columns(3)
        with col_b4:
            st.markdown("**Producción (MWh/mes)**")
            st.plotly_chart(create_budget_vs_real_chart(budget_vs_real, 'Aura III', 'produccion'), use_container_width=True, config={'displayModeBar': False}, key='budget_aiii_prod')
        with col_b5:
            st.markdown("**Precio Promedio (MX$/MWh)**")
            st.plotly_chart(create_budget_vs_real_chart(budget_vs_real, 'Aura III', 'precio'), use_container_width=True, config={'displayModeBar': False}, key='budget_aiii_precio')
        with col_b6:
            st.markdown("**Ingreso (MM MX$)**")
            st.plotly_chart(create_budget_vs_real_chart(budget_vs_real, 'Aura III', 'ingreso'), use_container_width=True, config={'displayModeBar': False}, key='budget_aiii_ingreso')
    else:
        st.info("Datos de presupuesto vs real no disponibles")
    
    # FOOTER
    st.markdown("---")
    c1, c2 = st.columns([3, 1])
    with c1:
        st.caption(f"☁️ Google Cloud Run • 🔒 Secure API • ⚡ Auto-refresh 5 min • 📊 Hoja: {current_sheet}")
    with c2:
        if st.button("🚪 Cerrar Sesión"):
            st.session_state.authenticated = False
            st.rerun()

if __name__ == "__main__":
    main()

