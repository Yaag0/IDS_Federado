import os
import base64
import dash
from dash import html, dcc
import dash_bootstrap_components as dbc

# Inicializamos la app con un tema oscuro (ideal para ciberseguridad)
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.CYBORG])
app.title = "IDS Dashboard"

def get_base64_image(image_path):
    """Carga una imagen local y la convierte a base64 para mostrarla en la web."""
    if not os.path.exists(image_path):
        return ""
    with open(image_path, 'rb') as f:
        encoded_image = base64.b64encode(f.read()).decode('utf-8')
    return f"data:image/png;base64,{encoded_image}"

# Cargar las imágenes generadas por tu main.py
cm_img = get_base64_image('reports/figures/confusion_matrix.png')
roc_img = get_base64_image('reports/figures/roc_curve.png')

app.layout = dbc.Container([
    dbc.Row([
        dbc.Col(html.H1("Panel de Control - IDS Federado PQC", className="text-center text-primary my-4"), width=12)
    ]),
    
    dbc.Row([
        dbc.Col(
            dbc.Card([
                dbc.CardHeader("Métricas del Modelo Global (Krum + PQC)", className="text-center fw-bold"),
                dbc.CardBody([
                    html.H4("Accuracy: 99.97%", className="text-success text-center"),
                    html.H5("Precision: 99.94%", className="text-center"),
                    html.H5("Recall: 100.00%", className="text-center"),
                    html.H5("F1-Score: 99.97%", className="text-center"),
                    html.P("El modelo es robusto ante envenenamiento y sus comunicaciones están aseguradas mediante cifrado híbrido Kyber+Dilithium+AES.", className="text-muted text-center mt-3")
                ])
            ], color="dark", outline=True),
            width=12, md=12, className="mb-4"
        )
    ]),

    dbc.Row([
        dbc.Col([
            dbc.Card([
                dbc.CardHeader("Matriz de Confusión", className="text-center"),
                dbc.CardBody(html.Img(src=cm_img, style={"width": "100%", "borderRadius": "5px"}))
            ], color="dark", outline=True)
        ], width=12, md=6, className="mb-4"),
        
        dbc.Col([
            dbc.Card([
                dbc.CardHeader("Curva ROC", className="text-center"),
                dbc.CardBody(html.Img(src=roc_img, style={"width": "100%", "borderRadius": "5px"}))
            ], color="dark", outline=True)
        ], width=12, md=6, className="mb-4")
    ]),
    
    dbc.Row([
        dbc.Col(html.Footer("Sistema de Detección de Intrusos - Arquitectura Segura y Distribuida", className="text-center text-muted my-4"), width=12)
    ])
], fluid=True)

if __name__ == '__main__':
    # Se ejecuta en el puerto 8050 por defecto
    app.run(debug=True)