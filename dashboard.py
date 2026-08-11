import os
import pandas as pd
from dash import Dash, dcc, html, Input, Output
import dash_bootstrap_components as dbc
import plotly.express as px

app = Dash(__name__, external_stylesheets=[dbc.themes.CYBORG])

def load_data():
    results_path = "results"
    met_path = os.path.join(results_path, "metricas_completas.csv")
    lat_path = os.path.join(results_path, "latencias_agregacion.csv")
    
    df_metrics = pd.read_csv(met_path) if os.path.exists(met_path) else pd.DataFrame()
    df_lat = pd.read_csv(lat_path) if os.path.exists(lat_path) else pd.DataFrame()
    return df_metrics, df_lat

app.layout = dbc.Container([
    html.H2("QSF-IDS", className="mt-4 mb-4 text-center", style={"fontFamily": "serif"}),
    
    dbc.Row([
        dbc.Col([
            dbc.Card([
                dbc.CardHeader("Comparativa de Métricas (F1, Accuracy)"),
                dbc.CardBody(dcc.Graph(id="metrics-graph"))
            ])
        ], md=6),
        dbc.Col([
            dbc.Card([
                dbc.CardHeader("Análisis de Latencia PQC (ms) por Ronda"),
                dbc.CardBody(dcc.Graph(id="lat-graph"))
            ])
        ], md=6)
    ], className="mb-4"),
    
    dcc.Interval(id='interval-component', interval=5000, n_intervals=0)
], fluid=True)

@app.callback(
    Output('metrics-graph', 'figure'),
    Output('lat-graph', 'figure'),
    Input('interval-component', 'n_intervals')
)
def update_graphs(n):
    df_metrics, df_lat = load_data()
    
    if not df_metrics.empty:
        df_melt = df_metrics.melt(id_vars="escenario", value_vars=["accuracy", "precision", "recall", "f1"])
        fig_metrics = px.bar(df_melt, x="variable", y="value", color="escenario", barmode="group", template="plotly_dark")
    else:
        fig_metrics = px.bar(title="Esperando datos...", template="plotly_dark")

    if not df_lat.empty:
        fig_lat = px.line(df_lat, x="ronda", y=["t_actual_mediana", "t_simulado_he"], markers=True, template="plotly_dark")
        fig_lat.update_yaxes(title="Tiempo (ms)")
    else:
        fig_lat = px.line(title="Esperando latencias...", template="plotly_dark")
        
    return fig_metrics, fig_lat

if __name__ == '__main__':
    app.run(debug=True, port=8050)