import dash
from dash import dcc, html
from dash.dependencies import Input, Output
import dash_bootstrap_components as dbc
import plotly.graph_objs as go
import pandas as pd
import os

app = dash.Dash(__name__, external_stylesheets=[dbc.themes.CYBORG])
app.title = "Quantum IDS - Monitor en Tiempo Real"

app.layout = dbc.Container([
    dbc.Row([
        dbc.Col(html.H1("Quantum Security IDS - Aprendizaje Federado", className="text-center text-primary mt-4 mb-4"), width=12)
    ]),
    
    # Componente de intervalo que se dispara cada 2 segundos para actualizar los datos
    dcc.Interval(
        id='interval-component',
        interval=2*1000, # en milisegundos (2 segundos)
        n_intervals=0
    ),
    
    dbc.Row([
        dbc.Col(
            dbc.Card([
                dbc.CardHeader("Métricas de Nodos Locales (En Vivo)", className="fw-bold"),
                dbc.CardBody([dcc.Graph(id='live-loss-graph')])
            ], className="shadow-lg"), 
            width=12
        )
    ]),
    
    dbc.Row([
        dbc.Col(html.P("Estado del Túnel: Activo (Kyber-768 + AES-256-GCM)", className="text-success text-center mt-3"), width=12)
    ])
], fluid=True)

# Callback para actualizar la gráfica automáticamente leyendo el archivo CSV
@app.callback(
    Output('live-loss-graph', 'figure'),
    [Input('interval-component', 'n_intervals')]
)
def update_graph_live(n):
    fig = go.Figure()
    
    # Verificamos si el archivo de métricas ya existe
    if os.path.exists("metrics.csv"):
        df = pd.read_csv("metrics.csv")
        
        if not df.empty:
            rondas = df['Ronda']
            
            fig.add_trace(go.Scatter(x=rondas, y=df['Cliente_0'], mode='lines+markers', name='Cliente 0', line=dict(width=3)))
            fig.add_trace(go.Scatter(x=rondas, y=df['Cliente_1'], mode='lines+markers', name='Cliente 1', line=dict(width=3)))
            fig.add_trace(go.Scatter(x=rondas, y=df['Cliente_2'], mode='lines+markers', name='Cliente 2', line=dict(width=3)))

    fig.update_layout(
        title="Convergencia del Entrenamiento (Loss por Ronda)",
        xaxis_title="Ronda Federada",
        yaxis_title="Función de Pérdida (Loss)",
        template="plotly_dark",
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)'
    )
    
    # Forzamos que el eje X muestre estrictamente números enteros en las rondas
    fig.update_xaxes(dtick=1)
    
    return fig

if __name__ == '__main__':
    app.run(debug=True, port=8050)