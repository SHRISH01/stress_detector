import base64
import cv2
from dash import Dash, dcc, html
from dash.dependencies import Input, Output
import plotly.graph_objs as go

from gui.state import SharedState
from gui.processor import VideoProcessor


state = SharedState()
processor = None

app = Dash(__name__)

app.layout = html.Div([
    html.H2("Real-Time Stress Detection System"),

    html.Div([
        html.Button("Start Webcam", id="start"),
        html.Button("Stop", id="stop")
    ]),

    html.Div([
        html.Img(id="video", style={"width": "400px"})
    ]),

    html.Div([
        dcc.Graph(id="pulse", style={"width": "32%", "display": "inline-block"}),
        dcc.Graph(id="hr", style={"width": "32%", "display": "inline-block"}),
        dcc.Graph(id="stress", style={"width": "32%", "display": "inline-block"}),
    ]),

    html.H4(id="stats"),

    dcc.Interval(id="timer", interval=1000)
])


@app.callback(
    Output("video", "src"),
    Output("pulse", "figure"),
    Output("hr", "figure"),
    Output("stress", "figure"),
    Output("stats", "children"),
    Input("timer", "n_intervals")
)
def update_ui(_):
    img_src = None
    if state.frame is not None:
        _, buffer = cv2.imencode(".jpg", state.frame)
        img_src = "data:image/jpeg;base64," + base64.b64encode(buffer).decode()

    pulse_fig = go.Figure(
        data=[go.Scatter(y=list(state.pulse), mode="lines", name="rPPG Signal")]
    )
    pulse_fig.update_layout(title="rPPG Signal")

    hr_fig = go.Figure(
        data=[go.Scatter(y=list(state.hr_hist), mode="lines", name="Heart Rate")]
    )
    hr_fig.update_layout(title="Heart Rate (BPM)")

    stress_fig = go.Figure(
        data=[go.Scatter(y=list(state.stress_hist), mode="lines", name="Stress Index")]
    )
    stress_fig.update_layout(title="Stress Index (0–100)")

    stats = "Computing..."
    if state.hr:
        stats = f"HR: {state.hr:.1f} BPM"
    if state.stress:
        stats += f" | Stress: {state.stress:.1f}"

    return img_src, pulse_fig, hr_fig, stress_fig, stats


@app.callback(
    Input("start", "n_clicks"),
    Input("stop", "n_clicks"),
    prevent_initial_call=True
)
def control(start, stop):
    global processor
    if stop and processor:
        processor.stop()
        processor = None
    elif start:
        processor = VideoProcessor(0, state)
        processor.start()


if __name__ == "__main__":
    app.run(debug=False)
