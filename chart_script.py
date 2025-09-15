import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

# Create a flowchart showing the AI stock trading system architecture
fig = go.Figure()

# Define colors for different layers
colors = {
    'data_sources': '#1FB8CD',    # Strong cyan
    'data_processing': '#DB4545', # Bright red  
    'analysis': '#2E8B57',        # Sea green
    'decision': '#5D878F',        # Cyan
    'execution': '#D2BA4C',       # Moderate yellow
    'monitoring': '#B4413C',      # Moderate red
    'storage': '#964325'          # Dark orange
}

# Define positions and components
components = [
    # Data Sources Layer (y=7)
    {'name': 'Zerodha API', 'type': 'rect', 'x': 1, 'y': 7, 'layer': 'data_sources'},
    {'name': 'Yahoo Finance', 'type': 'rect', 'x': 3, 'y': 7, 'layer': 'data_sources'},
    {'name': 'Historical Data', 'type': 'rect', 'x': 5, 'y': 7, 'layer': 'data_sources'},
    
    # Data Processing Layer (y=6)
    {'name': 'Data Fetching', 'type': 'rect', 'x': 1, 'y': 6, 'layer': 'data_processing'},
    {'name': 'Data Cleaning', 'type': 'rect', 'x': 3, 'y': 6, 'layer': 'data_processing'},
    {'name': 'Feature Eng', 'type': 'rect', 'x': 5, 'y': 6, 'layer': 'data_processing'},
    
    # Analysis Layer (y=5)
    {'name': 'Tech Indicators', 'type': 'rect', 'x': 1, 'y': 5, 'layer': 'analysis'},
    {'name': 'ML Predictions', 'type': 'rect', 'x': 3, 'y': 5, 'layer': 'analysis'},
    {'name': 'Signal Gen', 'type': 'rect', 'x': 5, 'y': 5, 'layer': 'analysis'},
    
    # Decision Layer (y=4)
    {'name': 'Risk Mgmt', 'type': 'diamond', 'x': 1.5, 'y': 4, 'layer': 'decision'},
    {'name': 'Position Size', 'type': 'diamond', 'x': 3, 'y': 4, 'layer': 'decision'},
    {'name': 'Buy/Sell', 'type': 'diamond', 'x': 4.5, 'y': 4, 'layer': 'decision'},
    
    # Execution Layer (y=3)
    {'name': 'Order Place', 'type': 'rect', 'x': 1, 'y': 3, 'layer': 'execution'},
    {'name': 'Portfolio Mon', 'type': 'rect', 'x': 3, 'y': 3, 'layer': 'execution'},
    {'name': 'Trade Exec', 'type': 'rect', 'x': 5, 'y': 3, 'layer': 'execution'},
    
    # Monitoring Layer (y=2)
    {'name': 'Perf Tracking', 'type': 'rect', 'x': 1, 'y': 2, 'layer': 'monitoring'},
    {'name': 'Alerts', 'type': 'rect', 'x': 3, 'y': 2, 'layer': 'monitoring'},
    {'name': 'Logging', 'type': 'rect', 'x': 5, 'y': 2, 'layer': 'monitoring'},
    
    # Storage Layer (y=1)
    {'name': 'Database', 'type': 'cylinder', 'x': 1.5, 'y': 1, 'layer': 'storage'},
    {'name': 'Cache/Redis', 'type': 'cylinder', 'x': 3, 'y': 1, 'layer': 'storage'},
    {'name': 'File Storage', 'type': 'cylinder', 'x': 4.5, 'y': 1, 'layer': 'storage'}
]

# Add rectangles
for comp in components:
    if comp['type'] == 'rect':
        fig.add_shape(
            type="rect",
            x0=comp['x']-0.4, y0=comp['y']-0.15,
            x1=comp['x']+0.4, y1=comp['y']+0.15,
            line=dict(color=colors[comp['layer']], width=2),
            fillcolor=colors[comp['layer']],
            opacity=0.7
        )
    elif comp['type'] == 'diamond':
        # Create diamond shape using path
        fig.add_shape(
            type="path",
            path=f"M {comp['x']} {comp['y']+0.2} L {comp['x']+0.3} {comp['y']} L {comp['x']} {comp['y']-0.2} L {comp['x']-0.3} {comp['y']} Z",
            line=dict(color=colors[comp['layer']], width=2),
            fillcolor=colors[comp['layer']],
            opacity=0.7
        )
    elif comp['type'] == 'cylinder':
        # Represent cylinder as rounded rectangle
        fig.add_shape(
            type="rect",
            x0=comp['x']-0.35, y0=comp['y']-0.15,
            x1=comp['x']+0.35, y1=comp['y']+0.15,
            line=dict(color=colors[comp['layer']], width=2),
            fillcolor=colors[comp['layer']],
            opacity=0.7
        )

# Add text labels
for comp in components:
    fig.add_annotation(
        x=comp['x'], y=comp['y'],
        text=comp['name'],
        showarrow=False,
        font=dict(size=10, color="white", family="Arial Black"),
        align="center"
    )

# Add arrows showing data flow
arrows = [
    # From data sources to processing
    {'start': (1, 6.85), 'end': (1, 6.15)},
    {'start': (3, 6.85), 'end': (3, 6.15)},
    {'start': (5, 6.85), 'end': (5, 6.15)},
    
    # From processing to analysis
    {'start': (1, 5.85), 'end': (1, 5.15)},
    {'start': (3, 5.85), 'end': (3, 5.15)},
    {'start': (5, 5.85), 'end': (5, 5.15)},
    
    # From analysis to decision
    {'start': (1, 4.85), 'end': (1.5, 4.2)},
    {'start': (3, 4.85), 'end': (3, 4.2)},
    {'start': (5, 4.85), 'end': (4.5, 4.2)},
    
    # From decision to execution
    {'start': (1.5, 3.8), 'end': (1, 3.15)},
    {'start': (3, 3.8), 'end': (3, 3.15)},
    {'start': (4.5, 3.8), 'end': (5, 3.15)},
    
    # From execution to monitoring
    {'start': (1, 2.85), 'end': (1, 2.15)},
    {'start': (3, 2.85), 'end': (3, 2.15)},
    {'start': (5, 2.85), 'end': (5, 2.15)},
    
    # From monitoring to storage
    {'start': (1, 1.85), 'end': (1.5, 1.15)},
    {'start': (3, 1.85), 'end': (3, 1.15)},
    {'start': (5, 1.85), 'end': (4.5, 1.15)}
]

# Add arrows
for arrow in arrows:
    fig.add_annotation(
        x=arrow['end'][0], y=arrow['end'][1],
        ax=arrow['start'][0], ay=arrow['start'][1],
        xref='x', yref='y',
        axref='x', ayref='y',
        arrowhead=2,
        arrowsize=1,
        arrowwidth=2,
        arrowcolor='#333333',
        showarrow=True
    )

# Add layer labels on the left
layer_labels = [
    {'name': 'Data Sources', 'y': 7},
    {'name': 'Processing', 'y': 6},
    {'name': 'Analysis', 'y': 5},
    {'name': 'Decision', 'y': 4},
    {'name': 'Execution', 'y': 3},
    {'name': 'Monitoring', 'y': 2},
    {'name': 'Storage', 'y': 1}
]

for label in layer_labels:
    fig.add_annotation(
        x=0.2, y=label['y'],
        text=label['name'],
        showarrow=False,
        font=dict(size=11, color="#333333", family="Arial"),
        align="center",
        textangle=-90
    )

# Update layout
fig.update_layout(
    title="AI Trading System Architecture",
    xaxis=dict(range=[0, 6], showgrid=False, showticklabels=False, zeroline=False),
    yaxis=dict(range=[0.5, 7.5], showgrid=False, showticklabels=False, zeroline=False),
    plot_bgcolor='white',
    showlegend=False
)

# Remove axes
fig.update_xaxes(visible=False)
fig.update_yaxes(visible=False)

# Save the chart
fig.write_image("chart.png")
fig.write_image("chart.svg", format="svg")

fig.show()