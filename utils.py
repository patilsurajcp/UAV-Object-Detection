# utils.py
import pandas as pd
import plotly.graph_objects as go
from collections import Counter

def get_detection_stats(detections):
    """Calculate detection statistics"""
    if not detections:
        return {}

    class_counts = Counter([d['class'] for d in detections])
    avg_conf     = sum([d['confidence'] for d in detections]) / len(detections)

    return {
        'total'       : len(detections),
        'class_counts': dict(class_counts),
        'avg_conf'    : round(avg_conf, 3),
        'classes'     : list(class_counts.keys()),
    }


def plot_class_distribution(detections):
    """Bar chart of detected classes"""
    if not detections:
        return None

    class_counts = Counter([d['class'] for d in detections])
    sorted_items = sorted(class_counts.items(), key=lambda x: -x[1])
    classes      = [item[0] for item in sorted_items]
    counts       = [item[1] for item in sorted_items]

    colors = [
        '#FF6B6B', '#FF8E53', '#FFC300', '#2ECC71',
        '#1ABC9C', '#3498DB', '#9B59B6', '#E91E63',
        '#FF5722', '#607D8B'
    ]

    fig = go.Figure(data=[
        go.Bar(
            x            = classes,
            y            = counts,
            marker_color = colors[:len(classes)],
            text         = counts,
            textposition = 'outside',
        )
    ])

    fig.update_layout(
        title         = '📊 Detected Object Classes',
        xaxis_title   = 'Class',
        yaxis_title   = 'Count',
        plot_bgcolor  = 'rgba(0,0,0,0)',
        paper_bgcolor = 'rgba(0,0,0,0)',
        font          = dict(color='white', size=13),
        height        = 400,
        yaxis         = dict(
            gridcolor = 'rgba(255,255,255,0.1)',
            range     = [0, max(counts) + 1]
        ),
        xaxis         = dict(
            gridcolor = 'rgba(255,255,255,0.1)'
        ),
        margin        = dict(t=50, b=50, l=50, r=50)
    )
    return fig


def plot_confidence_distribution(detections):
    """Histogram of confidence scores"""
    if not detections:
        return None

    class_conf = {}
    for d in detections:
        cls  = d['class']
        conf = d['confidence']
        if cls not in class_conf:
            class_conf[cls] = []
        class_conf[cls].append(conf)

    colors = [
        '#FF6B6B', '#FF8E53', '#FFC300', '#2ECC71',
        '#1ABC9C', '#3498DB', '#9B59B6', '#E91E63',
        '#FF5722', '#607D8B'
    ]

    fig = go.Figure()
    for i, (cls, confs) in enumerate(class_conf.items()):
        fig.add_trace(go.Histogram(
            x            = confs,
            name         = cls,
            nbinsx       = 10,
            marker_color = colors[i % len(colors)],
            opacity      = 0.8,
        ))

    fig.update_layout(
        title         = '📈 Confidence Score Distribution',
        xaxis_title   = 'Confidence Score',
        yaxis_title   = 'Count',
        barmode       = 'overlay',
        plot_bgcolor  = 'rgba(0,0,0,0)',
        paper_bgcolor = 'rgba(0,0,0,0)',
        font          = dict(color='white', size=13),
        height        = 400,
        xaxis         = dict(
            range     = [0, 1],
            gridcolor = 'rgba(255,255,255,0.1)'
        ),
        yaxis         = dict(
            gridcolor = 'rgba(255,255,255,0.1)'
        ),
        legend        = dict(
            bgcolor   = 'rgba(0,0,0,0.5)',
            font      = dict(color='white')
        ),
        margin        = dict(t=50, b=50, l=50, r=50)
    )
    return fig


def plot_object_map(detections, img_width, img_height):
    """Show object locations on image map"""
    if not detections:
        return None

    colors_map = {
        'pedestrian'     : 'red',
        'people'         : 'orange',
        'bicycle'        : 'yellow',
        'car'            : 'lime',
        'van'            : 'cyan',
        'truck'          : 'royalblue',
        'tricycle'       : 'violet',
        'awning-tricycle': 'pink',
        'bus'            : 'coral',
        'motor'          : 'lightgray'
    }

    fig = go.Figure()

    fig.add_shape(
        type      = 'rect',
        x0=0, y0=0,
        x1=img_width,
        y1=img_height,
        line      = dict(color='white', width=2),
        fillcolor = 'rgba(0,0,30,0.9)'
    )

    class_groups = {}
    for d in detections:
        cls = d['class']
        if cls not in class_groups:
            class_groups[cls] = {'x': [], 'y': [], 'text': []}
        class_groups[cls]['x'].append(d['center_x'])
        class_groups[cls]['y'].append(img_height - d['center_y'])
        class_groups[cls]['text'].append(
            f"Class: {cls}<br>"
            f"Conf: {d['confidence']:.2f}<br>"
            f"X: {d['center_x']:.0f}px<br>"
            f"Y: {d['center_y']:.0f}px"
        )

    for cls, data in class_groups.items():
        fig.add_trace(go.Scatter(
            x         = data['x'],
            y         = data['y'],
            mode      = 'markers',
            name      = cls,
            text      = data['text'],
            hoverinfo = 'text',
            marker    = dict(
                size  = 14,
                color = colors_map.get(cls, 'white'),
                symbol= 'circle',
                line  = dict(width=2, color='white')
            )
        ))

    fig.update_layout(
        title         = '🗺️ Object Location Map',
        xaxis_title   = 'X Position (pixels)',
        yaxis_title   = 'Y Position (pixels)',
        plot_bgcolor  = 'rgba(0,0,30,0.9)',
        paper_bgcolor = 'rgba(0,0,0,0)',
        font          = dict(color='white', size=13),
        height        = 500,
        xaxis         = dict(
            range     = [0, img_width],
            gridcolor = 'rgba(255,255,255,0.1)'
        ),
        yaxis         = dict(
            range     = [0, img_height],
            gridcolor = 'rgba(255,255,255,0.1)'
        ),
        legend        = dict(
            bgcolor   = 'rgba(0,0,0,0.5)',
            font      = dict(color='white')
        ),
        margin        = dict(t=50, b=50, l=50, r=50)
    )
    return fig


def get_detection_table(detections):
    """Create detection results dataframe"""
    if not detections:
        return pd.DataFrame()

    rows = []
    for i, d in enumerate(detections):
        rows.append({
            '#'         : i + 1,
            'Class'     : d['class'],
            'Confidence': f"{d['confidence']:.3f}",
            'Center X'  : f"{d['center_x']:.0f}px",
            'Center Y'  : f"{d['center_y']:.0f}px",
            'Width'     : f"{d['width']:.0f}px",
            'Height'    : f"{d['height']:.0f}px",
        })
    return pd.DataFrame(rows)