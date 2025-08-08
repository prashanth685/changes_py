# import os
# from PIL import Image, ImageDraw

# def create_icon(filename, draw_function, size=(24, 24), output_dir=r"C:\Users\Prashanth S\Desktop\new_one\icons"):
#     # Create a transparent image
#     img = Image.new('RGBA', size, (0, 0, 0, 0))
#     draw = ImageDraw.Draw(img)
#     draw_function(draw, size)
#     # Ensure the output directory exists
#     os.makedirs(output_dir, exist_ok=True)
#     # Save the icon
#     img.save(os.path.join(output_dir, filename), "PNG")

# def draw_tag(draw, size):
#     w, h = size
#     # Draw a tag shape
#     draw.polygon([w/4, h/4, 3*w/4, h/4, 3*w/4, 3*h/4, w/4, 3*h/4], outline="gray", width=2)
#     draw.ellipse([w/4-3, h/4-3, w/4+3, h/4+3], fill="gray")

# def draw_clock(draw, size):
#     w, h = size
#     # Draw a circle for the clock face
#     draw.ellipse([4, 4, w-4, h-4], outline="gray", width=2)
#     # Draw clock hands
#     draw.line([w/2, h/2, w/2, h/4], fill="gray", width=2)  # Hour hand
#     draw.line([w/2, h/2, 3*w/4, h/2], fill="gray", width=2)  # Minute hand

# def draw_table(draw, size):
#     w, h = size
#     # Draw a 2x2 grid
#     draw.rectangle([4, 4, w-4, h-4], outline="gray", width=2)
#     draw.line([4, h/2, w-4, h/2], fill="gray", width=2)
#     draw.line([w/2, 4, w/2, h-4], fill="gray", width=2)

# def draw_waveform(draw, size):
#     w, h = size
#     # Draw a simple waveform
#     points = [(x, h/2 + 4 * (1 if x % 8 < 4 else -1)) for x in range(4, w-4, 2)]
#     draw.line(points, fill="gray", width=2)

# def draw_waterfall(draw, size):
#     w, h = size
#     # Draw stacked bars
#     for y in range(6, h-6, 4):
#         draw.rectangle([4, y, w-4, y+2], fill="gray")

# def draw_orbit(draw, size):
#     w, h = size
#     # Draw a planet with an orbit path
#     draw.ellipse([w/2-6, h/2-6, w/2+6, h/2+6], outline="gray", width=2)  # Orbit path
#     draw.ellipse([w/2-2, h/2-2, w/2+2, h/2+2], fill="gray")  # Planet

# def draw_trend(draw, size):
#     w, h = size
#     # Draw an upward trend line
#     draw.line([4, h-4, w-4, 4], fill="gray", width=2)

# def draw_multi_trend(draw, size):
#     w, h = size
#     # Draw two trend lines
#     draw.line([4, h-4, w-4, 4], fill="gray", width=2)
#     draw.line([4, h-8, w-4, 8], fill="gray", width=2)

# def draw_bode(draw, size):
#     w, h = size
#     # Draw a frequency graph (axes with a curve)
#     draw.line([4, h-4, w-4, h-4], fill="gray", width=2)  # X-axis
#     draw.line([4, h-4, 4, 4], fill="gray", width=2)  # Y-axis
#     points = [(x, h/2 + 3 * (1 if x % 8 < 4 else -1)) for x in range(4, w-4, 2)]
#     draw.line(points, fill="gray", width=2)

# def draw_history(draw, size):
#     w, h = size
#     # Draw a timeline with dots
#     draw.line([4, h/2, w-4, h/2], fill="gray", width=2)
#     for x in range(8, w-8, 6):
#         draw.ellipse([x-2, h/2-2, x+2, h/2+2], fill="gray")

# def draw_report_time(draw, size):
#     w, h = size
#     # Draw a document with a small clock
#     draw.rectangle([6, 4, w-6, h-4], outline="gray", width=2)
#     draw.line([6, 4, 8, 6], fill="gray", width=2)  # Folded corner
#     draw.ellipse([w/2-4, h/2-4, w/2+4, h/2+4], outline="gray", width=2)  # Clock
#     draw.line([w/2, h/2, w/2, h/2-2], fill="gray", width=1)  # Hour hand

# def draw_report(draw, size):
#     w, h = size
#     # Draw a document
#     draw.rectangle([6, 4, w-6, h-4], outline="gray", width=2)
#     draw.line([6, 4, 8, 6], fill="gray", width=2)  # Folded corner
#     for y in range(8, h-8, 4):
#         draw.line([8, y, w-8, y], fill="gray", width=1)  # Text lines

# # Generate all icons
# icon_functions = [
#     # ("tag.png", draw_tag),
#     ("clock.png", draw_clock),
#     ("table.png", draw_table),
#     ("waveform.png", draw_waveform),
#     ("waterfall.png", draw_waterfall),
#     ("orbit.png", draw_orbit),
#     ("trend.png", draw_trend),
#     ("multi-trend.png", draw_multi_trend),
#     ("bode.png", draw_bode),
#     ("history.png", draw_history),
#     ("report-time.png", draw_report_time),
#     ("report.png", draw_report),
# ]

# for filename, draw_func in icon_functions:
#     create_icon(filename, draw_func)

# print("Icons generated successfully in 'C:\\Users\\Prashanth S\\Desktop\\new_one\\icons'.")





import os
from PIL import Image, ImageDraw
import math

def create_icon(filename, draw_function, size=(64, 64)):
    # Create a transparent image
    img = Image.new('RGBA', size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)
    draw_function(draw, size)
    # Ensure the icons folder exists
    os.makedirs("icons", exist_ok=True)
    # Save the icon
    img.save(os.path.join("icons", filename), "PNG")

def draw_clock(draw, size):
    w, h = size
    # Draw a circle for the clock face
    draw.ellipse([8, 8, w-8, h-8], outline="#ffb300", width=3)
    # Draw clock hands
    draw.line([w/2, h/2, w/2, h/4], fill="#ffb300", width=3)  # Hour hand
    draw.line([w/2, h/2, w*3/4, h/2], fill="#ffb300", width=2)  # Minute hand

def draw_table(draw, size):
    w, h = size
    # Draw a 3x3 grid
    draw.rectangle([8, 8, w-8, h-8], outline="#64b5f6", width=2)
    draw.line([8, h/3, w-8, h/3], fill="#64b5f6", width=2)
    draw.line([8, 2*h/3, w-8, 2*h/3], fill="#64b5f6", width=2)
    draw.line([w/3, 8, w/3, h-8], fill="#64b5f6", width=2)
    draw.line([2*w/3, 8, 2*w/3, h-8], fill="#64b5f6", width=2)

def draw_report_time(draw, size):
    w, h = size
    # Draw a document with a small clock
    draw.rectangle([12, 8, w-12, h-8], outline="#4db6ac", width=2)
    draw.line([12, 12, 16, 16], fill="#4db6ac", width=2)  # Folded corner
    draw.ellipse([w/2-10, h/2-10, w/2+10, h/2+10], outline="#4db6ac", width=2)  # Clock
    draw.line([w/2, h/2, w/2, h/2-5], fill="#4db6ac", width=2)  # Hour hand

def draw_waveform(draw, size):
    w, h = size
    # Draw a sine wave
    points = [(x, h/2 + 15 * (1 if x % 20 < 10 else -1)) for x in range(8, w-8, 2)]
    draw.line(points, fill="#ba68c8", width=3)

def draw_waterfall(draw, size):
    w, h = size
    # Draw stacked bars
    for y in range(15, h-15, 12):
        draw.rectangle([8, y, w-8, y+8], fill="#4dd0e1")

def draw_ruler(draw, size):
    w, h = size
    # Draw a horizontal ruler
    draw.line([8, h/2, w-8, h/2], fill="#4dd0e1", width=3)
    for x in range(8, w-8, 10):
        draw.line([x, h/2-5, x, h/2+5], fill="#4dd0e1", width=2)

def draw_orbit(draw, size):
    w, h = size
    # Draw a planet with an orbit path
    draw.ellipse([w/2-15, h/2-15, w/2+15, h/2+15], outline="#f06292", width=2)  # Orbit path
    draw.ellipse([w/2-5, h/2-5, w/2+5, h/2+5], fill="#f06292")  # Planet

def draw_trend(draw, size):
    w, h = size
    # Draw an upward trend line
    draw.line([8, h-8, w-8, 8], fill="#aed581", width=3)

def draw_multi_trend(draw, size):
    w, h = size
    # Draw two trend lines
    draw.line([8, h-8, w-8, 8], fill="#ff8a65", width=3)
    draw.line([8, h-16, w-8, 16], fill="#ff8a65", width=2)

def draw_bode(draw, size):
    w, h = size
    # Draw a frequency graph (sine wave with axes)
    draw.line([8, h-8, w-8, h-8], fill="#7986cb", width=2)  # X-axis
    draw.line([8, h-8, 8, 8], fill="#7986cb", width=2)  # Y-axis
    points = [(x, h/2 + 10 * (1 if x % 20 < 10 else -1)) for x in range(8, w-8, 2)]
    draw.line(points, fill="#7986cb", width=3)

def draw_history(draw, size):
    w, h = size
    # Draw a timeline with dots
    draw.line([8, h/2, w-8, h/2], fill="#ef5350", width=3)
    for x in range(20, w-20, 15):
        draw.ellipse([x-3, h/2-3, x+3, h/2+3], fill="#ef5350")

def draw_report(draw, size):
    w, h = size
    # Draw a document
    draw.rectangle([12, 8, w-12, h-8], outline="#ab47bc", width=2)
    draw.line([12, 12, 16, 16], fill="#ab47bc", width=2)  # Folded corner
    for y in range(20, h-20, 10):
        draw.line([16, y, w-16, y], fill="#ab47bc", width=1)  # Text lines

def draw_polar_icon(draw, size):
    w, h = size
    cx, cy = w // 2, h // 2
    radius = min(w, h) // 2 - 4  # Leave padding

    # Draw outer circle
    draw.ellipse([cx - radius, cy - radius, cx + radius, cy + radius], outline="#42a5f5", width=2)

    # Draw radial lines
    for angle_deg in range(0, 360, 45):  # Every 45 degrees
        angle_rad = math.radians(angle_deg)
        x = cx + radius * math.cos(angle_rad)
        y = cy + radius * math.sin(angle_rad)
        draw.line([cx, cy, x, y], fill="#42a5f5", width=1)

    # Draw concentric circles
    for r in range(radius // 3, radius, radius // 3):
        draw.ellipse([cx - r, cy - r, cx + r, cy + r], outline="#90caf9", width=1)

# Generate all icons
icon_functions = [
    ("clock.png", draw_clock),
    ("table.png", draw_table),
    ("report-time.png", draw_report_time),
    ("waveform.png", draw_waveform),
    ("waterfall.png", draw_waterfall),
    ("ruler.png", draw_ruler),
    ("orbit.png", draw_orbit),
    ("trend.png", draw_trend),
    ("multi-trend.png", draw_multi_trend),
    ("bode.png", draw_bode),
    ("history.png", draw_history),
    ("report.png", draw_report),
    ("polar.png",draw_polar_icon),
]

for filename, draw_func in icon_functions:
    create_icon(filename, draw_func)

print("Icons generated successfully in the 'icons' folder.")