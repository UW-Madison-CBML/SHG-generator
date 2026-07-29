import random
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.collections as mcoll
import streamlit as st
from streamlit_drawable_canvas import st_canvas

st.set_page_config(layout="wide")
st.title("Interactive Canvas: Synthetic Second Harmonic Generation images")

# --- INITIALIZE SESSION STATE ---
if "canvas_key" not in st.session_state:
    st.session_state["canvas_key"] = 0
if "initial_drawing" not in st.session_state:
    st.session_state["initial_drawing"] = None

# Setup layout columns
col1, col2 = st.columns([1, 1])

# --- CONTROL BUTTONS IN SIDEBAR ---
col_btn1, col_btn2 = st.sidebar.columns(2)

# 🎲 Randomize Scene Button (Appends random items to existing canvas objects)
if col_btn1.button("🎲 Randomize", use_container_width=True):
    width = st.session_state.get("canvas_w_val", 500)
    height = st.session_state.get("canvas_h_val", 500)
    
    # Retrieve existing objects from current canvas session state so we don't wipe them
    current_key = f"canvas_{st.session_state['canvas_key']}"
    existing_objects = []
    
    if current_key in st.session_state and st.session_state[current_key] is not None:
        canvas_state = st.session_state[current_key]
        if isinstance(canvas_state, dict):
            json_data = canvas_state.get("json_data")
            if json_data and isinstance(json_data, dict):
                existing_objects = json_data.get("objects", [])
        elif hasattr(canvas_state, "json_data") and canvas_state.json_data:
            existing_objects = canvas_state.json_data.get("objects", [])
    elif st.session_state.get("initial_drawing") and "objects" in st.session_state["initial_drawing"]:
        existing_objects = st.session_state["initial_drawing"]["objects"]

    # Start with existing objects and add new random objects
    random_objects = list(existing_objects)
    
    # 1. Generate 0 to 10 Random Seed Points
    num_points = random.randint(0, 10)
    for _ in range(num_points):
        px = random.randint(50, width - 50)
        py = random.randint(50, height - 50)
        random_objects.append({
            "type": "circle",
            "left": px - 5,
            "top": py - 5,
            "radius": 5,
            "scaleX": 1,
            "scaleY": 1,
            "stroke": "#FF0000",
            "strokeWidth": 2,
            "fill": "rgba(0, 0, 0, 0)"
        })
        
    # 2. Generate 0 to 5 Random Obstacle Circles
    num_obstacles = random.randint(0, 5)
    for _ in range(num_obstacles):
        cx = random.randint(80, width - 80)
        cy = random.randint(80, height - 80)
        rad = random.randint(25, 65)
        random_objects.append({
            "type": "circle",
            "left": cx - rad,
            "top": cy - rad,
            "radius": rad,
            "scaleX": 1,
            "scaleY": 1,
            "stroke": "#000000",
            "strokeWidth": 2,
            "fill": "rgba(0, 0, 0, 0)"
        })

    # 3. Generate 2 to 5 Random Intensity Regions
    num_regions = random.randint(2, 5)
    for _ in range(num_regions):
        rx = random.randint(80, width - 80)
        ry = random.randint(80, height - 80)
        rad = random.randint(40, 90)
        random_objects.append({
            "type": "circle",
            "left": rx - rad,
            "top": ry - rad,
            "radius": rad,
            "scaleX": 1,
            "scaleY": 1,
            "stroke": "#00FFCC",
            "strokeWidth": 2,
            "fill": "rgba(0, 0, 0, 0)"
        })

    # Update session state with combined objects and force canvas redraw
    st.session_state["initial_drawing"] = {"objects": random_objects}
    st.session_state["canvas_key"] += 1
    st.rerun()

# 🗑️ Clear Canvas Button (The ONLY way to reset the canvas)
if col_btn2.button("🗑️ Clear Canvas", use_container_width=True):
    st.session_state["initial_drawing"] = None
    st.session_state["canvas_key"] += 1
    st.rerun()

st.sidebar.markdown("---")

# --- SIDEBAR & CANVAS SETTINGS ---
st.sidebar.header("Canvas Dimensions")
canvas_width = st.sidebar.slider("Canvas Width", 300, 800, 500, key="canvas_w_val")
canvas_height = st.sidebar.slider("Canvas Height", 300, 800, 500, key="canvas_h_val")

st.sidebar.header("Drawing Mode")
mode_selection = st.sidebar.radio(
    "Active Tool",
    ["point", "circle", "region"],
    format_func=lambda x: {
        "point": "Points (Density Seeds)",
        "circle": "Circles (Wells)",
        "region": "Intensity Regions (Masking Zones)"
    }[x],
)

stroke_width = 2
circle_color = "#FF0000"

if mode_selection == "point":
    drawing_mode = "point"
elif mode_selection == "circle":
    drawing_mode = "circle"
    circle_color = st.sidebar.color_picker("Circle Outline Color", "#000000")
    stroke_width = st.sidebar.slider("Circle Line Width", 1, 10, 2)
elif mode_selection == "region":
    drawing_mode = "circle"
    circle_color = "#00FFCC"
    stroke_width = 2

with col1:
    st.subheader("1. Interactive Canvas")
    canvas_result = st_canvas(
        fill_color="rgba(0, 0, 0, 0)",
        stroke_width=stroke_width,
        stroke_color=circle_color,
        background_color="#ffffff",
        height=canvas_height,
        width=canvas_width,
        drawing_mode=drawing_mode,
        initial_drawing=st.session_state["initial_drawing"],
        point_display_radius=5,
        key=f"canvas_{st.session_state['canvas_key']}",
    )

# --- EXTRACT CANVAS OBJECTS ---
points = []
drawn_circles = []
intensity_regions = []

if canvas_result.json_data is not None:
    objects = canvas_result.json_data.get("objects", [])
    
    point_idx = 1
    circle_idx = 1
    region_idx = 1
    
    for obj in objects:
        obj_type = obj.get("type")
        radius = obj.get("radius", 5)
        scale_x = obj.get("scaleX", 1)
        scale_y = obj.get("scaleY", 1)
        stroke = obj.get("stroke", "#FF0000")
        
        center_x = obj.get("left", 0) + (radius * scale_x)
        center_y = obj.get("top", 0) + (radius * scale_y)
        
        if obj_type == "circle" and radius == 5 and scale_x == 1 and scale_y == 1 and stroke == "#FF0000":
            point_id = f"pt_{point_idx}_{int(center_x)}_{int(center_y)}"
            points.append({"id": point_id, "x": center_x, "y": center_y, "index": point_idx})
            point_idx += 1
        elif obj_type == "circle":
            effective_radius = radius * max(scale_x, scale_y)
            if stroke == "#00FFCC":
                region_id = f"reg_{region_idx}_{int(center_x)}_{int(center_y)}"
                intensity_regions.append({
                    "id": region_id, "index": region_idx,
                    "x": center_x, "y": center_y, "radius": effective_radius
                })
                region_idx += 1
            else:
                drawn_circles.append({
                    "index": circle_idx, "x": center_x, "y": center_y,
                    "radius": effective_radius, "color": stroke, "stroke_width": obj.get("strokeWidth", 2)
                })
                circle_idx += 1

# --- GAUSSIAN DENSITY CONTROLS ---
st.sidebar.header("Gaussian Point Parameters")
density_params = []
if points:
    st.sidebar.markdown("**Configure parameters per seed point:**")
    for pt in points:
        pid = pt["id"]
        px, py = int(pt["x"]), int(pt["y"])
        
        with st.sidebar.expander(f"Point {pt['index']} at ({px}, {py})", expanded=True):
            intensity = st.slider("Intensity (Amplitude)", 0.1, 10.0, 1.0, 0.1, key=f"intensity_{pid}")
            sigma_x = st.slider("X Spread (σ_x)", 5.0, 150.0, 60.0, 1.0, key=f"sigma_x_{pid}")
            sigma_y = st.slider("Y Spread (σ_y)", 5.0, 150.0, 20.0, 1.0, key=f"sigma_y_{pid}")
            angle_deg = st.slider("Rotation (°)", 0.0, 180.0, 0.0, 5.0, key=f"angle_{pid}")

            density_params.append({
                "x": pt["x"], "y": pt["y"], "sigma_x": sigma_x, "sigma_y": sigma_y,
                "angle": angle_deg, "intensity": intensity
            })
else:
    st.sidebar.info("Switch Active Tool to 'Points' to seed the density map.")

# --- INTENSITY REGION MASK CONTROLS ---
region_params = []
if intensity_regions:
    st.sidebar.header("Intensity Region Masking")
    for reg in intensity_regions:
        rid = reg["id"]
        rx, ry = int(reg["x"]), int(reg["y"])
        with st.sidebar.expander(f"Intensity Region {reg['index']} at ({rx}, {ry})", expanded=True):
            reg_intensity = st.slider("Region Line Brightness", 0.0, 1.0, 0.2, 0.05, key=f"reg_int_{rid}")
            reg_feather = st.slider("Edge Feathering (Softness)", 0.0, 100.0, 20.0, 5.0, key=f"reg_feather_{rid}")
            region_params.append({
                "x": reg["x"], "y": reg["y"], "radius": reg["radius"],
                "intensity": reg_intensity, "feather": reg_feather
            })

# --- SPLINE CONTROLS ---
st.sidebar.header("Streamline Spline Settings")
spline_density = st.sidebar.slider("Seed Grid Density", 0.5, 4.0, 1.5, 0.1)
spline_max_length = st.sidebar.slider("Spline Max Length", 0.5, 10.0, 4.0, 0.5)

spline_amplitude = st.sidebar.slider("Spline Squiggle Amplitude", 0.0, 2.0, 0.4, 0.05)
spline_freq = st.sidebar.slider("Squiggle Frequency", 0.01, 0.2, 0.04, 0.01)
spline_line_width = st.sidebar.slider("Line Width (Thickness)", 0.5, 5.0, 1.5, 0.1)

# Smooth softness cushion slider for circle obstacle deflection
st.sidebar.header("Obstacle Softness")
obstacle_softness = st.sidebar.slider("Deflection Softness Range", 2.0, 8.0, 4.5, 0.5)

# --- COMPUTATION FUNCTIONS ---
def generate_density_map(params, width, height):
    x = np.arange(0, width)
    y = np.arange(0, height)
    xx, yy = np.meshgrid(x, y)
    
    density_map = np.zeros((height, width), dtype=np.float64)
    for p in params:
        px, py = p["x"], p["y"]
        rad = np.radians(p["angle"])
        cos_t, sin_t = np.cos(rad), np.sin(rad)
        dx, dy = xx - px, yy - py
        x_rot = dx * cos_t + dy * sin_t
        y_rot = -dx * sin_t + dy * cos_t
        gaussian = p["intensity"] * np.exp(-((x_rot ** 2) / (2 * p["sigma_x"] ** 2) + (y_rot ** 2) / (2 * p["sigma_y"] ** 2)))
        density_map += gaussian
        
    return xx, yy, density_map


def generate_guided_vector_field(density_map, drawn_circles, xx, yy, softness=4.5):
    dD_dy, dD_dx = np.gradient(density_map)

    U = -dD_dy.copy()
    V = dD_dx.copy()

    for circ in drawn_circles:
        cx, cy = circ["x"], circ["y"]
        R = circ["radius"]

        dx, dy = xx - cx, yy - cy
        r = np.hypot(dx, dy)
        r_safe = np.maximum(r, 1e-3)

        outer_R = R * softness
        in_range = (r > 0) & (r < outer_R)

        if np.any(in_range):
            nx = dx[in_range] / r_safe[in_range]
            ny = dy[in_range] / r_safe[in_range]

            v_dot_n = U[in_range] * nx + V[in_range] * ny
            norm_dist = np.clip((r[in_range] - R) / (outer_R - R), 0.0, 1.0)
            cushion_weight = 0.5 * (1.0 + np.cos(np.pi * norm_dist))

            U[in_range] -= v_dot_n * nx * cushion_weight
            V[in_range] -= v_dot_n * ny * cushion_weight

        inside = r <= R
        if np.any(inside):
            decay = (r[inside] / R) ** 2
            U[inside] *= decay
            V[inside] *= decay

    return U, V


def add_squiggliness(U, V, xx, yy, amplitude, freq):
    if amplitude == 0:
        return U, V

    speed = np.hypot(U, V)
    speed_nonzero = np.where(speed == 0, 1.0, speed)

    perp_u = -V / speed_nonzero
    perp_v = U / speed_nonzero

    wave = np.sin(2 * np.pi * freq * (xx + yy))

    return U + (amplitude * speed * wave * perp_u), V + (amplitude * speed * wave * perp_v)


def render_binary_mask_with_regions(ax, x_1d, y_1d, U_sq, V_sq, density, maxlength, linewidth, region_params):
    strm = ax.streamplot(
        x_1d, y_1d, U_sq, V_sq,
        density=density, maxlength=maxlength,
        linewidth=linewidth, color='white', arrowstyle='-'
    )
    
    lines = strm.lines.get_paths()
    if not lines or not region_params:
        return

    strm.lines.remove()
    segments = []
    colors = []

    for path in lines:
        verts = path.vertices
        if len(verts) < 2:
            continue

        for i in range(len(verts) - 1):
            p1, p2 = verts[i], verts[i + 1]
            mid_x, mid_y = (p1[0] + p2[0]) / 2.0, (p1[1] + p2[1]) / 2.0
            seg_intensity = 1.0

            for reg in region_params:
                dist = np.hypot(mid_x - reg["x"], mid_y - reg["y"])
                r = reg["radius"]
                f = reg["feather"]
                target_int = reg["intensity"]

                if dist <= r:
                    seg_intensity = min(seg_intensity, target_int)
                elif dist < (r + f) and f > 0:
                    t = (dist - r) / f
                    smooth_val = target_int + (1.0 - target_int) * (3 * t**2 - 2 * t**3)
                    seg_intensity = min(seg_intensity, smooth_val)

            segments.append([p1, p2])
            colors.append((seg_intensity, seg_intensity, seg_intensity, 1.0))

    lc = mcoll.LineCollection(segments, colors=colors, linewidths=linewidth)
    ax.add_collection(lc)


# --- OUTPUT TAB RENDERING ---
with col2:
    tab1, tab2, tab3, tab5, tab4 = st.tabs([
        "Scalar Density Map", 
        "Vector Field Quiver", 
        "Streamline Splines", 
        "Binary Spline Mask",
        "Drawn Circles Geometry"
    ])

    # --- TAB 1: DENSITY MAP ---
    with tab1:
        st.subheader("Generated Density Map")
        if points:
            xx, yy, density_map = generate_density_map(density_params, canvas_width, canvas_height)
            
            fig1, ax1 = plt.subplots(figsize=(6, 6 * (canvas_height / canvas_width)))
            im = ax1.imshow(density_map, cmap="viridis", origin="upper", extent=[0, canvas_width, canvas_height, 0])
            
            for pt in points:
                ax1.scatter(pt["x"], pt["y"], color="red", marker="x", s=50)
                ax1.text(pt["x"] + 8, pt["y"] - 8, f"P{pt['index']}", color="white", fontweight="bold", fontsize=10,
                         bbox=dict(boxstyle="round,pad=0.2", fc="black", alpha=0.5))
            
            for circ in drawn_circles:
                ax1.add_patch(patches.Circle((circ["x"], circ["y"]), radius=circ["radius"], fill=False, edgecolor=circ["color"], linewidth=circ["stroke_width"]))

            for reg in intensity_regions:
                ax1.add_patch(patches.Circle((reg["x"], reg["y"]), radius=reg["radius"], fill=False, edgecolor="#00FFCC", linestyle="--", linewidth=1.5))

            plt.colorbar(im, ax=ax1, label="Density Level")
            ax1.set_title("Scalar Density Field")
            st.pyplot(fig1)
        else:
            st.info("Select 'Points' mode on the left to seed the density map or click '🎲 Randomize'.")

    # --- TAB 2: GUIDED VECTOR FIELD ---
    with tab2:
        st.subheader("Vector Field Following Density & Circumventing Circles")
        if points:
            xx, yy, density_map = generate_density_map(density_params, canvas_width, canvas_height)
            subsample = st.slider("Grid Subsampling Step", 5, 30, 12)

            U, V = generate_guided_vector_field(density_map, drawn_circles, xx, yy, softness=obstacle_softness)

            fig2, ax2 = plt.subplots(figsize=(6, 6 * (canvas_height / canvas_width)))

            speed = np.hypot(U, V)
            speed_nonzero = np.where(speed == 0, 1.0, speed)

            ax2.imshow(density_map, cmap="gray", alpha=0.3, origin="upper", extent=[0, canvas_width, canvas_height, 0])
            ax2.quiver(
                xx[::subsample, ::subsample], yy[::subsample, ::subsample],
                (U / speed_nonzero)[::subsample, ::subsample], (V / speed_nonzero)[::subsample, ::subsample],
                speed[::subsample, ::subsample], cmap="cool", pivot="mid"
            )

            for circ in drawn_circles:
                ax2.add_patch(patches.Circle((circ["x"], circ["y"]), radius=circ["radius"], fill=True, facecolor="crimson", alpha=0.3, edgecolor=circ["color"], linewidth=circ["stroke_width"]))

            for pt in points:
                ax2.scatter(pt["x"], pt["y"], color="red", marker="o", s=30, zorder=5)

            ax2.set_xlim(0, canvas_width)
            ax2.set_ylim(canvas_height, 0)
            ax2.set_title("Density Flow & Circumvention Quiver Plot")
            st.pyplot(fig2)
        else:
            st.info("Place seed points on the canvas or click '🎲 Randomize'.")

    # --- TAB 3: STREAMLINE SPLINES ---
    with tab3:
        st.subheader("Generated Splines along the Guided Field")
        if points:
            xx, yy, density_map = generate_density_map(density_params, canvas_width, canvas_height)
            U, V = generate_guided_vector_field(density_map, drawn_circles, xx, yy, softness=obstacle_softness)
            U_sq, V_sq = add_squiggliness(U, V, xx, yy, amplitude=spline_amplitude, freq=spline_freq)

            x_1d = np.arange(0, canvas_width)
            y_1d = np.arange(0, canvas_height)
            speed = np.hypot(U, V)

            fig4, ax4 = plt.subplots(figsize=(6, 6 * (canvas_height / canvas_width)))
            ax4.imshow(density_map, cmap="inferno", alpha=0.25, origin="upper", extent=[0, canvas_width, canvas_height, 0])

            ax4.streamplot(
                x_1d, y_1d, U_sq, V_sq,
                density=spline_density, maxlength=spline_max_length,
                linewidth=spline_line_width, color=speed, cmap="autumn",
                arrowstyle="->", arrowsize=1.2
            )

            for circ in drawn_circles:
                ax4.add_patch(patches.Circle((circ["x"], circ["y"]), radius=circ["radius"], fill=True, facecolor="black", alpha=0.5, edgecolor=circ["color"], linewidth=circ["stroke_width"]))

            for pt in points:
                ax4.scatter(pt["x"], pt["y"], color="cyan", marker="*", s=60, zorder=5)

            ax4.set_xlim(0, canvas_width)
            ax4.set_ylim(canvas_height, 0)
            ax4.set_title("Streamline Spline Flow Field")
            st.pyplot(fig4)
        else:
            st.info("Place seed points on the canvas or click '🎲 Randomize'.")

    # --- TAB 5: BINARY SPLINE MASK ---
    with tab5:
        st.subheader("Binary Image Mask with Regional Intensity Filtering")
        if points:
            xx, yy, density_map = generate_density_map(density_params, canvas_width, canvas_height)
            U, V = generate_guided_vector_field(density_map, drawn_circles, xx, yy, softness=obstacle_softness)
            U_sq, V_sq = add_squiggliness(U, V, xx, yy, amplitude=spline_amplitude, freq=spline_freq)

            x_1d = np.arange(0, canvas_width)
            y_1d = np.arange(0, canvas_height)

            fig_bin, ax_bin = plt.subplots(figsize=(6, 6 * (canvas_height / canvas_width)))
            fig_bin.patch.set_facecolor('black')
            ax_bin.set_facecolor('black')

            render_binary_mask_with_regions(
                ax_bin, x_1d, y_1d, U_sq, V_sq,
                density=spline_density,
                maxlength=spline_max_length,
                linewidth=spline_line_width,
                region_params=region_params
            )

            ax_bin.set_xlim(0, canvas_width)
            ax_bin.set_ylim(canvas_height, 0)
            ax_bin.axis('off')
            
            st.pyplot(fig_bin)
        else:
            st.info("Place seed points on the canvas or click '🎲 Randomize'.")

    # --- TAB 4: DRAWN CIRCLES OVERVIEW ---
    with tab4:
        st.subheader("Drawn Circles Geometry")
        if drawn_circles or intensity_regions:
            if drawn_circles:
                st.markdown("**Obstacle Circles:**")
                st.dataframe(drawn_circles, use_container_width=True)
            if intensity_regions:
                st.markdown("**Intensity Mask Regions:**")
                st.dataframe(intensity_regions, use_container_width=True)
            
            fig3, ax3 = plt.subplots(figsize=(6, 6 * (canvas_height / canvas_width)))
            ax3.set_xlim(0, canvas_width)
            ax3.set_ylim(canvas_height, 0)
            ax3.set_facecolor("#ffffff")
            
            for circ in drawn_circles:
                ax3.add_patch(patches.Circle((circ["x"], circ["y"]), radius=circ["radius"], fill=False, edgecolor=circ["color"], linewidth=circ["stroke_width"]))
                ax3.text(circ["x"], circ["y"], f"C{circ['index']}", color="black", fontweight="bold", fontsize=10, ha="center", va="center")

            for reg in intensity_regions:
                ax3.add_patch(patches.Circle((reg["x"], reg["y"]), radius=reg["radius"], fill=False, edgecolor="#00FFCC", linestyle="--", linewidth=2))
                ax3.text(reg["x"], reg["y"], f"R{reg['index']}", color="#008080", fontweight="bold", fontsize=10, ha="center", va="center")

            ax3.set_title("Geometric Circle Overlay")
            st.pyplot(fig3)
        else:
            st.info("Switch 'Active Tool' in the sidebar or click '🎲 Randomize'.")