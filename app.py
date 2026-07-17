import streamlit as st
import streamlit.components.v1 as components

from floorplan_core import (
    ROOM_TYPES, FURN_TYPES, THEMES,
    Furniture, generate_plan, render_svg, ModelError,
)

st.set_page_config(page_title="Floorplan AI", layout="wide", page_icon="\U0001F4D0")

# ---------------------------------------------------------------- session state

if "plan" not in st.session_state:
    st.session_state.plan = None
if "selected_room_id" not in st.session_state:
    st.session_state.selected_room_id = None
if "theme_key" not in st.session_state:
    st.session_state.theme_key = "blueprint"

# ---------------------------------------------------------------- sidebar

with st.sidebar:
    st.markdown("### \U0001F4D0 Floorplan AI")
    st.caption("Generative interior layouts")

    brief = st.text_area(
        "Describe the space",
        value=(
            "Tech startup office for about 30 people. Open workstation area, two "
            "meeting rooms (one small, one large), a reception with waiting seats, "
            "a manager's office, a kitchen/break area, a storage room, and two "
            "restrooms."
        ),
        height=130,
    )

    col1, col2 = st.columns(2)
    with col1:
        sqft = st.number_input("Total area (sq ft)", min_value=400, max_value=50000,
                                value=3200, step=100)
    with col2:
        shape_label = st.selectbox("Shape", ["Wide rectangle", "Square", "Deep rectangle"])
    aspect = {"Wide rectangle": 1.3, "Square": 1.0, "Deep rectangle": 0.75}[shape_label]

    st.markdown("#### Model backend")

    def _secret(name: str) -> str:
        return st.secrets.get(name, "") if hasattr(st, "secrets") else ""

    # Preset providers: (label, backend, base_url, default_model, secrets_key_name)
    PROVIDER_PRESETS = {
        "Anthropic (Claude)": ("anthropic", None, "claude-sonnet-5", "ANTHROPIC_API_KEY"),
        "Groq (Llama 3.3, free tier)": ("custom", "https://api.groq.com/openai/v1/chat/completions",
                                         "llama-3.3-70b-versatile", "GROQ_API_KEY"),
        "OpenRouter (many free models)": ("custom", "https://openrouter.ai/api/v1/chat/completions",
                                           "meta-llama/llama-3.3-70b-instruct:free", "OPENROUTER_API_KEY"),
        "Cerebras (fast, free tier)": ("custom", "https://api.cerebras.ai/v1/chat/completions",
                                        "llama3.3-70b", "CEREBRAS_API_KEY"),
        "Together AI": ("custom", "https://api.together.xyz/v1/chat/completions",
                         "meta-llama/Llama-3.3-70B-Instruct-Turbo-Free", "TOGETHER_API_KEY"),
        "Ollama (local, no key)": ("custom", "http://localhost:11434/v1/chat/completions",
                                    "llama3.1:8b", None),
        "Other custom endpoint": ("custom", "", "", None),
    }

    provider_label = st.selectbox("Provider", list(PROVIDER_PRESETS.keys()), index=1)
    backend, preset_url, preset_model, secret_name = PROVIDER_PRESETS[provider_label]
    preset_key = _secret(secret_name) if secret_name else ""

    if backend == "anthropic":
        api_key = st.text_input(
            "Anthropic API key", value=preset_key, type="password",
            key=f"apikey_{provider_label}",
            help="Set ANTHROPIC_API_KEY in .streamlit/secrets.toml to skip pasting this every time.",
        )
        model = st.text_input("Model", value=preset_model, key=f"model_{provider_label}")
        backend_kwargs = {"api_key": api_key, "model": model}
        if not api_key:
            st.warning("No Anthropic key found — paste one above, or pick a different provider.")
    else:
        base_url = st.text_input("Endpoint URL", value=preset_url, key=f"url_{provider_label}")
        model = st.text_input("Model name", value=preset_model, key=f"model_{provider_label}")
        needs_key = provider_label != "Ollama (local, no key)"
        api_key = st.text_input(
            "API key" + ("" if needs_key else " (optional)"), value=preset_key, type="password",
            key=f"apikey_{provider_label}",
            help=(f"Set {secret_name} in .streamlit/secrets.toml to skip pasting this every time."
                  if secret_name else "Ollama running locally doesn't need a key."),
        )
        backend_kwargs = {"base_url": base_url, "model": model, "api_key": api_key or None}
        if needs_key and not api_key:
            st.warning(f"No key found for {provider_label} — paste one above, "
                       f"or add {secret_name} to .streamlit/secrets.toml.")

    generate_clicked = st.button("Generate floor plan", type="primary", use_container_width=True)

    if generate_clicked:
        if not brief.strip():
            st.error("Describe the space first.")
        else:
            with st.spinner("Drafting layout\u2026"):
                try:
                    plan = generate_plan(brief.strip(), float(sqft), aspect, backend, **backend_kwargs)
                    st.session_state.plan = plan
                    st.session_state.selected_room_id = None
                    st.success(f"Plan generated \u2014 {len(plan.rooms)} rooms.")
                except ModelError as e:
                    st.error(str(e))
                except Exception as e:
                    st.error(f"Generation failed: {e}")

    st.markdown("---")
    st.caption(
        "The generation call is isolated in `floorplan_core.generate_plan()`. "
        "Switch backend to a custom endpoint to run this against any "
        "OpenAI-compatible open-source model server instead of Claude."
    )

# ---------------------------------------------------------------- main area

plan = st.session_state.plan

top_l, top_r = st.columns([3, 1])
with top_l:
    if plan:
        st.markdown(f"**{brief.strip()[:80]}**")
        st.caption(f"{len(plan.rooms)} rooms \u00b7 {sqft} sq ft \u00b7 {shape_label.lower()}")
    else:
        st.markdown("**No plan yet**")
        st.caption("Describe a space in the sidebar and generate a layout.")
with top_r:
    theme_key = st.selectbox("Theme", list(THEMES.keys()),
                              index=list(THEMES.keys()).index(st.session_state.theme_key))
    st.session_state.theme_key = theme_key

main_col, edit_col = st.columns([2.4, 1])

with main_col:
    if plan is None:
        st.info("Your generated floor plan will appear here.")
    else:
        svg = render_svg(plan, st.session_state.theme_key, st.session_state.selected_room_id)
        components.html(svg, height=700, scrolling=True)
        st.download_button("Download SVG", data=svg, file_name="floorplan.svg",
                            mime="image/svg+xml", use_container_width=True)

with edit_col:
    if plan is None:
        st.caption("Room editor appears once a plan is generated.")
    else:
        st.markdown("#### Rooms")
        names = [f"{r.name} ({round(r.w*r.h*plan.sqft_per_unit2)} sf)" for r in plan.rooms]
        ids = [r.id for r in plan.rooms]
        default_idx = ids.index(st.session_state.selected_room_id) if st.session_state.selected_room_id in ids else 0
        if names:
            picked = st.radio("Select a room to edit", names, index=default_idx,
                               label_visibility="collapsed")
            st.session_state.selected_room_id = ids[names.index(picked)]

        room = next((r for r in plan.rooms if r.id == st.session_state.selected_room_id), None)
        if room:
            st.markdown("---")
            st.markdown(f"##### Edit \u201c{room.name}\u201d")
            room.name = st.text_input("Name", value=room.name, key=f"name_{room.id}")
            room.type = st.selectbox("Type (sets color)", ROOM_TYPES,
                                      index=ROOM_TYPES.index(room.type) if room.type in ROOM_TYPES else 0,
                                      key=f"type_{room.id}")

            st.markdown("**Furniture**")
            add_col1, add_col2 = st.columns([2, 1])
            with add_col1:
                new_f_type = st.selectbox("Add item", FURN_TYPES, key=f"newf_{room.id}",
                                           label_visibility="collapsed")
            with add_col2:
                if st.button("Add", key=f"addbtn_{room.id}", use_container_width=True):
                    room.furniture.append(Furniture(
                        id=f"f_{len(room.furniture)}_{new_f_type}_{room.id}",
                        type=new_f_type, x=room.w / 2, y=room.h / 2, rot=0,
                    ))
                    st.rerun()

            for fu in list(room.furniture):
                with st.expander(f"{fu.type} \u2014 ({round(fu.x)}, {round(fu.y)})"):
                    fu.x = st.slider("X position", 0.0, float(room.w), float(min(fu.x, room.w)),
                                      key=f"fx_{fu.id}")
                    fu.y = st.slider("Y position", 0.0, float(room.h), float(min(fu.y, room.h)),
                                      key=f"fy_{fu.id}")
                    fu.rot = st.select_slider("Rotation", [0, 90, 180, 270], value=fu.rot,
                                               key=f"frot_{fu.id}")
                    if st.button("Remove", key=f"delf_{fu.id}"):
                        room.furniture.remove(fu)
                        st.rerun()

            st.markdown("---")
            if st.button("Delete room", key=f"delroom_{room.id}", use_container_width=True):
                plan.rooms = [r for r in plan.rooms if r.id != room.id]
                st.session_state.selected_room_id = None
                st.rerun()
