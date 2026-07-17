"""
Core logic for the AI floor plan generator.

Kept free of any Streamlit imports so it can be unit tested and reused
(e.g. from a CLI or a different UI) independently of the app shell.
"""
from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from typing import Any

import requests

CANVAS_W = 920
CANVAS_H = 640
MARGIN = 30

THEMES: dict[str, dict[str, Any]] = {
    "blueprint": {
        "wall": "#3D444B",
        "bg": "#FFFFFF",
        "types": {
            "office": "#E6ECF6", "meeting": "#E3F1EC", "workstation": "#EEF0F1",
            "reception": "#F7ECD8", "storage": "#ECECE9", "kitchen": "#EAF2E2",
            "restroom": "#F3E4E6", "corridor": "#F2F3F1", "lounge": "#F0EEF7",
            "server": "#E6E9F6", "phonebooth": "#F1E6F0", "other": "#EFEFEC",
        },
    },
    "modern": {
        "wall": "#232323",
        "bg": "#FCFCFA",
        "types": {
            "office": "#DCE7FB", "meeting": "#D8F0E6", "workstation": "#F2F2EE",
            "reception": "#FBE6C2", "storage": "#E6E6E1", "kitchen": "#DEF0CE",
            "restroom": "#F6D9DE", "corridor": "#EFEFEA", "lounge": "#E9E2F7",
            "server": "#DBE1FB", "phonebooth": "#F6DDF0", "other": "#EAEAE5",
        },
    },
    "warm": {
        "wall": "#4A3B2E",
        "bg": "#FBF8F3",
        "types": {
            "office": "#F1E3D0", "meeting": "#E7EEDB", "workstation": "#F5EFE4",
            "reception": "#F0D9B5", "storage": "#E9E2D4", "kitchen": "#E3ECD3",
            "restroom": "#F0DAD8", "corridor": "#F3EEE3", "lounge": "#EBE1F0",
            "server": "#E2E6EE", "phonebooth": "#F0DEE9", "other": "#EFE8DC",
        },
    },
}

ROOM_TYPES = ["office", "meeting", "workstation", "reception", "storage", "kitchen",
              "restroom", "corridor", "lounge", "server", "phonebooth", "other"]
FURN_TYPES = ["desk4", "table-round", "table-rect", "chair", "shelf", "plant",
              "counter", "locker", "sink", "rack"]


# ---------------------------------------------------------------- data model

@dataclass
class Furniture:
    id: str
    type: str
    x: float
    y: float
    rot: float = 0


@dataclass
class Door:
    wall: str  # top | bottom | left | right
    pos: float


@dataclass
class Room:
    id: str
    name: str
    x: float
    y: float
    w: float
    h: float
    type: str = "other"
    furniture: list[Furniture] = field(default_factory=list)
    doors: list[Door] = field(default_factory=list)


@dataclass
class Plan:
    bw: float
    bh: float
    rooms: list[Room] = field(default_factory=list)
    sqft_per_unit2: float = 1.0


# ---------------------------------------------------------------- prompting

def build_system_prompt() -> str:
    return (
        'You are a compact JSON floor-plan generator for an interior design tool. '
        'Output ONLY minified JSON, no markdown fences, no prose, no explanation. '
        'Schema: {"bw":900,"bh":640,"r":[{"i":"r1","n":"Room Name","x":0,"y":0,"w":0,"h":0,'
        '"t":"type","f":[["ftype",relX,relY,rot]],"d":[["wall",posAlongWall]]}]} '
        "bw/bh = building width/height in the same units as room x/y/w/h (building is exactly bw x bh). "
        "Rooms: x,y is top-left corner in building coordinates. Rooms must NOT overlap and must stay "
        "within 0..bw and 0..bh. Leave no meaningless gaps: adjacent rooms should share walls where "
        "sensible, like a real architectural plan. "
        "t (room type) must be one of: " + ", ".join(ROOM_TYPES) + ". "
        "f (furniture) items are relative to the room top-left corner (0,0 to w,h). ftype must be one of: "
        + ", ".join(FURN_TYPES) + ". rot is 0, 90, 180 or 270. Include 1-5 sensible furniture items per "
        "room based on its type and size. "
        'd (doors) items are ["wall",posAlongWall] where wall is one of top,bottom,left,right and '
        "posAlongWall is the distance in units from that wall's starting corner to the door center. "
        "Give every room at least one door on a wall that borders a corridor, another room, or the "
        "building exterior. "
        "Use 6 to 9 rooms total, sized realistically relative to the described headcount and total area. "
        "Keep the JSON compact: short numbers, no whitespace beyond what JSON requires, no trailing "
        "commentary."
    )


def build_user_prompt(brief: str, bw: float, bh: float) -> str:
    return f"Building canvas is exactly {bw:.0f} x {bh:.0f} units. Brief: {brief} Return the JSON now."


# ---------------------------------------------------------------- model calls

class ModelError(RuntimeError):
    pass


def call_anthropic(system_prompt: str, user_prompt: str, api_key: str,
                    model: str = "claude-sonnet-5", max_tokens: int = 1500,
                    timeout: int = 60) -> str:
    if not api_key:
        raise ModelError("No Anthropic API key set. Add one in the sidebar, or switch backend.")
    resp = requests.post(
        "https://api.anthropic.com/v1/messages",
        headers={
            "x-api-key": api_key,
            "anthropic-version": "2023-06-01",
            "content-type": "application/json",
        },
        json={
            "model": model,
            "max_tokens": max_tokens,
            "system": system_prompt,
            "messages": [{"role": "user", "content": user_prompt}],
        },
        timeout=timeout,
    )
    if not resp.ok:
        raise ModelError(f"Anthropic API returned {resp.status_code}: {resp.text[:300]}")
    data = resp.json()
    for block in data.get("content", []):
        if block.get("type") == "text":
            return block["text"]
    raise ModelError("No text content in Anthropic response.")


def call_openai_compatible(system_prompt: str, user_prompt: str, base_url: str,
                            model: str, api_key: str | None = None,
                            timeout: int = 120) -> str:
    """Works with Ollama, vLLM, LM Studio, TGI, or any OpenAI-compatible
    /v1/chat/completions server — this is how you plug in an open-source
    model such as Llama 3, Mistral, or Qwen."""
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    resp = requests.post(
        base_url,
        headers=headers,
        json={
            "model": model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "stream": False,
        },
        timeout=timeout,
    )
    if not resp.ok:
        raise ModelError(f"Custom endpoint returned {resp.status_code}: {resp.text[:300]}")
    data = resp.json()
    try:
        return data["choices"][0]["message"]["content"]
    except (KeyError, IndexError) as exc:
        raise ModelError("Unexpected response shape from custom endpoint.") from exc


# ---------------------------------------------------------------- parsing

def extract_json(text: str) -> dict:
    t = text.strip()
    if t.startswith("```"):
        t = t.split("```", 2)[1]
        if t.lower().startswith("json"):
            t = t[4:]
    start, end = t.find("{"), t.rfind("}")
    if start == -1 or end == -1:
        raise ModelError("Could not find JSON in model output.")
    return json.loads(t[start:end + 1])


def _clamp(v, lo, hi):
    if not isinstance(v, (int, float)):
        v = lo
    return min(max(v, lo), hi)


def normalize_plan(raw: dict, sqft: float) -> Plan:
    bw = raw.get("bw") or (CANVAS_W - MARGIN * 2)
    bh = raw.get("bh") or (CANVAS_H - MARGIN * 2)
    rooms: list[Room] = []
    for idx, rm in enumerate(raw.get("r", [])):
        furniture = [
            Furniture(id=f"f{idx}_{fi}", type=f[0] if len(f) > 0 else "chair",
                      x=f[1] if len(f) > 1 else 0, y=f[2] if len(f) > 2 else 0,
                      rot=f[3] if len(f) > 3 else 0)
            for fi, f in enumerate(rm.get("f", []))
        ]
        doors = [Door(wall=d[0] if len(d) > 0 else "bottom", pos=d[1] if len(d) > 1 else 10)
                 for d in rm.get("d", [])]
        rooms.append(Room(
            id=rm.get("i") or f"room{idx}",
            name=rm.get("n") or "Room",
            x=_clamp(rm.get("x"), 0, bw), y=_clamp(rm.get("y"), 0, bh),
            w=max(20, rm.get("w") or 100), h=max(20, rm.get("h") or 100),
            type=rm.get("t") or "other",
            furniture=furniture, doors=doors,
        ))
    area_units = bw * bh
    scale = math.sqrt(sqft / area_units) if area_units > 0 else 1.0
    return Plan(bw=bw, bh=bh, rooms=rooms, sqft_per_unit2=scale * scale)


def generate_plan(brief: str, sqft: float, aspect: float, backend: str, **backend_kwargs) -> Plan:
    """High-level entry point: builds prompts, calls the chosen backend,
    parses the result into a Plan. backend is 'anthropic' or 'custom'."""
    bh = round(math.sqrt((CANVAS_W - MARGIN * 2) * (CANVAS_H - MARGIN * 2) / aspect))
    bw = round(bh * aspect)
    w = min(bw, CANVAS_W - MARGIN * 2)
    h = min(bh, CANVAS_H - MARGIN * 2)

    system_prompt = build_system_prompt()
    user_prompt = build_user_prompt(brief, w, h)

    if backend == "anthropic":
        text = call_anthropic(system_prompt, user_prompt, **backend_kwargs)
    elif backend == "custom":
        text = call_openai_compatible(system_prompt, user_prompt, **backend_kwargs)
    else:
        raise ModelError(f"Unknown backend: {backend}")

    raw = extract_json(text)
    return normalize_plan(raw, sqft)


# ---------------------------------------------------------------- SVG rendering

def _esc(s: str) -> str:
    return (str(s).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
            .replace('"', "&quot;"))


def _furniture_svg(fu: Furniture, wall_color: str) -> str:
    s = f'<g transform="translate({fu.x},{fu.y}) rotate({fu.rot})">'
    t = fu.type
    if t == "desk4":
        s += (f'<rect x="-18" y="-18" width="36" height="36" rx="3" fill="#fff" '
              f'stroke="{wall_color}" stroke-width="1.3"/>'
              f'<line x1="-18" y1="0" x2="18" y2="0" stroke="{wall_color}" stroke-width="1"/>'
              f'<line x1="0" y1="-18" x2="0" y2="18" stroke="{wall_color}" stroke-width="1"/>')
        for cx, cy in [(-9, -9), (9, -9), (-9, 9), (9, 9)]:
            s += f'<circle cx="{cx}" cy="{cy}" r="3" fill="none" stroke="{wall_color}" stroke-width="0.9"/>'
    elif t == "table-round":
        s += f'<circle cx="0" cy="0" r="20" fill="#fff" stroke="{wall_color}" stroke-width="1.3"/>'
        for i in range(6):
            a = (i / 6) * 2 * math.pi
            s += (f'<circle cx="{math.cos(a)*27:.1f}" cy="{math.sin(a)*27:.1f}" r="3.4" '
                  f'fill="none" stroke="{wall_color}" stroke-width="0.9"/>')
    elif t == "table-rect":
        s += f'<rect x="-30" y="-14" width="60" height="28" rx="2" fill="#fff" stroke="{wall_color}" stroke-width="1.3"/>'
        for i in range(-2, 3):
            s += (f'<circle cx="{i*11}" cy="-20" r="3" fill="none" stroke="{wall_color}" stroke-width="0.9"/>'
                  f'<circle cx="{i*11}" cy="20" r="3" fill="none" stroke="{wall_color}" stroke-width="0.9"/>')
    elif t == "chair":
        s += (f'<rect x="-6" y="-6" width="12" height="12" rx="2" fill="#fff" stroke="{wall_color}" stroke-width="1"/>'
              f'<line x1="-6" y1="-6" x2="6" y2="-6" stroke="{wall_color}" stroke-width="2"/>')
    elif t == "shelf":
        s += f'<rect x="-22" y="-9" width="44" height="18" fill="#fff" stroke="{wall_color}" stroke-width="1.2"/>'
        for dy in (-4.5, 0, 4.5):
            s += f'<line x1="-22" y1="{dy}" x2="22" y2="{dy}" stroke="{wall_color}" stroke-width="0.7"/>'
    elif t == "plant":
        s += f'<circle cx="0" cy="0" r="9" fill="#fff" stroke="{wall_color}" stroke-width="1.1"/>'
        for i in range(6):
            a = (i / 6) * 2 * math.pi
            s += (f'<line x1="0" y1="0" x2="{math.cos(a)*7:.1f}" y2="{math.sin(a)*7:.1f}" '
                  f'stroke="{wall_color}" stroke-width="0.8"/>')
    elif t == "counter":
        s += f'<path d="M -28,-10 H 28 V 10 H 0 V -10" fill="#fff" stroke="{wall_color}" stroke-width="1.2"/>'
    elif t == "locker":
        s += f'<rect x="-16" y="-20" width="32" height="40" fill="#fff" stroke="{wall_color}" stroke-width="1.2"/>'
        for dx in (-8, 0, 8):
            s += f'<line x1="{dx}" y1="-20" x2="{dx}" y2="20" stroke="{wall_color}" stroke-width="0.7"/>'
    elif t == "sink":
        s += (f'<rect x="-14" y="-9" width="28" height="18" rx="4" fill="#fff" stroke="{wall_color}" stroke-width="1.1"/>'
              f'<circle cx="0" cy="0" r="5" fill="none" stroke="{wall_color}" stroke-width="0.9"/>')
    elif t == "rack":
        s += f'<rect x="-10" y="-22" width="20" height="44" fill="#fff" stroke="{wall_color}" stroke-width="1.2"/>'
        for dy in range(-16, 17, 6):
            s += f'<line x1="-10" y1="{dy}" x2="10" y2="{dy}" stroke="{wall_color}" stroke-width="0.6"/>'
    else:
        s += f'<circle cx="0" cy="0" r="6" fill="none" stroke="{wall_color}" stroke-width="1"/>'
    s += "</g>"
    return s


def _door_svg(room: Room, door: Door, wall_color: str, bg_color: str) -> str:
    w = 26
    pos = door.pos
    if door.wall == "top":
        x1, y1 = room.x + pos, room.y
        x2, y2 = x1, y1 + w
        arc = f"M {x1} {y1} A {w} {w} 0 0 1 {x1+w} {y1}"
    elif door.wall == "bottom":
        x1, y1 = room.x + pos, room.y + room.h
        x2, y2 = x1, y1 - w
        arc = f"M {x1} {y1} A {w} {w} 0 0 0 {x1+w} {y1}"
    elif door.wall == "left":
        x1, y1 = room.x, room.y + pos
        x2, y2 = x1 + w, y1
        arc = f"M {x1} {y1} A {w} {w} 0 0 0 {x1} {y1+w}"
    else:
        x1, y1 = room.x + room.w, room.y + pos
        x2, y2 = x1 - w, y1
        arc = f"M {x1} {y1} A {w} {w} 0 0 1 {x1} {y1+w}"
    return (
        f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" stroke="{bg_color}" stroke-width="6"/>'
        f'<path d="{arc}" fill="none" stroke="{wall_color}" stroke-width="1" stroke-dasharray="2,2"/>'
        f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" stroke="{wall_color}" stroke-width="1.4"/>'
    )


def render_svg(plan: Plan, theme_key: str = "blueprint", selected_room_id: str | None = None) -> str:
    theme = THEMES.get(theme_key, THEMES["blueprint"])
    wall = theme["wall"]
    bg = theme["bg"]
    W, H = plan.bw + MARGIN * 2, plan.bh + MARGIN * 2

    parts = [f'<svg viewBox="0 0 {W} {H}" width="{min(920, W):.0f}" '
             f'xmlns="http://www.w3.org/2000/svg" font-family="Inter, sans-serif">']
    parts.append(f'<rect x="0" y="0" width="{W}" height="{H}" fill="{bg}"/>')

    grid = []
    x = 0
    while x <= plan.bw:
        grid.append(f'<line x1="{x+MARGIN}" y1="{MARGIN}" x2="{x+MARGIN}" y2="{plan.bh+MARGIN}" '
                     f'stroke="#000" stroke-width="0.15" opacity="0.06"/>')
        x += 20
    y = 0
    while y <= plan.bh:
        grid.append(f'<line x1="{MARGIN}" y1="{y+MARGIN}" x2="{plan.bw+MARGIN}" y2="{y+MARGIN}" '
                     f'stroke="#000" stroke-width="0.15" opacity="0.06"/>')
        y += 20
    parts.append("<g>" + "".join(grid) + "</g>")

    parts.append(f'<g transform="translate({MARGIN},{MARGIN})">')
    parts.append(f'<rect x="0" y="0" width="{plan.bw}" height="{plan.bh}" fill="none" '
                 f'stroke="{wall}" stroke-width="7" rx="6"/>')

    for rm in plan.rooms:
        fill = theme["types"].get(rm.type, theme["types"]["other"])
        is_sel = rm.id == selected_room_id
        stroke = "#B8802E" if is_sel else wall
        sw = 3 if is_sel else 4.5
        parts.append(f'<g>')
        parts.append(f'<rect x="{rm.x}" y="{rm.y}" width="{rm.w}" height="{rm.h}" '
                     f'fill="{fill}" stroke="{stroke}" stroke-width="{sw}"/>')
        for fu in rm.furniture:
            fx = _clamp(fu.x, 4, rm.w - 4)
            fy = _clamp(fu.y, 4, rm.h - 4)
            fu_c = Furniture(id=fu.id, type=fu.type, x=fx, y=fy, rot=fu.rot)
            parts.append(f'<g transform="translate({rm.x},{rm.y})">' + _furniture_svg(fu_c, wall) + "</g>")
        for d in rm.doors:
            parts.append(_door_svg(rm, d, wall, bg))
        area_sqft = round(rm.w * rm.h * plan.sqft_per_unit2)
        parts.append(f'<text x="{rm.x+8}" y="{rm.y+16}" font-size="12" font-weight="600" '
                     f'fill="#3B4B6B">{_esc(rm.name)}</text>')
        parts.append(f'<text x="{rm.x+8}" y="{rm.y+29}" font-size="9.5" fill="#8B939B" '
                     f'font-family="JetBrains Mono, monospace">{area_sqft} sq ft</text>')
        parts.append("</g>")

    cx, cy = plan.bw - 26, 24
    parts.append(
        f'<g transform="translate({cx},{cy})">'
        f'<circle r="16" fill="none" stroke="{wall}" stroke-width="1"/>'
        f'<line x1="0" y1="12" x2="0" y2="-12" stroke="{wall}" stroke-width="1"/>'
        f'<polygon points="0,-12 -3,-6 3,-6" fill="{wall}"/>'
        f'<text x="0" y="-19" text-anchor="middle" font-size="8" fill="#8B939B">N</text>'
        "</g>"
    )

    parts.append("</g></svg>")
    return "".join(parts)
