"""
USDA Crop Stress Analyst - Comparative Moisture Index Analysis
Powered by HP ZGX Nano AI Station

A Vision Language Model demo for on-prem agricultural remote sensing analysis.
Compares paired Sentinel-2 moisture index imagery (stress year vs. normal year),
identifies regional crop stress patterns, quantifies impact extent, and recommends
USDA program intervention priorities.

Uses Qwen3-VL-8B-Instruct-FP8 served via vLLM. Both images are passed to the
model in a single multimodal call so the comparison reasoning happens in one
forward pass rather than two independent analyses stitched together.

Compliance by Architecture: all imagery, all inference, all analyst output
stays on the ZGX Nano. Zero cloud dependency.
"""

import os
import io
import base64
import random
import string
import time
import logging
import traceback
from datetime import datetime, timezone
from pathlib import Path

from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
import httpx


# -- Logging setup -----------------------------------------------------------
# Robust logging from the start: every pipeline stage logs INFO with timing,
# failures get full tracebacks. No silent failures.
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("usda-crop-stress")

# -- Configuration -----------------------------------------------------------
HOST = os.environ.get("HOST", "0.0.0.0")
PORT = int(os.environ.get("PORT", 8000))
VLLM_BASE_URL = os.environ.get("VLLM_BASE_URL", "http://localhost:8090/v1")
VLLM_MODEL = os.environ.get("VLLM_MODEL", "/models/Qwen3-VL-8B-Instruct-FP8")
VLLM_TIMEOUT = float(os.environ.get("VLLM_TIMEOUT", "300"))

# -- Agricultural regions matching the demo imagery --------------------------
# These align with the Sentinel-2 moisture index footprints over the
# Ogallala-aquifer-irrigated High Plains, where the imagery for this demo
# was captured. Synthetic coordinates fall within each region's bounding box.
REGIONS = {
    "ok_tx_panhandle": {
        "name": "Oklahoma / Texas Panhandle",
        "lat_range": (36.4, 37.0),
        "lon_range": (-102.0, -100.5),
        "landmarks": ["Guymon, OK", "Liberal, KS", "Hooker, OK", "Beaver, OK"],
        "primary_crops": ["Winter wheat", "Grain sorghum", "Corn (irrigated)", "Cotton"],
        "irrigation": "Heavy center-pivot from Ogallala Aquifer",
        "fsa_counties": ["Texas County OK", "Cimarron County OK", "Seward County KS"],
    },
    "western_kansas": {
        "name": "Western Kansas (Garden City / Dodge City)",
        "lat_range": (37.6, 38.1),
        "lon_range": (-101.0, -100.0),
        "landmarks": ["Dodge City, KS", "Garden City, KS", "Hutchinson, KS"],
        "primary_crops": ["Winter wheat", "Corn (irrigated)", "Grain sorghum", "Alfalfa"],
        "irrigation": "Extensive center-pivot, Ogallala Aquifer dependent",
        "fsa_counties": ["Finney County KS", "Ford County KS", "Gray County KS"],
    },
    "western_oklahoma": {
        "name": "Western Oklahoma (Boise City / Clayton)",
        "lat_range": (36.5, 37.1),
        "lon_range": (-103.3, -102.4),
        "landmarks": ["Boise City, OK", "Clayton, NM", "Dalhart, TX"],
        "primary_crops": ["Winter wheat", "Grain sorghum", "Rangeland"],
        "irrigation": "Mixed dryland and limited center-pivot",
        "fsa_counties": ["Cimarron County OK", "Union County NM"],
    },
    "tx_panhandle": {
        "name": "Texas Panhandle (Dalhart / Texhoma)",
        "lat_range": (35.8, 36.4),
        "lon_range": (-102.8, -101.7),
        "landmarks": ["Dalhart, TX", "Texhoma, TX", "Perryton, TX"],
        "primary_crops": ["Corn (irrigated)", "Cotton", "Winter wheat", "Cattle"],
        "irrigation": "High-intensity center-pivot, declining Ogallala",
        "fsa_counties": ["Dallam County TX", "Hartley County TX", "Sherman County TX"],
    },
    "sw_nebraska": {
        "name": "Southwest Nebraska",
        "lat_range": (40.2, 40.9),
        "lon_range": (-101.5, -100.0),
        "landmarks": ["McCook, NE", "Imperial, NE", "Grant, NE"],
        "primary_crops": ["Corn (irrigated)", "Soybeans", "Winter wheat", "Dry beans"],
        "irrigation": "Center-pivot, Ogallala-dependent",
        "fsa_counties": ["Chase County NE", "Hayes County NE", "Perkins County NE"],
    },
    "southern_high_plains": {
        "name": "Southern High Plains (regional view)",
        "lat_range": (35.0, 38.0),
        "lon_range": (-103.5, -100.0),
        "landmarks": ["Multi-state regional analysis"],
        "primary_crops": ["Winter wheat", "Corn", "Grain sorghum", "Cotton"],
        "irrigation": "Mixed dryland and center-pivot",
        "fsa_counties": ["Multi-county regional view"],
    },
}

# -- Stress severity tiers ---------------------------------------------------
# Replaces the manned/unmanned classification from the source demo with a
# domain-appropriate severity scale derived from VLM analysis output.
STRESS_TIERS = {
    "EXTREME": {
        "label": "EXTREME DROUGHT STRESS",
        "color_hex": "#8B0000",
        "description": "Vast majority of agricultural land showing severe moisture deficit. Crop failure widespread on non-irrigated land.",
    },
    "SEVERE": {
        "label": "SEVERE DROUGHT STRESS",
        "color_hex": "#D94040",
        "description": "Most rangeland and dryland showing low moisture. Irrigated parcels are isolated islands of green.",
    },
    "MODERATE": {
        "label": "MODERATE STRESS",
        "color_hex": "#E8A317",
        "description": "Mixed pattern of stressed and healthy land. Irrigated agriculture still productive but dryland under pressure.",
    },
    "MILD": {
        "label": "MILD / LOCALIZED STRESS",
        "color_hex": "#D4A843",
        "description": "Localized stress patches. Most agricultural land showing adequate moisture for crop development.",
    },
    "HEALTHY": {
        "label": "HEALTHY / ADEQUATE MOISTURE",
        "color_hex": "#3DBD5D",
        "description": "Strong moisture availability across the region. Both irrigated and dryland agriculture in good condition.",
    },
}

# -- FastAPI setup -----------------------------------------------------------
app = FastAPI(
    title="USDA Crop Stress Analyst - Comparative Moisture Index Analysis",
    description="VLM-powered comparative analysis of paired Sentinel-2 moisture index imagery",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

FRONTEND_DIR = os.environ.get("FRONTEND_DIR", "/app/frontend")
frontend_path = Path(FRONTEND_DIR)
if frontend_path.exists():
    app.mount("/static", StaticFiles(directory=frontend_path), name="static")
else:
    logger.warning("Frontend directory not found at %s", FRONTEND_DIR)

http_client = httpx.AsyncClient(timeout=VLLM_TIMEOUT)


# -- VLM prompt --------------------------------------------------------------
# The system prompt is the heart of the demo. It establishes the analyst
# persona, explains the moisture index color scale, and asks for a structured
# differential analysis suitable for a USDA stakeholder.
#
# Curtis's principle from prior work: focused single-call prompts with plain
# text labeled output beat complex multi-field JSON schemas. We use that
# pattern here -- one prompt, both images, labeled lines parsed downstream.
COMPARATIVE_ANALYSIS_PROMPT = """You are a senior agricultural remote sensing analyst working for the U.S. Department of Agriculture. You have been provided two Sentinel-2 satellite moisture index images covering the same geographic region in different growing seasons.

IMAGE INTERPRETATION GUIDE
Both images are rendered using a Normalized Difference Moisture Index (NDMI) color scale derived from Sentinel-2 bands B8A and B11:
- Deep BLUE      = high vegetation moisture content (healthy crops, full canopy)
- LIGHT BLUE / CYAN = adequate moisture (productive cropland)
- GREEN / YELLOW = transitional moisture (stress emerging)
- ORANGE         = moderate moisture deficit (drought stress)
- RED            = severe moisture deficit (failed crops, bare soil, rangeland die-off)

Visible features in High Plains agricultural imagery typically include:
- Center-pivot irrigation circles (geometric circular shapes, often the most vivid blue when irrigated)
- Rectangular dryland field grids (rely entirely on precipitation)
- Native rangeland (irregular, often most stressed in drought years)
- River corridors (linear blue features following drainage)

REGIONAL CONTEXT
{region_context}

ANALYTICAL TASK
Compare IMAGE A and IMAGE B. Identify which image shows greater moisture stress, quantify the visible extent of stress, identify the most vulnerable agricultural areas, and recommend USDA program intervention priorities.

OUTPUT FORMAT
Answer each prompt on its own line in exactly this format. Be specific to what you actually observe in the images, not generic.

STRESS_COMPARISON: [Which image (A or B) shows greater moisture stress? Briefly justify based on visible color distribution.]
STRESS_TIER: [One of: EXTREME, SEVERE, MODERATE, MILD, HEALTHY -- the severity tier of the more-stressed image]
STRESSED_AREA_PERCENT: [Approximate percentage of the more-stressed image showing orange-to-red (moisture-deficient) coloration. Give a single number followed by %.]
IRRIGATION_OBSERVATION: [Describe what you see regarding center-pivot irrigation circles in both images. Are they visually distinct from surrounding land in the stress year?]
DRYLAND_VS_IRRIGATED: [Compare how dryland fields versus irrigated fields are responding in the stress image. What does this tell us about the failure pattern?]
VULNERABLE_AREAS: [Identify the most vulnerable agricultural areas visible in the stress image -- specific landmarks, directions, or land-use patterns.]
PROGRAM_PRIORITY: [Which USDA program category is most relevant given what you see: CROP_INSURANCE (RMA), LIVESTOCK_FORAGE (FSA LFP), CONSERVATION_SUPPORT (NRCS EQIP), or DROUGHT_MONITOR_ESCALATION. Pick one.]
ANALYST_SUMMARY: [Two to three sentence comparative summary suitable for a USDA stakeholder briefing.]"""


# -- vLLM interaction --------------------------------------------------------

async def query_vlm_two_images(
    image_a_b64: str,
    image_b_b64: str,
    prompt: str,
    max_tokens: int = 1024,
) -> tuple[str, dict]:
    """
    Send TWO images + a prompt to Qwen3-VL in a single multimodal call.

    The comparison reasoning happens in one forward pass -- this is critical
    for the demo narrative. Two independent single-image analyses stitched
    together do not produce the same quality of differential analysis.

    Returns (content_text, usage_dict).
    """
    payload = {
        "model": VLLM_MODEL,
        "messages": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/jpeg;base64,{image_a_b64}"},
                    },
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/jpeg;base64,{image_b_b64}"},
                    },
                    {"type": "text", "text": prompt},
                ],
            }
        ],
        "max_tokens": max_tokens,
        "temperature": 0.3,
        "chat_template_kwargs": {"enable_thinking": False},
    }

    t0 = time.time()
    try:
        logger.info("Submitting two-image comparison call to vLLM (max_tokens=%d)...", max_tokens)
        response = await http_client.post(
            f"{VLLM_BASE_URL}/chat/completions",
            json=payload,
        )
        response.raise_for_status()
        data = response.json()
        content = data["choices"][0]["message"]["content"].strip()
        usage = data.get("usage", {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0})
        elapsed = time.time() - t0
        logger.info(
            "vLLM call complete in %.2fs (prompt=%d, completion=%d, total=%d)",
            elapsed,
            usage.get("prompt_tokens", 0),
            usage.get("completion_tokens", 0),
            usage.get("total_tokens", 0),
        )
        return content, usage
    except httpx.HTTPStatusError as e:
        logger.error("vLLM HTTP error %d: %s", e.response.status_code, e.response.text[:500])
        raise
    except Exception as e:
        logger.error("vLLM call failed after %.2fs: %s", time.time() - t0, e)
        logger.debug(traceback.format_exc())
        raise


# -- VLM output parsing ------------------------------------------------------

PARSED_FIELDS = [
    "stress_comparison",
    "stress_tier",
    "stressed_area_percent",
    "irrigation_observation",
    "dryland_vs_irrigated",
    "vulnerable_areas",
    "program_priority",
    "analyst_summary",
]


def parse_vlm_output(raw_response: str) -> dict:
    """Parse the labeled-line VLM output into a dict.

    Tolerant of minor formatting drift -- matches by label prefix only,
    case-insensitive. Missing fields get explicit "Not provided" sentinels
    rather than empty strings so the report renders cleanly.
    """
    parsed = {field: "Not provided by model" for field in PARSED_FIELDS}

    for line in raw_response.strip().split("\n"):
        line = line.strip()
        if not line:
            continue
        for field in PARSED_FIELDS:
            tag = field.upper() + ":"
            if line.upper().startswith(tag):
                value = line[len(tag):].strip()
                if value:
                    parsed[field] = value
                break

    logger.info("Parsed VLM output fields: %s", {k: (v[:60] + "..." if len(v) > 60 else v) for k, v in parsed.items()})
    return parsed


# -- Classification ----------------------------------------------------------

def classify_stress_tier(parsed: dict) -> tuple[str, dict]:
    """Map the VLM's STRESS_TIER output to one of the canonical tiers.

    Defensive matching -- the model might write "Severe drought stress" or
    "SEVERE" or "severe." We normalize to one of the five canonical keys.
    """
    raw = (parsed.get("stress_tier") or "").upper()

    # Match in order from most-severe to least-severe so that ambiguous
    # responses bias toward more conservative (more-stressed) classification.
    for tier in ["EXTREME", "SEVERE", "MODERATE", "MILD", "HEALTHY"]:
        if tier in raw:
            return tier, STRESS_TIERS[tier]

    logger.warning("Could not classify stress tier from: '%s', defaulting to MODERATE", raw)
    return "MODERATE", STRESS_TIERS["MODERATE"]


def parse_stressed_area_percent(parsed: dict) -> float | None:
    """Extract a numeric percentage from the model's STRESSED_AREA_PERCENT field."""
    raw = parsed.get("stressed_area_percent", "")
    if not raw or raw == "Not provided by model":
        return None
    # Strip everything except digits and decimal points
    digits = "".join(c for c in raw if c.isdigit() or c == ".")
    if not digits:
        return None
    try:
        value = float(digits)
        if 0 <= value <= 100:
            return value
    except ValueError:
        pass
    return None


# -- USDA program recommendations -------------------------------------------

USDA_PROGRAM_DETAILS = {
    "CROP_INSURANCE": {
        "agency": "Risk Management Agency (RMA)",
        "program": "Federal Crop Insurance Program",
        "actions": [
            "Coordinate with Approved Insurance Providers (AIPs) on anticipated claim surge",
            "Flag affected counties for accelerated loss adjuster deployment",
            "Pull Yield Protection (YP) and Revenue Protection (RP) policy concentration data",
            "Coordinate with RMA Regional Office on Prevented Planting determinations",
        ],
    },
    "LIVESTOCK_FORAGE": {
        "agency": "Farm Service Agency (FSA)",
        "program": "Livestock Forage Disaster Program (LFP)",
        "actions": [
            "Initiate D2+ drought designation review with US Drought Monitor",
            "Stage county committee meetings for grazing loss certifications",
            "Pre-position payment calculation data for eligible producers",
            "Coordinate with Emergency Conservation Program (ECP) for water-source assistance",
        ],
    },
    "CONSERVATION_SUPPORT": {
        "agency": "Natural Resources Conservation Service (NRCS)",
        "program": "Environmental Quality Incentives Program (EQIP) - Drought Initiative",
        "actions": [
            "Prioritize irrigation efficiency and soil-health practices in affected counties",
            "Open special signup for EQIP drought-resilience contracts",
            "Coordinate with state conservationist on Ogallala Aquifer Initiative allocations",
            "Stage technical assistance for cover crop and residue management transitions",
        ],
    },
    "DROUGHT_MONITOR_ESCALATION": {
        "agency": "USDA / NOAA / NDMC",
        "program": "U.S. Drought Monitor Coordination",
        "actions": [
            "Submit imagery and ground-truth data to National Drought Mitigation Center",
            "Coordinate weekly USDM author briefing for category escalation",
            "Cross-reference with NASS Crop Progress and Condition Report",
            "Brief Secretarial Disaster Designation review board",
        ],
    },
}


def build_program_recommendations(parsed: dict, region_data: dict) -> dict:
    """Build a structured USDA program recommendation block."""
    priority_raw = (parsed.get("program_priority") or "").upper()

    # Match against canonical program keys
    matched_program = None
    for key in USDA_PROGRAM_DETAILS:
        if key in priority_raw:
            matched_program = key
            break

    if not matched_program:
        # Default fallback based on stress tier -- if model didn't pick a
        # clean program key, infer from severity.
        tier = parsed.get("stress_tier", "").upper()
        if "EXTREME" in tier or "SEVERE" in tier:
            matched_program = "DROUGHT_MONITOR_ESCALATION"
        elif "MODERATE" in tier:
            matched_program = "CROP_INSURANCE"
        else:
            matched_program = "CONSERVATION_SUPPORT"
        logger.info("Program priority defaulted to %s based on stress tier", matched_program)

    program = USDA_PROGRAM_DETAILS[matched_program]

    return {
        "priority_key": matched_program,
        "agency": program["agency"],
        "program": program["program"],
        "actions": program["actions"],
        "affected_counties": region_data.get("fsa_counties", []),
    }


# -- Region / synthetic geolocation helpers ---------------------------------

def build_region_context(region_key: str) -> tuple[str, dict]:
    """Build the region context block injected into the VLM prompt."""
    region = REGIONS.get(region_key, REGIONS["southern_high_plains"])
    landmarks = ", ".join(region["landmarks"])
    crops = ", ".join(region["primary_crops"])
    context = (
        f"Region: {region['name']}\n"
        f"Key landmarks visible: {landmarks}\n"
        f"Primary agricultural production: {crops}\n"
        f"Irrigation profile: {region['irrigation']}"
    )
    return context, region


def generate_synthetic_centroid(region_data: dict) -> dict:
    """Generate a synthetic centroid lat/lon within the region's bounding box.

    Used for the report header so the demo has plausible-looking
    geographic metadata. Not derived from the imagery EXIF (Sentinel-2
    tiles cover ~100km on a side so a centroid is fine for demo purposes).
    """
    lat = random.uniform(*region_data["lat_range"])
    lon = random.uniform(*region_data["lon_range"])

    lat_dir = "N" if lat >= 0 else "S"
    lon_dir = "E" if lon >= 0 else "W"

    lat_abs, lon_abs = abs(lat), abs(lon)
    lat_deg = int(lat_abs)
    lat_min = int((lat_abs - lat_deg) * 60)
    lat_sec = int(((lat_abs - lat_deg) * 60 - lat_min) * 60)
    lon_deg = int(lon_abs)
    lon_min = int((lon_abs - lon_deg) * 60)
    lon_sec = int(((lon_abs - lon_deg) * 60 - lon_min) * 60)

    return {
        "decimal": {"lat": round(lat, 6), "lon": round(lon, 6)},
        "dms": f"{lat_deg}\u00b0{lat_min:02d}'{lat_sec:02d}\"{lat_dir}, {lon_deg}\u00b0{lon_min:02d}'{lon_sec:02d}\"{lon_dir}",
        "region_name": region_data["name"],
        "landmarks": region_data["landmarks"],
        "primary_crops": region_data["primary_crops"],
    }


def generate_report_id() -> str:
    """Generate a USDA-style report identifier."""
    prefix = random.choice(["USDA-CSA", "RMA-DROUGHT", "FSA-LFP", "NDMC-WATCH"])
    date_str = datetime.now(timezone.utc).strftime("%Y%m%d")
    seq = ''.join(random.choices(string.digits, k=4))
    return f"{prefix}-{date_str}-{seq}"


# -- Image preprocessing -----------------------------------------------------

def prepare_image_for_vlm(image: Image.Image, max_dim: int = 1024) -> str:
    """Convert PIL image to base64 JPEG, resizing if needed.

    Qwen3-VL tokenizes vision input at roughly (pixels)/(14^2 * 4) tokens.
    A 1024x1024 image is ~1280 vision tokens. Two such images plus prompt
    and output fit comfortably in an 8192-token context.
    """
    if image.mode != "RGB":
        image = image.convert("RGB")

    if max(image.size) > max_dim:
        ratio = max_dim / max(image.size)
        new_size = (int(image.size[0] * ratio), int(image.size[1] * ratio))
        image = image.resize(new_size, Image.LANCZOS)
        logger.info("Resized image to %s", new_size)

    buffer = io.BytesIO()
    image.save(buffer, format="JPEG", quality=92)
    return base64.b64encode(buffer.getvalue()).decode("utf-8")


# -- Main analysis pipeline --------------------------------------------------

async def run_comparative_analysis(
    image_a: Image.Image,
    image_b: Image.Image,
    region_key: str,
    image_a_label: str,
    image_b_label: str,
    custom_guidance: str = "",
) -> dict:
    """Full comparative analysis pipeline.

    Steps:
      1. Preprocess both images to base64.
      2. Build region context and assemble final prompt.
      3. Single VLM call with both images.
      4. Parse labeled output into structured fields.
      5. Classify stress tier and resolve USDA program priority.
      6. Assemble final report.
    """
    pipeline_t0 = time.time()
    logger.info("=" * 60)
    logger.info("Starting comparative analysis | region=%s", region_key)
    logger.info("  Image A label: %s", image_a_label)
    logger.info("  Image B label: %s", image_b_label)

    # 1. Preprocess
    image_a_b64 = prepare_image_for_vlm(image_a)
    image_b_b64 = prepare_image_for_vlm(image_b)

    # 2. Build prompt
    region_context, region_data = build_region_context(region_key)

    # Inject the user-provided image labels so the model knows which is which
    label_block = (
        f"Image A is labeled by the analyst as: {image_a_label}\n"
        f"Image B is labeled by the analyst as: {image_b_label}"
    )

    final_prompt = COMPARATIVE_ANALYSIS_PROMPT.format(region_context=region_context)
    final_prompt += f"\n\nLABEL CONTEXT\n{label_block}"
    if custom_guidance:
        final_prompt += f"\n\nADDITIONAL ANALYST GUIDANCE\n{custom_guidance}"

    # 3. VLM call
    raw_output, usage = await query_vlm_two_images(
        image_a_b64,
        image_b_b64,
        final_prompt,
        max_tokens=1024,
    )
    logger.debug("Raw VLM output:\n%s", raw_output)

    # 4. Parse
    parsed = parse_vlm_output(raw_output)

    # 5. Classify and resolve priority
    tier_key, tier_data = classify_stress_tier(parsed)
    stressed_pct = parse_stressed_area_percent(parsed)
    program_recommendation = build_program_recommendations(parsed, region_data)

    # 6. Assemble report
    centroid = generate_synthetic_centroid(region_data)
    report = {
        "report_id": generate_report_id(),
        "classification": "UNCLASSIFIED // FOR DEMONSTRATION PURPOSES ONLY",
        "generated_at_utc": datetime.now(timezone.utc).strftime("%d %b %Y %H%MZ").upper(),
        "image_a_label": image_a_label,
        "image_b_label": image_b_label,
        "region": {
            "key": region_key,
            "name": region_data["name"],
            "centroid_dms": centroid["dms"],
            "centroid_decimal": centroid["decimal"],
            "landmarks": region_data["landmarks"],
            "primary_crops": region_data["primary_crops"],
            "irrigation_profile": region_data["irrigation"],
        },
        "assessment": {
            "stress_tier_key": tier_key,
            "stress_tier_label": tier_data["label"],
            "stress_tier_color": tier_data["color_hex"],
            "stress_tier_description": tier_data["description"],
            "stressed_area_percent": stressed_pct,
            "stress_comparison": parsed["stress_comparison"],
            "irrigation_observation": parsed["irrigation_observation"],
            "dryland_vs_irrigated": parsed["dryland_vs_irrigated"],
            "vulnerable_areas": parsed["vulnerable_areas"],
            "analyst_summary": parsed["analyst_summary"],
        },
        "usda_recommendation": program_recommendation,
        "raw_vlm_output": raw_output,
        "token_usage": usage,
        "pipeline_seconds": round(time.time() - pipeline_t0, 2),
    }

    logger.info(
        "Analysis complete in %.2fs | tier=%s | stressed_area=%s%% | priority=%s",
        report["pipeline_seconds"],
        tier_key,
        stressed_pct if stressed_pct is not None else "n/a",
        program_recommendation["priority_key"],
    )
    logger.info("=" * 60)
    return report


# -- API routes --------------------------------------------------------------

@app.get("/", response_class=HTMLResponse)
async def root():
    """Serve the main application page."""
    index_path = Path(FRONTEND_DIR) / "index.html"
    if index_path.exists():
        return HTMLResponse(content=index_path.read_text())
    return HTMLResponse(content="<h1>USDA Crop Stress Analyst</h1><p>index.html not found at " + str(index_path) + "</p>")


@app.get("/api/health")
async def health_check():
    """Health check endpoint. Also verifies vLLM is responsive."""
    vllm_healthy = False
    vllm_error = None
    try:
        resp = await http_client.get(
            f"{VLLM_BASE_URL.replace('/v1', '')}/health",
            timeout=5.0,
        )
        vllm_healthy = resp.status_code == 200
    except Exception as e:
        vllm_error = str(e)

    return {
        "status": "healthy" if vllm_healthy else "degraded",
        "vllm_server": "ready" if vllm_healthy else "not ready",
        "vllm_error": vllm_error,
        "model": "Qwen3-VL-8B-Instruct-FP8",
        "inference_engine": "vLLM",
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }


@app.get("/api/regions")
async def get_regions():
    """Get available agricultural regions."""
    return {
        "regions": [
            {
                "id": k,
                "name": v["name"],
                "landmarks": v["landmarks"],
                "primary_crops": v["primary_crops"],
            }
            for k, v in REGIONS.items()
        ]
    }


@app.post("/api/analyze")
async def analyze_endpoint(
    image_a: UploadFile = File(..., description="First moisture index image"),
    image_b: UploadFile = File(..., description="Second moisture index image"),
    region: str = Form("ok_tx_panhandle"),
    image_a_label: str = Form("August 2023 (stress year)"),
    image_b_label: str = Form("August 2019 (normal year)"),
    custom_guidance: str = Form(""),
):
    """Run a comparative moisture index analysis on a paired image set."""

    # Validate region
    if region not in REGIONS:
        logger.warning("Unknown region '%s', defaulting to ok_tx_panhandle", region)
        region = "ok_tx_panhandle"

    # Validate content types
    for img_field, img_upload in [("image_a", image_a), ("image_b", image_b)]:
        if not img_upload.content_type or not img_upload.content_type.startswith("image/"):
            raise HTTPException(status_code=400, detail=f"{img_field} must be an image")

    # Load images
    try:
        data_a = await image_a.read()
        data_b = await image_b.read()
        for label, data in [("image_a", data_a), ("image_b", data_b)]:
            if len(data) > 20 * 1024 * 1024:
                raise HTTPException(status_code=400, detail=f"{label} too large (max 20MB)")
        img_a = Image.open(io.BytesIO(data_a))
        img_b = Image.open(io.BytesIO(data_b))
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Image loading failed: %s", e)
        raise HTTPException(status_code=400, detail=f"Could not load images: {e}")

    # Run pipeline
    try:
        report = await run_comparative_analysis(
            image_a=img_a,
            image_b=img_b,
            region_key=region,
            image_a_label=image_a_label.strip() or "Image A",
            image_b_label=image_b_label.strip() or "Image B",
            custom_guidance=custom_guidance.strip(),
        )
        return JSONResponse(content=report)
    except Exception as e:
        logger.error("Pipeline failed: %s\n%s", e, traceback.format_exc())
        raise HTTPException(status_code=500, detail=f"Analysis failed: {e}")


@app.on_event("startup")
async def startup_event():
    host_ip = os.environ.get("HOST_IP", "")
    banner = "\n" + "=" * 64 + "\n"
    banner += "  USDA Crop Stress Analyst - Comparative Moisture Index\n"
    banner += "  Qwen3-VL-8B-Instruct-FP8 | HP ZGX Nano | vLLM\n"
    banner += "  Compliance by Architecture: 100% on-prem inference\n"
    banner += "=" * 64 + "\n"
    if host_ip:
        banner += f"\n  \u27a1  http://{host_ip}:{PORT}\n"
    else:
        banner += f"\n  \u27a1  http://localhost:{PORT}\n"
    banner += "=" * 64 + "\n"
    print(banner)
    logger.info("Service started | model=%s | vllm=%s", VLLM_MODEL, VLLM_BASE_URL)


@app.on_event("shutdown")
async def shutdown_event():
    await http_client.aclose()
    logger.info("Service shutting down")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host=HOST, port=PORT)
