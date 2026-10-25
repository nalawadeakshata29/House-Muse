import os
import base64
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

from modules.hf_space_client import call_hf_space_redesign
from modules.caption_engine import ROOM_FURNITURE_REQUIREMENTS, inject_room_furniture_into_prompt
from modules.controlnet_generator import DEFAULT_STYLE_PROMPTS, DEFAULT_NEGATIVE_PROMPT

BASE_DIR = Path(__file__).resolve().parent
test_image = BASE_DIR / ".." / "frontend" / "public" / "test" / "test 1.jpg"
if not test_image.exists():
    test_image = BASE_DIR / ".." / "frontend" / "public" / "test" / "unfurnished room.jpeg"
if not test_image.exists():
    test_image = BASE_DIR / ".." / "frontend" / "public" / "test" / "test.jpg"

space_id = os.getenv("HF_SPACE_ID", "arnav1711/house-muse-interior-sdxl")
hf_token = os.getenv("HF_API_TOKEN")

print(f"Testing HF Space: {space_id}")
print(f"Using image: {test_image}")

# ---------------------------------------------------------------------------
# Test representative styles with room-appropriate furniture requirements
# ---------------------------------------------------------------------------
room_type = "living_room"
req_furniture = ROOM_FURNITURE_REQUIREMENTS[room_type]

base_prompts = {
    "mid_century": DEFAULT_STYLE_PROMPTS["mid_century"],
    "contemporary_indian": DEFAULT_STYLE_PROMPTS["contemporary_indian"],
    "industrial": DEFAULT_STYLE_PROMPTS["industrial"],
}

# Inject room-specific furniture right after refurnishing line
test_styles = []
for style, base_p in base_prompts.items():
    injected_p = inject_room_furniture_into_prompt(base_p, room_type)
    test_styles.append((style, base_p, injected_p))

output_dir = BASE_DIR / "test_outputs"
after_dir = output_dir / "after"
output_dir.mkdir(parents=True, exist_ok=True)
after_dir.mkdir(parents=True, exist_ok=True)

passed = 0
failed = 0

for style, before_prompt, prompt in test_styles:
    print(f"\n==================================================")
    print(f"[Test] Style: {style} (Room Type: {room_type})")
    print(f"[Before Prompt]:\n  {before_prompt}")
    print(f"[After Prompt]:\n  {prompt}")
    print(f"==================================================")
    print(f"[Test] Generating style '{style}' with controlnet_scale=0.55 and window preservation prompt...")
    response = call_hf_space_redesign(
        space_id=space_id,
        image_input=str(test_image),
        style=style,
        prompt=prompt,
        negative_prompt=DEFAULT_NEGATIVE_PROMPT,
        controlnet_scale=0.55,
        lora_scale=0.80,
        num_inference_steps=30,
        hf_token=hf_token,
    )

    if response["success"]:
        img_data = base64.b64decode(response["image_b64"])
        # Save both to test_outputs/ and test_outputs/after/
        out_file = output_dir / f"test_output_{style}.jpg"
        after_file = after_dir / f"test_output_{style}_after.jpg"
        with open(out_file, "wb") as f:
            f.write(img_data)
        with open(after_file, "wb") as f:
            f.write(img_data)
        print(f"PASS: {style} -> {out_file} & {after_file}")
        passed += 1
    else:
        print(f"FAIL: {style} -> {response['error']}")
        failed += 1

print(f"\nTotal passed: {passed}/{len(test_styles)}")
print(f"Total failed: {failed}/{len(test_styles)}")