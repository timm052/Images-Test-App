import os
import math
import base64
import io
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from flask import Flask, request, jsonify, render_template

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 32 * 1024 * 1024  # 32MB limit

ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg", "webp", "bmp"}


def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


def pil_to_cv(pil_img):
    return cv2.cvtColor(np.array(pil_img.convert("RGB")), cv2.COLOR_RGB2BGR)


def cv_to_pil(cv_img):
    return Image.fromarray(cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB))


def detect_and_match(img_a_cv, img_b_cv, max_features=2000):
    """Detect keypoints and match between two images using ORB + AKAZE fallback."""
    gray_a = cv2.cvtColor(img_a_cv, cv2.COLOR_BGR2GRAY)
    gray_b = cv2.cvtColor(img_b_cv, cv2.COLOR_BGR2GRAY)

    # Try ORB first (fast, patent-free)
    detector = cv2.ORB_create(max_features)
    kp_a, desc_a = detector.detectAndCompute(gray_a, None)
    kp_b, desc_b = detector.detectAndCompute(gray_b, None)

    if desc_a is None or desc_b is None or len(kp_a) < 4 or len(kp_b) < 4:
        # Fallback: AKAZE
        detector = cv2.AKAZE_create()
        kp_a, desc_a = detector.detectAndCompute(gray_a, None)
        kp_b, desc_b = detector.detectAndCompute(gray_b, None)

    if desc_a is None or desc_b is None or len(kp_a) < 4 or len(kp_b) < 4:
        return None, None, None

    # BFMatcher with Hamming distance (suits binary descriptors)
    matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
    raw_matches = matcher.knnMatch(desc_a, desc_b, k=2)

    # Lowe's ratio test
    good = []
    for pair in raw_matches:
        if len(pair) == 2:
            m, n = pair
            if m.distance < 0.75 * n.distance:
                good.append(m)

    return kp_a, kp_b, good


def compute_alignment(kp_a, kp_b, good_matches, img_a_shape):
    """
    Use homography to compute translation, rotation, and scale.
    Returns a dict with alignment parameters.
    """
    if len(good_matches) < 4:
        return None

    pts_a = np.float32([kp_a[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
    pts_b = np.float32([kp_b[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)

    H, mask = cv2.findHomography(pts_b, pts_a, cv2.RANSAC, 5.0)
    if H is None:
        return None

    inliers = int(mask.sum()) if mask is not None else 0

    # Decompose homography: H = [[a, b, tx], [c, d, ty], [0, 0, 1]]
    a, b = H[0, 0], H[0, 1]
    c, d = H[1, 0], H[1, 1]
    tx, ty = H[0, 2], H[1, 2]

    scale_x = math.sqrt(a**2 + c**2)
    scale_y = math.sqrt(b**2 + d**2)
    scale = (scale_x + scale_y) / 2.0

    angle_rad = math.atan2(c, a)
    angle_deg = math.degrees(angle_rad)

    h, w = img_a_shape[:2]

    return {
        "tx": tx,
        "ty": ty,
        "angle_deg": angle_deg,
        "scale": scale,
        "inliers": inliers,
        "total_matches": len(good_matches),
        "H": H,
        "mask": mask,
        "pts_a": pts_a,
        "pts_b": pts_b,
        "img_width": w,
        "img_height": h,
    }


def build_instructions(result):
    """Convert numerical alignment into human-readable camera movement instructions."""
    instructions = []
    tx, ty = result["tx"], result["ty"]
    angle = result["angle_deg"]
    scale = result["scale"]
    w, h = result["img_width"], result["img_height"]

    # Threshold: ignore sub-pixel shifts
    px_threshold = max(w, h) * 0.005

    if abs(tx) > px_threshold:
        direction = "right" if tx > 0 else "left"
        pct = abs(tx) / w * 100
        instructions.append(f"Move camera {direction} by ~{abs(tx):.0f}px ({pct:.1f}% of frame width)")

    if abs(ty) > px_threshold:
        direction = "down" if ty > 0 else "up"
        pct = abs(ty) / h * 100
        instructions.append(f"Move camera {direction} by ~{abs(ty):.0f}px ({pct:.1f}% of frame height)")

    angle_threshold = 0.5  # degrees
    if abs(angle) > angle_threshold:
        direction = "clockwise" if angle < 0 else "counter-clockwise"
        instructions.append(f"Rotate camera {direction} by ~{abs(angle):.1f}°")

    scale_threshold = 0.02
    if abs(scale - 1.0) > scale_threshold:
        if scale > 1.0:
            instructions.append(f"Move camera closer (zoom in) by ~{(scale - 1) * 100:.1f}%")
        else:
            instructions.append(f"Move camera farther away (zoom out) by ~{(1 - scale) * 100:.1f}%")

    if not instructions:
        instructions.append("Images are already well-aligned — no significant camera movement needed.")

    return instructions


def draw_arrow(draw, start, end, color, width=3, arrow_size=12):
    """Draw a line with an arrowhead."""
    draw.line([start, end], fill=color, width=width)
    # Arrowhead
    dx = end[0] - start[0]
    dy = end[1] - start[1]
    length = math.hypot(dx, dy)
    if length == 0:
        return
    ux, uy = dx / length, dy / length
    px, py = -uy, ux  # perpendicular
    tip = end
    left = (tip[0] - arrow_size * ux + arrow_size * 0.5 * px,
            tip[1] - arrow_size * uy + arrow_size * 0.5 * py)
    right = (tip[0] - arrow_size * ux - arrow_size * 0.5 * px,
             tip[1] - arrow_size * uy - arrow_size * 0.5 * py)
    draw.polygon([tip, left, right], fill=color)


def generate_result_image(img_a_pil, img_b_pil, kp_a, kp_b, good_matches, result, instructions):
    """
    Compose a side-by-side visualization with:
    - Matched keypoints drawn on each image
    - An aligned overlay panel
    - An instruction panel
    """
    target_h = 400
    def resize_keep_ar(img, height):
        w, h = img.size
        scale = height / h
        return img.resize((int(w * scale), height), Image.LANCZOS), scale

    img_a_r, scale_a = resize_keep_ar(img_a_pil, target_h)
    img_b_r, scale_b = resize_keep_ar(img_b_pil, target_h)

    w_a, h_a = img_a_r.size
    w_b, h_b = img_b_r.size

    # ---- Panel 1 & 2: keypoint matches ----
    cv_a_r = pil_to_cv(img_a_r)
    cv_b_r = pil_to_cv(img_b_r)

    # Scale keypoints to resized images
    kp_a_scaled = [cv2.KeyPoint(kp.pt[0] * scale_a, kp.pt[1] * scale_a, kp.size) for kp in kp_a]
    kp_b_scaled = [cv2.KeyPoint(kp.pt[0] * scale_b, kp.pt[1] * scale_b, kp.size) for kp in kp_b]

    match_img = cv2.drawMatches(
        cv_a_r, kp_a_scaled,
        cv_b_r, kp_b_scaled,
        good_matches[:40],
        None,
        matchColor=(0, 255, 100),
        singlePointColor=(200, 200, 0),
        matchesMask=None,
        flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS,
    )
    panel_match = cv_to_pil(match_img)  # width = w_a + w_b

    # ---- Panel 3: overlay (B warped onto A) ----
    H = result["H"]
    # Scale H to resized image coordinates
    S_a = np.array([[scale_a, 0, 0], [0, scale_a, 0], [0, 0, 1]], dtype=np.float64)
    S_b_inv = np.array([[1/scale_b, 0, 0], [0, 1/scale_b, 0], [0, 0, 1]], dtype=np.float64)
    H_scaled = S_a @ H @ S_b_inv

    cv_b_warped = cv2.warpPerspective(cv_b_r, H_scaled, (w_a, h_a))
    # Blend A and warped B
    overlay = cv2.addWeighted(cv_a_r, 0.55, cv_b_warped, 0.45, 0)
    # Draw difference heatmap tint
    diff = cv2.absdiff(cv_a_r, cv_b_warped)
    diff_colored = cv2.applyColorMap(cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY), cv2.COLORMAP_HOT)
    overlay = cv2.addWeighted(overlay, 0.75, diff_colored, 0.25, 0)
    panel_overlay = cv_to_pil(overlay)

    # ---- Panel 4: instruction panel ----
    instr_w = w_a + w_b
    instr_h = 220
    instr_img = Image.new("RGB", (instr_w, instr_h), color=(15, 20, 35))
    draw = ImageDraw.Draw(instr_img)

    try:
        font_title = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 18)
        font_body = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 14)
    except Exception:
        font_title = ImageFont.load_default()
        font_body = font_title

    title = "Camera Alignment Instructions (move Image B camera to match Image A)"
    draw.text((16, 12), title, fill=(255, 220, 60), font=font_title)

    y = 44
    metrics = [
        f"Translation: dx={result['tx']:+.1f}px, dy={result['ty']:+.1f}px",
        f"Rotation: {result['angle_deg']:+.2f}°",
        f"Scale factor: {result['scale']:.4f}",
        f"Matched keypoints: {result['inliers']} inliers / {result['total_matches']} matches",
    ]
    for m in metrics:
        draw.text((16, y), m, fill=(160, 210, 255), font=font_body)
        y += 22

    y += 6
    draw.line([(16, y), (instr_w - 16, y)], fill=(60, 70, 100), width=1)
    y += 10

    for i, instr in enumerate(instructions, 1):
        bullet = f"  {i}. {instr}"
        draw.text((16, y), bullet, fill=(100, 255, 160), font=font_body)
        y += 22
        if y > instr_h - 20:
            break

    # ---- Draw movement arrow on overlay panel ----
    draw_ov = ImageDraw.Draw(panel_overlay)
    cx, cy = w_a // 2, h_a // 2
    arrow_end = (
        int(cx + min(result["tx"] * scale_a, w_a * 0.35)),
        int(cy + min(result["ty"] * scale_a, h_a * 0.35)),
    )
    if abs(arrow_end[0] - cx) > 3 or abs(arrow_end[1] - cy) > 3:
        draw_arrow(draw_ov, (cx, cy), arrow_end, color=(255, 80, 80), width=4, arrow_size=14)
        draw_ov.text((cx + 4, cy - 20), "Δ move", fill=(255, 80, 80), font=font_body)

    # ---- Assemble final canvas ----
    top_w = w_a + w_b  # match panel width
    overlay_padded = Image.new("RGB", (top_w, target_h), (10, 10, 10))
    # Center the overlay panel
    ox = (top_w - w_a) // 2
    overlay_padded.paste(panel_overlay, (ox, 0))

    # Label the overlay panel
    draw_op = ImageDraw.Draw(overlay_padded)
    draw_op.text((ox + 6, 6), "Overlay: A (solid) + B warped + diff heatmap", fill=(255, 220, 60), font=font_body)

    canvas_h = target_h + target_h + instr_h + 6  # match row + overlay row + instructions
    canvas = Image.new("RGB", (top_w, canvas_h), (10, 10, 10))
    canvas.paste(panel_match, (0, 0))
    canvas.paste(overlay_padded, (0, target_h + 3))
    canvas.paste(instr_img, (0, target_h * 2 + 6))

    # Label the match panel
    draw_c = ImageDraw.Draw(canvas)
    draw_c.text((8, 6), "Image A — keypoints & matches", fill=(255, 220, 60), font=font_body)
    draw_c.text((w_a + 8, 6), "Image B — keypoints & matches", fill=(255, 220, 60), font=font_body)

    return canvas


def image_to_data_url(pil_img, fmt="PNG"):
    buf = io.BytesIO()
    pil_img.save(buf, format=fmt)
    b64 = base64.b64encode(buf.getvalue()).decode()
    mime = "image/png" if fmt == "PNG" else "image/jpeg"
    return f"data:{mime};base64,{b64}"


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/analyze", methods=["POST"])
def analyze():
    if "image_a" not in request.files or "image_b" not in request.files:
        return jsonify({"error": "Both image_a and image_b are required."}), 400

    file_a = request.files["image_a"]
    file_b = request.files["image_b"]

    if not (allowed_file(file_a.filename) and allowed_file(file_b.filename)):
        return jsonify({"error": "Unsupported file type. Use PNG, JPG, WEBP, or BMP."}), 400

    try:
        img_a_pil = Image.open(file_a.stream).convert("RGB")
        img_b_pil = Image.open(file_b.stream).convert("RGB")
    except Exception as e:
        return jsonify({"error": f"Could not open image: {e}"}), 400

    img_a_cv = pil_to_cv(img_a_pil)
    img_b_cv = pil_to_cv(img_b_pil)

    kp_a, kp_b, good_matches = detect_and_match(img_a_cv, img_b_cv)

    if good_matches is None or len(good_matches) < 4:
        return jsonify({
            "error": (
                "Not enough matching keypoints found between the two images "
                f"({'0' if good_matches is None else len(good_matches)} matches). "
                "Ensure the images share overlapping content."
            )
        }), 422

    result = compute_alignment(kp_a, kp_b, good_matches, img_a_cv.shape)
    if result is None:
        return jsonify({"error": "Could not compute homography. The images may be too different."}), 422

    instructions = build_instructions(result)

    result_img = generate_result_image(img_a_pil, img_b_pil, kp_a, kp_b, good_matches, result, instructions)
    result_data_url = image_to_data_url(result_img)

    return jsonify({
        "result_image": result_data_url,
        "instructions": instructions,
        "metrics": {
            "translation_x": round(result["tx"], 2),
            "translation_y": round(result["ty"], 2),
            "rotation_deg": round(result["angle_deg"], 3),
            "scale": round(result["scale"], 4),
            "inlier_matches": result["inliers"],
            "total_matches": result["total_matches"],
        },
    })


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)
