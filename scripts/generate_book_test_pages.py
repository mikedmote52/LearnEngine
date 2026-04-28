"""Generate two test cases of book chapter pages for Phase B verification.

Test 1: physics textbook chapter with a diagram (Newton's laws, public domain content).
Test 2: prose chapter from a public-domain novel (Pride and Prejudice ch.1, Project Gutenberg).

Outputs base64-encoded JPEGs (data URIs) into /tmp/book_test_*.json so the test
harness can POST them to /api/analyze-book-chapter.
"""
import base64
import io
import json
import os
from PIL import Image, ImageDraw, ImageFont

OUT_DIR = "/tmp"


def find_font(size):
    candidates = [
        "/System/Library/Fonts/Supplemental/Times New Roman.ttf",
        "/System/Library/Fonts/Times.ttc",
        "/System/Library/Fonts/Supplemental/Georgia.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSerif.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/TTF/DejaVuSerif.ttf",
    ]
    for c in candidates:
        if os.path.exists(c):
            try:
                return ImageFont.truetype(c, size)
            except Exception:
                pass
    return ImageFont.load_default()


def render_page(text_lines, page_no, chapter, draw_diagram=False, draw_equation=False, w=1024, h=1320):
    img = Image.new("RGB", (w, h), color=(252, 250, 245))
    d = ImageDraw.Draw(img)
    title_font = find_font(28)
    body_font = find_font(20)
    page_font = find_font(14)

    # Header
    if page_no == 1:
        d.text((w // 2 - 200, 60), chapter, font=title_font, fill=(20, 20, 20))
        y = 130
    else:
        d.text((60, 40), chapter, font=page_font, fill=(120, 120, 120))
        y = 90
    # Page no footer
    d.text((w - 100, h - 50), "p. " + str(page_no), font=page_font, fill=(140, 140, 140))

    # Body
    for line in text_lines:
        d.text((80, y), line, font=body_font, fill=(20, 20, 20))
        y += 32

    if draw_diagram:
        # A simple "diagram": a labeled force diagram
        cx, cy = w // 2, h - 320
        d.rectangle([cx - 40, cy - 30, cx + 40, cy + 30], outline=(30, 30, 30), width=3)
        # Force arrow F (right)
        d.line([(cx + 40, cy), (cx + 180, cy)], fill=(180, 30, 30), width=4)
        d.polygon([(cx + 180, cy - 8), (cx + 200, cy), (cx + 180, cy + 8)], fill=(180, 30, 30))
        d.text((cx + 100, cy - 30), "F", font=title_font, fill=(180, 30, 30))
        # Acceleration arrow a (right, longer)
        d.line([(cx + 40, cy + 60), (cx + 220, cy + 60)], fill=(30, 30, 180), width=4)
        d.polygon([(cx + 220, cy + 52), (cx + 240, cy + 60), (cx + 220, cy + 68)], fill=(30, 30, 180))
        d.text((cx + 130, cy + 30), "a", font=title_font, fill=(30, 30, 180))
        # Caption
        d.text((cx - 200, cy + 110), "Fig. " + str(page_no) + ".1: Force and resulting acceleration of a body of mass m.", font=page_font, fill=(60, 60, 60))

    if draw_equation:
        d.text((w // 2 - 60, h - 200), "F = m · a", font=ImageFont.truetype(find_font(40).path if hasattr(find_font(40), 'path') else "", 40) if hasattr(find_font(40), 'path') else find_font(40), fill=(20, 20, 20))

    return img


def to_data_uri(img, quality=85):
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=quality, optimize=True)
    return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode("ascii")


# ============== Test 1: physics textbook with diagrams + equations ==============

physics_pages = [
    {
        "page_no": 1,
        "chapter": "Chapter 4 — Newton's Laws of Motion",
        "draw_diagram": False, "draw_equation": False,
        "text": [
            "Sir Isaac Newton (1643–1727) formulated three laws that together describe",
            "the relationship between motion and the forces that produce it. These laws",
            "form the cornerstone of classical mechanics and remain accurate for objects",
            "moving at speeds far less than the speed of light.",
            "",
            "4.1 The First Law of Motion",
            "",
            "Often called the law of inertia, Newton's first law states that an object",
            "at rest will remain at rest, and an object in uniform motion will continue",
            "in uniform motion in a straight line, unless acted upon by an external,",
            "unbalanced force.",
            "",
            "An important consequence is that an object's natural state is to maintain",
            "its current velocity. Only an unbalanced force can change that velocity.",
            "Friction, gravity, and normal forces are the most common forces in",
            "everyday experience that are responsible for changes in motion.",
        ],
    },
    {
        "page_no": 2,
        "chapter": "Chapter 4 — Newton's Laws of Motion",
        "draw_diagram": True, "draw_equation": False,
        "text": [
            "4.2 The Second Law of Motion",
            "",
            "Newton's second law quantifies how a net force acting on a body changes",
            "the body's velocity. The law is most commonly written as a relationship",
            "between force, mass, and acceleration. The acceleration produced by a",
            "net force on a body is directly proportional to the magnitude of the net",
            "force, in the same direction as the net force, and inversely proportional",
            "to the mass of the body.",
            "",
            "When a single force F acts on a body of mass m, the resulting acceleration",
            "a points in the same direction as F. Doubling the force doubles the",
            "acceleration; doubling the mass halves the acceleration for the same force.",
        ],
    },
    {
        "page_no": 3,
        "chapter": "Chapter 4 — Newton's Laws of Motion",
        "draw_diagram": False, "draw_equation": True,
        "text": [
            "Mathematically the second law is expressed as the equation shown below.",
            "Force is measured in newtons (N), where one newton equals one kilogram",
            "meter per second squared. Mass is measured in kilograms, and acceleration",
            "in meters per second squared.",
            "",
            "When multiple forces act on a body, the net force is the vector sum of",
            "all the individual forces. Only this net force determines the body's",
            "acceleration. Internal forces between parts of the same body cancel",
            "according to Newton's third law and do not appear in the net.",
        ],
    },
    {
        "page_no": 4,
        "chapter": "Chapter 4 — Newton's Laws of Motion",
        "draw_diagram": False, "draw_equation": False,
        "text": [
            "4.3 The Third Law of Motion",
            "",
            "Newton's third law states that for every action there is an equal and",
            "opposite reaction. Forces always come in pairs: when body A exerts a",
            "force on body B, body B exerts an equal force back on body A in the",
            "opposite direction.",
            "",
            "A common misconception is that these forces cancel out. They do not,",
            "because they act on different bodies. The action acts on body B, while",
            "the reaction acts on body A. To find the net force on a single body, only",
            "the forces acting on that body are summed.",
            "",
            "Example: when you push against a wall, the wall pushes back on you with",
            "an equal force. The wall does not move because of friction with the floor",
            "and its connection to the building, not because the third-law pair cancels.",
        ],
    },
    {
        "page_no": 5,
        "chapter": "Chapter 4 — Newton's Laws of Motion",
        "draw_diagram": False, "draw_equation": False,
        "text": [
            "4.4 Summary and Applications",
            "",
            "Newton's three laws together provide a complete description of motion",
            "for objects much larger than atoms and slower than the speed of light.",
            "The laws apply to projectile motion, planetary orbits, machine design,",
            "and countless engineering problems.",
            "",
            "Key takeaways:",
            "  - Inertia: objects resist changes to their velocity.",
            "  - Force-mass-acceleration: F = m·a quantifies how forces change motion.",
            "  - Action-reaction: forces always come in equal and opposite pairs",
            "    acting on different bodies.",
            "",
            "In the next chapter, we apply these laws to specific systems including",
            "inclined planes, pulleys, and friction-dominated motion.",
        ],
    },
]

# ============== Test 2: prose chapter (Pride and Prejudice ch.1) ==============

prose_text = """It is a truth universally acknowledged, that a single man in possession
of a good fortune, must be in want of a wife.

However little known the feelings or views of such a man may be on his
first entering a neighbourhood, this truth is so well fixed in the minds
of the surrounding families, that he is considered the rightful property
of some one or other of their daughters.

"My dear Mr. Bennet," said his lady to him one day, "have you heard
that Netherfield Park is let at last?"

Mr. Bennet replied that he had not.

"But it is," returned she; "for Mrs. Long has just been here, and she
told me all about it."

Mr. Bennet made no answer.

"Do not you want to know who has taken it?" cried his wife impatiently.

"YOU want to tell me, and I have no objection to hearing it."

This was invitation enough.

"Why, my dear, you must know, Mrs. Long says that Netherfield is taken
by a young man of large fortune from the north of England; that he came
down on Monday in a chaise and four to see the place, and was so much
delighted with it that he agreed with Mr. Morris immediately; that he
is to take possession before Michaelmas, and some of his servants are
to be in the house by the end of next week."

"What is his name?"

"Bingley."

"Is he married or single?"

"Oh! single, my dear, to be sure! A single man of large fortune; four
or five thousand a year. What a fine thing for our girls!"

"How so? how can it affect them?"

"My dear Mr. Bennet," replied his wife, "how can you be so tiresome!
You must know that I am thinking of his marrying one of them."

"Is that his design in settling here?"

"Design! nonsense, how can you talk so! But it is very likely that he
MAY fall in love with one of them, and therefore you must visit him as
soon as he comes."
"""
prose_lines = [l for l in prose_text.split("\n")]

prose_pages = []
chunk_size = 16
for pi, start in enumerate(range(0, len(prose_lines), chunk_size)):
    chunk = prose_lines[start:start + chunk_size]
    prose_pages.append({
        "page_no": pi + 1,
        "chapter": "Pride and Prejudice — Chapter 1",
        "draw_diagram": False, "draw_equation": False,
        "text": chunk,
    })


def render_set(pages):
    return [to_data_uri(render_page(p["text"], p["page_no"], p["chapter"], p["draw_diagram"], p["draw_equation"])) for p in pages]


physics_imgs = render_set(physics_pages)
prose_imgs = render_set(prose_pages)

with open(os.path.join(OUT_DIR, "book_test_physics.json"), "w") as f:
    json.dump({"images": physics_imgs, "chapter_title": "Newton's Laws of Motion"}, f)
with open(os.path.join(OUT_DIR, "book_test_prose.json"), "w") as f:
    json.dump({"images": prose_imgs, "chapter_title": "Pride and Prejudice — Chapter 1"}, f)

print("physics pages:", len(physics_imgs), "first uri len:", len(physics_imgs[0]))
print("prose pages:", len(prose_imgs), "first uri len:", len(prose_imgs[0]))
