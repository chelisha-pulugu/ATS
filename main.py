import os
from flask import Flask, request, render_template, jsonify
from google import genai
import PyPDF2
from werkzeug.utils import secure_filename

# ---------------- CONFIG ----------------
UPLOAD_FOLDER = "uploads"
ALLOWED_EXTENSIONS = {"pdf"}

os.makedirs(UPLOAD_FOLDER, exist_ok=True)

client = genai.Client(
    api_key="_API_KEY_"   
)

app = Flask(__name__)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER


# ---------------- HELPERS ----------------
def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


def extract_text_from_pdf(pdf_path):
    text = ""
    with open(pdf_path, "rb") as file:
        reader = PyPDF2.PdfReader(file)
        for page in reader.pages:
            extracted = page.extract_text()
            if extracted:
                text += extracted + "\n"
    return text.strip()


def gemini_call(prompt):
    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=prompt
    )
    return response.text.strip()


# ---------------- GEMINI TASKS ----------------
def parse_resume(resume_text):
    prompt = f"""
You are a professional resume parser.

Extract and summarize:
- Skills
- Experience summary
- Education
- Tools & technologies

Resume Text:
{resume_text}

Return clean bullet points.
"""
    return gemini_call(prompt)


def parse_job_description(jd_text):
    prompt = f"""
Extract from the job description:
- Required skills
- Responsibilities
- Preferred qualifications

Job Description:
{jd_text}

Return clean bullet points.
"""
    return gemini_call(prompt)


def ats_match(parsed_resume, parsed_jd):
    prompt = f"""
You are an ATS system.

Compare resume and job description.

Resume:
{parsed_resume}

Job Description:
{parsed_jd}

STRICT FORMAT:
1. Match percentage (0-100)
2. Matching skills
3. Missing skills
4. Strengths
5. Improvement suggestions
"""
    return gemini_call(prompt)


# ---------------- ROUTES ----------------
@app.route("/")
def home():
    return render_template("index.html")


@app.route("/analyze", methods=["POST"])
def analyze():
    try:
        if "resume" not in request.files:
            return jsonify({"error": "Resume PDF is required"}), 400

        resume_file = request.files["resume"]
        jd_text = request.form.get("job_description", "").strip()

        if resume_file.filename == "" or not allowed_file(resume_file.filename):
            return jsonify({"error": "Invalid PDF file"}), 400

        if not jd_text:
            return jsonify({"error": "Job description is required"}), 400

        # Save file safely
        filename = secure_filename(resume_file.filename)
        pdf_path = os.path.join(app.config["UPLOAD_FOLDER"], filename)
        resume_file.save(pdf_path)

        # Extract resume text
        resume_text = extract_text_from_pdf(pdf_path)

        if not resume_text:
            return jsonify({"error": "Could not read PDF content"}), 400

        # Gemini processing
        parsed_resume = parse_resume(resume_text)
        parsed_jd = parse_job_description(jd_text)
        ats_result = ats_match(parsed_resume, parsed_jd)

        return jsonify({
            "parsed_resume": parsed_resume,
            "parsed_job_description": parsed_jd,
            "ats_result": ats_result
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ---------------- RUN ----------------
if __name__ == "__main__":
    app.run(debug=True, port=8080)

