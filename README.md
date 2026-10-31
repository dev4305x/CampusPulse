# CampusPulse

### AI-Powered Student Feedback Intelligence

> From student voices to actionable campus insights.

CampusPulse is a working AI-powered student feedback analysis prototype.

## 🚀 Quick Start

### 1. Clone the Repository

```bash
git clone https://github.com/dev4305x/CampusPulse.git
cd CampusPulse
```

### 2. Create Virtual Environment

```bash
python -m venv venv
```

Activate it:

**Windows PowerShell**
```powershell
.\venv\Scripts\Activate.ps1
```

If required:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\venv\Scripts\Activate.ps1
```

### 3. Install Dependencies

```bash
pip install -r campuspulse_requirements.txt
```

### 4. Launch the Application

```bash
streamlit run campuspulse_app.py
```

The CampusPulse dashboard will open in your browser.

---

## 🧪 Testing the Prototype

### Student Feedback

1. Open **Student Feedback**.
2. Enter a natural-language complaint or suggestion.
3. Optionally enter a location.
4. Click **Analyze & Submit**.
5. Check the generated:
   - Sentiment
   - Category
   - Confidence
   - Fine-grained aspect
   - Related complaints

### Try These Examples

```text
C Block Wi-Fi keeps disconnecting every night after 7 PM.
```

```text
I'm really tired of having exams scheduled so frequently.
```

```text
The library is fine but there are no seats during evening hours.
```

```text
The laboratory PCs are extremely slow.
```

### Admin Dashboard

Open **Admin Intelligence** to view:

- Feedback statistics
- Sentiment distribution
- Category distribution
- Recurring issues
- Feedback trends
- Fine-grained aspects
- Live submissions

---

## Real campus feedback training

The curated Google Form export is stored at `data/campuspulse_real_feedback_curated.csv`. It contains the 27 responses, with timestamps and other form fields kept as metadata. The training script uses only the `feedback` text as a model feature.

Two Positive-labeled narratives that describe serious problems are marked `pending_human_relabel` and are excluded from candidate training. One exact text duplicate (`Bad`) already exists in the public labeled dataset; it is excluded from the new-response evaluation to prevent optimistic results.

Run the reproducible candidate training and evaluation with:

```bash
python train_campuspulse_real_feedback.py
```

This writes candidate sentiment and category models under `campuspulse_models/` and metrics to `data/campuspulse_real_feedback_evaluation.json`. The candidate files do not replace the live V4 models. The accompanying Markdown report explains the evaluation and promotion decision. The 24-row out-of-fold evaluation is descriptive only; it is too small and too sparse in positive sentiment and rare categories to establish campus-wide performance.

---

## ✅ Verified Environment

- Python 3.12
- Streamlit
- Scikit-learn
- Pandas
- SQLite
