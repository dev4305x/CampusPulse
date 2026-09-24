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

## ✅ Verified Environment

- Python 3.12
- Streamlit
- Scikit-learn
- Pandas
- SQLite
