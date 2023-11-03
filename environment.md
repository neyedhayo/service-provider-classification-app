# Python Virtual Environment (with Venv Environment)

**Description:**
Creating a new vitual environment for a new machine learning project is good practice.

A step-by-step how to;

**Step 1:** Creating the new environment

```bash
python3 -m venv my_virtualenv
```

---

## Running the FastAPI

**Description:**
Running the fastapi application;

```bash
uvicorn app.main:app --reload
```

**`"OR without reload"`**

```bash
uvicorn app.main:app 
```
