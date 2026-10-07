# Troubleshooting

- If packages are missing: activate `.venv` and run `pip install -r requirements.txt`.
- If the model is missing: run `python scripts/train_model.py`.
- If port 5000 is busy: change the port in `scripts/run_app.py`.
- If validation fails: use supported numeric ranges and categories shown by the web form.
