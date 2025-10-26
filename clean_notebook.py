import json

notebook_file = "Face_expression_recognition.ipynb"
cleaned_file = "Face_expression_recognition_notebook.ipynb"

with open(notebook_file, "r", encoding="utf-8") as f:
    data = json.load(f)

# Check if widgets exist in metadata
widgets = data.get("metadata", {}).get("widgets", {})

# Add missing "state" key to each widget
for widget_id, widget_data in widgets.items():
    if "state" not in widget_data:
        widget_data["state"] = {}

# Save cleaned notebook
with open(cleaned_file, "w", encoding="utf-8") as f:
    json.dump(data, f, indent=2)

print(f"Notebook cleaned and saved as {cleaned_file}")
