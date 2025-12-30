# Dialogue Structure Model

This is a lightweight tool that figures out what individual lines in a chat log actually *are*—whether they're normal text, code, system logs, or just empty noise.

It's mainly built for analyzing conversations with **coding agents**. When you're trying to investigate an agent's behavior, you usually don't want to wade through pages of stack traces or random boilerplate code. This model helps you spot those "technical blocks" so you can shrink them down and focus on the actual conversation.

## Project Structure

- `model/`: The brains of the operation (classifier, features, training).
- `dataset/`: Where the data lives.
- `requirements.txt`: The stuff you need to install.

## Setup

Install the python dependencies:

```bash
pip install -r requirements.txt
```

## Usage

Throw a list of strings (lines) at it, and it'll tell you what each line looks like. It does this by looking at "shape" features (like entropy and character ratios) rather than trying to read the words.

```python
from model import classify_lines

lines = [
    "So, how does this look?",
    "def hello(): return 'world'",
    "2023-12-30 [INFO] Something happened",
    "   "
]

labels = classify_lines(lines)
# Output: ['text', 'code', 'logs', 'none']
```

## Training

If you want to train your own model, you need a dataset. The script handles the feature extraction for you, you just need to provide the raw text and the labels (see next section).

1. **Get your data ready**: Put a JSONL file in `dataset/` (see the format below).
2. **Run the training**:
   ```bash
   python -m model.train
   ```
   This will pop out a `trained_model.pkl` in the `model/` folder.

### Data Format

The trainer expects a JSONL file. Each line just needs `text` and `label`:

```json
{"text": "Is this a bug?", "label": "text"}
{"text": "Error: null pointer", "label": "logs"}
{"text": "return true;", "label": "code"}
```

**Heads up on Data**: The production weights involved a private dataset, so I haven't included that here. I've added a `sample.jsonl` so you can see what the format looks like, but for training a model for real results, you'll want to curate a few thousand lines from your own logs.

## Under the Hood

It uses a lightweight **Random Forest Classifier** that looks at statistical features rather than semantic embeddings:
- **Entropy**: Code and logs essentially look "denser" than conversational English.
- **Symbol Ratios**: Lots of brackets and semicolons usually means code.
- **Indentation**: Leading whitespace is a dead giveaway for Python or JSON.

The training works on an 80/20 split and will print out a scorecard (precision/recall) when it's done training.
