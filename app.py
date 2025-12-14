from flask import Flask, render_template, request
import torch
import torch.nn as nn
from transformers import BertTokenizerFast, AutoModel

app = Flask(__name__)

# ======================
# DEVICE
# ======================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ======================
# LOAD MODEL & TOKENIZER
# ======================
tokenizer = BertTokenizerFast.from_pretrained("bert-base-uncased")
bert = AutoModel.from_pretrained("bert-base-uncased")

class BERT_Arch(nn.Module):
    def __init__(self, bert):
        super().__init__()
        self.bert = bert
        self.dropout = nn.Dropout(0.1)
        self.fc1 = nn.Linear(768, 512)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(512, 2)
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, sent_id, mask):
        cls = self.bert(sent_id, attention_mask=mask)['pooler_output']
        x = self.fc1(cls)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        return self.softmax(x)

model = BERT_Arch(bert)
model.load_state_dict(torch.load("model/c2_new_model_weights.pt", map_location=device))
model.to(device)
model.eval()

# ======================
# ROUTES
# ======================
@app.route("/", methods=["GET", "POST"])
def index():
    prediction = None
    confidence = None

    if request.method == "POST":
        text = request.form["news"]

        tokens = tokenizer(
            text,
            max_length=15,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )

        with torch.no_grad():
            outputs = model(
                tokens["input_ids"].to(device),
                tokens["attention_mask"].to(device)
            )

        probs = torch.exp(outputs)
        conf, pred = torch.max(probs, dim=1)

        prediction = "FAKE" if pred.item() == 1 else "TRUE"
        confidence = round(conf.item() * 100, 2)

    return render_template("index.html",
                prediction=prediction,
                confidence=confidence)

if __name__ == "__main__":
    app.run(debug=True)