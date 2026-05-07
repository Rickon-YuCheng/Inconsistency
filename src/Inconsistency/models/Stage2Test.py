import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from torch.utils.data import DataLoader
from tqdm import tqdm
import warnings
import csv
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

from Stage1Tr import daicwoz_dataset, collate_fn
from Stage2Main import whole_model, D_MODEL, NHEAD

warnings.filterwarnings("ignore", category=FutureWarning)


WEIGHT_PATH = "stage2BestWeights.pth"
SAVE_RESULT_PATH = "stage2TestResult.csv"


def test(model, test_loader, loss_dep, device):
    model.eval()

    totDepLoss = 0.0
    valid_batches = 0

    patient_arr = []
    true_arr = []
    pred_mean_arr = []
    pred_vote_arr = []
    prob_arr = []
    feature_arr = []


    pbar = tqdm(
        test_loader,
        desc="Testing",
        leave=False,
        unit="batch",
    )

    with torch.inference_mode():
        for data in pbar:
            if data is None:
                continue

            xa, xt, aMask, tMask, atei_label, dep_label, Patient = data
            xa = xa.to(device)
            xt = xt.to(device)
            aMask = aMask.to(device)
            tMask = tMask.to(device)
            dep_label = dep_label.to(device)

            with torch.autocast(
                device_type="cuda",
                enabled=(device == "cuda"),
            ):
                _, dep_logits, feature = model(xa, xt, aMask, tMask, return_feature=True)

                patient_dep = dep_logits.mean(dim=0)

                # gpt
                feat = feature.detach().cpu().float()

                # 如果是 [T, 128]
                if feat.dim() == 2:
                    feat = feat.mean(dim=0)

                # 如果是 [1,128]（其實這個也會被處理成一樣）
                # elif feat.dim() == 1:
                #     pass

                feature_arr.append(feat.numpy())
                # ===

                L_Depression = loss_dep(
                    patient_dep.unsqueeze(0),
                    dep_label.unsqueeze(0),
                )

            dep_prob = torch.softmax(patient_dep, dim=-1)

            dep_pred_mean = patient_dep.argmax(dim=-1)
            dep_pred_vote = majority_vote(dep_logits)

            totDepLoss += L_Depression.item()
            valid_batches += 1

            patient_arr.append(int(Patient))
            true_arr.append(int(dep_label.item()))
            pred_mean_arr.append(int(dep_pred_mean.item()))
            pred_vote_arr.append(int(dep_pred_vote.item()))
            prob_arr.append(dep_prob.detach().cpu().numpy())

            cur_metrics = get_metrics(true_arr, pred_vote_arr)

            pbar.set_postfix({
                "dep_loss": totDepLoss / valid_batches,
                "vote_acc": cur_metrics["acc"],
                "vote_f1": cur_metrics["f1"],
            })

    avg_dep_loss = totDepLoss / max(valid_batches, 1)
    mean_metrics = get_metrics(true_arr, pred_mean_arr)
    vote_metrics = get_metrics(true_arr, pred_vote_arr)

    return {
        "dep_loss": avg_dep_loss,

        "mean_acc": mean_metrics["acc"],
        "mean_pre": mean_metrics["pre"],
        "mean_rec": mean_metrics["rec"],
        "mean_f1": mean_metrics["f1"],

        "vote_acc": vote_metrics["acc"],
        "vote_pre": vote_metrics["pre"],
        "vote_rec": vote_metrics["rec"],
        "vote_f1": vote_metrics["f1"],

        "patient": np.array(patient_arr),
        "true": np.array(true_arr),
        "pred_mean": np.array(pred_mean_arr),
        "pred_vote": np.array(pred_vote_arr),
        "prob": np.array(prob_arr),

        "feature": np.stack(feature_arr, axis=0),
    }

def get_metrics(y_true, y_pred):
    return {
        "acc": accuracy_score(y_true, y_pred),
        "pre": precision_score(y_true, y_pred, average="macro", zero_division=0),
        "rec": recall_score(y_true, y_pred, average="macro", zero_division=0),
        "f1": f1_score(y_true, y_pred, average="macro", zero_division=0),
    }


def majority_vote(dep_logits):
    seg_pred = dep_logits.argmax(dim=-1)

    if seg_pred.dim() == 0:
        return seg_pred

    return torch.mode(seg_pred).values

def plot_tsne(features, labels):
    features = np.asarray(features)
    labels = np.asarray(labels)

    if len(features) < 5:
        print("[t-SNE] samples too few, skip.")
        return

    perplexity = min(30, max(2, len(features) // 3))

    z = TSNE(
        n_components=2,
        perplexity=perplexity,
        learning_rate="auto",
        init="pca",
        random_state=42,
    ).fit_transform(features)

    names = {
        0: "Healthy",
        1: "Mild",
        2: "Moderate",
    }

    markers = {
        0: "o",
        1: "^",
        2: "s",
    }

    plt.figure(figsize=(6, 5))

    for cls in [0, 1, 2]:
        idx = labels == cls
        plt.scatter(
            z[idx, 0],
            z[idx, 1],
            marker=markers[cls],
            label=names[cls],
            alpha=0.85,
            s=35,
        )

    plt.legend()
    plt.title("t-SNE of Final Hidden Feature")
    plt.xticks([])
    plt.yticks([])
    plt.tight_layout()
    plt.savefig("tsne_test.png", dpi=300, bbox_inches="tight")
    plt.close()

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"

    testDS = daicwoz_dataset(fold="test")

    test_loader = DataLoader(
        testDS,
        batch_size=1,
        shuffle=False,
        collate_fn=collate_fn,
    )

    print("=" * 80)
    print(f"Device      : {device}")
    print(f"Test samples: {len(testDS)}")
    print(f"Weight path : {WEIGHT_PATH}")
    print("=" * 80)
    

    model = whole_model(D_MODEL, NHEAD).to(device)

    model.load_state_dict(
        torch.load(
            WEIGHT_PATH,
            map_location=device,
        )
    )

    loss_dep = nn.CrossEntropyLoss()

    result = test(
        model=model,
        test_loader=test_loader,
        loss_dep=loss_dep,
        device=device,
    )

    report_text = classification_report(
        result["true"],
        result["pred_mean"],
        labels=[0, 1, 2],
        target_names=["class_0", "class_1", "class_2"],
        zero_division=0,
    )

    cm = confusion_matrix(
        result["true"],
        result["pred_mean"],
        labels=[0, 1, 2]
    )

    print("\n[Classification Report]")
    print(report_text)

    print("\n[Confusion Matrix]")
    print(cm)

    print("=" * 80)
    print(f"Test Dep Loss: {result['dep_loss']:.4f}")

    print(
        f"[Mean Logits] "
        f"Acc: {result['mean_acc']:.4f} | "
        f"Pre: {result['mean_pre']:.4f} | "
        f"Rec: {result['mean_rec']:.4f} | "
        f"F1: {result['mean_f1']:.4f}"
    )

    print(
        f"[Vote]        "
        f"Acc: {result['vote_acc']:.4f} | "
        f"Pre: {result['vote_pre']:.4f} | "
        f"Rec: {result['vote_rec']:.4f} | "
        f"F1: {result['vote_f1']:.4f}"
    )

    print("=" * 80)

    plot_tsne(result["feature"], result["true"])

    print("=" * 80)

    with open(SAVE_RESULT_PATH, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)

        writer.writerow([
            "patient",
            "ground_truth",
            "pred_mean",
            "pred_vote",
            "prob_0",
            "prob_1",
            "prob_2",
        ])

        for patient, true, pred_mean, pred_vote, prob in zip(
            result["patient"],
            result["true"],
            result["pred_mean"],
            result["pred_vote"],
            result["prob"],
        ):
            writer.writerow([
                int(patient),
                int(true),
                int(pred_mean),
                int(pred_vote),
                float(prob[0]),
                float(prob[1]),
                float(prob[2]),
            ])

if __name__ == "__main__":
    main()