import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
import warnings
import csv

from Stage1Tr import daicwoz_dataset, collate_fn
from Stage2Main import whole_model, D_MODEL, NHEAD

warnings.filterwarnings("ignore", category=FutureWarning)


WEIGHT_PATH = "stage2BestWeights.pth"
SAVE_RESULT_PATH = "stage2TestResult.csv"


def test(model, test_loader, loss_dep, device):
    model.eval()

    totDepLoss = 0.0
    correct_dep = 0
    valid_batches = 0

    patient_arr = []
    true_arr = []
    pred_arr = []
    prob_arr = []

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
                _, dep_logits = model(xa, xt, aMask, tMask)

                patient_dep = dep_logits.mean(dim=0)

                L_Depression = loss_dep(
                    patient_dep.unsqueeze(0),
                    dep_label.unsqueeze(0),
                )

            dep_prob = torch.softmax(patient_dep, dim=-1)
            dep_pred = patient_dep.argmax(dim=-1)

            totDepLoss += L_Depression.item()
            correct_dep += int(dep_pred.item() == dep_label.item())
            valid_batches += 1

            # 如果你的 collate_fn 沒有回傳 patient，這裡先用 batch index 代替
            patient_arr.append(int(Patient))
            true_arr.append(int(dep_label.item()))
            pred_arr.append(int(dep_pred.item()))
            prob_arr.append(dep_prob.detach().cpu().numpy())

            pbar.set_postfix({
                "dep_loss": totDepLoss / valid_batches,
                "dep_acc": correct_dep / valid_batches,
            })

    avg_dep_loss = totDepLoss / max(valid_batches, 1)
    dep_acc = correct_dep / max(valid_batches, 1)

    return {
        "dep_loss": avg_dep_loss,
        "dep_acc": dep_acc,
        "patient": np.array(patient_arr),
        "true": np.array(true_arr),
        "pred": np.array(pred_arr),
        "prob": np.array(prob_arr),
    }


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

    print("=" * 80)
    print(f"Test Dep Loss: {result['dep_loss']:.4f}")
    print(f"Test Dep Acc : {result['dep_acc']:.4f}")
    print("=" * 80)

    with open(SAVE_RESULT_PATH, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)

        writer.writerow([
            "patient",
            "ground_truth",
            "prediction",
            "prob_0",
            "prob_1",
            "prob_2",
        ])

        for patient, true, pred, prob in zip(
            result["patient"],
            result["true"],
            result["pred"],
            result["prob"],
        ):
            writer.writerow([
                int(patient),
                int(true),
                int(pred),
                float(prob[0]),
                float(prob[1]),
                float(prob[2]),
            ])

if __name__ == "__main__":
    main()