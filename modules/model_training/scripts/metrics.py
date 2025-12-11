from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

# Your file path
log_file = "/Users/noahbruderer/local_work_files/rosette_paper/results_old/model_training_old/champion_baseline/events.out.tfevents.1756721030.Noahs-MacBook-Pro.local.84474.0"

# Load the file
event_acc = EventAccumulator(log_file)
event_acc.Reload()

# Get the metrics
# Note: Based on your previous code, the tags should be "IoU/validation" and "F1_Score/validation"
try:
    f1_scores = event_acc.Scalars("F1_Score/validation")
    iou_scores = event_acc.Scalars(
        "IoU/validation"
    )  # Dice is often IoU or very close to it in these logs

    # Find the max score (best epoch)
    best_f1 = max(s.value for s in f1_scores)
    best_iou = max(s.value for s in iou_scores)

    print(f"BEST F1 SCORE: {best_f1:.4f}")
    print(f"BEST IoU (Dice): {best_iou:.4f}")

    # Check at which step (epoch) the best F1 happened
    best_step = [s.step for s in f1_scores if s.value == best_f1][0]
    print(f"Achieved at Epoch: {best_step}")

except KeyError:
    print("Could not find exact tags. Available tags:", event_acc.Tags()["scalars"])
