def evaluate_model(model, y_test, y_pred, y_pred_proba, model_name, training_time):
    """
    Calculate all evaluation metrics for a trained model
    Args:
        model: trained model
        y_test: true labels
        y_pred: predicted labels
        y_pred_proba: predicted probabilities
        model_name: name of the model
        training_time: time taken to train
    """
    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_pred_proba)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    mcc = matthews_corrcoef(y_test, y_pred)

    # Calculate confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    tn, fp, fn, tp = cm.ravel()

    # Print results
    print(f"\n{'='*60}")
    print(f"Evaluation: {model_name}")
    print('='*60)
    print(f"Training Time: {training_time:.4f} seconds")
    print(f"Accuracy:  {accuracy:.4f}")
    print(f"AUC Score: {auc:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1 Score:  {f1:.4f}")
    print(f"MCC Score: {mcc:.4f}")
    print(f"Confusion Matrix: TN={tn}, FP={fp}, FN={fn}, TP={tp}")
    print('='*60)

    return {
        'Model': model_name,
        'Accuracy': accuracy,
        'AUC Score': auc,
        'Precision': precision,
        'Recall': recall,
        'F1 Score': f1,
        'MCC Score': mcc,
        'Training Time': training_time,
        'Confusion Matrix': cm,
        'TN': tn, 'FP': fp, 'FN': fn, 'TP': tp
    }

# Initialize storage for results and predictions
results = []
trained_models = {}
model_predictions = {}  # Store predictions for each model

print("✓ Evaluation function defined")
print("✓ Ready to train models")
