def health_status(predicted_rul):
    if predicted_rul <= 10:
        return "critical"
    elif predicted_rul <= 30:
        return "warning"
    else:
        return "healthy"
